"""Owned localhost llama.cpp server with verified local artifact identity."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import socket
import subprocess
import time
import httpx
from backend.core.artifacts import inspect_gguf, sha256


class LlamaCpp:
    name = 'llama.cpp'
    prompt_template = 'chatml-v1: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n'

    def __init__(self, artifact, executable='llama-server', context=1024,
                 source_sha256=None, log_path='.feather/llama-server.log', startup_timeout=120):
        self.artifact = inspect_gguf(artifact)
        self.source_sha256 = source_sha256
        self.context = context
        binary = shutil.which(str(executable))
        if not binary:
            raise ValueError('llama-server unavailable; supply --server-executable')
        if not 128 <= context <= 4096:
            raise ValueError('Managed server context must be between 128 and 4096')
        self.binary = binary
        self.executable_sha256 = sha256(binary)
        self.benchmark_settings = {"prompt_template": self.prompt_template, "cache_prompt": False, "threads": 2, "parallel": 1, "server_executable_sha256": self.executable_sha256}
        version = subprocess.run([binary, '--version'], capture_output=True, text=True, timeout=10, check=True)
        self._version = (version.stdout + version.stderr).strip()
        if not self._version:
            raise ValueError('llama-server returned no build version')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.log = path.open('w')
        self.command = [binary, '-m', self.artifact['path'], '--host', '127.0.0.1', '--port', str(port),
                        '-c', str(context), '-np', '1', '-t', '2', '--no-warmup']
        self.process = subprocess.Popen(self.command, stdout=self.log, stderr=subprocess.STDOUT)
        self.client = httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=120)
        deadline = time.monotonic() + startup_timeout
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f'llama-server exited {self.process.returncode}; inspect {path}')
                try:
                    if self.client.get('/health').status_code == 200:
                        return
                except httpx.TransportError:
                    pass
                time.sleep(.1)
            raise TimeoutError(f'llama-server startup timed out; inspect {path}')
        except BaseException:
            self.close()
            raise

    def models(self):
        return [{'name': Path(self.artifact['path']).name, 'size': self.artifact['size_bytes'],
                 'digest': self.artifact['sha256'], 'artifact_sha256': self.artifact['sha256'],
                 'details': {'tensor_types': self.artifact['tensor_types'], 'parameter_count': self.artifact['parameter_count']}}]

    def version(self):
        return self._version

    def generate(self, model, prompt, options):
        if model != self.models()[0]['name'] or options.get('num_ctx', self.context) != self.context:
            raise ValueError('Model or context does not match managed server')
        started = time.perf_counter()
        first = None
        output = []
        final = None
        # Shared explicit template for baseline and quantized variants; not inferred from model name.
        formatted = f'<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n'
        body = {'prompt': formatted, 'stream': True, 'n_predict': options.get('num_predict', 128),
                'seed': options.get('seed', 42), 'temperature': options.get('temperature', 0),
                'cache_prompt': False, 'stop': ['<|im_end|>', '<|im_start|>']}
        with self.client.stream('POST', '/completion', json=body) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith('data: '):
                    continue
                item = json.loads(line[6:])
                if item.get('error'):
                    raise RuntimeError(str(item['error']))
                if item.get('content'):
                    if first is None:
                        first = time.perf_counter() - started
                    output.append(item['content'])
                if item.get('stop'):
                    final = item
        if final is None:
            raise RuntimeError('Incomplete llama.cpp stream')
        timings = final.get('timings', {})
        raw = {'eval_count': timings.get('predicted_n'), 'eval_duration': timings.get('predicted_ms', 0) * 1e6,
               'prompt_eval_count': timings.get('prompt_n'), 'prompt_eval_duration': timings.get('prompt_ms', 0) * 1e6,
               'native': final}
        return {'output': ''.join(output), 'ttft_seconds': first, 'wall_seconds': time.perf_counter() - started, 'runtime': raw}

    def close(self):
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        finally:
            self.client.close()
            self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
