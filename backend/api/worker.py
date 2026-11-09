"""One bounded local job per OS process; never runs model-generated code directly."""
from __future__ import annotations
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time


def execute(request):
    kind, payload, config = request['kind'], dict(request['payload']), request['config']
    root = Path(request['root'])
    if kind == 'quantization':
        from backend.quantization.gguf_adapter import convert
        source = Path(payload['source']).resolve(strict=True)
        model_root = Path(config['model_root']).resolve()
        if not source.is_relative_to(model_root):
            raise ValueError('Source must be inside configured model directory')
        artifacts = root / 'artifacts'
        artifacts.mkdir(exist_ok=True)
        return convert(source, artifacts / request['id'], target=payload['target'],
                       executable=config['quantizer'], timeout=config['job_timeout'])
    if kind in {'benchmark', 'evaluation'}:
        selection = payload.pop('backend', 'ollama')
        artifact_job_id = payload.pop('artifact_job_id', None)
        if selection == 'llama.cpp':
            from backend.api.jobs import Jobs
            from backend.core.artifacts import inspect_gguf
            from backend.backends.llama_cpp import LlamaCpp
            artifact_job = Jobs(root, config).get(artifact_job_id)
            if artifact_job['kind'] != 'quantization' or artifact_job['status'] != 'completed':
                raise ValueError('Artifact job must be a completed quantization')
            manifest = artifact_job['result']
            artifact_path = Path(manifest['artifact']['path']).resolve(strict=True)
            if not artifact_path.is_relative_to(root / 'artifacts'):
                raise ValueError('Artifact must be inside managed artifact storage')
            inspected = inspect_gguf(artifact_path)
            if inspected['sha256'] != manifest['artifact']['sha256']:
                raise ValueError('Artifact changed after conversion')
            backend = LlamaCpp(artifact_path, executable=config['server_executable'], context=payload['context'],
                               source_sha256=manifest['source']['sha256'],
                               log_path=root / 'jobs' / request['id'] / 'llama-server.log')
            payload['model'] = backend.models()[0]['name']
        elif selection == 'ollama':
            from backend.backends.ollama import Ollama
            backend = Ollama(config['ollama_url'])
        else:
            raise ValueError('Unsupported inference backend')
        try:
            if kind == 'benchmark':
                from backend.storage.database import Database
                from backend.benchmarks.runner import run
                db = Database(root / 'experiments.sqlite3')
                try:
                    return run(backend, db, **payload)
                finally:
                    db.connection.close()
            from backend.evaluation.runner import evaluate
            from backend.storage.evaluations import EvaluationStore
            result = evaluate(backend, **payload)
            EvaluationStore(root / 'experiments.sqlite3').save(result)
            return result
        finally:
            backend.close()
    raise ValueError('Unsupported job kind')


def main():
    path = Path(sys.argv[1])
    request = json.loads(path.read_text())
    os.environ['FEATHER_JOB_ID'] = request['id']
    def terminate(signum, frame):
        raise KeyboardInterrupt('Job cancelled or API stopping')
    signal.signal(signal.SIGTERM, terminate)
    parent_pid = request['parent_pid']
    def watch_parent():
        while True:
            time.sleep(.25)
            if os.getppid() != parent_pid:
                os.killpg(os.getpgrp(), signal.SIGKILL)
    threading.Thread(target=watch_parent, daemon=True).start()
    try:
        result = execute(request)
        (path.parent / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    except BaseException as error:
        (path.parent / 'error.json').write_text(json.dumps({'error': f'{type(error).__name__}: {error}'}))
        raise


if __name__ == '__main__':
    main()
