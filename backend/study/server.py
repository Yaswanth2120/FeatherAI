"""One llama-server session under /usr/bin/time -l with load latency and stage memory snapshots.

Implements the generate()/models()/version() interface used by the evaluation runner, so
quality and performance measurements share the same server configuration.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time

import httpx
import psutil

from backend.core.artifacts import sha256
from backend.diagnostics.memory import RssSampler, delta, footprint, parse_time_l, system_snapshot

TEMPLATE = 'chatml-v1: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n'
RUNTIME_OVERHEAD_BYTES = 512 * 1024 ** 2


def runtime_admission(artifact: dict) -> dict:
    """Pre-flight estimate for llama-server with --load-mode none at context <= 2048.

    Basis: weights are fully resident (artifact size) plus a 512 MiB allowance for KV cache, compute
    buffers and runtime. Measured SmolLM2 F16 overhead at context 2048 was 214 MiB
    (472 MiB peak footprint - 258 MiB artifact).
    """
    return {"bytes": artifact["size_bytes"] + RUNTIME_OVERHEAD_BYTES, "rule": "artifact size + 512 MiB (v0.2)"}


def docker_vm_running() -> bool:
    return any("com.docker.virtualization" in " ".join(p.info.get("cmdline") or [])
               for p in psutil.process_iter(["cmdline"]))


def metal_evidence(text: str, offloaded) -> dict:
    device = re.search(r"ggml_metal_device_init: GPU name:\s*(\S+) \(([^)]+)\)", text)
    layers = [int(offloaded.group(1)), int(offloaded.group(2))] if offloaded else None
    return {"device": device.group(2) if device else None, "offloaded_layers": layers,
            "verified": bool(device and layers and layers[0] == layers[1])}


def buffer_sizes(text: str) -> dict:
    sizes = {}
    for device, size in re.findall(r"(\w+?)(?:_Mapped)? model buffer size =\s*([\d.]+) MiB", text):
        sizes[f"model_{device}"] = sizes.get(f"model_{device}", 0) + float(size)
    for device, size in re.findall(r"(\w+) KV buffer size =\s*([\d.]+) MiB", text):
        sizes[f"kv_{device}"] = sizes.get(f"kv_{device}", 0) + float(size)
    for device, size in re.findall(r"(\w+) compute buffer size =\s*([\d.]+) MiB", text):
        sizes[f"compute_{device}"] = sizes.get(f"compute_{device}", 0) + float(size)
    kv = re.search(r"llama_kv_cache: size =.*?\((\s*\d+) cells.*?K \((\w+)\).*?V \((\w+)\)", text)
    if kv:
        sizes["kv_cells"], sizes["kv_type_k"], sizes["kv_type_v"] = int(kv.group(1)), kv.group(2), kv.group(3)
    return sizes


class ServerSession:
    name = "llama.cpp"
    prompt_template = TEMPLATE

    def __init__(self, artifact: dict, binary: str, log_path: Path, context=2048, threads=4, load_mode="none",
                 source_sha256: str | None = None, startup_timeout=180, request_timeout=300):
        self.artifact = artifact
        self.binary = str(Path(binary).resolve())
        self.log_path = Path(log_path)
        self.context, self.threads, self.load_mode = context, threads, load_mode
        self.source_sha256 = source_sha256
        self.startup_timeout, self.request_timeout = startup_timeout, request_timeout
        self.settings = {"context": context, "threads": threads, "load_mode": load_mode, "parallel": 1, "cache_ram_mib": 0,
                         "fit": "off", "server_warmup": False, "n_gpu_layers": "default (all)", "prompt_template": TEMPLATE,
                         "executable_sha256": sha256(self.binary)}
        version = subprocess.run([self.binary, "--version"], capture_output=True, text=True, timeout=20)
        self._version = (version.stdout + version.stderr).strip()
        self.record: dict = {}
        self.wrapper = None
        self.server_pid = None
        self.sampler = None
        self.client = None
        self.log = None

    # Evaluation-runner interface -------------------------------------------------------------
    def models(self):
        return [{"name": Path(self.artifact["path"]).name, "size": self.artifact["size_bytes"],
                 "digest": self.artifact["sha256"], "artifact_sha256": self.artifact["sha256"],
                 "details": {"tensor_types": self.artifact["tensor_types"], "parameter_count": self.artifact["parameter_count"]}}]

    def version(self):
        return self._version

    # Lifecycle --------------------------------------------------------------------------------
    def start(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.command = [self.binary, "-m", self.artifact["path"], "--host", "127.0.0.1", "--port", str(self.port),
                        "-c", str(self.context), "-np", "1", "-t", str(self.threads), "--load-mode", self.load_mode,
                        "--cache-ram", "0", "--fit", "off", "--no-warmup", "-lv", "4"]
        self.before = system_snapshot()
        admission = runtime_admission(self.artifact)
        if admission["bytes"] > self.before["available_bytes"]:
            raise MemoryError(f"Runtime initialization rejected before start: estimate {admission['bytes']} bytes "
                              f"({admission['rule']}) exceeds available {self.before['available_bytes']} bytes. "
                              "Close other applications or use a smaller variant.")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("w")
        started = time.monotonic()
        self.wrapper = subprocess.Popen(["/usr/bin/time", "-l", *self.command], stdout=self.log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
        self.client = httpx.Client(base_url=f"http://127.0.0.1:{self.port}", timeout=self.request_timeout)
        self.server_pid = None
        try:
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                if self.wrapper.poll() is not None:
                    raise RuntimeError(f"llama-server exited {self.wrapper.returncode}; inspect {self.log_path}")
                if self.server_pid is None:
                    children = psutil.Process(self.wrapper.pid).children()
                    if children:
                        self.server_pid = children[0].pid
                        self.sampler = RssSampler(self.server_pid, .1).start()
                try:
                    if self.client.get("/health").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(.05)
            else:
                raise TimeoutError(f"llama-server startup timed out; inspect {self.log_path}")
            self.load_seconds = time.monotonic() - started
            self.snapshots = {"after_load": self._footprint()}
            self.sampler.mark("after_load")
            return self
        except BaseException:
            self.stop()
            raise

    def _footprint(self):
        return {"time": time.time(), "footprint_bytes": footprint(self.server_pid) if self.server_pid else None}

    def mark(self, name: str):
        self.snapshots[name] = self._footprint()
        self.sampler.mark(name)

    def generate(self, model, prompt, options):
        if model != self.models()[0]["name"] or options.get("num_ctx", self.context) != self.context:
            raise ValueError("Model or context does not match managed server")
        body = {"prompt": TEMPLATE.split(": ", 1)[1].format(prompt=prompt), "stream": True,
                "n_predict": options.get("num_predict", 128), "seed": options.get("seed", 42),
                "temperature": options.get("temperature", 0), "cache_prompt": False,
                "ignore_eos": bool(options.get("ignore_eos", False)), "stop": [] if options.get("ignore_eos") else ["<|im_end|>", "<|im_start|>"]}
        started = time.perf_counter()
        first, output, final = None, [], None
        with self.client.stream("POST", "/completion", json=body) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                item = json.loads(line[6:])
                if item.get("error"):
                    raise RuntimeError(str(item["error"]))
                if item.get("content"):
                    if first is None:
                        first = time.perf_counter() - started
                    output.append(item["content"])
                if item.get("stop"):
                    final = item
        if final is None:
            raise RuntimeError("Incomplete llama.cpp stream")
        timings = final.get("timings", {})
        return {"output": "".join(output), "ttft_seconds": first, "wall_seconds": time.perf_counter() - started,
                "runtime": {"eval_count": timings.get("predicted_n"), "eval_duration": timings.get("predicted_ms", 0) * 1e6,
                            "prompt_eval_count": timings.get("prompt_n"), "prompt_eval_duration": timings.get("prompt_ms", 0) * 1e6,
                            "native": {k: final.get(k) for k in ("stop_type", "truncated", "tokens_predicted", "tokens_evaluated", "timings")}}}

    def stop(self) -> dict:
        if self.wrapper is None:  # rejected before the process started
            self.record = {"started": False, "settings": self.settings, "runtime_version": self._version}
            return self.record
        if getattr(self, "sampler", None):
            self.sampler.stop()
        if self.server_pid:
            try:
                os.kill(self.server_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            self.wrapper.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(self.wrapper.pid, signal.SIGKILL)
            self.wrapper.wait()
        self.client.close()
        self.log.close()
        after = system_snapshot()
        text = self.log_path.read_text(errors="replace")
        offloaded = re.search(r"offloaded (\d+)/(\d+) layers to GPU", text)
        self.record = {
            "load_seconds": getattr(self, "load_seconds", None), "command": self.command, "settings": self.settings,
            "runtime_version": self._version, "time_l": parse_time_l(text),
            "snapshots": getattr(self, "snapshots", {}),
            "sampled_rss_phase_peaks": self.sampler.phase_peaks() if getattr(self, "sampler", None) else {},
            "metal": metal_evidence(text, offloaded),
            "buffers_mib": buffer_sizes(text),
            "system_before": {k: self.before[k] for k in ("available_bytes", "swap", "pressure_level", "thermal")},
            "system_after": {k: after[k] for k in ("available_bytes", "swap", "pressure_level", "thermal")},
            "system_delta": delta(self.before, after), "docker_vm_running": docker_vm_running(),
            "exit_code": self.wrapper.returncode}
        return self.record

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        if not self.record:
            self.stop()
