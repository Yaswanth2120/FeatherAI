"""Drive one llama-server process under /usr/bin/time -l.

One process = one run: load model, wait for /health, send warm-up
request(s) (discarded), send the measured prompts sequentially, then
SIGTERM the server so /usr/bin/time prints its resource report.
"""
from __future__ import annotations

import http.client
import json
import re
import subprocess
import time
from pathlib import Path

from .sysinfo import ROOT, parse_time_l, snapshot

# Evidence that the Metal backend initialised and every layer was offloaded.
# Strings taken from this build's log at -lv 4 (llama.cpp v0.9.0, cd959ba).
_METAL_INIT = re.compile(r"ggml_metal_init: found device: (.+)")
_OFFLOADED = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")
_EMBED = re.compile(r"MTL : EMBED_LIBRARY = (\d)")
_MTL_BUF = re.compile(r"MTL0 (model|KV|compute) buffer size =\s+([\d.]+) MiB")


def check_metal(log: str) -> dict:
    dev = _METAL_INIT.search(log)
    m = _OFFLOADED.search(log)
    off = (int(m.group(1)), int(m.group(2))) if m else None
    emb = _EMBED.search(log)
    return {"metal_device": dev.group(1).strip() if dev else None,
            "offloaded_layers": list(off) if off else None,
            "embed_library": int(emb.group(1)) if emb else None,
            "mtl0_buffers_mib": {k: float(v) for k, v in _MTL_BUF.findall(log)},
            "ok": bool(dev) and off is not None and off[0] == off[1] and off[1] > 0}


def parse_sse_events(lines):
    """Yield decoded JSON objects from `data: {...}` SSE lines."""
    for raw in lines:
        line = raw.decode() if isinstance(raw, bytes) else raw
        line = line.strip()
        if line.startswith("data: "):
            payload = line[6:]
            if payload == "[DONE]":
                return
            yield json.loads(payload)


def build_cmd(cfg: dict, model: Path, mode: str) -> list[str]:
    s = cfg["llama_server"]
    cmd = ["/usr/bin/time", "-l", str(ROOT / s["binary"]),
           "-m", str(model), "--host", s["host"], "--port", str(s["port"]),
           "-c", str(s["ctx_size"]), "-ngl", str(s["n_gpu_layers"]),
           "-t", str(s["threads"]), "-np", str(s["parallel"]),
           *s.get("extra_args", [])]
    # llama.cpp v0.9.0 replaced --no-mmap with --load-mode {auto,none,mmap,...};
    # pass it explicitly in both modes (auto would resolve to mmap).
    load_mode = {"no_mmap": "none", "mmap": "mmap"}.get(mode)
    if load_mode is None:
        raise ValueError(mode)
    cmd += ["--load-mode", load_mode]
    return cmd


def _wait_ready(host, port, proc, timeout):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early (code {proc.returncode})")
        try:
            c = http.client.HTTPConnection(host, port, timeout=2)
            c.request("GET", "/health")
            if c.getresponse().status == 200:
                return time.monotonic() - t0
        except OSError:
            pass
        time.sleep(0.25)
    raise TimeoutError("llama-server not ready")


def complete(host, port, prompt: str, req: dict, seed: int, timeout: float) -> dict:
    """Streamed /completion. TTFT is client wall-clock from request send to
    the first streamed chunk that carries generated text."""
    body = json.dumps({
        "prompt": prompt, "stream": True, "seed": seed,
        "n_predict": req["n_predict"], "ignore_eos": req["ignore_eos"],
        "temperature": req["temperature"], "cache_prompt": req["cache_prompt"],
    })
    c = http.client.HTTPConnection(host, port, timeout=timeout)
    t0 = time.perf_counter()
    c.request("POST", "/completion", body, {"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        raise RuntimeError(f"/completion HTTP {r.status}: {r.read()[:300]!r}")
    ttft = None
    final = None
    n_chunks = 0
    for ev in parse_sse_events(r):
        if ttft is None and ev.get("content"):
            ttft = time.perf_counter() - t0
        n_chunks += 1
        if ev.get("stop"):
            final = ev
    wall = time.perf_counter() - t0
    if final is None or "timings" not in final:
        raise RuntimeError("stream ended without final timings")
    t = final["timings"]
    return {
        "ttft_s_client": ttft,
        "wall_s": wall,
        "prompt_n": t.get("prompt_n"),
        "prompt_ms": t.get("prompt_ms"),
        "prompt_tok_s": t.get("prompt_per_second"),
        "decode_n": t.get("predicted_n"),
        "decode_ms": t.get("predicted_ms"),
        "decode_tok_s": t.get("predicted_per_second"),
        "tokens_cached": final.get("tokens_cached"),
        "stop_type": final.get("stop_type"),
        "stream_chunks": n_chunks,
    }


def run_server(cfg: dict, model: Path, mode: str, prompts: list[dict], seed: int,
               log_path: Path, warmup_prompts: list[dict]) -> dict:
    """prompts / warmup_prompts: [{"id": ..., "prompt": ...}].

    System snapshots are taken at phase boundaries (outside any request) so
    paging can be attributed to load, warm-up, measured requests, or shutdown.
    """
    s, req = cfg["llama_server"], cfg["request"]
    cmd = build_cmd(cfg, model, mode)
    out = {"cmd": cmd, "warmup": [], "requests": [], "error": None, "phase_snapshots": {}}

    def mark(name):
        out["phase_snapshots"][name] = {"t_monotonic": time.monotonic(), "snap": snapshot()}

    with open(log_path, "w") as log:
        t_launch = time.monotonic()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            out["load_s"] = _wait_ready(s["host"], s["port"], proc, cfg["server_ready_timeout_s"])
            mark("after_load")
            # Warm-up prompts are disjoint from every measured prompt set.
            for i in range(cfg["warmup_requests"]):
                wp = warmup_prompts[i % len(warmup_prompts)]
                out["warmup"].append(complete(s["host"], s["port"], wp["prompt"], req,
                                              seed, cfg["request_timeout_s"])
                                     | {"prompt_id": wp["id"]})
            mark("after_warmup")
            for i, p in enumerate(prompts):
                r = complete(s["host"], s["port"], p["prompt"], req, seed + i,
                             cfg["request_timeout_s"])
                out["requests"].append(r | {"prompt_index": i, "prompt_id": p["id"]})
            mark("after_measured")
        except Exception as e:  # recorded; run is marked invalid
            out["error"] = f"{type(e).__name__}: {e}"
        finally:
            # proc is /usr/bin/time; signal its child (llama-server) so time reports.
            kids = subprocess.run(["pgrep", "-P", str(proc.pid)], capture_output=True,
                                  text=True).stdout.split()
            for k in kids:
                try:
                    subprocess.run(["kill", "-TERM", k])
                except Exception:
                    pass
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                for k in kids:
                    subprocess.run(["kill", "-KILL", k])
                proc.wait(timeout=30)
        out["process_wall_s"] = time.monotonic() - t_launch
    text = log_path.read_text(errors="replace")
    try:
        out["time_l"] = parse_time_l(text)
    except ValueError as e:
        out["time_l"] = None
        out["error"] = out["error"] or f"time -l: {e}"
    out["metal"] = check_metal(text)
    out["exit_code"] = proc.returncode
    return out
