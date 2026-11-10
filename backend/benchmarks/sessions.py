"""Session-based benchmark for API jobs, using the same method as `feather quant-study`.

Each session is a fresh llama-server process under /usr/bin/time -l (load latency, peak footprint,
Metal check) followed by warmups and measured requests with exactly ``tokens`` output tokens.
Trials carry their session index so uncertainty is computed with the session-clustered bootstrap.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import uuid

from backend.core.hardware import inspect
from backend.study.benchmark import PROMPT_PATH, session_flags, trial_metrics
from backend.study.server import ServerSession

METHOD = "sessions-v2"
LEGACY_METHOD = "legacy-sampled-rss-v1"
METHOD_NOTE = ("sessions-v2: fresh llama-server per session (--load-mode none, /usr/bin/time -l peak footprint including "
               "startup), warmups excluded, fixed-length outputs (ignore_eos), session-clustered bootstrap 95% CI for medians.")
LEGACY_NOTE = ("legacy-sampled-rss-v1: one server process, 50 ms sampled process RSS during requests (startup excluded), "
               "no confidence interval. Not directly comparable with sessions-v2 results.")


def run_sessions(artifact: dict, server: str, db, log_dir: Path, source_sha256: str | None, sessions=4, measured=5,
                 warmups=1, cooldown=10, context=2048, tokens=128, seed=42, threads=4) -> dict:
    if not (1 <= sessions <= 10 and 1 <= measured <= 20 and 0 <= warmups <= 3 and 0 <= cooldown <= 120):
        raise ValueError("Invalid session benchmark design")
    prompt = PROMPT_PATH.read_text()
    options = {"seed": seed, "temperature": 0, "num_ctx": context, "num_predict": tokens, "ignore_eos": True}
    identifier = str(uuid.uuid4())
    hardware = inspect()
    model = {"name": Path(artifact["path"]).name, "size": artifact["size_bytes"], "digest": artifact["sha256"],
             "artifact_sha256": artifact["sha256"],
             "details": {"tensor_types": artifact["tensor_types"], "parameter_count": artifact["parameter_count"]}}
    probe = ServerSession(artifact, server, log_dir / "version.log", context=context, threads=threads)
    backend_settings = dict(probe.settings, method=METHOD)
    design = {"sessions": sessions, "measured_per_session": measured, "warmups_per_session": warmups, "cooldown_seconds": cooldown}
    identity = {"options": options, "backend": "llama.cpp", "runtime_version": probe.version(), "design": design,
                "hardware": {k: hardware[k] for k in ["chip", "architecture", "os", "physical_ram_bytes"]},
                "backend_settings": backend_settings}
    metadata = {"created_at": datetime.now(timezone.utc).isoformat(), "method": METHOD, "method_note": METHOD_NOTE,
                "model": model, "backend": "llama.cpp", "runtime_version": probe.version(), "hardware": hardware,
                "source_sha256": source_sha256, "backend_settings": backend_settings, "design": design,
                "settings_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "options": options,
                "repeats": sessions * measured, "warmups": sessions * warmups,
                "packages": {p: importlib.metadata.version(p) for p in ["feather-ai", "httpx", "numpy", "psutil"]},
                "memory_metric": "median over sessions of /usr/bin/time -l peak memory footprint (--load-mode none)",
                "sessions": []}
    db.start(identifier, metadata)
    ordinal = 0
    for index in range(1, sessions + 1):
        if index > 1 and cooldown:
            time.sleep(cooldown)
        session = ServerSession(artifact, server, log_dir / f"session-{index:02d}.log", context=context, threads=threads,
                                source_sha256=source_sha256)
        entry = {"session": index, "error": None}
        try:
            session.start()
            name = session.models()[0]["name"]
            for position in range(warmups + measured):
                trial = {"ordinal": ordinal, "session": index, "warmup": position < warmups, "started_at": time.time()}
                try:
                    trial.update(trial_metrics(session.generate(name, prompt, options)))
                    if trial["output_tokens"] != tokens:
                        trial["error"] = f"output token count {trial['output_tokens']} != {tokens}"
                except Exception as error:  # recorded and excluded, never imputed
                    trial["error"] = f"{type(error).__name__}: {error}"[:300]
                db.save(identifier, ordinal, trial)
                ordinal += 1
        except Exception as error:
            entry["error"] = f"{type(error).__name__}: {error}"[:500]
            # A session that never produced trials still counts against completeness.
            for position in range(warmups + measured):
                db.save(identifier, ordinal, {"ordinal": ordinal, "session": index, "warmup": position < warmups,
                                              "error": f"session failed: {entry['error']}"})
                ordinal += 1
        finally:
            record = session.stop() if not session.record else session.record
        entry.update(load_seconds=record.get("load_seconds"), time_l=record.get("time_l"), metal=record.get("metal"),
                     buffers_mib=record.get("buffers_mib"), started=record.get("started", True),
                     flags=session_flags(record) if "system_delta" in record else ["incomplete"])
        metadata["sessions"].append(entry)
    peaks = [s["time_l"].get("peak_memory_footprint_bytes") for s in metadata["sessions"] if s.get("time_l")]
    peaks = sorted(p for p in peaks if p)
    if peaks:
        metadata["peak_memory_bytes"] = int(peaks[len(peaks) // 2]) if len(peaks) % 2 else int((peaks[len(peaks) // 2 - 1] + peaks[len(peaks) // 2]) / 2)
    with db.connection:
        db.connection.execute("UPDATE experiments SET metadata=? WHERE id=?", (json.dumps(metadata), identifier))
    from backend.benchmarks.runner import summarize
    report = db.report(identifier)
    report["summary"] = summarize(report["trials"], metadata)
    return report
