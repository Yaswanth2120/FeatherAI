from contextlib import nullcontext
import json
from backend.benchmarks.system_monitor import ProcessMonitor
import hashlib
import importlib.metadata
import time
import uuid
from datetime import datetime, timezone
import numpy as np
from backend.core.hardware import inspect

PROMPT = "Write a Python binary_search(items, target) function. Return the index or -1. Output only the function."


def summarize(trials, metadata=None):
    """Descriptive summary; session-clustered 95% CI for the median only for the sessions-v2 method.

    Legacy reports keep their original method label and never receive a confidence interval.
    """
    from backend.benchmarks.sessions import LEGACY_METHOD, LEGACY_NOTE, METHOD, METHOD_NOTE
    from backend.study.stats import describe, median_ci
    method = (metadata or {}).get("method") or LEGACY_METHOD
    measured = [t for t in trials if not t["warmup"] and not t.get("error")]
    result = {"method": method, "method_note": METHOD_NOTE if method == METHOD else LEGACY_NOTE,
              "successful_trials": len(measured), "failed_trials": sum(bool(t.get("error")) for t in trials if not t["warmup"])}
    for key in ["wall_seconds", "ttft_seconds", "decode_tokens_per_second", "prefill_tokens_per_second"]:
        values = [t[key] for t in measured if t.get(key) is not None]
        if not values:
            result[key] = None
            continue
        entry = {"median": float(np.median(values)), "p95": float(np.percentile(values, 95)), "min": min(values), "max": max(values)}
        if method == METHOD:
            groups = {}
            for t in measured:
                if t.get(key) is not None:
                    groups.setdefault(t["session"], []).append(t[key])
            entry.update({k: v for k, v in describe(values).items() if k in ("mean", "sd", "n")},
                         median_ci95=median_ci(list(groups.values())), sessions_used=len(groups),
                         ci_method="two-stage session-clustered bootstrap, 10000 resamples, seed 20261109")
        result[key] = entry
    return result


def run(backend, db, model, repeats=5, warmups=1, context=2048, tokens=128, seed=42, prompt=PROMPT):
    if repeats < 1 or warmups < 0 or context < 1 or tokens < 1:
        raise ValueError("Repeats, context and tokens must be positive; warmups must be nonnegative")
    models = backend.models()
    selected = next((m for m in models if m["name"] == model), None)
    if selected is None:
        raise ValueError(f"Model {model!r} is not installed; run feather models list")
    identifier = str(uuid.uuid4())
    options = {"seed": seed, "temperature": 0, "num_ctx": context, "num_predict": tokens}
    hardware = inspect()
    settings_identity = {"options": options, "backend": getattr(backend, "name", "ollama"), "runtime_version": backend.version(),
                         "hardware": {key: hardware[key] for key in ["chip", "architecture", "os", "physical_ram_bytes"]}, "warmups": warmups, "repeats": repeats, "backend_settings": getattr(backend, "benchmark_settings", {})}
    metadata = {"created_at": datetime.now(timezone.utc).isoformat(), "method": "legacy-sampled-rss-v1", "model": selected,
        "backend": getattr(backend, "name", "ollama"), "runtime_version": backend.version(), "hardware": hardware,
        "source_sha256": getattr(backend, "source_sha256", None),
        "backend_settings": getattr(backend, "benchmark_settings", {}),
        "settings_sha256": hashlib.sha256(json.dumps(settings_identity, sort_keys=True).encode()).hexdigest(),
        "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "options": options, "repeats": repeats, "warmups": warmups,
        "packages": {p: importlib.metadata.version(p) for p in ["feather-ai", "httpx", "numpy", "psutil"]},
        "memory_note": "Managed llama.cpp: sampled process-tree RSS every 50ms during requests; excludes startup peak and unaccounted GPU allocations, may miss short peaks. Ollama process memory is not measured.",
        "load_note": "Warmups are excluded; model residency before the first request is unknown."}
    db.start(identifier, metadata)
    for ordinal in range(warmups + repeats):
        trial = {"ordinal": ordinal, "warmup": ordinal < warmups, "started_at": time.time()}
        try:
            monitor = ProcessMonitor(backend.process.pid) if getattr(backend, "process", None) else None
            with monitor if monitor else nullcontext():
                trial.update(backend.generate(model, prompt, options))
            if monitor:
                trial.update(peak_memory_bytes=monitor.peak, memory_samples=monitor.samples, memory_metric="sampled_process_tree_rss")
            raw = trial["runtime"]
            for name, count, duration in [("decode", "eval_count", "eval_duration"), ("prefill", "prompt_eval_count", "prompt_eval_duration")]:
                trial[f"{name}_tokens_per_second"] = raw[count] / (raw[duration] / 1e9) if count in raw and raw.get(duration, 0) > 0 else None
        except Exception as error:
            trial["error"] = f"{type(error).__name__}: {error}"
        trial["system_after"] = inspect()
        db.save(identifier, ordinal, trial)
    peaks = [t.get("peak_memory_bytes") for t in db.report(identifier)["trials"] if not t["warmup"] and not t.get("error") and t.get("peak_memory_bytes")]
    if peaks:
        metadata["peak_memory_bytes"] = max(peaks)
        metadata["memory_metric"] = "sampled_process_tree_rss"
        with db.connection:
            db.connection.execute("UPDATE experiments SET metadata=? WHERE id=?", (json.dumps(metadata), identifier))
    report = db.report(identifier)
    report["summary"] = summarize(report["trials"])
    return report
