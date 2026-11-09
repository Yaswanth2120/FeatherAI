"""Build task04 comparison evidence from an exported v0.2 study (values copied, never estimated)."""
from __future__ import annotations

import json
from pathlib import Path


def build(study_dir: Path) -> dict:
    state = json.loads((study_dir / "study.json").read_text())
    phases = state["phases"]
    sessions = [json.loads(line) for line in (study_dir / "benchmark" / "sessions.jsonl").read_text().splitlines()]
    analysis = phases["benchmark"]["analysis"]["primary_all_sessions"]
    design = phases["benchmark"]["design"]
    artifacts = {"F16": (state["source"]["sha256"], state["source"]["size_bytes"])}
    for job in phases["convert"]["jobs"]:
        artifacts[job["target"]] = (job["artifact"]["sha256"], job["artifact"]["size_bytes"])
    runs = phases["quality"]["runs"]
    perplexity = phases["perplexity"]["results"]
    configurations = []
    for label, (digest, size) in artifacts.items():
        mine = [s for s in sessions if s["variant"] == label]
        trials = [{**t, "session": s["session"], "round": s["round"]} for s in mine for t in s["trials"]]
        versions = {s["server"]["runtime_version"] for s in mine}
        ppl = perplexity[label].get("ppl") if label == "F16" else perplexity[label].get("mean_ppl_q")
        entry = analysis[label]
        configurations.append({
            "label": label, "source_sha256": state["source"]["sha256"], "artifact_sha256": digest, "file_size_bytes": size,
            "peak_memory_bytes": int(entry["peak_memory_footprint_bytes"]["median"]),
            "memory_metric": "median over sessions of /usr/bin/time -l peak memory footprint (--load-mode none)",
            **{m: entry[m]["median"] for m in ("ttft_seconds", "prefill_tokens_per_second", "decode_tokens_per_second", "wall_seconds")},
            "median_ci95": {m: entry[m]["median_ci95"] for m in ("ttft_seconds", "prefill_tokens_per_second", "decode_tokens_per_second", "wall_seconds")},
            "coding_correctness": runs[f"coding-v1-{label}"]["summary"]["accuracy"],
            "coding_status": runs[f"coding-v1-{label}"]["status"],
            "reasoning_accuracy": runs[f"reasoning-v1-{label}"]["summary"]["accuracy"],
            "perplexity": ppl, "mean_kld": perplexity[label].get("mean_kld"),
            "repeats": sum(not t["warmup"] for t in trials), "warmups": sum(t["warmup"] for t in trials),
            "context_length": design["context"], "decoding_settings": design["options"],
            "runtime_version": versions.pop() if len(versions) == 1 else sorted(versions),
            "hardware": {k: state["hardware"][k] for k in ("chip", "architecture", "physical_ram_bytes", "os")},
            "raw_trials": trials})
    return {"kind": "real-run", "status": "completed", "study": str(study_dir), "toolchain_commit": state["toolchain_commit"],
            "note": "Derived from experiments/studies/<id>/; medians and CIs follow the pre-registered v0.2 protocol.",
            "configurations": configurations}
