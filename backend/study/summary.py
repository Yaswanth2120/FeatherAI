"""Compact, status-explicit summary of an exported study for the API and dashboard."""
from __future__ import annotations

import json
from pathlib import Path

PHASES = ("convert", "validate", "dryrun", "benchmark", "quality", "perplexity")


def summarize_study(path: Path) -> dict:
    state = json.loads(path.read_text())
    phases = state.get("phases", {})
    status = {phase: phases.get(phase, {}).get("status", "not_run") for phase in PHASES}
    sensitivity = (phases.get("benchmark", {}).get("analysis") or {}).get("sensitivity_unflagged_sessions", {})
    summary = {"id": path.parent.name, "study": state.get("study"), "phases": status,
               "complete": all(value == "completed" for value in status.values()),
               "source": {k: (state.get("source") or {}).get(k) for k in ("path", "sha256", "size_bytes", "architecture", "parameter_count")},
               "toolchain_commit": state.get("toolchain_commit"), "conditions": state.get("conditions"), "variants": {}}
    variants = summary["variants"]
    variants["F16"] = {"artifact_size_bytes": (state.get("source") or {}).get("size_bytes"), "sha256": (state.get("source") or {}).get("sha256")}
    for job in phases.get("convert", {}).get("jobs", []):
        variants[job["target"]] = {"artifact_size_bytes": job["artifact"]["size_bytes"], "sha256": job["artifact"]["sha256"],
                                   "quantization_peak_rss_bytes": (job.get("memory") or {}).get("max_rss_bytes"),
                                   "storage_reduction_fraction": job.get("storage_reduction_fraction")}
    primary = (phases.get("benchmark", {}).get("analysis") or {}).get("primary_all_sessions", {})
    for label, entry in primary.items():
        variant = variants.setdefault(label, {})
        for metric in ("decode_tokens_per_second", "prefill_tokens_per_second", "ttft_seconds", "wall_seconds"):
            m = entry.get(metric) or {}
            variant[metric] = {k: m.get(k) for k in ("median", "median_ci95", "mean", "sd", "p95", "n")}
            ratio = next((v for k, v in m.items() if k.startswith("ratio_to_")), None)
            if ratio:
                variant[metric]["ratio_to_baseline"] = ratio
        variant["peak_memory_footprint_bytes"] = (entry.get("peak_memory_footprint_bytes") or {}).get("median")
        variant["load_seconds_repeat_median"] = (entry.get("load_seconds_repeat") or {}).get("median")
        variant["load_seconds_first_median"] = (entry.get("load_seconds_first_in_study") or {}).get("median")
        variant["failed_trials"] = entry.get("failed_trials")
        unflagged = (sensitivity.get(label) or {}).get("decode_tokens_per_second") or {}
        variant["sensitivity_unflagged"] = {"sessions_used": unflagged.get("sessions_used", 0), "n": unflagged.get("n"),
                                            "decode_median": unflagged.get("median"), "median_ci95": unflagged.get("median_ci95")}
        variant["flagged_sessions"] = entry.get("flagged_sessions")
        variant["sessions"] = entry.get("sessions")
    quality = phases.get("quality", {})
    for key, run in (quality.get("runs") or {}).items():
        suite, _, label = key.rpartition("-")
        variants.setdefault(label, {}).setdefault("quality", {})[suite] = {
            "status": run.get("status"), "reason": run.get("reason"), "accuracy": (run.get("summary") or {}).get("accuracy"),
            "not_scored": (run.get("summary") or {}).get("not_scored"),
            "accuracy_ci95": (run.get("summary") or {}).get("accuracy_ci95"), "items": (run.get("summary") or {}).get("items"),
            "outcomes": (run.get("summary") or {}).get("outcomes"), "sandbox_status": (run.get("sandbox") or {}).get("status")}
    summary["quality_comparisons"] = {
        key: {"status": value.get("status"), "reason": value.get("reason"),
              "categories": {c: {k: v.get(k) for k in ("difference", "difference_ci95", "informative", "validity", "evidence", "mcnemar_exact_p")}
                             for c, v in (value.get("categories") or {}).items()}}
        for key, value in (quality.get("comparisons") or {}).items()}
    for label, result in (phases.get("perplexity", {}).get("results") or {}).items():
        variants.setdefault(label, {})["perplexity"] = {**{k: result.get(k) for k in ("ppl", "ppl_se", "mean_ppl_q", "mean_kld", "mean_kld_se", "same_top_p", "ppl_ratio", "exit_code")},
                                                        "status": result.get("status", "completed" if result.get("exit_code") == 0 else "failed")}
    return summary
