"""Reproducible flat exports (CSV) from a study.json; numbers are copied, never recomputed or rounded."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from backend.study.summary import summarize_study


def export_tables(study_json: Path, destination: Path) -> list[Path]:
    summary = summarize_study(study_json)
    destination.mkdir(parents=True, exist_ok=True)
    written = []
    performance = destination / "summary-performance.csv"
    with performance.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "artifact_size_bytes", "sha256", "sessions", "flagged_sessions", "failed_trials", "metric", "n",
                         "median", "median_ci95_low", "median_ci95_high", "mean", "sd", "p95", "ratio_to_F16", "ratio_ci95_low", "ratio_ci95_high"])
        for name, v in summary["variants"].items():
            for metric in ("decode_tokens_per_second", "prefill_tokens_per_second", "ttft_seconds", "wall_seconds"):
                m = v.get(metric)
                if not m:
                    continue
                ratio = m.get("ratio_to_baseline") or {}
                ci = m.get("median_ci95") or [None, None]
                rci = ratio.get("ci95") or [None, None]
                writer.writerow([name, v.get("artifact_size_bytes"), v.get("sha256"), v.get("sessions"), v.get("flagged_sessions"),
                                 v.get("failed_trials"), metric, m.get("n"), m.get("median"), ci[0], ci[1], m.get("mean"), m.get("sd"),
                                 m.get("p95"), ratio.get("ratio"), rci[0], rci[1]])
    written.append(performance)
    memory = destination / "summary-memory-quality.csv"
    with memory.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "peak_memory_footprint_bytes_median", "load_seconds_first", "load_seconds_repeat_median",
                         "reasoning_status", "reasoning_accuracy", "reasoning_ci95", "coding_status", "coding_accuracy", "coding_ci95",
                         "mean_kld", "mean_kld_se", "same_top_p_percent", "ppl_base", "ppl_ratio"])
        for name, v in summary["variants"].items():
            q = v.get("quality", {})
            r, c, p = q.get("reasoning-v1", {}), q.get("coding-v1", {}), v.get("perplexity", {})
            writer.writerow([name, v.get("peak_memory_footprint_bytes"), v.get("load_seconds_first_median"), v.get("load_seconds_repeat_median"),
                             r.get("status", "not_run"), r.get("accuracy"), json.dumps(r.get("accuracy_ci95")),
                             c.get("status", "not_run"), c.get("accuracy"), json.dumps(c.get("accuracy_ci95")),
                             p.get("mean_kld"), p.get("mean_kld_se"), p.get("same_top_p"), p.get("ppl"), p.get("ppl_ratio")])
    written.append(memory)
    comparisons = destination / "summary-quality-comparisons.csv"
    with comparisons.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["comparison", "status", "category", "difference", "ci95_low", "ci95_high", "mcnemar_exact_p", "informative", "validity", "evidence", "reason"])
        for key, value in summary["quality_comparisons"].items():
            if value["status"] != "completed":
                writer.writerow([key, value["status"], "", "", "", "", "", "", "", "", value.get("reason")])
                continue
            for category, row in value["categories"].items():
                ci = row.get("difference_ci95") or [None, None]
                writer.writerow([key, "completed", category, row["difference"], ci[0], ci[1], row["mcnemar_exact_p"], row["informative"],
                                 row["validity"], row["evidence"], ""])
    written.append(comparisons)
    (destination / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    written.append(destination / "summary.json")
    return written
