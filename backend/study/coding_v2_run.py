"""Pre-registered coding-v2 run on an exported study's variants (docs/DECISIONS.md, coding-v2 analysis plan).

Sequential: F16 screening dry run, then held-out for F16, Q8_0, Q4_K_M. Resumable per run.
Analysis: exact two-sided McNemar with Holm across {Q8_0, Q4_K_M}; paired bootstrap 95% and 90% CIs
(10 000 resamples, seed 20261110); equivalence within +/-10 pp only if the 90% CI lies inside it;
baseline validity (F16 >= 10 solved and >= 25%).
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np

from backend.core.artifacts import inspect_gguf
from backend.evaluation.runner import evaluate_dataset
from backend.evaluation.scoring import paired_comparison
from backend.study.run import _tree_digest
from backend.study.server import ServerSession

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261110
MARGIN = 0.10


def holm(pvalues: dict[str, float], alpha=0.05) -> dict[str, dict]:
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    result, rejecting, m = {}, True, len(ordered)
    running = 0.0
    for rank, (name, p) in enumerate(ordered):
        adjusted = min(1.0, max(running, (m - rank) * p))
        running = adjusted
        rejecting = rejecting and p <= alpha / (m - rank)
        result[name] = {"p": p, "holm_adjusted_p": adjusted, "reject_at_family_0.05": rejecting}
    return result


def bootstrap_ci(base: list[bool], cand: list[bool], level: float) -> list[float]:
    d = np.asarray(cand, float) - np.asarray(base, float)
    idx = np.random.default_rng(SEED).integers(0, len(d), size=(10_000, len(d)))
    boot = d[idx].mean(axis=1)
    tail = (1 - level) / 2 * 100
    return [float(np.percentile(boot, tail)), float(np.percentile(boot, 100 - tail))]


def analyze(reports: dict[str, dict]) -> dict:
    base = {r["task_id"]: bool(r["passed"]) for r in reports["F16"]["results"]}
    solved = sum(base.values())
    valid = solved >= 10 and solved / len(base) >= 0.25
    comparisons, pvalues = {}, {}
    for label in ("Q8_0", "Q4_K_M"):
        cand = {r["task_id"]: bool(r["passed"]) for r in reports[label]["results"]}
        assert cand.keys() == base.keys()
        ids = sorted(base)
        a, b = [base[i] for i in ids], [cand[i] for i in ids]
        stats = paired_comparison(a, b, resamples=10_000, seed=SEED)
        ci90 = bootstrap_ci(a, b, 0.90)
        stats.update(difference_ci90=ci90,
                     equivalence_within_10pp=bool(-MARGIN < ci90[0] and ci90[1] < MARGIN),
                     equivalence_label=("equivalent within +/-10 pp" if -MARGIN < ci90[0] and ci90[1] < MARGIN else "not shown equivalent"))
        comparisons[label] = stats
        pvalues[label] = stats["mcnemar_exact_p"]
    adjusted = holm(pvalues)
    for label in comparisons:
        comparisons[label].update(adjusted[label])
    return {"baseline_solved": solved, "items": len(base), "baseline_informative": valid, "comparisons": comparisons,
            "pass_rates": {k: sum(bool(r["passed"]) for r in v["results"]) / len(v["results"]) for k, v in reports.items()}}


def run(study_dir: Path, tools: Path, out: Path, log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "run.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "kind": "real-run", "suite": "coding-v2", "created_at": datetime.now(timezone.utc).isoformat(), "runs": {},
        "repository_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "repository_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()),
        "working_tree_diff_sha256": _tree_digest(ROOT), "study": str(study_dir.relative_to(ROOT))}
    study = json.loads((study_dir / "study.json").read_text())
    variants = {"F16": study["source"]["path"]}
    variants.update({job["target"]: job["artifact"]["path"] for job in study["phases"]["convert"]["jobs"]})
    expected = {"F16": study["source"]["sha256"], **{job["target"]: job["artifact"]["sha256"] for job in study["phases"]["convert"]["jobs"]}}
    plan = [("F16", "screening")] + [(label, "heldout") for label in ("F16", "Q8_0", "Q4_K_M")]
    for label, split in plan:
        key = f"{split}-{label}"
        if state["runs"].get(key, {}).get("status") == "completed":
            continue
        artifact = inspect_gguf(variants[label])
        artifact.pop("tensors", None)
        if artifact["sha256"] != expected[label]:
            raise ValueError(f"{label} hash changed: {artifact['sha256']}")
        log(f"coding-v2 {key}")
        session = ServerSession(artifact, str(tools / "llama-server"), out / "logs" / f"{key}.log", context=2048,
                                source_sha256=study["source"]["sha256"]).start()
        with session:
            report = evaluate_dataset(session, session.models()[0]["name"], "coding-v2", split=split, seed=42, context=2048, tokens=384,
                                      progress=lambda d, t, r: log(f"  {key}: {d}/{t}") if d % 60 == 0 or d == t else None)
            report["server_session"] = session.stop()
        report.pop("hardware", None)
        (out / f"{key}.json").write_text(json.dumps(report, indent=1) + "\n")
        state["runs"][key] = {"status": report["status"], "artifact_sha256": artifact["sha256"], "summary": report["summary"],
                              "file": f"{key}.json"}
        state_path.write_text(json.dumps(state, indent=1) + "\n")
    held = {label: json.loads((out / f"heldout-{label}.json").read_text()) for label in ("F16", "Q8_0", "Q4_K_M")}
    if all(state["runs"][f"heldout-{l}"]["status"] == "completed" for l in held):
        state["analysis"] = analyze(held)
    else:
        state["analysis"] = {"status": "insufficient_evidence", "reason": "Not every held-out item was scored"}
    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    state_path.write_text(json.dumps(state, indent=1) + "\n")
    return state


if __name__ == "__main__":
    import sys
    result = run(ROOT / sys.argv[1], ROOT / ".feather/tools/llama.cpp/build/bin", ROOT / sys.argv[2], log=lambda m: print(m, flush=True))
    print(json.dumps(result.get("analysis"), indent=1))
