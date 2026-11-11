"""Pre-registered reasoning-v2 replication of the Q8_0 reasoning signal (docs/DECISIONS.md).

Settings match the reasoning-v1 study run: context 2048, temperature 0, seed 42, n_predict 256,
ChatML template, llama-server --load-mode none. Order: F16 screening dry run (format check only,
never analysed), then held-out F16, Q8_0, Q4_K_M, sequentially. Resumable; every resumption,
exclusion and infrastructure failure is recorded by the runner itself.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from backend.core.artifacts import inspect_gguf
from backend.evaluation.datasets import SHA256, load
from backend.evaluation.runner import evaluate_dataset
from backend.evaluation.scoring import paired_comparison
from backend.study.coding_v2_run import SEED, bootstrap_ci, holm
from backend.study.run import _tree_digest
from backend.study.server import ServerSession

ROOT = Path(__file__).resolve().parents[2]
SUITE = "reasoning-v2"
MARGIN = 0.05
SETTINGS = {"context": 2048, "seed": 42, "temperature": 0, "tokens": 256}


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def classify(q8: dict) -> str:
    """Outcome mapping registered in docs/DECISIONS.md before any reasoning-v2 output existed."""
    if q8["reject_at_family_0.05"] and q8["difference"] < 0:
        return "replicated"
    if q8["reject_at_family_0.05"] and q8["difference"] > 0:
        return "opposite direction (Q8_0 better than F16)"
    if q8["equivalent_within_5pp"]:
        return "not detected (equivalent within ±5 pp)"
    return "inconclusive"


def analyze(reports: dict[str, dict]) -> dict:
    base = {r["task_id"]: r for r in reports["F16"]["results"]}
    solved = sum(bool(r["passed"]) for r in base.values())
    comparisons, pvalues = {}, {}
    for label in ("Q8_0", "Q4_K_M"):
        cand = {r["task_id"]: r for r in reports[label]["results"]}
        if cand.keys() != base.keys():
            raise ValueError(f"{label} item set differs from F16")
        ids = sorted(base)
        a, b = [bool(base[i]["passed"]) for i in ids], [bool(cand[i]["passed"]) for i in ids]
        stats = paired_comparison(a, b, resamples=10_000, seed=SEED)
        ci90 = bootstrap_ci(a, b, 0.90)
        stats.update(difference_ci90=ci90, equivalent_within_5pp=bool(-MARGIN < ci90[0] and ci90[1] < MARGIN))
        exploratory = {}
        for category in sorted({r["item_category"] for r in base.values()}):
            cids = [i for i in ids if base[i]["item_category"] == category]
            ca, cb = [bool(base[i]["passed"]) for i in cids], [bool(cand[i]["passed"]) for i in cids]
            c = paired_comparison(ca, cb, resamples=10_000, seed=SEED)
            exploratory[category] = {k: c[k] for k in ("items", "baseline_accuracy", "candidate_accuracy", "difference",
                                                       "difference_ci95", "baseline_only_correct", "candidate_only_correct",
                                                       "mcnemar_exact_p")}
        stats["exploratory_by_category"] = exploratory
        comparisons[label] = stats
        pvalues[label] = stats["mcnemar_exact_p"]
    for label, adjusted in holm(pvalues).items():
        comparisons[label].update(adjusted)
    discordance = (comparisons["Q8_0"]["baseline_only_correct"] + comparisons["Q8_0"]["candidate_only_correct"]) / len(base)
    return {"items": len(base), "baseline_solved": solved,
            "baseline_informative": solved >= 10 and solved / len(base) >= 0.25,
            "pass_rates": {k: sum(bool(r["passed"]) for r in v["results"]) / len(v["results"]) for k, v in reports.items()},
            "comparisons": comparisons, "q8_discordance": discordance,
            "power_note": ("observed Q8_0 discordance >= 0.10: the registered design may be underpowered" if discordance >= 0.10
                           else "observed Q8_0 discordance within the registered planning range"),
            "q8_signal": classify(comparisons["Q8_0"])}


def run(study_dir: Path, tools: Path, out: Path, log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "run.json"
    provenance = {"at": datetime.now(timezone.utc).isoformat(), "repository_commit": _git("rev-parse", "HEAD"),
                  "repository_dirty": bool(_git("status", "--porcelain")), "working_tree_diff_sha256": _tree_digest(ROOT)}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        pending = [k for k in state["plan"] if state["runs"].get(k, {}).get("status") != "completed"]
        if pending:
            state.setdefault("resumptions", []).append({**provenance, "pending_runs": pending,
                                                        "note": "Recorded automatically by the runner on restart."})
    else:
        state = {"kind": "real-run", "suite": SUITE, "dataset_sha256": SHA256[SUITE], "settings": SETTINGS,
                 "plan": ["screening-F16", "heldout-F16", "heldout-Q8_0", "heldout-Q4_K_M"], "runs": {},
                 "started": provenance, "study": str(study_dir.relative_to(ROOT))}
    if provenance["repository_dirty"]:
        raise RuntimeError("Refusing to run from a dirty working tree")
    load(SUITE)  # verifies the frozen dataset hash
    study = json.loads((study_dir / "study.json").read_text())
    paths = {"F16": study["source"]["path"], **{j["target"]: j["artifact"]["path"] for j in study["phases"]["convert"]["jobs"]}}
    expected = {"F16": study["source"]["sha256"], **{j["target"]: j["artifact"]["sha256"] for j in study["phases"]["convert"]["jobs"]}}
    state_path.write_text(json.dumps(state, indent=1) + "\n")
    for key in state["plan"]:
        if state["runs"].get(key, {}).get("status") == "completed":
            continue
        split, label = key.split("-", 1)
        artifact = inspect_gguf(paths[label])
        artifact.pop("tensors", None)
        if artifact["sha256"] != expected[label]:
            raise ValueError(f"{label} hash changed: {artifact['sha256']}")
        log(f"{SUITE} {key}")
        session = ServerSession(artifact, str(tools / "llama-server"), out / "logs" / f"{key}.log", context=SETTINGS["context"],
                                source_sha256=study["source"]["sha256"]).start()
        with session:
            report = evaluate_dataset(session, session.models()[0]["name"], SUITE, split=split, seed=SETTINGS["seed"],
                                      context=SETTINGS["context"], tokens=SETTINGS["tokens"],
                                      progress=lambda d, t, r: log(f"  {key}: {d}/{t}") if d % 100 == 0 or d == t else None)
            report["server_session"] = session.stop()
        report.pop("hardware", None)
        (out / f"{key}.json").write_text(json.dumps(report, indent=1) + "\n")
        failures = [r["task_id"] for r in report["results"] if r["status"] != "scored"]
        state["runs"][key] = {"status": report["status"], "artifact_sha256": artifact["sha256"], "summary": report["summary"],
                              "file": f"{key}.json", "infrastructure_failures": failures, "provenance": provenance}
        state_path.write_text(json.dumps(state, indent=1) + "\n")
    held = {label: json.loads((out / f"heldout-{label}.json").read_text()) for label in ("F16", "Q8_0", "Q4_K_M")}
    if all(state["runs"][f"heldout-{label}"]["status"] == "completed" for label in held):
        state["analysis"] = analyze(held)
    else:
        state["analysis"] = {"status": "insufficient_evidence", "reason": "Not every held-out item was scored",
                             "q8_signal": "inconclusive"}
    state["finished_at"] = datetime.now(timezone.utc).isoformat()
    state_path.write_text(json.dumps(state, indent=1) + "\n")
    return state


if __name__ == "__main__":
    import sys
    result = run(ROOT / sys.argv[1], ROOT / ".feather/tools/llama.cpp/build/bin", ROOT / sys.argv[2], log=lambda m: print(m, flush=True))
    print(json.dumps({k: v for k, v in result["analysis"].items() if k != "comparisons"}, indent=1))
