"""Freeze coding-v2: run every MBPP reference inside the Docker sandbox, then write coding-v2.json once."""
import hashlib
import json
from pathlib import Path

from backend.evaluation.coding import score_code
from backend.evaluation.datasets.coding_v2 import build, candidates
from backend.evaluation.sandbox import discover

OUT = Path(__file__).resolve().parent / "coding-v2.json"


def main():
    if OUT.exists():
        print("coding-v2.json exists; not overwritten", hashlib.sha256(OUT.read_bytes()).hexdigest())
        return
    backend = discover()
    if backend.status()["status"] != "available":
        raise SystemExit(f"Sandbox unavailable: {backend.status().get('reason')}")
    items, _ = candidates()
    results, failures = {}, []
    for index, item in enumerate(items, 1):
        scored = score_code(item["reference"], {"function": item["function"], "cases": item["cases"], "comparison": "python-eq"}, sandbox=backend)
        results[item["id"]] = scored.get("passed") is True
        if not results[item["id"]]:
            failures.append((item["id"], scored.get("outcome"), scored.get("reason")))
        if index % 50 == 0:
            print(f"{index}/{len(items)}", flush=True)
    data = build(results)
    OUT.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    Path(OUT.with_suffix(".reference-check.json")).write_text(json.dumps(
        {"checked": len(items), "passed": sum(results.values()), "failures": failures}, indent=1) + "\n")
    counts = {split: sum(i["split"] == split for i in data["items"]) for split in ("screening", "heldout")}
    print("written", counts, {k: len(v) for k, v in data["excluded"].items()}, hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
