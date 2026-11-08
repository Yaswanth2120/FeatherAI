"""Apply the pre-declared page-in rule to an idle-pilot JSON.

    uv run python scripts/derive_pagein_limit.py results/sessions/<id>/pilot-*.json

Rule (configs/thresholds.yaml): limit = max(2 x highest pilot-window rate, 10 pages/s),
from 20 consecutive 30 s windows under measured-run conditions. Prints the limit.
Refuses smoke pilots, pilots with gate failures, or the wrong window design.
"""
import json, sys

EXPECT_WINDOWS, EXPECT_S, TOL_S, FLOOR, FACTOR = 20, 30.0, 2.0, 10.0, 2.0
MIN_UPTIME_S = 900  # must equal thresholds.yaml validity_gates.min_uptime_before_session_s


def derive(pilot: dict) -> dict:
    problems = []
    if pilot.get("kind") != "idle_pilot":
        problems.append("not an idle_pilot file")
    if pilot.get("smoke"):
        problems.append("smoke pilot (gates not enforced)")
    if pilot.get("gate_failures_ignored"):
        problems.append(f"gate failures: {pilot['gate_failures_ignored']}")
    w = pilot.get("windows", [])
    if len(w) != EXPECT_WINDOWS:
        problems.append(f"{len(w)} windows, expected {EXPECT_WINDOWS}")
    off = [x["index"] for x in w if abs(x["wall_s"] - EXPECT_S) > TOL_S]
    if off:
        problems.append(f"windows not ~{EXPECT_S:.0f} s: {off}")
    up = w[0].get("pre", {}).get("uptime_s") if w else None
    if up is None:
        problems.append("no uptime recorded (pilot predates the settle gate)")
    elif up < MIN_UPTIME_S:
        problems.append(f"pilot started at {up/60:.1f} min uptime < {MIN_UPTIME_S/60:.0f} min")
    rates = [x["pagein_rate_pages_per_s"] for x in w]
    hi = max(rates) if rates else None
    limit = max(FACTOR * hi, FLOOR) if hi is not None else None
    return {"ok": not problems, "problems": problems, "n_windows": len(w),
            "highest_window_rate": hi, "limit_pages_per_s": limit,
            "rates": rates, "page_size": pilot.get("page_size")}


if __name__ == "__main__":
    r = derive(json.load(open(sys.argv[1])))
    print(json.dumps({k: v for k, v in r.items() if k != "rates"}, indent=2))
    print("window rates (pages/s):", [round(x, 3) for x in r["rates"]])
    sys.exit(0 if r["ok"] else 2)
