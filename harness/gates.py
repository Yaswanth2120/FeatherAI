"""Pre-run gates (abort) and post-run validity checks (invalidate / flag).

All numeric limits come from configs/thresholds.yaml; nothing here is hard-coded.
"""
from __future__ import annotations


def find_nulls(d, prefix="") -> list[str]:
    out = []
    if isinstance(d, dict):
        for k, v in d.items():
            out += find_nulls(v, f"{prefix}{k}.")
    elif d is None:
        out.append(prefix.rstrip("."))
    return out


def pre_run_failures(snap: dict, thr: dict, model_bytes: int | None,
                     blockers: list[str]) -> list[str]:
    g = thr["validity_gates"]
    fails = []
    if blockers:
        fails.append(f"blocking processes running: {', '.join(blockers)}")
    if g["require_ac_power"] and not snap["power"]["ac_power"]:
        fails.append("not on AC power")
    if g["require_low_power_mode_off"] and snap["lowpowermode"] != 0:
        fails.append(f"Low Power Mode is {snap['lowpowermode']!r} (need 0)")
    if g["require_no_thermal_warning"] and not snap["therm"]["ok"]:
        fails.append(f"thermal check not clean: {snap['therm']['unexpected_lines']}")
    lim = g["swap_used_at_start_max_mb"]
    if lim is None:
        fails.append("swap_used_at_start_max_mb is null")
    elif snap["swap"]["used_mb"] > lim:
        fails.append(f"swap used {snap['swap']['used_mb']:.1f} MB > {lim} MB")
    if model_bytes is not None:
        head = g["free_memory_headroom_mb"]
        if head is None:
            fails.append("free_memory_headroom_mb is null (precheck undefined)")
        else:
            need = model_bytes + head * 2**20
            if snap["available_bytes"] < need:
                fails.append(f"available {snap['available_bytes']/2**20:.0f} MB < "
                             f"model {model_bytes/2**20:.0f} MB + headroom {head} MB")
    return fails


def post_run_validity(pre: dict, post: dict, server: dict | None, thr: dict,
                      wall_s: float) -> dict:
    g = thr["validity_gates"]
    swap_growth = post["swap"]["used_mb"] - pre["swap"]["used_mb"]
    pageins = post["vm_stat"]["pageins"] - pre["vm_stat"]["pageins"]
    rate = pageins / wall_s if wall_s > 0 else None
    invalid = []
    lim = g["swap_growth_during_run_max_mb"]
    if lim is None or swap_growth > lim:
        invalid.append(f"swap grew {swap_growth:.1f} MB (limit {lim})")
    if server is not None:
        if server.get("error"):
            invalid.append(f"server/request error: {server['error']}")
        if not server.get("metal", {}).get("ok"):
            invalid.append(f"Metal backend not confirmed: {server.get('metal')}")
        if server.get("time_l") is None:
            invalid.append("no /usr/bin/time -l report")
    pr_lim = g["pagein_rate_contaminated_pages_per_s"]
    contaminated = None if pr_lim is None or rate is None else rate > pr_lim
    return {
        "valid": not invalid,
        "invalid_reasons": invalid,
        "swap_growth_mb": swap_growth,
        "pageins_delta": pageins,
        "pageouts_delta": post["vm_stat"]["pageouts"] - pre["vm_stat"]["pageouts"],
        "swapins_delta": post["vm_stat"]["swapins"] - pre["vm_stat"]["swapins"],
        "swapouts_delta": post["vm_stat"]["swapouts"] - pre["vm_stat"]["swapouts"],
        "pagein_rate_pages_per_s": rate,
        "paging_contaminated": contaminated,   # None = threshold not yet calibrated
        "thermal_after_run_ok": post["therm"]["ok"],
    }
