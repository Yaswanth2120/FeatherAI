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


def _vm_delta(a: dict, b: dict) -> dict:
    return {k: b["vm_stat"][k] - a["vm_stat"][k]
            for k in ("pageins", "pageouts", "swapins", "swapouts")}


def phase_paging(pre: dict, post: dict, phases: dict, t_pre: float, t_post: float) -> dict:
    """vm_stat deltas and durations per phase. Phases missing (e.g. after an
    error) are skipped; boundaries are pre -> after_load -> after_warmup ->
    after_measured -> post."""
    pts = [("pre", t_pre, pre)]
    for name in ("after_load", "after_warmup", "after_measured"):
        if name in phases:
            pts.append((name, phases[name]["t_monotonic"], phases[name]["snap"]))
    pts.append(("post", t_post, post))
    label = {("pre", "after_load"): "load", ("after_load", "after_warmup"): "warmup",
             ("after_warmup", "after_measured"): "measured",
             ("after_measured", "post"): "shutdown"}
    out = {}
    for (n0, t0, s0), (n1, t1, s1) in zip(pts, pts[1:]):
        out[label.get((n0, n1), f"{n0}->{n1}")] = _vm_delta(s0, s1) | {"duration_s": t1 - t0}
    return out


def post_run_validity(pre: dict, post: dict, server: dict | None, thr: dict,
                      wall_s: float, t_pre: float | None = None,
                      t_post: float | None = None) -> dict:
    """Swap growth is judged over the whole process lifetime. The page-in
    contamination rate is judged ONLY over the measured-request window
    (after load and warm-up), so model-load page-ins don't count; for the idle
    pilot (server=None) the whole window is the measured window."""
    g = thr["validity_gates"]
    swap_growth = post["swap"]["used_mb"] - pre["swap"]["used_mb"]
    whole = _vm_delta(pre, post)
    invalid = []
    lim = g["swap_growth_during_run_max_mb"]
    if lim is None or swap_growth > lim:
        invalid.append(f"swap grew {swap_growth:.1f} MB (limit {lim})")
    phases = None
    if server is None:
        win_pageins, win_s = whole["pageins"], wall_s
    else:
        phases = phase_paging(pre, post, server.get("phase_snapshots", {}),
                              t_pre if t_pre is not None else 0.0,
                              t_post if t_post is not None else wall_s)
        m = phases.get("measured")
        win_pageins, win_s = (m["pageins"], m["duration_s"]) if m else (None, None)
        if server.get("error"):
            invalid.append(f"server/request error: {server['error']}")
        if not server.get("metal", {}).get("ok"):
            invalid.append(f"Metal backend not confirmed: {server.get('metal')}")
        if server.get("time_l") is None:
            invalid.append("no /usr/bin/time -l report")
        if m is None:
            invalid.append("measured-request window incomplete")
    rate = win_pageins / win_s if win_pageins is not None and win_s else None
    pr_lim = g["pagein_rate_contaminated_pages_per_s"]
    contaminated = None if pr_lim is None or rate is None else rate > pr_lim
    return {
        "valid": not invalid,
        "invalid_reasons": invalid,
        "swap_growth_mb": swap_growth,
        "whole_run_vm_delta": whole,
        "phase_paging": phases,                      # load / warmup / measured / shutdown
        "load_pageins": phases["load"]["pageins"] if phases and "load" in phases else None,
        "measured_window_pageins": win_pageins,
        "measured_window_s": win_s,
        "pagein_rate_pages_per_s": rate,             # measured-request window only
        "paging_contaminated": contaminated,         # None = threshold not yet calibrated
        "thermal_after_run_ok": post["therm"]["ok"],
    }
