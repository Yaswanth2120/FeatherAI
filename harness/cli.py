"""FeatherAI harness CLI. Normally invoked via scripts/measure.sh.

  python -m harness.cli session-start [--smoke]
  python -m harness.cli pilot   --session DIR [--windows N] [--window-s S]
  python -m harness.cli matrix  --session DIR --matrix FILE
  python -m harness.cli run     --session DIR --model F --label L --mode M --seed N --prompt-set A
  python -m harness.cli check
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

from . import gates, sysinfo
from .llama_server import run_server
from .sysinfo import ROOT

CSV_FIELDS = [
    "run_id", "session", "smoke", "label", "mode", "seed", "prompt_set", "model_sha256",
    "valid", "invalid_reasons", "paging_contaminated",
    "peak_footprint_bytes", "max_rss_bytes",
    "decode_tok_s_median", "prompt_tok_s_median", "ttft_s_median", "load_s",
    "load_pageins", "measured_window_pageins", "measured_window_s",
    "pagein_rate_pages_per_s", "swap_growth_mb", "prompt_token_mismatches",
    "llama_cpp_commit", "mlx_version", "mlx_lm_version", "macos_version",
]


def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text())


def now_id() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def write_json(p: Path, obj) -> None:
    p.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")


def append_csv(p: Path, row: dict) -> None:
    new = not p.exists()
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def load_prompts(name: str) -> list[dict]:
    p = ROOT / "configs" / "prompt_sets" / f"{name}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def expected_tokens(label: str | None) -> dict:
    """{prompt_id: {"tokens", "text_sha256"}} for the configured tokenizer."""
    p = ROOT / "configs" / "prompt_sets" / "token_counts.yaml"
    if not label or not p.exists():
        return {}
    return (load_yaml(p).get(label) or {}).get("prompts", {})


def token_check(server: dict, prompts: list[dict], label: str | None) -> dict:
    """Compare server prompt_n with the recorded count for each prompt."""
    exp = expected_tokens(label)
    by_id = {p["id"]: p for p in prompts}
    rows, mismatches = [], []
    for r in server["warmup"] + server["requests"]:
        pid = r.get("prompt_id")
        e = exp.get(pid)
        stale = e is not None and pid in by_id and e["text_sha256"] != sysinfo.hashlib.sha256(
            by_id[pid]["prompt"].encode()).hexdigest()
        row = {"prompt_id": pid, "expected": e["tokens"] if e else None,
               "server_prompt_n": r.get("prompt_n"), "count_stale": stale}
        rows.append(row)
        if e is None or stale or e["tokens"] != r.get("prompt_n"):
            mismatches.append(pid)
    return {"tokenizer_label": label, "per_prompt": rows, "mismatches": mismatches}


def prereg_failures(smoke: bool) -> list[str]:
    """Model runs require: prereg-v1 tag exists, thresholds.yaml identical to
    the tagged version, no nulls, clean harness tree, llama.cpp at pinned commit."""
    fails = []
    pins = load_yaml(ROOT / "configs" / "pins.yaml")
    env = sysinfo.environment()
    if env["llama_cpp_commit"] != pins["llama_cpp"]["commit"]:
        fails.append(f"llama.cpp HEAD {env['llama_cpp_commit']} != pinned "
                     f"{pins['llama_cpp']['commit']}")
    if smoke:
        return fails
    if env["prereg_tag_commit"] is None:
        fails.append("git tag prereg-v1 does not exist")
    else:
        r = subprocess.run(["git", "diff", "--quiet", "prereg-v1", "--",
                            "configs/thresholds.yaml"], cwd=ROOT)
        if r.returncode != 0:
            fails.append("configs/thresholds.yaml differs from prereg-v1")
    nulls = gates.find_nulls(load_yaml(ROOT / "configs" / "thresholds.yaml"))
    if nulls:
        fails.append(f"thresholds.yaml has undecided values: {nulls}")
    if env["harness_dirty"]:
        fails.append("working tree has uncommitted changes")
    return fails


# ------------------------------------------------------------------ commands
def cmd_session_start(a) -> int:
    root = ROOT / "results" / ("smoke" if a.smoke else "sessions")
    root.mkdir(parents=True, exist_ok=True)
    boot = sysinfo.boot_time()
    thr = load_yaml(ROOT / "configs" / "thresholds.yaml")
    if not a.smoke and thr["validity_gates"]["reboot_once_per_session"]:
        for s in root.glob("*/session.json"):
            if json.loads(s.read_text())["environment"]["boot_time"] == boot:
                print(f"ABORT: session {s.parent.name} already used this boot "
                      f"({boot}). Reboot first.", file=sys.stderr)
                return 2
    d = root / now_id()
    d.mkdir()
    write_json(d / "session.json", {"started": dt.datetime.now().isoformat(),
                                    "smoke": a.smoke,
                                    "environment": sysinfo.environment()})
    print(d)
    return 0


def _session(a) -> tuple[Path, bool]:
    d = Path(a.session)
    meta = json.loads((d / "session.json").read_text())
    return d, meta["smoke"]


def cmd_pilot(a) -> int:
    """Idle-harness pilot: no model loaded; measure background page-in rate
    and swap drift with the harness otherwise in its measurement posture."""
    d, smoke = _session(a)
    thr = load_yaml(ROOT / "configs" / "thresholds.yaml")
    snap = sysinfo.snapshot()
    fails = gates.pre_run_failures(snap, thr, None, [] if smoke else sysinfo.blocking_processes())
    if fails and not smoke:
        print("ABORT pilot:\n  " + "\n  ".join(fails), file=sys.stderr)
        return 2
    windows = []
    for i in range(a.windows):
        pre = sysinfo.snapshot()
        t0 = time.monotonic()
        time.sleep(a.window_s)
        wall = time.monotonic() - t0
        post = sysinfo.snapshot()
        v = gates.post_run_validity(pre, post, None, thr, wall)
        windows.append({"index": i, "wall_s": wall, "pre": pre, "post": post,
                        "note": "idle window: whole window = measured window",
                        "pageins_delta": v["measured_window_pageins"],
                        "pagein_rate_pages_per_s": v["pagein_rate_pages_per_s"],
                        "swap_growth_mb": v["swap_growth_mb"]})
        print(f"window {i}: {v['measured_window_pageins']} pageins in {wall:.0f}s "
              f"= {v['pagein_rate_pages_per_s']:.3f} pages/s, swap {v['swap_growth_mb']:+.1f} MB")
    rates = [w["pagein_rate_pages_per_s"] for w in windows]
    out = {"kind": "idle_pilot", "smoke": smoke, "gate_failures_ignored": fails if smoke else [],
           "environment": sysinfo.environment(), "page_size": snap["vm_stat"]["page_size"],
           "windows": windows,
           "summary": {"n": len(rates), "min": min(rates), "median": statistics.median(rates),
                       "max": max(rates)}}
    p = d / f"pilot-{now_id()}.json"
    write_json(p, out)
    print(f"wrote {p}\nsummary (pages/s): {out['summary']}")
    return 0


def do_run(d: Path, smoke: bool, model: Path, label: str, mode: str, seed: int,
           prompt_set: str, bench: Path | None = None) -> dict:
    bench = bench or ROOT / "configs" / "bench.yaml"
    cfg = load_yaml(bench)
    thr = load_yaml(ROOT / "configs" / "thresholds.yaml")
    run_id = f"{now_id()}_{label}_{mode}_s{seed}_{prompt_set}"
    rec = {"run_id": run_id, "session": d.name, "smoke": smoke, "label": label,
           "mode": mode, "seed": seed, "prompt_set": prompt_set,
           "MEASURED": not smoke,
           "note": "SMOKE TEST: functional check only, not a measurement" if smoke else None,
           "started": dt.datetime.now().isoformat(),
           "environment": sysinfo.environment(),
           "config": {"bench": cfg,
                      "bench_path": str(bench.relative_to(ROOT)),
                      "bench_sha256": sysinfo.sha256_file(bench),
                      "prompt_set_sha256": sysinfo.sha256_file(
                          ROOT / "configs" / "prompt_sets" / f"{prompt_set}.jsonl"),
                      "warmup_prompt_set_sha256": sysinfo.sha256_file(
                          ROOT / "configs" / "prompt_sets" /
                          f"{cfg.get('warmup_prompt_set', 'warmup')}.jsonl")}}
    # Hash also pre-reads the file, so page-cache state is the same for every run.
    rec["model"] = {"path": str(model), "bytes": model.stat().st_size,
                    "sha256": sysinfo.sha256_file(model)}
    fails = prereg_failures(smoke)
    pre = sysinfo.snapshot()
    fails += gates.pre_run_failures(pre, thr, rec["model"]["bytes"],
                                    [] if smoke else sysinfo.blocking_processes())
    rec["pre"] = pre
    rec["gate_failures"] = fails
    if fails and not smoke:
        rec["status"] = "aborted_by_gate"
        write_json(d / f"{run_id}.json", rec)
        print(f"ABORT {run_id}:\n  " + "\n  ".join(fails), file=sys.stderr)
        return rec
    prompts = load_prompts(prompt_set)
    warm = load_prompts(cfg.get("warmup_prompt_set", "warmup"))
    t0 = time.monotonic()
    server = run_server(cfg, model, mode, prompts, seed, d / f"{run_id}.log", warm)
    t1 = time.monotonic()
    post = sysinfo.snapshot()
    v = gates.post_run_validity(pre, post, server, thr, t1 - t0, t_pre=t0, t_post=t1)
    rec["prompt_tokens"] = token_check(server, prompts + warm, cfg.get("tokenizer_label"))
    reqs = server["requests"]

    def med(k):
        xs = [r[k] for r in reqs if r.get(k) is not None]
        return statistics.median(xs) if xs else None

    tl = server.get("time_l") or {}
    rec |= {"status": "completed", "post": post, "server": server, "validity": v,
            "metrics": {"peak_footprint_bytes": tl.get("peak_footprint_bytes"),
                        "max_rss_bytes": tl.get("max_rss_bytes"),
                        "page_faults_process": tl.get("page_faults"),
                        "decode_tok_s_median": med("decode_tok_s"),
                        "prompt_tok_s_median": med("prompt_tok_s"),
                        "ttft_s_median": med("ttft_s_client"),
                        "load_s": server.get("load_s"),
                        "primary_memory_metric": mode == "no_mmap"},
            "available_bytes_formula": sysinfo.AVAILABLE_FORMULA}
    write_json(d / f"{run_id}.json", rec)
    e, m = rec["environment"], rec["metrics"]
    append_csv(d / "runs.csv", {
        "run_id": run_id, "session": d.name, "smoke": smoke, "label": label, "mode": mode,
        "seed": seed, "prompt_set": prompt_set, "model_sha256": rec["model"]["sha256"],
        "valid": v["valid"], "invalid_reasons": " | ".join(v["invalid_reasons"]),
        "paging_contaminated": v["paging_contaminated"], **m,
        "load_pageins": v["load_pageins"],
        "measured_window_pageins": v["measured_window_pageins"],
        "measured_window_s": v["measured_window_s"],
        "pagein_rate_pages_per_s": v["pagein_rate_pages_per_s"],
        "prompt_token_mismatches": " ".join(rec["prompt_tokens"]["mismatches"]),
        "swap_growth_mb": v["swap_growth_mb"], **{k: e[k] for k in
        ("llama_cpp_commit", "mlx_version", "mlx_lm_version", "macos_version")}})
    print(f"{'SMOKE ' if smoke else ''}{run_id}: valid={v['valid']} "
          f"footprint={m['peak_footprint_bytes']} decode={m['decode_tok_s_median']} "
          f"prompt={m['prompt_tok_s_median']} ttft={m['ttft_s_median']} "
          f"load_pageins={v['load_pageins']} measured_rate={v['pagein_rate_pages_per_s']} "
          f"token_mismatches={rec['prompt_tokens']['mismatches']} reasons={v['invalid_reasons']}")
    return rec


def cmd_run(a) -> int:
    d, smoke = _session(a)
    if a.bench and not smoke:
        print("ABORT: --bench override is only allowed in smoke sessions", file=sys.stderr)
        return 2
    rec = do_run(d, smoke, Path(a.model).resolve(), a.label, a.mode, a.seed, a.prompt_set,
                 (ROOT / a.bench) if a.bench else None)
    return 0 if rec.get("status") == "completed" else 2


def cmd_matrix(a) -> int:
    d, smoke = _session(a)
    cfg = load_yaml(ROOT / "configs" / "bench.yaml")
    thr = load_yaml(ROOT / "configs" / "thresholds.yaml")
    cooldown = 0 if smoke else thr["validity_gates"]["cooldown_between_runs_s"]
    jobs = [(r, mode) for r in load_yaml(Path(a.matrix))["runs"] for mode in cfg["modes"]]
    for i, (r, mode) in enumerate(jobs):
        if i > 0 and cooldown:
            print(f"cooldown {cooldown}s ...", flush=True)
            time.sleep(cooldown)
        bench = ROOT / r["bench"] if r.get("bench") else None
        if bench and not smoke:
            print("ABORT: per-run bench override is only allowed in smoke sessions", file=sys.stderr)
            return 2
        rec = do_run(d, smoke, (ROOT / r["model"]).resolve(), r["label"], mode,
                     int(r["seed"]), r["prompt_set"], bench)
        if rec.get("status") != "completed":
            print("stopping matrix: run aborted by gate", file=sys.stderr)
            return 2
    return 0


def cmd_check(a) -> int:
    snap = sysinfo.snapshot()
    thr = load_yaml(ROOT / "configs" / "thresholds.yaml")
    fails = prereg_failures(False) + gates.pre_run_failures(
        snap, thr, None, sysinfo.blocking_processes())
    print(json.dumps({"environment": sysinfo.environment(),
                      "swap": snap["swap"], "available_mb": snap["available_bytes"] / 2**20,
                      "power": snap["power"]["ac_power"], "lowpowermode": snap["lowpowermode"],
                      "therm_ok": snap["therm"]["ok"], "gate_failures": fails}, indent=2))
    return 0 if not fails else 2


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="harness")
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("session-start"); s.add_argument("--smoke", action="store_true")
    s.set_defaults(f=cmd_session_start)
    s = sp.add_parser("pilot"); s.add_argument("--session", required=True)
    s.add_argument("--windows", type=int, default=20); s.add_argument("--window-s", type=float, default=30)
    s.set_defaults(f=cmd_pilot)
    s = sp.add_parser("matrix"); s.add_argument("--session", required=True)
    s.add_argument("--matrix", required=True); s.set_defaults(f=cmd_matrix)
    s = sp.add_parser("run"); s.add_argument("--session", required=True)
    for k in ("--model", "--label", "--prompt-set"):
        s.add_argument(k, required=True)
    s.add_argument("--mode", choices=["no_mmap", "mmap"], required=True)
    s.add_argument("--seed", type=int, required=True); s.add_argument("--bench")
    s.set_defaults(f=cmd_run)
    s = sp.add_parser("check"); s.set_defaults(f=cmd_check)
    a = p.parse_args(argv)
    return a.f(a)


if __name__ == "__main__":
    sys.exit(main())
