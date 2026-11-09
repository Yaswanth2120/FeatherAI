"""Repeated, order-balanced performance measurement across variants of one source model."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import random
import time

from backend.study.server import ServerSession
from backend.study.stats import describe, median_ci, ratio_ci

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "benchmark.txt"
METRICS = ("decode_tokens_per_second", "prefill_tokens_per_second", "ttft_seconds", "wall_seconds")
SWAP_GROWTH_FLAG_BYTES = 64 * 1024 ** 2


def round_orders(labels: list[str], rounds: int, seed: int) -> list[list[str]]:
    """Seeded shuffle of all permutations, cycled: every variant appears once per round."""
    permutations = [list(p) for p in itertools.permutations(labels)]
    random.Random(seed).shuffle(permutations)
    return [permutations[i % len(permutations)] for i in range(rounds)]


def trial_metrics(result: dict) -> dict:
    runtime = result["runtime"]
    decode = runtime["eval_count"] / (runtime["eval_duration"] / 1e9) if runtime.get("eval_duration") else None
    prefill = runtime["prompt_eval_count"] / (runtime["prompt_eval_duration"] / 1e9) if runtime.get("prompt_eval_duration") else None
    return {"decode_tokens_per_second": decode, "prefill_tokens_per_second": prefill, "ttft_seconds": result["ttft_seconds"],
            "wall_seconds": result["wall_seconds"], "output_tokens": runtime["eval_count"], "prompt_tokens": runtime["prompt_eval_count"]}


def session_flags(record: dict) -> list[str]:
    flags = []
    growth = record["system_delta"].get("swap_used_delta_bytes")
    if growth is not None and growth > SWAP_GROWTH_FLAG_BYTES:
        flags.append("memory_pressure:swap_growth")
    levels = [record["system_before"].get("pressure_level"), record["system_after"].get("pressure_level")]
    if any(level is not None and level >= 2 for level in levels):
        flags.append("memory_pressure:level")
    if not record["metal"]["verified"]:
        flags.append("metal_unverified")
    if record.get("docker_vm_running"):
        flags.append("docker_vm_running")
    return flags


def run_benchmark(variants: dict[str, dict], server: str, out_dir: Path, source_sha256: str, rounds=5, warmups=1,
                  measured=4, cooldown=30, seed=20261109, context=2048, tokens=128, threads=4, log=print) -> dict:
    """``variants`` maps label -> inspected artifact. Writes sessions.jsonl and trials.csv incrementally."""
    prompt = PROMPT_PATH.read_text()
    options = {"num_ctx": context, "num_predict": tokens, "seed": 42, "temperature": 0, "ignore_eos": True}
    orders = round_orders(list(variants), rounds, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions_path, trials_path = out_dir / "sessions.jsonl", out_dir / "trials.csv"
    sessions_path.write_text("")  # one benchmark run per file; never mix stale sessions with a new analysis
    trials_path.write_text("variant,round,session,ordinal,warmup,decode_tokens_per_second,prefill_tokens_per_second,"
                           "ttft_seconds,wall_seconds,output_tokens,prompt_tokens,error\n")
    sessions = []
    seen = set()
    index = 0
    for round_number, order in enumerate(orders, 1):
        for label in order:
            if sessions:
                time.sleep(cooldown)
            index += 1
            artifact = variants[label]
            session = ServerSession(artifact, server, out_dir / f"session-{index:02d}-{label}.log", context=context,
                                    threads=threads, source_sha256=source_sha256)
            entry = {"session": index, "round": round_number, "variant": label, "first_load_in_study": label not in seen,
                     "artifact_sha256": artifact["sha256"], "trials": []}
            seen.add(label)
            try:
                session.start()
                model = session.models()[0]["name"]
                for ordinal in range(warmups + measured):
                    trial = {"ordinal": ordinal, "warmup": ordinal < warmups}
                    try:
                        trial.update(trial_metrics(session.generate(model, prompt, options)))
                        if trial["output_tokens"] != tokens:
                            trial["error"] = f"output token count {trial['output_tokens']} != {tokens}"
                    except Exception as error:  # recorded, excluded, never imputed
                        trial["error"] = f"{type(error).__name__}: {error}"[:300]
                    entry["trials"].append(trial)
                    if ordinal == warmups - 1:
                        session.mark("after_warmup")
                session.mark("after_measured")
            except Exception as error:
                entry["error"] = f"{type(error).__name__}: {error}"[:500]
            finally:
                entry["server"] = session.stop() if not session.record else session.record
            entry["flags"] = session_flags(entry["server"]) if "system_delta" in entry["server"] else ["incomplete"]
            sessions.append(entry)
            with sessions_path.open("a") as stream:
                stream.write(json.dumps(entry) + "\n")
            with trials_path.open("a") as stream:
                for t in entry["trials"]:
                    stream.write(",".join(str(x) for x in [label, round_number, index, t["ordinal"], t["warmup"],
                                 *[t.get(k, "") for k in METRICS], t.get("output_tokens", ""), t.get("prompt_tokens", ""),
                                 '"' + t.get("error", "").replace('"', "'") + '"']) + "\n")
            measured_ok = [t for t in entry["trials"] if not t["warmup"] and not t.get("error")]
            decode = [round(t["decode_tokens_per_second"], 1) for t in measured_ok]
            log(f"session {index:02d} round {round_number} {label}: load {entry['server'].get('load_seconds') or 0:.2f}s "
                f"peak footprint {(entry['server'].get('time_l', {}).get('peak_memory_footprint_bytes') or 0) / 2**20:.0f} MiB "
                f"decode {decode} flags {entry['flags']}{' ERROR ' + entry['error'] if entry.get('error') else ''}")
    return {"design": {"rounds": rounds, "warmups_per_session": warmups, "measured_per_session": measured,
                       "cooldown_seconds": cooldown, "seed": seed, "orders": orders, "prompt_sha256": _sha(prompt),
                       "options": options, "context": context, "threads": threads},
            "sessions": sessions, "analysis": analyze(sessions, list(variants))}


def _sha(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def analyze(sessions: list[dict], labels: list[str], baseline: str | None = None) -> dict:
    baseline = baseline or labels[0]

    def groups(label, metric, exclude_flagged=False):
        out = []
        for s in sessions:
            if s["variant"] != label or s.get("error") or (exclude_flagged and any(f.startswith("memory_pressure") for f in s["flags"])):
                continue
            values = [t[metric] for t in s["trials"] if not t["warmup"] and not t.get("error") and t.get(metric) is not None]
            if values:
                out.append(values)
        return out

    result = {}
    for scope, exclude in [("primary_all_sessions", False), ("sensitivity_unflagged_sessions", True)]:
        per_variant = {}
        for label in labels:
            mine = [s for s in sessions if s["variant"] == label]
            entry = {"sessions": len(mine), "failed_sessions": sum(bool(s.get("error")) for s in mine),
                     "failed_trials": sum(bool(t.get("error")) for s in mine for t in s["trials"] if not t["warmup"]),
                     "flagged_sessions": sum(any(f.startswith("memory_pressure") for f in s["flags"]) for s in mine)}
            for metric in METRICS:
                g = groups(label, metric, exclude)
                flat = [v for values in g for v in values]
                entry[metric] = {**(describe(flat) or {}), "median_ci95": median_ci(g) if g else None, "sessions_used": len(g)}
                if label != baseline and g and groups(baseline, metric, exclude):
                    entry[metric]["ratio_to_" + baseline] = ratio_ci(g, groups(baseline, metric, exclude))
            used = [s for s in mine if not s.get("error") and not (exclude and any(f.startswith("memory_pressure") for f in s["flags"]))]
            peaks = [s["server"]["time_l"].get("peak_memory_footprint_bytes") for s in used if s["server"].get("time_l")]
            rss = [s["server"]["time_l"].get("max_rss_bytes") for s in used if s["server"].get("time_l")]
            entry["peak_memory_footprint_bytes"] = describe([p for p in peaks if p])
            entry["max_rss_bytes"] = describe([p for p in rss if p])
            entry["load_seconds_first_in_study"] = describe([s["server"]["load_seconds"] for s in used if s["first_load_in_study"] and s["server"].get("load_seconds")])
            entry["load_seconds_repeat"] = describe([s["server"]["load_seconds"] for s in used if not s["first_load_in_study"] and s["server"].get("load_seconds")])
            per_variant[label] = entry
        result[scope] = per_variant
    return result
