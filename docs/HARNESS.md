# FeatherAI benchmark harness

## Setup (once)
    scripts/build_llama.sh      # llama.cpp v0.9.0 @ cd959bae99ff4bed22bef2604f45241c09bbb8c8, Metal (embedded library)
    uv sync --frozen            # Python 3.12, mlx 0.32.0, mlx-lm 0.31.3 (uv.lock)
    uv run pytest -q

## Measuring (you, from Terminal.app, VS Code and Claude Code closed)
1. Reboot. Plug in AC power. Turn Low Power Mode off.
2. `scripts/measure.sh check` prints environment and gate status, and runs nothing.
3. `scripts/measure.sh pilot` runs the idle-harness pilot (default 5 x 300 s windows, no model)
   and writes results/sessions/<id>/pilot-*.json.
4. Use the pilot to fill `pagein_rate_contaminated_pages_per_s` by the pre-declared rule, commit, then `git tag prereg-v1`.
5. Reboot again (one session per boot), then `scripts/measure.sh matrix configs/matrix.yaml`.

## What one run is
One llama-server process under `/usr/bin/time -l`:
load -> /health -> 1 warm-up request (prompt set `warmup`, discarded) -> each prompt in
the measured set, streamed, sequentially -> SIGTERM -> parse the time -l report.
Each matrix entry runs once per mode (`no_mmap` = `--load-mode none` = primary memory
metric, then `mmap`), with a 300 s cooldown between processes.

| Metric | Source |
|---|---|
| peak_footprint_bytes (primary, no_mmap) | `/usr/bin/time -l` "peak memory footprint" |
| max_rss_bytes | `/usr/bin/time -l` "maximum resident set size" |
| decode tok/s | server `timings.predicted_per_second`, median over prompts |
| prompt tok/s | server `timings.prompt_per_second`, median over prompts |
| TTFT | client wall clock from request send to first streamed token, median |
| page-ins, pageouts, swapins/outs | `vm_stat` delta across the process lifetime |
| swap growth | `sysctl vm.swapusage` used delta |
| thermal | `pmset -g therm` before (gate) and after (recorded) |

## Gates
Abort before the run: VS Code or Claude Code running; not on AC power; Low Power Mode on;
`pmset -g therm` output not clean (fails closed); swap used above the limit; available memory
below model size + headroom; llama.cpp not at the pinned commit; thresholds.yaml differs from
`prereg-v1` or contains nulls; uncommitted changes.
Invalidate after the run: swap growth above the limit; Metal not confirmed (needs
`ggml_metal_init: found device` and N/N layers offloaded); a request error; no time -l report.
Flag (keep the run): page-in rate above the calibrated limit, `paging_contaminated`.

## Smoke mode
`session-start --smoke` writes to results/smoke/ (gitignored). Gates are logged, not enforced;
there's no cooldown; records carry `"MEASURED": false`. It's a functional check only.
