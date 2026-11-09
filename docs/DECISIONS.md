# Design decisions

- Preserve a single repository with Python under backend; terminal command remains feather.
- Reuse the current Ollama benchmark and SQLite modules rather than replace them.
- Build llama.cpp inside ignored .feather/tools rather than install machine-wide packages. Pin and record its source commit.
- Prefer official Qwen 0.5B FP16 GGUF source for the first conversion; downloading higher-precision GGUF is source import, not a downloaded quantized result.
- Heavy inference and conversion run sequentially on the 8 GiB host.
- Generated Python code requires a restricted Docker container. Docker daemon absence is reported as unsupported, with no host-execution fallback.
- Autonomous Codex execution uses documented exec JSON mode and structured independent review. Local commits are permitted; remote publication is not.
- Dashboard and API stay local. Sites hosting registration/deployment is explicitly excluded by the user's no-publishing instruction.

Sources: https://learn.chatgpt.com/docs/agent-configuration/subagents and https://learn.chatgpt.com/docs/non-interactive-mode; llama.cpp quantization usage https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md.

## v0.2 decisions — recorded 2026-11-09, before any v0.2 held-out output or Qwen conversion existed

### Automation
- The Codex model is never hardcoded. Resolution: `FEATHER_CODEX_MODEL`, then `automation/config.yaml`, then a model accepted by a live `feather automation doctor --probe --select`, then the CLI default.
- Model-entitlement, outdated-CLI, authentication and usage-limit failures are non-retryable: the task becomes `blocked` after one invocation and its dependents do not start.
- Evidence (`experiments/automation-model-probes.json`): codex-cli 0.142.3 with a ChatGPT login accepts none of the eight listed models. Upgrading the CLI changes user-level software outside this repository and is left to the user.

### Coding sandbox
- Only `DockerExecutionBackend` may run generated code. Settings: pinned image `docker.io/library/python@sha256:e064671c…` (python:3.12-alpine), `--network none`, `--read-only`, `--cap-drop ALL`, `no-new-privileges`, uid 65534, 128 MiB memory with no swap, 0.5 CPU, 32 pids, 2 s per test case, 20 s wall time, 64 KiB output. It refuses to run when the daemon, seccomp or the pinned image cannot be confirmed. There is no host fallback and no automatic image pull.

### Quantization memory admission (revised from evidence)
- v0.1 rule: 2 × source + 256 MiB. This was an unmeasured guess, and it is what rejected Qwen2.5-Coder-0.5B. That was a pre-flight estimate at the quantization stage; `llama-quantize` never ran.
- Measured (`experiments/memory/quantize-smollm.json`): `llama-quantize` peak max RSS is 1.01–1.11 × source size and peak footprint is 0.92–0.94 × source size.
- v0.2 rule: **1.5 × source + 256 MiB**, about 35% above the largest measured RSS ratio. The guard stays enforced against available memory.
- The next conversion re-checks the rule. If any conversion's measured max RSS exceeds 1.5 × source, the rule reverts and the violation is recorded.

### Quantization study v0.2 (pre-registered)
- **Model and variants.** Source is Qwen2.5-Coder-0.5B-Instruct F16 GGUF from the official repository `Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF`, revision `ebb20151`, local SHA-256 `bb69b303…`. Variants are F16 (the source itself), plus Q8_0 and Q4_K_M created by Feather from that source with llama.cpp `9d340ec3`. Downloaded quantizations are never used.
- **Server settings for every variant.** llama-server with `--load-mode none` (weights in anonymous memory, so they count in the footprint), `-c 2048`, `-np 1`, `-t 4`, `--cache-ram 0`, `--fit off`, `--no-warmup`, and Metal offload of all layers, verified from the log. Chat template is fixed ChatML. Requests use temperature 0, seed 42 and `cache_prompt: false`.
- **Benchmark design.**
  - 5 rounds. Each round runs one server session per variant, in a randomized balanced order (seeded shuffle of the 6 permutations, seed 20261109).
  - Each session has 1 warm-up and then 4 measured requests, giving 20 measured trials per variant.
  - Each request uses the fixed prompt in `backend/study/prompts/benchmark.txt` with `n_predict 128` and `ignore_eos`.
  - There is a 30 s cooldown between sessions.
- **Recorded per session.**
  - Load latency (spawn to `/health` OK). The first session per variant is labelled "first load in study" because page-cache state is uncontrolled; true cold loads would need cache purging, which needs sudo, and is not claimed.
  - Peak memory footprint and max RSS from `/usr/bin/time -l`.
  - Footprint snapshots after load, after warm-up and after the measured requests.
  - Swap change, memory-pressure level, `pmset -g therm` output, and whether the Docker VM was running.
- **Session flags.** A session is flagged `memory_pressure` when swap use grows by more than 64 MiB during it, or the pressure level is ≥ 2. Flagged sessions are kept.
  - Primary analysis uses all completed sessions.
  - A sensitivity analysis excludes flagged sessions.
  - Failed requests are counted and excluded, never imputed.
- **Statistics.**
  - Per variant: median, mean, SD, p95, min and max.
  - 95% CI for the median from a two-stage cluster bootstrap: resample sessions, then trials within each session. 10,000 resamples, seed 20261109.
  - Variant/F16 ratios of medians, with CIs from independent bootstraps paired by resample index.
  - Uncertainty is never computed from rounded aggregates.
- **Quality.**
  - The held-out splits of `reasoning-v1` (160 items) and `coding-v1` (30 items) are run once per variant: context 2048, seed 42, temperature 0, `n_predict` 256 for reasoning and 384 for coding.
  - Before any held-out output exists, the screening splits are run on F16 only, as a format dry run. Prompts may change only before held-out outputs exist; any change creates a new dataset version.
  - Comparisons are paired: bootstrap over items plus exact McNemar.
  - A category is informative only if F16 solves at least 10 held-out items and at least 25% of them; otherwise it is labelled uninformative.
  - "Detectable change" means the paired 95% CI excludes 0. There are no pass/fail gates.
- **Perplexity and KLD.**
  - Tool: `llama-perplexity` at the same commit, `-c 512 --chunks 20`.
  - Corpus: a fixed list of CPython 3.12.9 standard-library files, concatenated in order and hashed. It is code-domain text the model may have seen in training, so it is used for relative KLD and PPL between variants, not as an absolute capability measure.
  - KLD is computed against the F16 base logits.
- **Measurement conditions.** These measurements are run by the agent while VS Code and other user applications are open. They are labelled accordingly, and `feather study` reruns the whole protocol from Terminal for cleaner conditions.
- **1.5B model.** Attempted only if the user approves the roughly 3.1 GB F16 download, and only if the measured 0.5B F16 peak footprint × 3 + 1 GiB fits within available memory. Otherwise it is recorded as not attempted, with the reason.

### Amendment 1 (2026-11-09, recorded AFTER the SmolLM2 study results existed — post hoc, disclosed as such)
- **Model substitution.** The pre-registered primary model, Qwen2.5-Coder-0.5B, could not run. Both the quantization pre-flight guard and the runtime pre-flight guard rejected it (`experiments/studies/qwen2.5-coder-0.5b/study.json`).
  - The full protocol was executed unchanged on SmolLM2-135M-Instruct F16 (`f535f83e…`), the v0.1 fallback source.
  - The Qwen study stays pre-registered and BLOCKED; it was not replaced.
  - Results for SmolLM2 are reported as a protocol-conformant study of a different model, not as evidence about Qwen.
- **Order design.** With 5 rounds and 6 permutations, the first five shuffled permutations were used. F16 ran first in a round once and last twice. This follows mechanically from the pre-registered "seeded shuffle, cycled" rule; it is not position-balanced and is disclosed in the study report.
- **Sensitivity analysis.** 13 of 15 sessions carried a memory-pressure flag. The unflagged-only analysis is not estimable for F16 or Q4_K_M; this is reported, not worked around.
- **Provenance.** The study code (`backend/study/`) was uncommitted at run time, on top of commit `1f6f8f1`. After the run, `run.py` was changed only to fix `toolchain_commit` resolution and to handle blocked phases. The corrected `toolchain_commit` is recorded under `provenance_corrections`. Future runs record `working_tree_diff_sha256` and the untracked files.
- **Review fixes after the run.** Sandbox exit classification, the accuracy denominator, phase status for partial or failed runs, and the perplexity exit-code check were all fixed after the study. Each was verified not to change any recorded SmolLM2 result: every item scored, every exit code 0, no infrastructure failures, and one benchmark run of 15 sessions.
