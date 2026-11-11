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

### task08 end-to-end workflow (recorded 2026-11-09, before the workflow ran)
- **One workflow, one identifier.** A single SmolLM2-135M run goes through the local API under one `workflow_id`: import (source hash verified), quantization jobs for Q8_0 and Q4_K_M, generation validation of each artifact, benchmark jobs, held-out `coding-v1` evaluation jobs in the Docker sandbox, a recommendation, and JSON/CSV exports. Historical runs are never reused.
- **Recommendation constraints.** Fixed in `experiments/e2e/recommendation-constraints.json`: objective fastest, ≤ 1 GiB, coding pass-rate lower 95% bound ≥ 0.70. A truthful `no_eligible_configuration` result is accepted; the constraints will not be relaxed.
- **Settings.** Context 2048, seed 42 and temperature 0 for every inference stage. Benchmark: 5 measured requests plus 1 warm-up per artifact. Coding: 384 tokens.
- **Stricter task08 gate.**
  - The v0.1 gate checked only seven true/false flags.
  - The v0.2 gate also requires that the source and artifact hashes chain consistently across all stages, that each stage record belongs to the `workflow_id`, that the sandbox was available and every coding item was scored, and that the constraints file is unchanged.
  - It also requires that the exported files exist with their recorded hashes, and that the interrupted-job and failure regression tests pass.
  - This makes the specification stricter; nothing is relaxed.

### coding-v2 design (recorded 2026-11-10, before any coding-v2 data is downloaded or any model has seen it)
- **Goal.** Detect a paired drop in pass rate between a quantized variant and F16 on the same problems, using an exact two-sided McNemar test at α = 0.05 with 80% power.
- **Sizing.** n ≈ (z₀.₉₇₅·√ψ + z₀.₈·√(ψ − δ²))² / δ², where δ is the drop and ψ the discordant-pair rate.
  - Measured ψ on coding-v1 (SmolLM2): 0.10 for Q8_0 and 0.20 for Q4_K_M.
  - δ = 10 pp needs about 152 problems at ψ = 0.20 and about 171 at ψ = 0.25.
  - δ = 5 pp needs about 312 problems at ψ = 0.10 and about 626 at ψ = 0.20.
  - **Target:** at least 300 held-out problems. That detects a 10 pp drop up to ψ ≈ 0.4, and a 5 pp drop only if ψ ≤ 0.10. A 5 pp drop at higher discordance is explicitly out of reach and will be reported as such.
- **Source.** Hand-writing 300+ independently checked problems is not feasible, so the problems come from the MBPP sanitized set:
  - Google Research, CC-BY-4.0; problems and tests written independently of this project.
  - Pinned by commit and file SHA-256.
  - An item is included only if all its tests are of the form `assert <name>(<literal args>) == <literal>`, parsed with `ast` and literal evaluation. Everything else (`math.isclose`, `set(...)`, side effects, imports in tests) is excluded and counted.
  - Expected values stay on the host. Tuples and sets are compared with an explicit canonical encoding, never via lossy JSON coercion.
  - The prompt shows the task text and the function signature from the first test's call. Test inputs and outputs are never shown.
  - *Correction (2026-11-10, before any model ran on coding-v2):* the frozen implementation states the function **name and positional-argument count** from the first test's call, not a full signature. No test values are shown.
- **Splits.** A seeded shuffle (seed 20261110) puts 40 problems in screening and the rest in held-out. The dataset is frozen and hash-pinned before any model runs on it.
- **Disclosed limitation.** MBPP is public, and models such as Qwen2.5-Coder may have seen it in training. That can inflate absolute pass rates. It affects variant-vs-F16 differences only through ceiling effects, so absolute pass rates must not be quoted as capability claims.
- **Correction and freeze record (2026-11-10).**
  - **License.** The design above calls MBPP "CC-BY-4.0". At the pinned commit, the google-research repository is Apache-2.0 and the MBPP README states no separate data license; CC-BY-4.0 is what the Hugging Face dataset card lists, and that was not verified. See `backend/evaluation/datasets/third_party/mbpp/NOTICE.md`.
  - **Freeze.** 401 problems passed the filter. 400 references passed their own tests in the sandbox; mbpp-123 failed and was excluded.
  - **Result.** 40 screening + **360 held-out**, SHA-256 `37e35bd0…`, frozen before any model saw it. Arguments and expected values use the type-preserving encoding, decoded inside the sandbox.

### coding-v2 run on Qwen2.5-Coder-0.5B: analysis plan (recorded 2026-11-10, before any model output on coding-v2)
- **Scope.** This run measures **coding** quality only. It cannot confirm or refute the Q8_0 *reasoning* signal from reasoning-v1; that needs the separate replication planned below.
- **Artifacts.** F16 source `bb69b303…`, plus Q8_0 `0d93a22a…` and Q4_K_M `3dc9c14a…` from `experiments/studies/qwen2.5-coder-0.5b-clean` (hashes re-verified before use).
- **Data and settings.**
  - `coding-v2` held-out split only: 360 problems, SHA-256 `37e35bd0…`. mbpp-123 is excluded.
  - Context 2048, temperature 0, seed 42, `n_predict` 384, ChatML template, llama-server `--load-mode none`.
  - Runs are sequential (F16, then Q8_0, then Q4_K_M) with the Docker sandbox available.
- **Screening dry run.** F16 on the 40 screening problems first, as a format check only. Prompts are frozen; any change requires `coding-v3`.
- **Primary hypotheses.**
  - H1: Q8_0 ≠ F16 and H2: Q4_K_M ≠ F16 in held-out pass rate.
  - Test: exact two-sided McNemar on paired outcomes, with Holm correction across {H1, H2} at family α = 0.05.
- **Effect estimates.** Paired difference with a paired bootstrap over problems: 10 000 resamples, seed 20261110, reporting both 95% and 90% CIs.
- **Equivalence.** The margin is ±10 pp, the design's detectable effect. "Equivalent within ±10 pp" is claimed only if the 90% CI lies entirely inside (−10, +10) pp, a TOST-style check. Otherwise the result is "not shown equivalent", even when there is no detectable difference.
- **Baseline validity.** Results are informative only if F16 solves at least 10 problems and at least 25%.
- **Reported, not tested.** Outcome categories (truncation, runtime errors, timeouts), per-problem outputs, and the unchanged screening-free held-out results. No other subgroup tests.

### Reasoning replication for the Q8_0 signal: plan (recorded 2026-11-10; nothing collected yet)
- **Signal to replicate.** On reasoning-v1 (160 held-out items), Q8_0 − F16 = −3.75 pp: 7 items correct only for F16 and 1 only for Q8_0. The paired bootstrap 95% CI was [−7.5, −0.6] and the exact McNemar p = 0.070, from one of 12 unadjusted comparisons. This is treated as hypothesis-generating only.
- **New data.** A new frozen version, `reasoning-v2`, built with the same generator code and the same four categories in equal shares.
  - It uses a different seed, and every question must differ from all reasoning-v1 questions.
  - Size: **150 held-out items per category (600 total) plus 40 screening items**.
  - Basis: discordance ψ ≈ 0.05–0.10 and δ = 3.75 pp, two-sided α = 0.025 (Holm over Q8_0 and Q4_K_M), 80% power, gives about 505–675 items. 600 sits inside that range. At ψ ≥ 0.10 the study may be underpowered, and that will be reported.
- **Primary hypothesis.** Two-sided: Q8_0's held-out pass rate on reasoning-v2 differs from F16's. The direction of interest is a drop, but the test is two-sided. Test: exact two-sided McNemar with Holm correction across {Q8_0, Q4_K_M}. Per-category results are exploratory only. *(Wording fixed 2026-11-10, before any data, to match the two-sided test.)*
- **Equivalence.** Margin ±5 pp, using the 90% paired-bootstrap CI.
- **Freeze rule.** The dataset is frozen and hash-pinned, and this plan committed, before any model runs on reasoning-v2.
- **Freeze record (2026-11-10).** `reasoning-v2` was built and frozen before any model ran on it: 600 held-out items (150 per category) + 40 screening, seed 20261110, no question shared with reasoning-v1, SHA-256 `16202bdd…`. Running it on the Qwen variants is the next step of the replication.
- **Execution and outcome mapping (recorded 2026-11-11, before any reasoning-v2 model output).**
  - **Settings.** Identical to the reasoning-v1 study: context 2048, temperature 0, seed 42, `n_predict` 256, ChatML, `--load-mode none`, strict `Answer:` extraction.
  - **Artifacts.** The hash-verified F16/Q8_0/Q4_K_M from `experiments/studies/qwen2.5-coder-0.5b-clean`.
  - **Order.** A screening dry run on F16 first (format check only; never analysed; prompts frozen), then held-out F16 → Q8_0 → Q4_K_M, sequentially, from a clean committed tree. The runner (`backend/study/reasoning_v2_run.py`) records resumptions and per-item infrastructure failures itself.
  - **Primary result, Q8_0 vs F16 on all 600 held-out items:**
    - **replicated:** Holm-adjusted exact McNemar rejects at family α = 0.05 *and* Q8_0 is lower;
    - **opposite direction:** rejects with Q8_0 higher;
    - **not detected:** not rejected *and* the 90% paired-bootstrap CI lies inside (−5, +5) pp (equivalent within ±5 pp);
    - **inconclusive:** otherwise, or if any held-out item is unscored.
  - **Reporting.** The baseline-validity rule (F16 solves at least 10 and at least 25%) is reported. Q4_K_M is reported with the same rules. Per-category results are exploratory. If observed Q8_0 discordance is ≥ 0.10, that is reported as possible under-power.
- **Execution bug fix (2026-11-11, before any reasoning-v2 model output).** The first launch attempts crashed with `KeyError: 'reasoning-v2'` before generating anything: the suite was missing from the evaluation runner's registry. Both attempts are recorded in the run file. The fix only registers the suite, with the same 256-token limit as reasoning-v1. Settings, data, order and analysis are unchanged.
