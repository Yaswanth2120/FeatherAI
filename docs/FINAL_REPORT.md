# Final report: Feather AI v0.2 (2026-11-09)

## Status

**v0.2 is not complete.** Two items in the definition of done are BLOCKED by external conditions:
- **Live Codex automation:** the CLI and account have no usable model.
- **The pre-registered Qwen study:** both memory guards reject it under current memory availability.

**Optional 1.5B experiment: BLOCKED** pending approval for a roughly 3.1 GB download and sufficient free memory; it was not attempted.

**Task queue:** task01–task09 are completed; each was adopted only after its acceptance gate passed and a commit-bound independent review approved it. task04 was adopted at `16fbe37` and task08 at `b862de3`. task09 was adopted at `4b704ce` after independent documentation review approved with no findings and its acceptance/regression gates re-ran.

Everything else is implemented and verified with real runs.

## Implementation

| Area | Result |
|---|---|
| **Automation** | Model resolution order: env var, then config, then probed selection, then CLI default; no hardcoded model. Model-entitlement, outdated-CLI, authentication and usage-limit failures block the task after one invocation and never retry. Every invocation is recorded. Commands: `feather automation doctor [--probe]`, and `feather automation selftest` (two dependent tasks in a throwaway repository). Live verification is **BLOCKED** (see below). |
| **Coding sandbox** | `ExecutionBackend` with Docker and Unavailable implementations. Settings: pinned image digest, no network, read-only root, `--cap-drop ALL`, `no-new-privileges`, uid 65534, 128 MiB / 0.5 CPU / 32 pids, per-case alarms, bounded output, OOM detection, forced container removal. If isolation can't be confirmed, the status is `blocked` and code is never run on the host. **Real isolation self-test: 9/9 probes passed.** |
| **Memory diagnostics** | `/usr/bin/time -l` peaks, footprint snapshots by stage, swap, memory pressure, vm_stat and thermal state. Every conversion records its memory. Separate pre-flight guards cover quantization and runtime. |
| **Evaluation** | Hash-pinned `reasoning-v1` (40 screening + 160 held-out) and `coding-v1` (10 screening + 30 held-out hidden-test problems). Strict answer extraction with invalid and truncated categories. Paired bootstrap plus McNemar comparisons. Pre-registered baseline-validity labels. |
| **Benchmark** | Order-balanced server sessions, 20 measured trials per variant. Reports median, mean, SD, p95, min and max, a session-clustered bootstrap CI and ratio CIs. Sensitivity analysis excludes flagged sessions. Raw JSONL and CSV are kept. |
| **API and dashboard** | `/api/status` (automation, sandbox, memory) and `/api/studies`. The Overview page has a status panel. Comparison Reports has one study panel per study, showing per-phase status, variants, CIs, quality validity and KLD. Blocked and incomplete phases are labelled. |

## Tests (final run)

| Check | Result |
|---|---|
| `uv run pytest` | **181 passed, 0 failed, 0 skipped.** The real-Docker sandbox tests and real-evidence end-to-end gate tests executed. One upstream Starlette deprecation warning. |
| `npm --prefix frontend run build` | **Passed** (the existing bundle-size warning remains) |
| Playwright `dashboard.spec.ts` against a live API with real saved data | **1 passed** |
| `automation/validate.py --task task04` | **Passes**; task04 adopted as completed at reviewed commit `16fbe37` |
| `automation/validate.py --task task08` | **Passes**: one clean-tree workflow (`8cab4706`), hash-chained across all stages, plus the interrupted-job and failure tests. task08 adopted as completed at reviewed commit `b862de3`. |

## Model experiments (measured; details in docs/studies/quantization-study-v0.2.md)

**SmolLM2-135M-Instruct (all phases complete).** F16 source `f535f83e…`; Q8_0 and Q4_K_M created by Feather; llama.cpp `9d340ec`.

| | F16 | Q8_0 | Q4_K_M |
|---|---:|---:|---:|
| Artifact size | 258.3 MiB | 138.1 MiB | 100.6 MiB |
| Peak footprint, median | 472.1 MiB | 300.5 MiB | 264.2 MiB |
| Decode tok/s, median [95% CI] | 220.8 [207.1, 225.8] | 252.6 [244.9, 256.4] | 262.8 [246.2, 268.5] |
| Decode ratio to F16 | — | 1.14 [1.10, 1.23] | 1.19 [1.11, 1.27] |
| Prefill ratio to F16 | — | 0.94 [0.90, 0.96] | 0.84 [0.79, 0.85] |
| Coding pass rate, held-out 30 | 36.7% [21.9, 54.5] | 26.7% [14.2, 44.4] | 36.7% [21.9, 54.5] |
| Reasoning, held-out 160 | 1.9% (uninformative baseline) | 1.9% | 1.3% |
| Mean KLD vs F16 | reference | 0.0012 | 0.0445 |

- **Coding differences vs F16:** Q8_0 −10 pp [−20, 0] (p = 0.25), Q4_K_M 0 pp [−16.7, +16.7]. No detectable change at n = 30.
- **Reasoning:** cannot measure quantization loss for this model.
- **Measurement conditions:** sessions ran under machine-wide memory pressure. Most sessions are flagged, so the sensitivity analysis could only be computed for Q8_0.

**Qwen2.5-Coder-0.5B-Instruct: BLOCKED.**
- The quantization pre-flight needs 2 168 074 048 bytes; 1 615 937 536 were available.
- The runtime pre-flight needs 1 803 296 640 bytes; 1 460 060 160 were available.
- Neither guard was relaxed.
- The v0.1 "Qwen exceeded memory" result was a pre-flight estimate, not an observed OOM.

## End-to-end workflow (task08)

One real SmolLM2-135M workflow (`8cab4706`, from clean commit `f957640`) ran through the local API, with a hash chain across every stage. Steps:
1. Import source `f535f83e`.
2. Quantization jobs create Q8_0 `a2c82248` and Q4_K_M `76ab8f09`.
3. Generation validation (runtime compatibility).
4. Benchmark jobs.
5. Held-out coding evaluation in the sandbox: 8/30 and 11/30.
6. Recommendation under constraints pre-registered in `c9db2cc`: **`no_eligible_configuration`**, because both coding lower bounds (0.142 and 0.219) are below 0.70.
7. Exports.

Run history:
- Failed run `3d138b53` (API never started) is preserved.
- Run `2377c241` is preserved but superseded because it ran from an uncommitted tree.
- The task08 gate verifies identity, hash chaining, raw job records, settings consistency, sandboxing, unchanged constraints committed before the run, and export hashes, plus the interrupted-job and failure tests.

## Known defects and limits

- **Benchmark memory figures differ by path.** API benchmark jobs (`backend/benchmarks/runner.py`) still use sampled process RSS (startup excluded) and 5 trials; only `feather quant-study` uses `/usr/bin/time -l` footprints and 20 clustered trials. So API benchmark numbers (for example the task08 decode medians) are not directly comparable with the study's.
- **Generation validation is a weak check.** It only requires non-empty output. It is a compatibility check, not a correctness check.
- **One dependency warning.** Starlette's TestClient emits a deprecation warning for httpx.
- **Large frontend bundle.** It is about 634 kB, which triggers a Vite warning.
- **iCloud Desktop.** The project lives in an iCloud-synced Desktop folder. Earlier, venv `.pth` files carried the hidden flag, which breaks the `feather` entry point; recreating them fixed it.

## Automation

- No real Codex task ran successfully, and no live dependent task advanced.
- Probing all eight listed models with codex-cli 0.142.3 and a ChatGPT login: four are refused for this account type and four need a newer CLI.
- Recovery, dependency ordering and non-retry blocking are verified with mocked tests only.

## Security

- Generated code runs only inside the hardened Docker container, and isolation was verified with real probes.
- With Docker unavailable, coding evaluation reports `blocked` and runs nothing.
- Docker is an isolation layer, not an absolute security boundary.

## Remaining blockers (need user action)

1. **Codex:** upgrade the CLI yourself (`codex update`), then run `uv run feather automation doctor --probe --probe-candidates --select` and `uv run feather automation selftest`.
2. **Qwen study:** run `feather quant-study` for Qwen from Terminal with VS Code, browsers and Docker closed, so at least about 2.1 GB is available at admission. Start Docker afterwards only for the quality phase.
3. **1.5B model:** needs your approval for a roughly 3.1 GB download, and more free memory than this machine had today.

## Commit references (v0.2, local only, nothing pushed)

| Commit | Change |
|---|---|
| `e9e02d3` | Automation: model selection, non-retryable failures, doctor, self-test |
| `f08f70a` | Docker sandbox and hash-pinned reasoning/coding datasets |
| `1f6f8f1` | Memory diagnostics, measured quantization memory, study pre-registration |
| `fc93634` | Study runner, SmolLM2 results, Qwen blocked diagnosis, first review fixes |
| `5a9c9b8`, `16fbe37` | Review fixes; reproducible task04 gate (task04 adopted at `16fbe37`) |
| `d31fd39`, `a9fcbf5` | Task state after task04 adoption. Note: `d31fd39` corrupted `PROGRESS.md`; it was repaired in `b862de3`. |
| `c9db2cc` | task08 design and recommendation constraints, pre-registered before any run |
| `9470e7d`, `533614f`, `f957640` | End-to-end workflow, stricter gate and review fixes |
| `d36ed7e` | Accepted workflow evidence (`8cab4706`) |
| `2eb0b49`, `b862de3` | Docs; PROGRESS repair (task08 adopted at `b862de3`) |
| `f94be82`, `4b704ce` | task09 documentation and reviewed fixes (task09 adopted at `4b704ce`) |

## Next engineering milestone

Run the pre-registered Qwen2.5-Coder-0.5B study and a Qwen end-to-end workflow under clean conditions. SmolLM2 is too weak for reasoning, so a larger coding set (well beyond 30 problems) is needed before small quality differences can be detected.

---

# Superseded: v0.1 final implementation report — 2026-11-09

> **Historical record.** These statements describe the v0.1 state and were superseded by the v0.2 report above: Docker unavailable, coding unmeasured, task04 failing.

## Status

**MVP incomplete.** Real FP16-to-Q4/Q8 quantization, generated-artifact inference, comparable timing runs, public reasoning evaluation, evidence-based recommendations, FastAPI and the five-page dashboard are implemented and exercised locally. Real coding accuracy and the complete original end-to-end acceptance gate remain blocked.

## Implemented and verified

- Preserved working baseline; the original15 tests still pass. No project recreation, nested FeatherAI directory, backup folder, push or publication.
- Seven documented project agent roles and bounded, resumable CLI orchestration with dependency scheduling, two maximum retries, persistent checkpoints, independent structured review, timeout enforcement, interruption recovery and reviewed local commits.
- Real llama.cpp toolchain built locally at commit 9d340ec341e079d2c28afd50871afce358cdbf48. Genuine higher-precision SmolLM2 weights converted by Feather to Q4_K_M and Q8_0; hashes, architecture, tokenizer, tensor structure and generation verified.
- Streaming benchmarks, 50ms process-tree RSS sampling, individual SQLite commits, JSON/CSV exports and reproducibility metadata.
- Public screening reasoning/coding frameworks, Wilson intervals, artifact/source comparison, and explicit unsupported perplexity. Generated code requires restricted Docker execution.
- Deterministic constraints, confidence/limitations and exact artifact/source evidence joins for recommendations.
- Localhost FastAPI process jobs with cancellation/timeouts/recovery, labeled container cleanup, body/origin/path guards and persisted experiment/evaluation history.
- React/TypeScript dashboard with actual hardware, runtime/artifact models, conversion controls, job history, charts, comparisons, evaluations and downloads.

## Real measurements

SmolLM2-135M-Instruct, same source/tokenizer/build, context1024, seed42, temperature0, max128 tokens, disabled prompt cache, one warmup + five trials/configuration, one slot and two CPU threads. Metal default offloading enabled. All final performance trials succeeded.

| Configuration | Artifact MiB | Sampled peak RSS MiB | Median TTFT ms | Median decode tok/s | Median wall s | Reasoning |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| FP16 | 258.34 | 395.84 | 11.08 | 223.36 | 0.5843 | 0/3 |
| Q4_K_M | 100.57 | 239.27 | 12.72 | 279.89 | 0.4702 | 0/3 |
| Q8_0 | 138.10 | 274.94 | 11.50 | 267.05 | 0.4913 | 0/3 |

These are short hot-model measurements, not sustained/cold-start results. Sampled process RSS excludes startup peaks and potentially GPU allocations; it does not establish total memory fit. All reasoning accuracy intervals are [0,0.5615] (Wilson95%, tiny public suite). No useful model quality was established. Coding and perplexity are unmeasured, not zero.

The diagnostic recommendation with min_quality0 selected the fastest Q4 measured run. Default70% minimum quality produces no eligible configuration. The ranking does not establish useful deployment quality.

## Tests and integration evidence

- Python regression and focused tests: **99 passed**; final results recorded in PROGRESS.md. Covers baseline, quantization error/provenance, token boundaries, native streaming/cleanup, scoring, isolation controls, evidence joins, recommendations, process jobs/cancel/timeout/restart, request guards, task scheduling/retries/resume/review/timeout.
- React TypeScript production build passed. Vite reports a bundle-size warning (approximately634kB uncompressed); no build failure.
- Live local Playwright test passed on desktop and mobile: navigation, saved reports, comparison, recommendation and JSON download; no page errors and no mobile horizontal overflow. Browser screenshots are ignored runtime artifacts under .feather/. The in-app browser automation tool was unavailable; installed Chrome was used through the test runner.
- Real API integration completed source import → new Q4 conversion → managed native benchmark → reasoning evaluation → recommendation → JSON export. See experiments/api-real-run.json.
- task02 real-quantization evidence gate passes against local weight hashes. task04 intentionally fails because coding_correctness is unavailable; task08 cannot pass while task04 is incomplete.

## Source provenance

The fallback FP16 source is bartowski/SmolLM2-135M-Instruct-GGUF revision 09816acd5d99df7be770d85ea30822623dab342c. Source SHA-256: `f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57`. This is a higher-precision imported source, not a downloaded quantization. Output hashes and raw results are in experiments/case-study.json and quantization-evidence.json. Model weights and compiled tools are ignored by Git. The earlier qwen3:1.7b Ollama smoke report is preserved separately and is not treated as a precision comparison.

## Blockers and remaining limitations

1. Official Qwen2.5 Coder0.5B source conversion was rejected by the RAM guard: estimate2,801,286,912 bytes vs1,807,138,816 available. The requested smaller-model fallback succeeded; the Qwen guard was never bypassed.
2. Docker executable exists but its daemon is unavailable. Generated-code scoring returns unsupported. Full coding-quality comparison, task04 acceptance and task08 complete integration remain blocked. No unrestricted host code fallback or machine security change was attempted.
3. Quality suites are public screening, not disjoint held-out/calibration corpora. All tested reasoning scores are0/3. Broader quality, supported model PPL, and useful quality-constrained recommendations require additional evidence.
4. MLX conversion, comprehensive Metal memory, thermal sampling, sustained benchmarks and load-latency distributions are not implemented.
5. Binaries/weights/local SQLite databases are not committed. Archived reports are reproducibility metadata, not independently rerunnable experiments without the pinned toolchain and source.
6. Live bounded Codex CLI review invocation exited1 because the selected account rejected configured gpt-6.1-sol (model_not_found). Runner tests/dry-run pass; unattended model execution remains blocked until CLI account/model compatibility is resolved. No account settings or credentials were changed.
7. Canonical pytest currently emits one upstream Starlette TestClient/httpx deprecation warning. Browser test assumes installed Chrome and populated local saved data. API locks serialize API jobs, not separately launched CLI experiments.

## Next priorities

Enable the isolated local coding runtime within the memory budget; run FP16/Q4/Q8 coding scoring and reattempt task04/task08. Add held-out datasets and a model with useful task quality, supported token-logprob/perplexity measurements, MLX support and more complete memory/thermal measurement.

## Commit references

`fb7fc4e` implements the local lab/API/dashboard; `9cadc2c` implements automation and agent definitions. The following evidence/documentation commit is recorded by Git history and PROGRESS.md. No remote pushes were made.
