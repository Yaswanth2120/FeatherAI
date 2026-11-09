# Final implementation report — 2026-11-09

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
