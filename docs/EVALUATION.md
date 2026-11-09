# Evaluation and measured recommendations

## Evaluation API

```python
from backend.backends.ollama import Ollama
from backend.evaluation.runner import evaluate, compare_evaluations
backend = Ollama()
try:
    report = evaluate(backend, 'qwen3:1.7b', suite='reasoning-small', repeats=1,
                      seed=42, context=1024, tokens=128)
finally:
    backend.close()
```

The caller persists the returned JSON report. Available suites are `reasoning-small` (three exact-answer arithmetic tasks) and `coding-small` (binary search, palindrome, stable deduplication). These are **public screening tasks**, not held-out benchmark claims. Each report records prompts, hashes, raw generations, task failures, model metadata/digest, runtime version, hardware, seed and decoding settings. Sampling temperature is zero; runtime nondeterminism can remain. Repeats are bounded to 20; context and generation tokens have explicit caps.

Reasoning accepts only a whitespace-normalized exact answer. Failed generations are tracked separately and never become passes. The summary reports scored/pass/failed/unsupported counts and a Wilson 95% interval. Repeating the same deterministic questions does not create independent evidence; these intervals are descriptive and must not be represented as a generalization guarantee.

`compare_evaluations` requires matching suite hash, settings hash, repeat count, completed task sets, and an explicit common `source_sha256`. Model digests alone do not prove common source weights. Attach `source_sha256` only from verified artifact lineage. Accuracy differences include a conservative difference of marginal Wilson bounds, explicitly not a calibrated paired confidence interval.

`backend.evaluation.scoring.perplexity(log_probabilities)` computes exp(mean negative log probability) only for explicitly provided held-out target-token log probabilities. None returns `unsupported`. Ollama adapter does not provide those probabilities, so its evaluation reports mark perplexity unsupported; generated output token probabilities must not be substituted for held-out teacher-forced scores.

## Generated-code isolation

Generated Python never executes in the host interpreter. Coding evaluation requires Docker and an already-present `python:3.12-alpine` image. The runner resolves its image ID and disables automatic pulls. If the daemon or image is unavailable, coding evaluation returns unsupported without running model generation.

The Docker process uses no host mounts, network none, a read-only root filesystem, no capabilities, no new privileges, non-root UID, 32 PIDs, 128 MiB RAM with no extra swap, half a CPU, and an 8 MiB noexec tmpfs. Python sets CPU, file-size and file-descriptor limits. The Docker client output is bounded to 64 KiB; generated code is limited to 16 KiB. A wall timeout kills the client and force-removes the uniquely named container. Cleanup failures are surfaced, not ignored. Expected answers remain on the host; only inputs and code enter the container. Unit-test outcomes compare returned values, not claims printed by model code.

This is a local container isolation boundary, not a claim of adversarial sandbox certification. No machine security setting is changed. See [Docker run documentation](https://docs.docker.com/reference/cli/docker/container/run) for the isolation flags.

## Recommendation API

```python
from backend.optimization.recommender import recommend
result = recommend(configurations, available_memory_bytes=4_000_000_000,
                   max_memory_bytes=3_000_000_000, min_quality=0.7,
                   max_latency_seconds=2.0, objective='fastest')
```

Each configuration is an object with:

| Field | Required measured evidence |
| --- | --- |
| `id` | Unique configuration identifier |
| `source_sha256` | Verified common source weights identity |
| `workload_sha256` | Common prompt/evaluation workload identity |
| `settings_sha256` | Common context/decoding/measurement policy identity |
| `peak_memory_bytes` | Observed runtime peak, not model file size |
| `quality` | Measured correctness fraction in [0,1] |
| `latency_seconds` | Measured representative request latency |
| `decode_tokens_per_second` | Measured generation throughput |
| `file_size_bytes` | Artifact file size |
| `measured_trials` | Positive measured performance-trial count |
| `quality_trials` | Positive scored evaluation-trial count |
| `quality_ci` | Optional [lower,upper] measured quality interval |

Missing, nonfinite, invalid, or duplicate evidence is rejected with reasons. Different source/workload/settings groups are not ranked together. Candidates must fit both available and maximum memory; meet minimum quality (using the interval lower bound when supplied); and satisfy the optional latency constraint. Objectives are `fastest`, `smallest`, `lowest_latency`, and `highest_quality`. Ties resolve by identifier. Reports return eligible rankings, rejection reasons, evidence limitations, and confidence metadata. Empty/incomplete evidence returns `insufficient_evidence`; complete evidence with no fit returns `no_eligible_configuration`. No metric is inferred from bit width and an Ollama benchmark without process peak memory is not sufficient evidence for a memory-fit recommendation.

## Verification in this environment

Unit tests cover scoring, uncertainty, unsupported Docker behavior, resource flags, timeout cleanup, bounded output, deterministic suite generation and failures, provenance comparisons, ranking constraints, finite metrics and missing evidence. Docker daemon is currently unavailable, so no real generated-code execution has been claimed. Real reasoning runs must be recorded separately by the integrator, sequentially with other heavy model work.

## Joining persisted evidence

`backend.optimization.evidence.configurations_from_reports(benchmarks, evaluations, suite='reasoning-small')` converts raw saved benchmark reports plus saved quality reports to the recommendation schema. It matches **both** verified `metadata.source_sha256` and `metadata.model.artifact_sha256` against the evaluation's `source_sha256` and `model.artifact_sha256`. Display names are never sufficient. It chooses the newest completed evaluation in the requested suite, deterministically by timestamp and ID; unsupported/partial evaluations do not qualify.

Performance medians are recomputed from raw measured trials; warmups are excluded. Incomplete benchmark trial counts and failed trials are marked insufficient. Measured memory must be present in benchmark `metadata.peak_memory_bytes`; no size-to-memory conversion is attempted. Quality and its Wilson bounds are recomputed from actual scored task rows. Workload identity includes benchmark prompt hash and quality-suite hash; settings identity includes the benchmark settings hash plus evaluation options, runtime version and repetition count. Missing lineage or measurements stay null and are rejected by the recommendation engine.

When hosted by the managed llama.cpp adapter, evaluation records the adapter's verified `source_sha256` and model `artifact_sha256` automatically. Imported Ollama models without verified provenance retain null source lineage. API worker containers carry an exact `feather.job` label from `FEATHER_JOB_ID`; `backend.evaluation.coding.cleanup_job(job_id)` removes only matching containers and raises if daemon cleanup cannot be confirmed. The API manager invokes this for cancellation/recovery in addition to the evaluator's normal `finally` cleanup.

Recommendation reports explicitly warn that sampled process RSS is not total GPU/unified-memory allocation and can miss peaks between samples. Evidence records retain the benchmark's `memory_metric` and `memory_note`. Setting `min_quality=0` permits diagnostic performance ranking even when every task fails; this must not be presented as a useful-quality recommendation. Wilson bounds are clamped to contain the observed proportion at all-fail/all-pass endpoints to avoid floating-point exclusion of 0 or 1; this does not change any scored outcome.
