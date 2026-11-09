# Architecture and API

A single local app serves the React build and a FastAPI REST API. SQLite stores performance trials, quality evaluations and job state. The CLI and API reuse the same Python benchmark, quantization and scoring modules. Heavy API work runs one subprocess at a time, protected by a per-data-directory worker lock. A separate bounded Codex development runner lives in automation; it is not the inference scheduler.

## Components

- `backend/core`: host inspection; full-file hashes; memory-mapped GGUF tensor, architecture and tokenizer validation.
- `backend/backends`: streaming Ollama and owned localhost llama.cpp server adapters.
- `backend/quantization`: teaching NumPy quantizer and real managed llama.cpp conversion.
- `backend/benchmarks`: warmups, repeated requests, statistics and 50ms process-tree RSS sampling for owned native servers.
- `backend/evaluation`: deterministic public coding/reasoning screening, Docker isolation, Wilson intervals and explicit unsupported perplexity.
- `backend/optimization`: exact artifact/source evidence joins and deterministic constrained ranking.
- `backend/storage`: SQLite raw performance and quality reports.
- `backend/api`: localhost/origin guards, bounded request bodies, typed requests, process jobs, cancellation and interruption recovery.
- `frontend`: five React/TypeScript views, Recharts and API-based job/report controls.
- `backend/evaluation/sandbox.py` (v0.2): `ExecutionBackend`, `DockerExecutionBackend` (pinned digest, no network, read-only, no capabilities, uid 65534, resource limits) and `UnavailableExecutionBackend` (status `blocked`, never executes).
- `backend/evaluation/datasets` (v0.2): frozen, hash-pinned `reasoning-v1` and `coding-v1`. `answers.py` holds strict answer extraction; `scoring.py` holds paired bootstrap and McNemar comparisons.
- `backend/diagnostics` (v0.2): `/usr/bin/time -l` peaks, footprint snapshots, swap, memory pressure, vm_stat and thermal state.
- `backend/study` (v0.2): managed llama-server sessions, the order-balanced benchmark, cluster-bootstrap statistics, llama-perplexity, resumable `feather quant-study`, and summary exports.
- `backend/e2e.py` (v0.2): a single `feather e2e` workflow through the real local API under one `workflow_id`.

## API usage

Start `uv run feather serve`. `/docs` provides the generated request schema. Relevant routes:

| Route | Purpose |
| --- | --- |
| GET `/api/health`, `/api/hardware` | Local service and hardware |
| GET `/api/models`, `/api/artifacts` | Installed Ollama models / managed conversions |
| POST `/api/jobs/quantization` | `{source:"source-f16.gguf", target:"Q4_K_M"}` inside model root |
| POST `/api/jobs/benchmark` | `{model:"qwen3:1.7b", repeats:5,warmups:1,context:1024,tokens:128,seed:42}` |
| POST `/api/jobs/evaluation` | `{model:"qwen3:1.7b", suite:"reasoning-small",repeats:1,context:1024,tokens:128,seed:42}`. `suite` may also be `coding-small`, `reasoning-v1` or `coding-v1`; the v1 suites run the held-out split. |
| GET `/api/jobs`, `/api/jobs/{id}` | Persisted lifecycle/results/errors |
| POST `/api/jobs/{id}/cancel` | Cancellation request |
| GET `/api/experiments`, `/api/evaluations` | Saved performance / quality |
| GET `/api/experiments/{id}/export?format=json` | Reproducible JSON; CSV supported |
| POST `/api/comparisons` | `{experiment_ids:["id-a","id-b"]}` |
| POST `/api/recommendations` | IDs plus available/max memory bytes, minimum quality, optional maximum latency, fastest/smallest/lowest_latency objective, and `quality_suite` (`reasoning-small`, `coding-small`, `reasoning-v1` or `coding-v1`; default `reasoning-small`) |
| GET `/api/status` | Automation health (offline doctor checks), sandbox isolation status, system memory state |
| GET `/api/studies` | Status-explicit summaries of `experiments/studies/*/study.json`. A malformed study is returned as `invalid`. |

Native benchmark/evaluation jobs additionally set `backend:"llama.cpp"` and `artifact_job_id` to a completed conversion UUID. Paths/hashes are checked against managed artifacts. Executables are server configuration, never client-supplied commands. Default model root is `models/`; use a filename relative to that directory. Backend configuration is available through `create_app(data_dir, model_root, ollama_url, quantizer, server_executable, job_timeout)` and CLI serve options.

## Lifecycles and recovery

API states include queued, running, completed, partial, unsupported, blocked (sandbox unavailable), failed, cancelled, interrupted and cleanup_required. Results remain available for partial/unsupported evaluations. On restart, interrupted jobs are not blindly replayed. Child process groups are reaped on cancel/timeout; generated-code Docker containers carry a job label and receive targeted daemon cleanup. Unconfirmed container cleanup halts the worker rather than scheduling more code. One API worker per data directory; do not run multiple Uvicorn workers.

Developer automation states are separate: pending, running, testing, reviewing, completed, blocked, failed. See automation README for bounded execution and resumption.

## Local security and resource boundaries

API binds localhost, validates Host and same-origin requests, and limits actual streamed body size to 64KiB. Model inputs must resolve inside the configured root, including symlink checks. Outputs use fresh job directories; shells are not used to construct model commands. These controls assume the local account and repository are trusted; they are not remote multi-user authentication.

Use the UI or study command sequentially. The API lock serializes API jobs; a separately launched CLI can still compete for the same machine. Do not run those heavy paths concurrently.
