# Feather AI engineering rules

Work in this single repository. Preserve working code and the `feather` CLI. Python application code belongs in `backend/`; React code belongs in `frontend/`.

Use specialized agents for independent tasks with exclusive file ownership. Architect maintains decisions; quantization owns adapters/provenance; evaluation owns scoring/isolation; backend owns local API/jobs; frontend owns dashboard; QA independently reviews acceptance and security; integrator runs checks and commits. Never run competing edits or concurrent heavy inference jobs.

Use actual measurements. Conversion exit code alone does not establish valid generation. Never substitute downloaded quantizations for Feather-created artifacts. Mark unsupported or incomplete results explicitly. Generated code runs only in an isolated, resource-limited container; never directly on the host. Bind services to localhost and constrain filesystem access.

Run focused tests and the full `uv run pytest` suite before commits. Frontend changes also require `npm --prefix frontend run build`. Commit only verified increments after review. Do not push, publish, purchase services or change machine-wide security. Do not run the automation runner recursively from a Codex task. Keep model files, runtime databases, logs, credentials and dependencies out of Git.

The task queue is in `automation/tasks.yaml`; execution evidence belongs in `docs/PROGRESS.md` and `docs/FINAL_REPORT.md`. Blockers do not authorize fabricated completion. Continue independent work when an experiment is blocked. Retry recoverable failures at most twice.
