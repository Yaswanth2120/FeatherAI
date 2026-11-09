# Progress — v0.2 (2026-11-09)

Status here is based only on acceptance gates and committed evidence.

| v0.2 item | Status | Evidence |
| --- | --- | --- |
| Codex automation recovery | **Code complete; live run BLOCKED** | `experiments/automation-model-probes.json`: with codex-cli 0.142.3 and a ChatGPT login, 4 models are refused for the account and 4 need a newer CLI. The doctor, model resolution and non-retryable blocking are covered by `tests/test_automation_config.py` (mocked). The live self-test has not run. |
| Secure coding evaluation | **Complete** | `backend/evaluation/sandbox.py`. Real Docker self-test: `experiments/sandbox-selftest.json` (9/9 probes; network blocked, read-only root, uid 65534, memory and fork limits, no leftover containers). Real-container tests in `tests/test_sandbox.py` ran, none skipped. |
| Qwen memory diagnosis | **Diagnosed; Qwen BLOCKED** | Rejected by the quantization and runtime pre-flight guards; no OOM was ever observed. Evidence: `experiments/memory/quantize-smollm.json`, `experiments/studies/qwen2.5-coder-0.5b/study.json`. |
| Reasoning evaluation rebuild | **Complete** | `reasoning-v1` and `coding-v1` are hash-pinned. The v0.1 0/3 result was caused both by an exact-match scoring defect and by model weakness (SmolLM2 solved 3/160 held-out items under strict scoring). |
| Benchmark reliability | **Complete** | 15 sessions, 20 measured trials per variant, session-clustered bootstrap. Raw trials in `experiments/studies/smollm2-135m/benchmark/`. |
| Quantization study | **Complete for SmolLM2-135M; primary Qwen study BLOCKED** | `docs/studies/quantization-study-v0.2.md`, `experiments/studies/` |
| API and dashboard integration | **Complete** | `/api/status` and `/api/studies`; status and study panels. The Playwright test asserts COMPLETE/INCOMPLETE and blocked labels against real data. |
| Final validation | 158 pytest passed, 0 skipped; frontend build passed; 1 browser test passed | See FINAL_REPORT.md |

**v0.1 task queue (`automation/tasks.yaml`)**

| Task | Status |
|---|---|
| task04 | The acceptance gate now **passes** on `experiments/comparison-evidence-v0.2.json`. It is not marked completed in the queue: adoption needs a commit-bound independent review record. |
| task08 | **Blocked.** No single real run has gone from import through coding evaluation to recommendation and export. The v0.2 study covers every stage except the recommendation. |
| task09 | **Blocked.** Depends on task08. |


---

# Progress — v0.1 (2026-11-09, superseded above where noted)

| Task | Status | Evidence |
| --- | --- | --- |
| 01 Repository audit | Verified | docs/AUDIT.md; original15 tests pass |
| 02 Real model quantization | Verified with smaller fallback | experiments/quantization-evidence.json; local hash gate passes; real nonempty generation |
| 03 Quality engine | Tooling implemented; real coding blocked | Unit tests; actual reasoning0/3/configuration; coding unsupported without Docker |
| 04 Performance + quality comparison | Blocked | Real FP16/Q4/Q8 timing/RSS reports; coding_correctness unavailable and acceptance fails intentionally |
| 05 Recommendations | Verified with limitations | Real artifact/source joins; diagnostic0-quality ranking and default-quality rejection |
| 06 FastAPI | Verified | Process integration/failure/recovery tests and experiments/api-real-run.json |
| 07 Dashboard | Verified | TypeScript build and live Playwright desktop/mobile workflow |
| 08 Complete end-to-end acceptance | Blocked | Reasoning-only path succeeds; required coding evidence missing |
| 09 Documentation/report | Written; full MVP sign-off blocked | README, architecture, methodology and FINAL_REPORT.md reflect incomplete acceptance |

Validation and independent review are required before local commits. No code has been pushed or published. Runtime task state is stored in ignored .feather/automation; committed task definitions remain a reproducible specification rather than fabricated completion markers.

## Validation and commits

- Final Python regression: **99 passed**, one upstream Starlette TestClient/httpx deprecation warning.
- Production TypeScript/Vite build passed; bundle-size warning remains.
- Live Chrome/Playwright workflow: **1 passed**, including desktop/mobile layout and report download.
- task02 acceptance passed. task04 acceptance failed as expected for missing measured coding_correctness.
- Independent core/API/UI review: approved after tokenizer, comparability, status and cleanup fixes; 37 focused native/artifact/API checks passed.
- Independent automation review: approved after cumulative timeout and blocker recording changes; 27 focused checks passed.
- `fb7fc4e`: local quantization/evaluation/API/dashboard implementation.
- `9cadc2c`: bounded resumable automation and seven role definitions.

Real evidence and this report are committed in a subsequent documentation increment. Runtime completion adoption is bound to that reviewed commit and reruns acceptance/regression gates; blocked tasks retain explicit evidence.

Live automation smoke: documented bounded `codex exec` was invoked read-only and exited1 with account `model_not_found` for configured gpt-6.1-sol. This is an external account/model blocker; dry-run/tests pass. No live unattended implementation task is claimed complete. See experiments/automation-cli-smoke.json.

## Final persisted queue

Validated tasks01,02,03,05,06,07 were adopted at reviewed commit `2d473a5`, with acceptance and regression checks rerun for each. Tasks04,08,09 are blocked with committed resource evidence; none is marked completed. `docs/TASK_STATE.json` records the final queue snapshot. The unavailable-coding and CLI-account blockers remain unchanged.

`2d473a5` commits the real conversion logs, raw model/API evidence, final report and resource/account blockers.
