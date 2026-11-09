# task09 correction: independent documentation re-audit

## Verdict and scope

**Approved, with no findings in the reviewed correction.** This is a fresh session-generated independent subagent review (`docs_reaudit`), not a cryptographic signature or external certification. It verifies the two documentation changes below, and rechecks the related earlier documentation corrections. It does not certify the whole application, the original review, or v0.2 completion.

Reviewed base commit: `3e386303fd298ca5cd585250fac9973b0b09e938`. The working diff at review time changed only `README.md:93` and `docs/FINAL_REPORT.md:82`. The integrator must bind the subsequent review record to the commit containing these exact reviewed contents. This report itself is the reviewer's only written file.

## Why a new review was necessary

The review record behind adoption at `4b704ce` stated that all five fixes had been verified, but the documentation still contained a broad Model Library claim and an obsolete bundle size. That earlier approval was overstated. This review does **not** retroactively validate it. Preserve the old record and adoption logs as superseded evidence, and use a newly commit-bound record for the corrected adoption.

## Assertion-by-assertion evidence

| Assertion | Direct evidence inspected | Result |
| --- | --- | --- |
| Model Library scope is now conditional, rather than every Feather artifact | `frontend/src/main.tsx:45` requests `/api/models` and `/api/artifacts`; `:59` combines those results into `runtimeModels`; `:67` renders that combined list. `backend/api/app.py:229–231` returns only nonempty results from completed quantization API jobs. The Model Library does not ingest the separate study summaries. | `README.md:93` now accurately describes installed Ollama models plus completed API-job artifacts when available; study-only artifacts are not automatically imported. Saying the page can only list Ollama models would also overstate the source evidence. |
| Current production JavaScript bundle size | Independently measured `frontend/dist/assets/index-mwCZaF0X.js` with `Path.stat().st_size`: **642,317 bytes**. The completed build log reports **642.32 kB** and the Vite chunk-size warning. | `docs/FINAL_REPORT.md:82` now states 642,317 bytes / 642.32 kB uncompressed. Historical v0.1 bundle descriptions are unchanged. |
| Current test count | Completed `.feather/task09-correction-pytest.log`: **181 passed**, one upstream Starlette TestClient/httpx deprecation warning, no failed/skipped result. | `docs/PROGRESS.md:33` and the current final-report test table agree. The full suite was run by the parent in this session; this reviewer inspected its completed log. |
| Dashboard evaluation suite description | `frontend/src/main.tsx:69` offers legacy three-item suites and `reasoning-v1` / `coding-v1` held-out suites, and displays the saved split. | `README.md:95` matches the implemented choices. |
| Dashboard recommendation default | `frontend/src/main.tsx:40–41` defaults to 70% and `coding-v1`; `:70` submits the chosen suite. `backend/api/app.py:301` chooses held-out for v1 suites when the split is omitted. The accepted workflow's `recommendation.json` requests coding-v1, heldout, 0.7 and records `no_eligible_configuration`. | `README.md:96` accurately names the default evidence and threshold; it does not claim useful model quality. |
| Evidence filenames | `automation/validate.py:152–153` prefers `comparison-evidence-v0.2.json`; `:208` requires `e2e-evidence-v0.2.json`. Independently checked all three files listed by the automation README exist. | `automation/README.md:21` points to the current v0.2 evidence files and the retained task02 quantization evidence. |
| Optional 1.5B status and v0.2 limitations | `README.md:9`, `docs/PROGRESS.md:28`, and `docs/FINAL_REPORT.md:9` explicitly state the optional experiment is blocked pending download approval and sufficient memory, and not attempted. Current report status still declares v0.2 incomplete and separately identifies live automation and Qwen blockers. | No false completion introduced. |

## Gate and diff checks

- This reviewer independently ran `uv run python automation/validate.py --task task09`: **exit 0**, including its recovery/failure subset (**7 passed, 19 deselected**, one dependency warning). It reads the local artifacts required by the gate; no model benchmark or conversion was started.
- This reviewer independently ran `git diff --check`: **exit 0**.
- Parent's completed production build log shows a successful build; reviewer independently verified the emitted bundle's exact byte count and hash.
- Parent's completed regression log records **181 passed**. No claim is made that this reviewer reran all 181 tests or the browser test. No fresh browser test was needed for these documentation-only edits.
- Existing measurement tables and commit-reference text are unchanged by the two-line correction. Their scientific validity was not re-audited here.

## Content fingerprints at review time

SHA-256 identifies the exact inspected contents. Build outputs and runtime logs remain ignored local files; these hashes do not make them portable or independently reproducible without the local inputs.

| File | SHA-256 |
| --- | --- |
| `README.md` | `adfd31d5600c212c9250c024773263a2a6ac8e77dbbccba27ab79d49067f9e4e` |
| `docs/FINAL_REPORT.md` | `1d73d290908afff60d7d7701ce3f48c9bee388a57d868ca9af0991ba11cda841` |
| `frontend/src/main.tsx` | `534b128b442684f9fcb0bcf2b26895bfc8a74c8560a55e1ba739059fc3409923` |
| `backend/api/app.py` | `212b88da8e9f4f2bcb10ad9fe6b1bbf89310c1d3dc38b546a4c26c097a156e5e` |
| `frontend/dist/assets/index-mwCZaF0X.js` | `9a280af74ad0bccca9b680aecc176d26308ba874d392b056780a778dd5ba9c09` |
| `.feather/task09-correction-pytest.log` | `0a691fd52449425863ca04eafff1fbfeab1f4f5a9142a06859cf850bf7e26904` |
| `.feather/task09-correction-build.log` | `7e8bd2c9cd3da0a8b6ba38b2913ccbd103a5fe03712de13596341b7bb4f00618` |
| `.feather/task09-correction-gate.log` | `a2fe2b50d4e0a6bdf89fc090e8d9741c24c12fbe74c1959f494081d661a11e84` |

## Integration condition

Approval covers the content fingerprints above. After committing the two fixes and this report, record the actual corrected commit in a fresh task09 review record, retain the prior approval as superseded, and let the orchestrator rerun adoption gates/regressions. Do not label the original adoption freshly verified or claim v0.2 complete. Any subsequent documentation-status checkpoint should accurately identify which commit was reviewed and adopted.
