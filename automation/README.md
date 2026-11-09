# Bounded local development runner

This runner uses the installed Codex CLI's documented `codex exec --json --sandbox workspace-write -C ROOT` interface. Independent review uses a separate read-only `codex exec` with a strict JSON schema. An exit code of zero alone never approves a review. CLI credentials are inherited from the existing Codex login; no credentials are copied into the repository.

Official references: [non-interactive execution](https://learn.chatgpt.com/docs/non-interactive-mode), [custom project agents](https://learn.chatgpt.com/docs/agent-configuration/subagents). Verified against installed `codex-cli 0.142.3`. Seven project-scoped TOML definitions inherit the parent model and permissions. An execution prompt selects the corresponding role instructions; the runner does not invent an unsupported `--agent` option.

Run from the existing project root after `uv sync`:

```sh
uv run python automation/orchestrator.py --dry-run
uv run python automation/orchestrator.py --status
uv run python automation/orchestrator.py --one --timeout 1800 --retries 2
uv run python automation/orchestrator.py --task task03
uv run python automation/orchestrator.py --timeout 1800 --retries 2
```

The final command processes eligible tasks sequentially. Independent tasks may continue after an ordinary failure, but critical Git/tree/lock failures stop the runner. No heavy inference runs concurrently through this runner. A clean Git working tree is required for a fresh task. Commit or otherwise preserve and inspect existing work before starting; the runner never cleans, stashes, resets or deletes it. Do not run this during another editor/agent's changes. The advisory lock excludes other runner processes, not unrelated editors.

Each task executes implementation, explicit acceptance commands, repository regression commands and independent review before a local commit. Code-review instructions alone are not a security boundary; the review process additionally runs in the CLI read-only sandbox. No dangerous permission bypass is used. Task configuration is trusted project code, not an API input. Command arrays run without a shell. The runner does not push, publish or buy services.

Acceptance gates reject missing real-run evidence. For real experiments, see `validate.py` for the schemas of `experiments/quantization-evidence.json`, `comparison-evidence.json`, and `e2e-evidence.json`. These files must represent actual measured runs with real artifacts; do not create placeholders to pass the gate. Unsupported perplexity requires an explicit reason. Code task acceptance additionally runs focused tests; independent review must verify the task's full requirements, not just test-file presence.

Progress, per-invocation argv/exit/timing metadata, CLI JSONL events, captured token usage when reported by Codex, review decisions, test logs and a final queue report persist under `.feather/automation/`. This path is already ignored by Git. Pricing/cost is not estimated: CLI token counts are recorded when available; account billing is not exposed by the CLI event stream. `tasks.yaml` is the versioned task specification; all tasks start pending and `.feather` state is the execution record. If the manifest changes after execution begins, the runner refuses to reinterpret old state. Inspect/archive that state before starting a different queue.

Stop with Ctrl-C, or send SIGTERM to the runner process. Its active child process group is terminated, with a bounded grace period before kill. Changes and logs are preserved. A crash/interruption leaves an active task blocked on recovery so it cannot blindly repeat model downloads or conversions. A completed reviewed commit is recovered if a crash occurred just before the final state checkpoint.

After inspecting interrupted/failed work and returning the tree to a clean state, explicitly resume:

```sh
uv run python automation/orchestrator.py --resume-task task03 --one
```

The attempt budget is persistent: one initial attempt and at most two retries in total, including interrupted attempts. Resumption never resets that budget. Exhausted tasks require a reviewed new task/queue; do not edit the state to fabricate completion. Exit 130 means interrupted, exit 2 means critical/configuration failure, and exit 1 means incomplete work remains. Dry-run/status do not invoke Codex, run tests, commit, or alter task statuses (the runner lock file may be created).

For work completed and independently reviewed in the interactive multi-agent session, import completion instead of repeating model experiments. First commit the reviewed work, keep the tree clean, and create an external review record under `.feather/` containing `{"task":"task01","commit":"FULL_CURRENT_HEAD_SHA","review":{"approved":true,"findings":[],"summary":"Actual independent review evidence"}}`. Then run:

```sh
uv run python automation/orchestrator.py --adopt-task task01 --review-record .feather/review-task01.json
```

Adoption requires completed dependencies, a review bound to the exact current commit, and successful task/regression gates. It records `origin: externally-reviewed` without spending another implementation attempt. This is an explicit integration action, not proof of reviewer identity; keep the actual review conversation/logs as supporting evidence. Do not use adoption to skip failed acceptance. Validation of source/output artifacts requires their ignored local weight files to remain available; archived JSON reports alone do not make that gate runnable on a fresh clone.

`--timeout` is the cumulative wall-clock budget for **each task attempt**, shared across implementation, acceptance/regression commands and independent review. Each invocation is clipped to the remaining budget, and no further invocation starts after it expires. Retries each receive a fresh attempt budget; with two retries, at most three bounded attempts occur. A process may require up to five additional seconds for termination cleanup. External adoption executes bounded checks but does not start an implementation attempt.

Record an externally observed blocker without faking task completion:

```sh
uv run python automation/orchestrator.py --block-task task04 --reason 'Docker daemon unavailable; coding correctness could not be measured' --evidence experiments/resource-blockers.json
```

The evidence file must already be committed/tracked and the working tree clean. The state records its SHA-256, current commit and reason as `origin: externally-observed`, preserving the attempt count. Completed tasks cannot be downgraded through this operation. Pending independent tasks remain eligible according to their dependencies. Use the same command for a blocked dependent integration/report task only when its reason and committed evidence accurately identify the unmet requirement.

## Model selection, preflight and self-test (v0.2)

The runner passes `-m <model>` only when a model is configured. Resolution order:
1. `FEATHER_CODEX_MODEL`.
2. `codex.model` in `automation/config.yaml`.
3. A model accepted by `feather automation doctor --probe --select`, stored in `.feather/automation/codex-model.json`.
4. The Codex CLI default.

`automation/config.yaml` also sets `task_timeout_seconds` and `max_attempts` (at most 3).

```bash
uv run feather automation doctor                                  # offline checks: CLI, login status, config, repository, queue, paths, tools
uv run feather automation doctor --probe                          # one small read-only request with the resolved model
uv run feather automation doctor --probe --probe-candidates --select  # try cached models in priority order, record the first accepted
uv run feather automation selftest                                # two dependent tasks via live Codex in a throwaway Git repository
```

Each Codex invocation is recorded in the task state: exit code, model and failure kind. Failures classified as `model_entitlement`, `cli_outdated`, `authentication` or `usage_limit` block the task after a single invocation. They are never retried, and dependent tasks do not start. Other failures keep the bounded retry budget. A manifest other than `tasks.yaml` keeps its own state file, `state-<name>.json`.

## Observed account blocker (re-diagnosed 2026-11-09)

All eight models listed in the local model cache were probed with codex-cli 0.142.3 and a ChatGPT login (`experiments/automation-model-probes.json`):
- **Refused for ChatGPT-account Codex use:** `gpt-6.1-sol` (the configured default), `gpt-6.1-terra`, `gpt-6-sol` and `gpt-6-luna`.
- **Need a newer CLI:** `gpt-6-astra`, `gpt-5.6-sol`, `gpt-5.6-terra` and `gpt-5.6-luna`. `codex doctor` reports 0.167.0 as available.

Live automation therefore remains **BLOCKED**. To unblock it:
1. Upgrade the CLI yourself (`codex update`).
2. Run `uv run feather automation doctor --probe --probe-candidates --select`.
3. Run `uv run feather automation selftest`.

The dependency ordering, model-rejection handling and interruption recovery are covered by mocked tests (`tests/test_automation_config.py`). They are not verified live.
