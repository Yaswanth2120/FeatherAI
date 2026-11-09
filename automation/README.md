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

## Observed account blocker

A real bounded read-only CLI smoke invocation on2026-11-09 exited1: the account rejected its configured `gpt-6.1-sol` model with `model_not_found` ("not supported when using Codex with a ChatGPT account"). See `experiments/automation-cli-smoke.json`. Runner unit tests and dry-run pass, but unattended model execution is not verified successful for this account. Configure a model supported by the selected CLI account before live execution; no automatic model substitution, credential change or paid fallback is performed. Interactive multi-agent reviews and validated completion adoption are separate from live CLI execution.
