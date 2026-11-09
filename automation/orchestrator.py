"""Sequential Codex task execution with durable checkpoints and explicit review gates.

This is a development runner, not an inference job manager. It never runs itself
recursively, pushes commits, publishes, or weakens the configured sandbox.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
STATES = {"pending", "running", "testing", "reviewing", "completed", "blocked", "failed"}
ACTIVE = {"running", "testing", "reviewing"}
IMPLEMENTATION_SCHEMA = {
    "type": "object", "properties": {
        "status": {"type": "string", "enum": ["completed", "blocked"]},
        "summary": {"type": "string"}, "blocker": {"type": "string"}},
    "required": ["status", "summary", "blocker"], "additionalProperties": False,
}
REVIEW_SCHEMA = {
    "type": "object", "properties": {
        "approved": {"type": "boolean"},
        "findings": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"}},
    "required": ["approved", "findings", "summary"], "additionalProperties": False,
}


class CriticalFailure(RuntimeError):
    """A failure for which automatic retry would risk unrelated work."""


def load_tasks(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid task YAML: {error}") from error
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("tasks"), list):
        raise ValueError("Expected version: 1 and a tasks list")
    ids = [task.get("id") for task in data["tasks"]]
    if any(not isinstance(i, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", i) for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Task IDs must be unique simple names")
    for task in data["tasks"]:
        if not isinstance(task.get("prompt"), str) or not isinstance(task.get("depends_on"), list):
            raise ValueError(f"Invalid task {task['id']}")
        if any(dep not in ids or dep == task["id"] for dep in task["depends_on"]):
            raise ValueError("Unknown or self dependency")
        validate_commands(task.get("acceptance"))
    validate_commands(data.get("regression"))
    remaining = {task["id"]: set(task["depends_on"]) for task in data["tasks"]}
    done: set[str] = set()
    while remaining:
        eligible = {key for key, deps in remaining.items() if deps <= done}
        if not eligible:
            raise ValueError("Task dependencies contain a cycle")
        done |= eligible
        remaining = {key: deps for key, deps in remaining.items() if key not in eligible}
    return data


def validate_commands(commands: Any) -> None:
    if not isinstance(commands, list) or not commands:
        raise ValueError("Acceptance and regression commands must be nonempty argv lists")
    for command in commands:
        if not isinstance(command, list) or not command or any(not isinstance(arg, str) or not arg or '\0' in arg for arg in command):
            raise ValueError("Each command must be a nonempty argv list; shell strings are unsupported")


def eligible_tasks(tasks: list[dict], state: dict) -> list[dict]:
    return [task for task in tasks if state[task["id"]]["status"] == "pending"
            and all(state[dep]["status"] == "completed" for dep in task["depends_on"])]


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def invoke(argv: list[str], root: Path, log: Path, timeout: int, env: dict | None = None) -> int:
    """Capture output and exit status; kill the whole process group on interruption."""
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    result: dict[str, Any] = {"argv": argv, "started_at": started, "timeout_seconds": timeout}
    with log.open("w") as stream:
        try:
            process = subprocess.Popen(argv, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True, env=env)
        except OSError as error:
            result.update(exit_code=127, error=f"{type(error).__name__}: {error}", finished_at=time.time())
            stream.write(result["error"] + "\n")
            atomic_json(log.with_suffix(log.suffix + ".invocation.json"), result)
            return 127
        try:
            result["exit_code"] = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            result["interrupted"] = type(error).__name__
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            result["exit_code"] = 124 if isinstance(error, subprocess.TimeoutExpired) else 130
            if isinstance(error, KeyboardInterrupt):
                raise
        finally:
            result["finished_at"] = time.time()
            atomic_json(log.with_suffix(log.suffix + ".invocation.json"), result)
    return result["exit_code"]


def token_usage(log: Path) -> list[dict]:
    usages = []
    if not log.exists():
        return usages
    for line in log.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
            usages.append(event["usage"])
    return usages


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def tree_digest(root: Path) -> str:
    """Include untracked contents so a reviewer cannot approve a different tree."""
    digest = hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=root))
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=root)
    for raw in sorted(untracked.split(b'\0')):
        if not raw:
            continue
        path = root / os.fsdecode(raw)
        digest.update(raw)
        digest.update(os.readlink(path).encode() if path.is_symlink() else path.read_bytes())
    return digest.hexdigest()


def review_approved(path: Path) -> bool:
    try:
        result = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return (set(result) == {"approved", "findings", "summary"}
            and result["approved"] is True and result["findings"] == []
            and isinstance(result["summary"], str) and bool(result["summary"].strip()))


class Runner:
    def __init__(self, root: Path, manifest: Path, timeout=1800, retries=2, executor=invoke):
        if not 0 <= retries <= 2 or timeout < 1:
            raise ValueError("Retries must be 0–2 and timeout must be positive")
        self.root = root.resolve()
        self.manifest = load_tasks(manifest)
        self.timeout, self.retries, self.execute = timeout, retries, executor
        self._deadline = None
        self.directory = self.root / ".feather" / "automation"
        self.state_path = self.directory / "state.json"
        if self.state_path.exists():
            saved = json.loads(self.state_path.read_text())
            if saved.get("manifest_sha256") != hashlib.sha256(manifest.read_bytes()).hexdigest():
                raise ValueError("Task manifest changed; inspect and archive existing state before starting a new queue")
            self.state = saved["tasks"]
        else:
            self.state = {task["id"]: {"status": "pending", "attempts": 0} for task in self.manifest["tasks"]}
        self.manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
        for task in self.manifest["tasks"]:
            entry = self.state.get(task["id"])
            if not entry or entry.get("status") not in STATES or not isinstance(entry.get("attempts"), int):
                raise ValueError("Invalid persisted task state")

    def save(self):
        atomic_json(self.state_path, {"manifest_sha256": self.manifest_hash, "tasks": self.state})

    def report(self):
        lines = ["# Automation execution report", "", "Status is based on recorded acceptance and review gates, not agent claims.", ""]
        for task in self.manifest["tasks"]:
            entry = self.state[task["id"]]
            lines += [f"- {task['id']} — {entry['status']} ({entry['attempts']} attempts)"]
            if entry.get("error"):
                lines += [f"  - {entry['error']}"]
            if entry.get("commit"):
                lines += [f"  - Commit: {entry['commit']}"]
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "report.md").write_text("\n".join(lines) + "\n")

    def recover(self):
        for task in self.manifest["tasks"]:
            entry = self.state[task["id"]]
            if entry["status"] in ACTIVE:
                # Recover a crash between validated commit and state-file checkpoint.
                if (entry.get("commit_intent") and not git(self.root, "status", "--porcelain")
                    and git(self.root, "log", "-1", "--format=%B") == entry["commit_intent"]
                    and git(self.root, "rev-parse", "HEAD^") == entry.get("start_head")):
                    entry.update(status="completed", commit=git(self.root, "rev-parse", "HEAD"))
                else:
                    entry.update(status="blocked", error="Interrupted execution; inspect preserved changes/logs. Clean or commit reviewed changes, then use --resume-task.")
        self.save()

    def resume(self, identifier: str):
        if identifier not in self.state:
            raise ValueError("Unknown task")
        entry = self.state[identifier]
        if entry["status"] not in {"blocked", "failed"}:
            raise ValueError("Only blocked or failed tasks may be resumed")
        if entry["attempts"] >= self.retries + 1:
            raise ValueError("Persisted attempt budget exhausted; do not silently reset retries")
        if git(self.root, "status", "--porcelain"):
            raise CriticalFailure("Resume requires a clean working tree after human inspection")
        entry.update(status="pending", error=None)
        self.save()

    def checked_execute(self, argv, root, log, timeout, env):
        remaining = self.timeout if self._deadline is None else self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Cumulative task attempt timeout exhausted before next invocation")
        return self.execute(argv, root, log, min(timeout, remaining), env)

    def block(self, identifier: str, reason: str, evidence: Path):
        """Record an observed external blocker; never turn observations into completion."""
        if identifier not in self.state:
            raise ValueError("Unknown task")
        if self.state[identifier]["status"] == "completed":
            raise ValueError("Cannot mark a completed task blocked")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 4000:
            raise ValueError("A concise nonempty blocker reason is required")
        if git(self.root, "status", "--porcelain"):
            raise CriticalFailure("Blocker recording requires committed evidence and a clean working tree")
        evidence = evidence.resolve(strict=True)
        if not evidence.is_relative_to(self.root) or not evidence.is_file():
            raise ValueError("Blocker evidence must be a file inside the repository")
        relative = evidence.relative_to(self.root).as_posix()
        try:
            git(self.root, "ls-files", "--error-unmatch", "--", relative)
        except subprocess.CalledProcessError as error:
            raise ValueError("Blocker evidence must be tracked in the reviewed repository") from error
        content = evidence.read_bytes()
        if not content:
            raise ValueError("Blocker evidence must not be empty")
        self.state[identifier].update(status="blocked", error=reason.strip(), origin="externally-observed",
            blocker_evidence={"path": relative, "sha256": hashlib.sha256(content).hexdigest(),
                              "commit": git(self.root, "rev-parse", "HEAD"), "recorded_at": time.time()})
        self.save()
        self.report()

    def adopt(self, identifier: str, review_record: Path):
        """Record externally integrated work only after local gates and commit-bound review."""
        if identifier not in self.state:
            raise ValueError("Unknown task")
        task = next(task for task in self.manifest["tasks"] if task["id"] == identifier)
        if not all(self.state[dep]["status"] == "completed" for dep in task["depends_on"]):
            raise ValueError("Cannot adopt a task before its dependencies")
        if git(self.root, "status", "--porcelain"):
            raise CriticalFailure("Adoption requires a clean reviewed working tree")
        record = json.loads(review_record.read_text())
        head = git(self.root, "rev-parse", "HEAD")
        if record.get("commit") != head or record.get("task") != identifier:
            raise ValueError("Review record must identify this task and current HEAD")
        review = record.get("review", {})
        if (set(review) != {"approved", "findings", "summary"} or review.get("approved") is not True
                or review.get("findings") != [] or not isinstance(review.get("summary"), str)
                or not review["summary"].strip()):
            raise ValueError("External review did not approve")
        directory = self.directory / identifier / "adoption"
        for index, command in enumerate(task["acceptance"] + self.manifest["regression"]):
            if self.execute(command, self.root, directory / f"check-{index}.log", self.timeout, None):
                raise ValueError("Adoption acceptance/regression check failed")
        if git(self.root, "status", "--porcelain") or git(self.root, "rev-parse", "HEAD") != head:
            raise CriticalFailure("Adoption checks changed the reviewed working tree")
        atomic_json(directory / "review.json", record)
        self.state[identifier].update(status="completed", commit=head, origin="externally-reviewed",
                                      error=None, review_record=str(review_record.resolve()))
        self.save()
        self.report()

    def codex(self, prompt, directory, name, review=False):
        final = directory / f"{name}-final.json"
        schema = directory / f"{name}-schema.json"
        argv = ["codex", "exec", "--json", "--sandbox", "read-only" if review else "workspace-write",
                "-C", str(self.root), "--output-last-message", str(final)]
        atomic_json(schema, REVIEW_SCHEMA if review else IMPLEMENTATION_SCHEMA)
        argv += ["--output-schema", str(schema), prompt]
        environment = dict(os.environ, FEATHER_AUTOMATION_CHILD="1")
        log = directory / f"{name}.jsonl"
        code = self.checked_execute(argv, self.root, log, self.timeout, environment)
        return code, final, token_usage(log)

    def run_task(self, task):
        entry = self.state[task["id"]]
        if git(self.root, "status", "--porcelain"):
            raise CriticalFailure("Refusing to run on a dirty working tree; preserve and review existing work first")
        start_head = git(self.root, "rev-parse", "HEAD")
        while entry["attempts"] < self.retries + 1:
            self._deadline = time.monotonic() + self.timeout
            entry.update(status="running", attempts=entry["attempts"] + 1, start_head=start_head)
            directory = self.directory / task["id"] / f"attempt-{entry['attempts']}"
            self.save()
            role = task.get("role", "integrator")
            prompt = (f"Act as the {role} role from .codex/agents. Complete only {task['id']}: {task.get('title', '')}.\n"
                      f"{task['prompt']}\nPrevious attempt result: {entry.get('error', 'none')}\n"
                      "Preserve working code. Do not commit, push, publish, modify automation state, invoke the automation runner, "
                      "spawn additional agents, change security settings or run untrusted code on the host. "
                      "Run heavy model experiments sequentially. If blocked, report exact evidence; never claim completion without meeting acceptance.")
            try:
                code, implementation, usage = self.codex(prompt, directory, "implementation")
                entry.setdefault("usage", []).extend(usage)
                if git(self.root, "rev-parse", "HEAD") != start_head:
                    raise CriticalFailure("Implementation changed HEAD; independent review is required")
                if code:
                    raise RuntimeError(f"Implementation exited {code}; see {directory}")
                try:
                    outcome = json.loads(implementation.read_text())
                except (OSError, ValueError) as error:
                    raise RuntimeError("Missing or invalid implementation outcome") from error
                if outcome.get("status") == "blocked" and outcome.get("blocker"):
                    entry.update(status="blocked", error=str(outcome["blocker"]))
                    self.save()
                    return
                if outcome.get("status") != "completed":
                    raise RuntimeError("Implementation did not report a valid outcome")
                entry["status"] = "testing"
                self.save()
                for index, command in enumerate(task["acceptance"] + self.manifest["regression"]):
                    code = self.checked_execute(command, self.root, directory / f"check-{index}.log", self.timeout, None)
                    if code:
                        raise RuntimeError(f"Acceptance/regression command {command!r} exited {code}")
                entry["status"] = "reviewing"
                self.save()
                before = tree_digest(self.root)
                code, final, usage = self.codex(
                    f"Independently review uncommitted changes for {task['id']}. Acceptance: {task['prompt']}\n"
                    "Inspect all tracked changes and untracked source files. Check correctness, security, tests and real evidence. "
                    "Do not edit, commit, execute generated model code, or invoke automation. Approve only when all acceptance criteria "
                    "are met and no actionable findings remain. Return strict JSON with approved, findings and summary.",
                    directory, "review", review=True)
                entry.setdefault("usage", []).extend(usage)
                if git(self.root, "rev-parse", "HEAD") != start_head or before != tree_digest(self.root):
                    raise CriticalFailure("Working tree or HEAD changed during review")
                if code or not review_approved(final):
                    raise RuntimeError(f"Review did not approve; inspect {final}")
                message = f"Complete {task['id']}: {task.get('title', task['id'])}"
                entry["commit_intent"] = message
                self.save()
                if git(self.root, "status", "--porcelain"):
                    if self.checked_execute(["git", "add", "--all"], self.root, directory / "git-add.log", self.timeout, None):
                        raise RuntimeError("Could not stage reviewed changes")
                    if self.checked_execute(["git", "commit", "-m", message], self.root, directory / "git-commit.log", self.timeout, None):
                        raise RuntimeError("Could not commit reviewed changes")
                entry.update(status="completed", commit=git(self.root, "rev-parse", "HEAD"), error=None)
                self.save()
                return
            except KeyboardInterrupt:
                entry.update(status="blocked", error="Interrupted; inspect changes and logs before explicit resumption")
                self.save()
                raise
            except CriticalFailure as error:
                entry.update(status="blocked", error=str(error))
                self.save()
                raise
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                entry.update(status="failed", error=f"{type(error).__name__}: {error}")
                self.save()
                # Retry only this task's preserved work; total attempts remain bounded.
        self.report()

    def run(self, one=False, identifier=None):
        while True:
            eligible = eligible_tasks(self.manifest["tasks"], self.state)
            if identifier:
                eligible = [task for task in eligible if task["id"] == identifier]
            if not eligible:
                break
            self.run_task(eligible[0])
            if one or identifier:
                break
        self.report()


@contextlib.contextmanager
def exclusive_lock(directory: Path):
    if directory.is_symlink() or directory.parent.is_symlink():
        raise CriticalFailure("Automation state directory must not be a symlink")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "runner.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CriticalFailure("Another automation runner holds the project lock") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--one", action="store_true")
    parser.add_argument("--task")
    parser.add_argument("--resume-task")
    parser.add_argument("--adopt-task")
    parser.add_argument("--review-record", type=Path)
    parser.add_argument("--block-task")
    parser.add_argument("--reason")
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    if os.environ.get("FEATHER_AUTOMATION_CHILD"):
        parser.exit(2, "Recursive automation is prohibited\n")
    try:
        with exclusive_lock(ROOT / ".feather" / "automation"):
            runner = Runner(ROOT, ROOT / "automation" / "tasks.yaml", args.timeout, args.retries)
            if args.dry_run or args.status:
                print(json.dumps({"tasks": runner.state, "eligible": [task["id"] for task in eligible_tasks(runner.manifest["tasks"], runner.state)],
                                  "timeout": args.timeout, "max_attempts": args.retries + 1}, indent=2))
                return
            def interrupted(signum, frame):
                raise KeyboardInterrupt
            signal.signal(signal.SIGTERM, interrupted)
            runner.recover()
            if args.block_task:
                if not args.reason or not args.evidence:
                    raise ValueError("--block-task requires --reason and --evidence")
                runner.block(args.block_task, args.reason, args.evidence)
                return
            if args.adopt_task:
                if not args.review_record:
                    raise ValueError("--adopt-task requires --review-record")
                runner.adopt(args.adopt_task, args.review_record)
                return
            if args.resume_task:
                runner.resume(args.resume_task)
            if args.task and args.task not in runner.state:
                raise ValueError("Unknown task")
            runner.run(args.one, args.task)
            if any(entry["status"] != "completed" for entry in runner.state.values()):
                parser.exit(1, "Queue has incomplete work; inspect .feather/automation/report.md\n")
    except KeyboardInterrupt:
        parser.exit(130, "Stopped safely; changes and invocation logs preserved\n")
    except (CriticalFailure, ValueError, OSError, subprocess.SubprocessError) as error:
        if 'runner' in locals():
            runner.report()
        parser.exit(2, f"Automation stopped: {error}\n")


if __name__ == "__main__":
    main()
