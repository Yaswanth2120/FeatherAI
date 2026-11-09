"""Two dependent tasks in a throwaway Git repository, used to verify live Codex automation.

The real project repository is never modified. Each task needs one implementation and one
read-only review invocation, so a live run makes four small Codex requests.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile

import yaml

from automation.orchestrator import CriticalFailure, Runner

CHECK = "import pathlib,sys; sys.exit(0 if pathlib.Path({path!r}).read_text() == {text!r} else 1)"
TASKS = {
    "version": 1,
    "tasks": [
        {"id": "selftest-alpha", "title": "Write alpha", "role": "integrator", "depends_on": [],
         "prompt": "Create the file selftest/alpha.txt containing exactly the text alpha followed by one newline. Change nothing else.",
         "acceptance": [["python3", "-c", CHECK.format(path="selftest/alpha.txt", text="alpha\n")]]},
        {"id": "selftest-beta", "title": "Write beta", "role": "integrator", "depends_on": ["selftest-alpha"],
         "prompt": "Read selftest/alpha.txt. Create selftest/beta.txt containing its text without the newline reversed "
                   "character by character, followed by one newline. Change nothing else.",
         "acceptance": [["python3", "-c", CHECK.format(path="selftest/beta.txt", text="ahpla\n")]]},
    ],
    "regression": [["git", "rev-parse", "HEAD"]],
}


def make_repository(directory: Path) -> Path:
    root = directory / "repo"
    root.mkdir()
    for command in (["git", "init", "-q"], ["git", "config", "user.name", "Feather Selftest"],
                    ["git", "config", "user.email", "selftest@example.invalid"]):
        subprocess.run(command, cwd=root, check=True)
    (root / ".gitignore").write_text(".feather/\n")
    (root / "README.md").write_text("Throwaway repository for Feather automation self-test.\n")
    (root / "automation").mkdir()
    (root / "automation" / "selftest.yaml").write_text(yaml.safe_dump(TASKS, sort_keys=False))
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "Self-test fixture"], cwd=root, check=True)
    return root


def run_selftest(model: str | None, executor=None, timeout=600, keep: Path | None = None) -> dict:
    directory = Path(keep) if keep else Path(tempfile.mkdtemp(prefix="feather-automation-selftest-"))
    root = make_repository(directory)
    kwargs = {"model": model} if executor is None else {"model": model, "executor": executor}
    runner = Runner(root, root / "automation" / "selftest.yaml", timeout=timeout, retries=0, **kwargs)
    error = None
    try:
        runner.run()
    except CriticalFailure as failure:
        error = str(failure)
    log = subprocess.check_output(["git", "log", "--format=%s"], cwd=root, text=True).splitlines()
    return {"repository": str(root), "model": model or "cli-default", "error": error,
            "tasks": {key: {k: v for k, v in value.items() if k in {"status", "attempts", "commit", "error", "failure_kind", "invocations"}}
                      for key, value in runner.state.items()},
            "commits": log, "dependent_advanced": runner.state["selftest-beta"]["status"] == "completed",
            "status": "passed" if all(v["status"] == "completed" for v in runner.state.values()) else "failed"}
