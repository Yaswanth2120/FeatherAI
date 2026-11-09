"""Preflight checks for Codex task automation. Never prints or reads credentials."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import tomllib

from automation.codex_config import ConfigError, classify_failure, load_config, resolve_model, selection_path
from automation.orchestrator import ROOT, eligible_tasks, load_tasks

PROBE_PROMPT = "Reply with exactly the single word OK and do nothing else."


def _run(argv, timeout=20, cwd=None):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        return result.returncode, (result.stdout + result.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return 127, f"{type(error).__name__}: {error}"


def _check(name, status, detail, required=True, **extra):
    return {"name": name, "status": status, "detail": detail, "required": required, **extra}


def cli_default_model() -> str | None:
    """Read only the top-level `model` key from the user's Codex config."""
    path = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "config.toml"
    try:
        value = tomllib.loads(path.read_text()).get("model")
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return value if isinstance(value, str) else None


def cached_models() -> list[str]:
    path = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "models_cache.json"
    try:
        models = json.loads(path.read_text()).get("models", [])
    except (OSError, ValueError):
        return []
    listed = [m for m in models if isinstance(m, dict) and m.get("visibility") == "list" and isinstance(m.get("slug"), str)]
    return [m["slug"] for m in sorted(listed, key=lambda m: m.get("priority", 1e9))]


def probe(model: str | None, timeout=150) -> dict:
    """One bounded read-only request in an empty temporary directory."""
    argv = ["codex", "exec", "--json", "--sandbox", "read-only", "--skip-git-repo-check"]
    if model:
        argv += ["-m", model]
    with tempfile.TemporaryDirectory(prefix="feather-codex-probe-") as directory:
        started = time.time()
        code, output = _run(argv + [PROBE_PROMPT], timeout=timeout, cwd=directory)
    completed = '"turn.completed"' in output
    failure = classify_failure(output)
    status = "accepted" if code == 0 and completed and not failure else "rejected"
    return {"model": model or "cli-default", "status": status, "exit_code": code, "turn_completed": completed,
            "failure": failure, "seconds": round(time.time() - started, 2), "probed_at": time.time()}


def run_checks(root: Path = ROOT, do_probe=False, probe_candidates=False, select=False) -> dict:
    checks = []
    code, version = _run(["codex", "--version"])
    has_cli = shutil.which("codex") is not None and code == 0
    checks.append(_check("codex_cli", "ok" if has_cli else "fail", version if has_cli else "Codex CLI not found on PATH"))
    if has_cli:
        code, status = _run(["codex", "login", "status"])
        first = status.splitlines()[0] if status else ""
        logged_in = code == 0 and "logged in" in first.lower() and "not logged in" not in first.lower()
        checks.append(_check("authentication", "ok" if logged_in else "fail",
                             first[:120] if logged_in else "Not logged in; run `codex login` yourself"))
    try:
        config = load_config(root)
        selection = resolve_model(root)
        checks.append(_check("automation_config", "ok", f"timeout {config['task_timeout_seconds']}s, max_attempts {config['max_attempts']}"))
    except ConfigError as error:
        config, selection = None, {"model": None, "source": "invalid configuration"}
        checks.append(_check("automation_config", "fail", str(error)))
    effective = selection["model"] or cli_default_model()
    checks.append(_check("model_configuration", "ok" if effective else "warn",
                         f"{effective or 'unknown'} (source: {selection['source']}"
                         + (", inherited from ~/.codex/config.toml)" if not selection["model"] and effective else ")"),
                         model=effective, source=selection["source"]))
    probes_path = root / ".feather" / "automation" / "model-probes.json"
    previous = {}
    if probes_path.is_file():
        try:
            previous = json.loads(probes_path.read_text())
        except ValueError:
            previous = {}
    known = previous.get(effective or "")
    if known and known.get("status") == "rejected" and not do_probe:
        failure = known.get("failure") or {}
        checks.append(_check("model_last_probe", "fail",
                             f"{effective} was rejected ({failure.get('kind', 'unknown')}): {failure.get('message', '')[:200]}",
                             action=failure.get("action")))
    try:
        head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
        checks.append(_check("repository", "warn" if dirty else "ok",
                             f"HEAD {head}" + (" (dirty tree: the runner refuses to start until it is clean)" if dirty else ", clean")))
    except (OSError, subprocess.CalledProcessError) as error:
        checks.append(_check("repository", "fail", f"git unavailable: {error}"))
    manifest = root / "automation" / "tasks.yaml"
    try:
        data = load_tasks(manifest)
        state_path = root / ".feather" / "automation" / "state.json"
        detail = f"{len(data['tasks'])} tasks; no saved state"
        status = "ok"
        if state_path.is_file():
            import hashlib
            saved = json.loads(state_path.read_text())
            if saved.get("manifest_sha256") != hashlib.sha256(manifest.read_bytes()).hexdigest():
                status, detail = "fail", "Saved queue state belongs to a different manifest; archive it before running"
            else:
                tasks = saved["tasks"]
                counts = {}
                for entry in tasks.values():
                    counts[entry["status"]] = counts.get(entry["status"], 0) + 1
                eligible = [task["id"] for task in eligible_tasks(data["tasks"], tasks)]
                detail = f"{counts}; eligible: {eligible or 'none'}"
        checks.append(_check("task_queue", status, detail))
    except (ValueError, OSError, KeyError) as error:
        checks.append(_check("task_queue", "fail", str(error)))
    directory = root / ".feather" / "automation"
    try:
        if directory.is_symlink():
            raise OSError("state directory is a symlink")
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory):
            pass
        checks.append(_check("logging_paths", "ok", str(directory.relative_to(root))))
    except OSError as error:
        checks.append(_check("logging_paths", "fail", str(error)))
    for tool, required in [("git", True), ("uv", True), ("npm", False), ("docker", False)]:
        checks.append(_check(f"dependency_{tool}", "ok" if shutil.which(tool) else ("fail" if required else "warn"),
                             shutil.which(tool) or "not found", required=required))
    checks.append(_check("recursion_guard", "fail" if os.environ.get("FEATHER_AUTOMATION_CHILD") else "ok",
                         "running inside an automation child" if os.environ.get("FEATHER_AUTOMATION_CHILD") else "not nested"))
    probes = []
    if do_probe and has_cli:
        candidates = [effective] if not probe_candidates else [effective, *[m for m in cached_models() if m != effective]]
        for candidate in candidates:
            result = probe(candidate if candidate else None)
            probes.append(result)
            previous[candidate or "cli-default"] = result
            if result["status"] == "accepted" or (result["failure"] or {}).get("kind") in {"authentication", "usage_limit"}:
                break
        directory.mkdir(parents=True, exist_ok=True)
        probes_path.write_text(json.dumps(previous, indent=2) + "\n")
        accepted = next((p for p in probes if p["status"] == "accepted"), None)
        if accepted and select:
            selection_path(root).write_text(json.dumps({**accepted, "cli_version": version}, indent=2) + "\n")
        last = probes[-1] if probes else None
        checks.append(_check("live_model_probe", "ok" if accepted else "fail",
                             f"accepted: {accepted['model']}" if accepted else
                             f"no accepted model; last failure {((last or {}).get('failure') or {}).get('kind', 'unknown')}",
                             action=None if accepted else ((last or {}).get("failure") or {}).get("action")))
    failing = [c for c in checks if c["required"] and c["status"] == "fail"]
    verified = any(c["name"] == "live_model_probe" and c["status"] == "ok" for c in checks)
    overall = "blocked" if failing else ("ready" if verified else "unverified")
    return {"status": overall, "checks": checks, "probes": probes,
            "note": None if verified else "Live model availability is verified only by `--probe` (one small request)."}


def format_report(result: dict) -> str:
    marks = {"ok": "✓", "warn": "!", "fail": "✗"}
    lines = [f"Feather automation doctor: {result['status'].upper()}"]
    for check in result["checks"]:
        lines.append(f"  {marks[check['status']]} {check['name']}: {check['detail']}")
        if check.get("action"):
            lines.append(f"      action: {check['action']}")
    for item in result["probes"]:
        kind = (item.get("failure") or {}).get("kind")
        lines.append(f"  probe {item['model']}: {item['status']}" + (f" ({kind})" if kind else ""))
    if result.get("note"):
        lines.append(f"  note: {result['note']}")
    return "\n".join(lines)
