"""Unit-test scoring of generated Python inside an ExecutionBackend; never in the host interpreter."""
import re
import shutil
import subprocess

from backend.evaluation.sandbox import (ExecutionBackend, Limits, _bounded_run, discover, from_legacy)  # noqa: F401

MAX_CODE = Limits.max_code_bytes
FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)(?:```|\Z)", re.S)

# Outcome categories for one problem.
OUTCOMES = ("passed", "partial_pass", "wrong_answer", "compile_error", "missing_function", "runtime_error",
            "timeout", "memory_limit", "output_limit", "code_too_large", "no_code")


def availability() -> dict:
    status = discover().status()
    return status if status["status"] == "available" else {**status, "status": "blocked"}


def extract_code(text: str) -> str:
    """Prefer the first ```python fenced block, then any fenced block, then the raw text."""
    blocks = FENCE.findall(text or "")
    for language, body in blocks:
        if language.lower() in {"python", "py", "python3"}:
            return body.strip("\n")
    if blocks:
        return blocks[0][1].strip("\n")
    return (text or "").strip()


def _equal(a, b) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    return a == b


def score_code(code: str, task: dict, timeout: float | None = None, sandbox=None, limits: Limits | None = None) -> dict:
    """Score one problem. ``task['cases']`` is a list of (arguments, expected) pairs kept on the host."""
    backend = from_legacy(sandbox)
    status = backend.status()
    if status["status"] != "available":
        return {"status": "blocked", "reason": status.get("reason", "Sandbox unavailable"), "backend": status.get("backend")}
    limits = limits or Limits(**({"wall_seconds": max(timeout, 1.0)} if timeout else {}))
    program = extract_code(code)
    base = {"status": "scored", "passed": False, "test_cases": len(task["cases"]), "cases_passed": 0,
            "sandbox_image": status.get("image_id"), "sandbox_backend": backend.name}
    if not program:
        return {**base, "outcome": "no_code", "reason": "Response contained no code"}
    if len(program.encode()) > limits.max_code_bytes:
        return {**base, "outcome": "code_too_large", "reason": "Generated code exceeds size limit"}
    # Expected answers remain on the host. The container receives code and test inputs only.
    record = backend.run_tests(program, task["function"], [case[0] for case in task["cases"]], limits)
    category = record.get("category")
    if category == "blocked":
        return {"status": "blocked", "reason": record.get("reason")}
    if category == "infrastructure_error":
        return {"status": "failed", "reason": record.get("reason"), "execution": record}
    if category != "executed":
        return {**base, "outcome": category, "reason": record.get("reason"), "execution": _slim(record)}
    harness = record["harness"]
    if harness.get("phase") in {"compile", "load"}:
        error = harness.get("error")
        outcome = ("compile_error" if harness["phase"] == "compile" else "missing_function" if error == "MissingFunction"
                   else "timeout" if error == "Timeout" else "memory_limit" if error == "MemoryError"
                   else "output_limit" if error == "OutputLimit" else "runtime_error")
        return {**base, "outcome": outcome, "reason": f"{error}: {harness.get('message', '')}".strip(": "),
                "execution": _slim(record)}
    if harness.get("error") == "OutputLimit":
        return {**base, "outcome": "output_limit", "reason": "Result exceeded output limit", "execution": _slim(record)}
    cases = harness.get("cases", [])
    results = []
    for index, (case, observed) in enumerate(zip(task["cases"], cases)):
        if observed.get("ok"):
            results.append({"index": index, "passed": _equal(observed["value"], case[1])})
        else:
            results.append({"index": index, "passed": False, "error": observed.get("error"),
                            "message": observed.get("message", "")[:200]})
    passed = sum(row["passed"] for row in results)
    complete = len(cases) == len(task["cases"])
    errors = [row["error"] for row in results if row.get("error")]
    outcome = ("passed" if complete and passed == len(task["cases"]) else
               "partial_pass" if passed else
               "timeout" if errors and all(e == "Timeout" for e in errors) else
               "memory_limit" if errors and all(e == "MemoryError" for e in errors) else
               "runtime_error" if errors else "wrong_answer")
    return {**base, "passed": outcome == "passed", "cases_passed": passed, "outcome": outcome,
            "case_results": results, "execution": _slim(record)}


def _slim(record: dict) -> dict:
    return {key: record.get(key) for key in ("container", "exit_code", "seconds", "category", "image_id")}


def _job_identifier(job_id: str) -> str:
    if not isinstance(job_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', job_id):
        raise ValueError('Invalid evaluation job identifier')
    return job_id


def cleanup_job(job_id: str) -> bool:
    """Remove only containers labeled for this job; fail if cleanup cannot be confirmed."""
    job_id = _job_identifier(job_id)
    binary = shutil.which('docker')
    if binary is None:
        # This executable is also required to create an evaluation container.
        return True
    try:
        result = subprocess.run([binary, 'ps', '-aq', '--filter', 'label=feather.job=' + job_id],
                                capture_output=True, text=True, timeout=5)
        if result.returncode:
            raise RuntimeError('Docker job cleanup cannot be confirmed: daemon unavailable')
        identifiers = result.stdout.split()
        if any(not re.fullmatch(r'[0-9a-f]{12,64}', identifier) for identifier in identifiers):
            raise RuntimeError('Docker returned an invalid container identity')
        if identifiers:
            removed = subprocess.run([binary, 'rm', '-f', *identifiers], capture_output=True, text=True, timeout=5)
            if removed.returncode:
                raise RuntimeError('Docker job container removal failed')
        return True
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f'Docker job cleanup not confirmed: {type(error).__name__}') from error


cleanup_job_containers = cleanup_job
