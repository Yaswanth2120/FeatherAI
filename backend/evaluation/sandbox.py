"""Execution backends for untrusted model-generated code.

Threat model: generated code is treated as hostile. It must never execute in a host
interpreter. The only supported backend is a locked-down Docker container (on macOS this is
a Linux container inside the Docker Desktop virtual machine). Docker is an isolation layer,
not an absolute security boundary: a compromised daemon, VM or kernel defeats it. When the
required isolation cannot be confirmed, execution is refused; there is no subprocess fallback.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import json
import os
import re
import selectors
import shutil
import subprocess
import tempfile
import time
import uuid

# python:3.12-alpine (CPython 3.12.16), pinned by registry digest on 2026-11-09.
PINNED_IMAGE = "docker.io/library/python@sha256:e064671ca452259bab50e8e2c7bd896e39e9a343bb06ca25234adf6fe18d0fc7"
OOM_EXIT = 137          # SIGKILL from the cgroup OOM killer (or docker kill)
CPU_LIMIT_EXIT = 152    # SIGXCPU after RLIMIT_CPU


class OutputLimitExceeded(ValueError):
    pass


@dataclass(frozen=True)
class Limits:
    memory_mb: int = 128
    cpus: float = 0.5
    pids: int = 32
    per_case_seconds: float = 2.0
    cpu_seconds: int = 10
    wall_seconds: float = 20.0
    max_output_bytes: int = 65536
    max_code_bytes: int = 16384
    tmpfs_mb: int = 8


# Runs inside the container. Expected answers never enter the container.
HARNESS = r'''
import json, os, resource, signal, sys
payload = json.loads(sys.stdin.read())
MAIN = os.getpid()
resource.setrlimit(resource.RLIMIT_CPU, (payload["cpu_seconds"], payload["cpu_seconds"]))
resource.setrlimit(resource.RLIMIT_FSIZE, (65536, 65536))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
out = os.fdopen(os.dup(1), "w")
sink = os.open(os.devnull, os.O_WRONLY)
os.dup2(sink, 1); os.dup2(sink, 2)
class CaseTimeout(BaseException):
    pass
def _alarm(signum, frame):
    raise CaseTimeout()
signal.signal(signal.SIGALRM, _alarm)
def emit(obj):
    if os.getpid() != MAIN:
        os._exit(0)  # forked children never report results
    text = json.dumps(obj)
    if len(text) > payload["max_output_bytes"]:
        text = json.dumps({"phase": obj.get("phase"), "error": "OutputLimit"})
    out.write(text); out.flush(); os._exit(0)
def describe(error):
    return {"error": type(error).__name__, "message": str(error)[:300]}
def canon(v, depth=0):
    if depth > 50:
        raise ValueError("too deep")
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, list):
        return [canon(x, depth + 1) for x in v]
    if isinstance(v, tuple):
        return {"__tuple__": [canon(x, depth + 1) for x in v]}
    if isinstance(v, (set, frozenset)):
        return {"__set__": sorted((canon(x, depth + 1) for x in v), key=lambda x: json.dumps(x, sort_keys=True))}
    if isinstance(v, dict):
        return {"__dict__": sorted(([canon(k, depth + 1), canon(x, depth + 1)] for k, x in v.items()), key=lambda x: json.dumps(x, sort_keys=True))}
    raise TypeError(type(v).__name__)
def uncanon(v):
    if isinstance(v, list):
        return [uncanon(x) for x in v]
    if isinstance(v, dict):
        if set(v) == {"__tuple__"}:
            return tuple(uncanon(x) for x in v["__tuple__"])
        if set(v) == {"__set__"}:
            return set(uncanon(x) for x in v["__set__"])
        if set(v) == {"__dict__"}:
            return {uncanon(k): uncanon(x) for k, x in v["__dict__"]}
    return v
try:
    compiled = compile(payload["code"], "<generated>", "exec")
except (SyntaxError, ValueError) as error:
    emit({"phase": "compile", **describe(error)})
namespace = {"__name__": "generated"}
signal.setitimer(signal.ITIMER_REAL, payload["per_case_seconds"])
try:
    exec(compiled, namespace)
except CaseTimeout:
    emit({"phase": "load", "error": "Timeout"})
except BaseException as error:
    emit({"phase": "load", **describe(error)})
finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
function = namespace.get(payload["function"])
if not callable(function):
    emit({"phase": "load", "error": "MissingFunction", "message": payload["function"]})
cases = []
for arguments in payload["inputs"]:
    if payload.get("canonical"):
        arguments = [uncanon(a) for a in arguments]
    signal.setitimer(signal.ITIMER_REAL, payload["per_case_seconds"])
    try:
        value = function(*arguments)
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            encoded = canon(value) if payload.get("canonical") else value
            json.dumps(encoded)
            cases.append({"ok": True, "value": encoded})
        except (TypeError, ValueError, RecursionError):
            cases.append({"ok": False, "error": "UnserializableResult", "message": type(value).__name__})
    except CaseTimeout:
        cases.append({"ok": False, "error": "Timeout"})
    except BaseException as error:
        cases.append({"ok": False, **describe(error)})
    finally:
        if os.getpid() != MAIN:
            os._exit(0)
        signal.setitimer(signal.ITIMER_REAL, 0)
emit({"phase": "cases", "cases": cases})
'''


def _bounded_run(command: list[str], payload: str, timeout: float, max_output: int = Limits.max_output_bytes) -> tuple[int, bytes]:
    """Run a client process with a wall-clock deadline and bounded combined output."""
    with tempfile.TemporaryFile() as source:
        source.write(payload.encode())
        source.seek(0)
        # Client/daemon diagnostics go to a separate bounded sink so they cannot corrupt the result frame.
        errors = tempfile.TemporaryFile()
        process = subprocess.Popen(command, stdin=source, stdout=subprocess.PIPE, stderr=errors)
        output = bytearray()
        deadline = time.monotonic() + timeout
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout)
                    for key, _ in selector.select(min(remaining, .1)):
                        chunk = os.read(key.fileobj.fileno(), 4096)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            break
                        output.extend(chunk)
                        if len(output) > max_output:
                            raise OutputLimitExceeded("Generated program exceeded output limit")
            return process.wait(timeout=max(.01, deadline - time.monotonic())), bytes(output)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            process.stdout.close()
            errors.close()


class ExecutionBackend(ABC):
    name = "abstract"

    @abstractmethod
    def status(self) -> dict:
        """Return {'status': 'available'|'blocked', 'backend': name, 'reason'?: str, ...}."""

    @abstractmethod
    def run_tests(self, code: str, function: str, inputs: list, limits: Limits = Limits(), canonical: bool = False) -> dict:
        """Execute untrusted ``code`` against ``inputs``; return a structured execution record."""


class UnavailableExecutionBackend(ExecutionBackend):
    name = "unavailable"

    def __init__(self, reason: str):
        self.reason = reason

    def status(self) -> dict:
        return {"status": "blocked", "backend": self.name, "reason": self.reason}

    def run_tests(self, code, function, inputs, limits=Limits(), canonical=False):
        return {"category": "blocked", "reason": self.reason}


def _job_label() -> list[str]:
    job = os.environ.get("FEATHER_JOB_ID")
    if not job:
        return []
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", job):
        raise ValueError("Invalid evaluation job identifier")
    return ["--label", "feather.job=" + job]


class DockerExecutionBackend(ExecutionBackend):
    name = "docker"

    def __init__(self, image: str = PINNED_IMAGE, binary: str | None = None, verified: dict | None = None):
        self.image = image
        self.binary = binary or shutil.which("docker")
        self._status = verified

    def _docker(self, *args, timeout=10):
        return subprocess.run([self.binary, *args], capture_output=True, text=True, timeout=timeout)

    def status(self) -> dict:
        if self._status is not None:
            return self._status
        blocked = lambda reason: {"status": "blocked", "backend": self.name, "reason": reason}
        if not self.binary:
            return blocked("Docker CLI is not installed; coding evaluation is disabled (no host fallback)")
        try:
            version = self._docker("version", "--format", "{{.Server.Version}}")
            if version.returncode:
                return blocked("Docker daemon is not reachable; start Docker Desktop. No host fallback is used.")
            info = self._docker("info", "--format", "{{json .SecurityOptions}}|{{.OperatingSystem}}|{{.MemTotal}}")
            security, operating_system, memory = (info.stdout.strip().split("|") + ["", "", ""])[:3]
            if info.returncode or "seccomp" not in security:
                return blocked("Docker daemon does not report seccomp; required isolation cannot be confirmed")
            image = self._docker("image", "inspect", "--format", "{{.Id}}", self.image)
            if image.returncode:
                return blocked(f"Pinned sandbox image {self.image} is not present locally; run `feather sandbox prepare` "
                               "(explicit pull, never automatic)")
        except (OSError, subprocess.TimeoutExpired) as error:
            return blocked(f"Docker unavailable: {type(error).__name__}")
        self._status = {"status": "available", "backend": self.name, "binary": self.binary, "image": self.image,
                        "image_id": image.stdout.strip(), "server_version": version.stdout.strip(),
                        "security_options": json.loads(security) if security.startswith("[") else security,
                        "daemon_os": operating_system, "daemon_memory_bytes": int(memory) if memory.isdigit() else None,
                        "isolation": "Linux container, no network, read-only root, all capabilities dropped, non-root user"}
        return self._status

    def command(self, name: str, limits: Limits) -> list[str]:
        status = self.status()
        return [self.binary, "run", "--pull=never", "--name", name, "-i",
                "--network", "none", "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--pids-limit", str(limits.pids), "--memory", f"{limits.memory_mb}m", "--memory-swap", f"{limits.memory_mb}m",
                "--cpus", str(limits.cpus), "--user", "65534:65534", "--log-driver", "none", "--ipc", "none",
                "--tmpfs", f"/tmp:rw,noexec,nosuid,nodev,size={limits.tmpfs_mb}m", "--workdir", "/tmp",
                *_job_label(), status["image_id"], "python", "-I", "-S", "-c", HARNESS]

    def run_tests(self, code, function, inputs, limits=Limits(), canonical=False):
        status = self.status()
        if status["status"] != "available":
            return {"category": "blocked", "reason": status["reason"]}
        name = "feather-eval-" + uuid.uuid4().hex
        payload = json.dumps({"code": code, "function": function, "inputs": inputs,
                              "per_case_seconds": limits.per_case_seconds, "cpu_seconds": limits.cpu_seconds,
                              "max_output_bytes": limits.max_output_bytes, "canonical": canonical})
        clock_start = time.monotonic()
        record: dict = {"container": name, "image_id": status["image_id"], "limits": asdict(limits)}
        try:
            code_, output = _bounded_run(self.command(name, limits), payload, limits.wall_seconds, limits.max_output_bytes)
            record["exit_code"] = code_
            state = self._state(name)
            started = state.get("started")  # the container process ran: its exit status belongs to generated code
            if state.get("oom") or (started and code_ == OOM_EXIT):
                record.update(category="memory_limit", reason="Container exceeded its memory limit" if state.get("oom") else "Container was killed (exit 137)")
            elif started and code_ == CPU_LIMIT_EXIT:
                record.update(category="timeout", reason="CPU time limit exceeded")
            elif not started:
                record.update(category="infrastructure_error", reason=f"Container did not start (docker exit {code_})")
            elif code_:
                record.update(category="runtime_error", reason=f"Generated program terminated the container with exit code {code_}")
            else:
                try:
                    harness = json.loads(output)
                    if not isinstance(harness, dict) or harness.get("phase") not in {"compile", "load", "cases"}:
                        raise ValueError
                    record.update(harness=harness, category="executed")
                except ValueError:
                    # The harness always emits a frame; a missing or garbled one means generated code interfered.
                    record.update(category="runtime_error", reason="Generated program suppressed or corrupted the result frame")
        except subprocess.TimeoutExpired:
            record.update(category="timeout", reason="Execution timeout")
        except OutputLimitExceeded:
            record.update(category="output_limit", reason="Generated program exceeded output limit")
        except OSError as error:
            record.update(category="infrastructure_error", reason=str(error)[:300])
        finally:
            record["seconds"] = round(time.monotonic() - clock_start, 3)
            self._remove(name)
        return record

    def _state(self, name) -> dict:
        """Container state after exit: whether its process started and whether it was OOM-killed."""
        try:
            result = self._docker("inspect", "--format", "{{.State.OOMKilled}}|{{.State.StartedAt}}", name, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            return {}
        if result.returncode:
            return {}
        oom, _, started = ((getattr(result, "stdout", "") or "").strip() + "|").partition("|")
        started = started.rstrip("|")
        return {"oom": oom == "true", "started": bool(started) and not started.startswith("0001-01-01")}

    def _remove(self, name):
        # A killed Docker client does not necessarily stop its container.
        try:
            cleanup = self._docker("rm", "-f", name, timeout=10)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"Container cleanup not confirmed for {name}: {type(error).__name__}") from error
        if cleanup.returncode and "No such container" not in (cleanup.stderr or ""):
            raise RuntimeError(f"Container cleanup failed for {name}; inspect Docker before continuing")


def discover(image: str = PINNED_IMAGE) -> ExecutionBackend:
    backend = DockerExecutionBackend(image)
    status = backend.status()
    return backend if status["status"] == "available" else UnavailableExecutionBackend(status["reason"])


def from_legacy(sandbox: dict | ExecutionBackend | None) -> ExecutionBackend:
    """Accept an ExecutionBackend, a verified status mapping, or None (discover)."""
    if isinstance(sandbox, ExecutionBackend):
        return sandbox
    if sandbox is None:
        return discover()
    if sandbox.get("status") == "available":
        verified = {"backend": "docker", **sandbox}
        verified.setdefault("image_id", sandbox.get("image_id"))
        return DockerExecutionBackend(binary=sandbox.get("binary"), verified=verified)
    return UnavailableExecutionBackend(sandbox.get("reason", "Sandbox unavailable"))
