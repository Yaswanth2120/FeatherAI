"""Docker isolation with bounded output; no generated code executes in the host interpreter."""
import json
import os
import re
import selectors
import shutil
import subprocess
import tempfile
import time
import uuid

IMAGE = 'python:3.12-alpine'
MAX_OUTPUT = 65536
MAX_CODE = 16384


def availability(image: str = IMAGE) -> dict:
    binary = shutil.which('docker')
    if binary is None:
        return {'status': 'unsupported', 'reason': 'Docker CLI is not installed'}
    try:
        result = subprocess.run([binary, 'image', 'inspect', '--format', '{{.Id}}', image], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'status': 'unsupported', 'reason': f'Docker unavailable: {type(error).__name__}'}
    if result.returncode:
        return {'status': 'unsupported', 'reason': f'Docker daemon or local image {image} unavailable; no automatic pull'}
    return {'status': 'available', 'binary': binary, 'image_id': result.stdout.strip()}


def _bounded_run(command: list[str], payload: str, timeout: float) -> tuple[int, bytes]:
    with tempfile.TemporaryFile() as source:
        source.write(payload.encode())
        source.seek(0)
        process = subprocess.Popen(command, stdin=source, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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
                        if len(output) > MAX_OUTPUT:
                            raise ValueError('Generated program exceeded output limit')
            return process.wait(timeout=max(.01, deadline - time.monotonic())), bytes(output)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            process.stdout.close()


def score_code(code: str, task: dict, timeout: float = 5, sandbox: dict | None = None) -> dict:
    sandbox = sandbox or availability()
    if sandbox['status'] != 'available':
        return sandbox.copy()
    if len(code.encode()) > MAX_CODE:
        return {'status': 'scored', 'passed': False, 'reason': 'Generated code exceeds size limit'}
    if code.strip().startswith('```'):
        parts = code.strip().splitlines()
        if parts[-1].strip() == '```':
            code = '\n'.join(parts[1:-1])
    name = 'feather-eval-' + uuid.uuid4().hex
    # Expected answers remain on the host. The container receives code and test inputs only.
    payload = {'code': code, 'function': task['function'], 'inputs': [case[0] for case in task['cases']]}
    harness = '''import contextlib, io, json, os, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
resource.setrlimit(resource.RLIMIT_FSIZE, (65536, 65536))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
payload = json.loads(sys.stdin.read())
namespace = {}
with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
    exec(compile(payload['code'], '<generated>', 'exec'), namespace)
    fn = namespace[payload['function']]
    results = [fn(*args) for args in payload['inputs']]
encoded = json.dumps(results)
if len(encoded) > 65536: raise ValueError('result too large')
sys.stdout.write(encoded)
'''
    command = [sandbox['binary'], 'run', '--rm', '--pull=never', '--name', name, '-i',
               '--network', 'none', '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
               '--pids-limit', '32', '--memory', '128m', '--memory-swap', '128m', '--cpus', '0.5',
               '--user', '65534:65534', '--log-driver', 'none', '--tmpfs', '/tmp:rw,noexec,nosuid,size=8m',
               *(['--label', 'feather.job=' + _job_identifier(os.environ['FEATHER_JOB_ID'])] if os.environ.get('FEATHER_JOB_ID') else []),
               sandbox['image_id'], 'python', '-I', '-c', harness]
    try:
        status, output = _bounded_run(command, json.dumps(payload), timeout)
        if status in {125, 126, 127}:
            return {'status': 'failed', 'reason': f'Docker infrastructure failed with code {status}'}
        if status:
            return {'status': 'scored', 'passed': False, 'reason': f'Container exited with code {status}'}
        actual = json.loads(output)
        expected = [case[1] for case in task['cases']]
        def equal(a, b):
            if type(a) is not type(b):
                return False
            if isinstance(a, list):
                return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
            return a == b
        matches = [equal(a, b) for a, b in zip(actual, expected)] if isinstance(actual, list) else []
        return {'status': 'scored', 'passed': isinstance(actual, list) and len(actual) == len(expected) and all(matches), 'test_cases': len(expected),
                'cases_passed': sum(matches),
                'sandbox_image': sandbox['image_id']}
    except subprocess.TimeoutExpired:
        return {'status': 'scored', 'passed': False, 'reason': 'Execution timeout'}
    except (ValueError, OSError) as error:
        return {'status': 'scored', 'passed': False, 'reason': str(error)[:500]}
    finally:
        # A killed Docker client does not necessarily stop its container.
        try:
            cleanup = subprocess.run([sandbox['binary'], 'rm', '-f', name], capture_output=True, text=True, timeout=5)
            if cleanup.returncode and 'No such container' not in cleanup.stderr:
                raise RuntimeError(f'Container cleanup failed for {name}; inspect Docker before continuing')
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f'Container cleanup not confirmed for {name}: {type(error).__name__}') from error


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
