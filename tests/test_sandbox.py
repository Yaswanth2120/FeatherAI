"""Sandbox behaviour. Classification is tested with a fake Docker; real isolation tests run only
when the pinned image and a reachable daemon are available (otherwise skipped, never passed)."""
import json
from types import SimpleNamespace
import pytest
import backend.evaluation.sandbox as sandbox
from backend.evaluation.coding import extract_code, score_code
from backend.evaluation.sandbox import DockerExecutionBackend, UnavailableExecutionBackend, discover

VERIFIED = {'status': 'available', 'binary': 'docker', 'image_id': 'sha256:test'}
TASK = {'function': 'f', 'cases': [([2], 4), ([3], 9), ([0], 0)]}


STARTED = '2026-11-09T21:00:00.000000000Z'
NEVER = '0001-01-01T00:00:00Z'


def fake_docker(monkeypatch, exit_code=0, output=b'', oom='false', started=STARTED):
    calls = []
    monkeypatch.setattr(sandbox, '_bounded_run', lambda command, payload, timeout, limit: (calls.append(command) or (exit_code, output)))
    monkeypatch.setattr(sandbox.subprocess, 'run', lambda command, **kw: (calls.append(command) or SimpleNamespace(
        returncode=0, stderr='', stdout=f'{oom}|{started}' if command[1] == 'inspect' else '')))
    return calls


@pytest.mark.parametrize('exit_code,oom,outcome', [(137, 'false', 'memory_limit'), (0, 'true', 'memory_limit'),
                                                   (152, 'false', 'timeout'), (1, 'false', 'runtime_error')])
def test_exit_status_classification(monkeypatch, exit_code, oom, outcome):
    calls = fake_docker(monkeypatch, exit_code, b'{}', oom)
    result = score_code('def f(x): return x', TASK, sandbox=VERIFIED)
    assert result['status'] == 'scored' and result['outcome'] == outcome and result['passed'] is False
    assert calls[-1][1:3] == ['rm', '-f'], 'container must always be removed'


def test_infrastructure_failure_is_not_a_score(monkeypatch):
    fake_docker(monkeypatch, 125, b'', started=NEVER)
    assert score_code('def f(x): return x', TASK, sandbox=VERIFIED)['status'] == 'failed'


@pytest.mark.parametrize('exit_code,output', [(125, b''), (127, b''), (0, b''), (0, b'not json'), (0, b'[1, 2]')])
def test_generated_code_cannot_disguise_failure_as_infrastructure(monkeypatch, exit_code, output):
    # A started container's exit status and stdout are controlled by generated code.
    fake_docker(monkeypatch, exit_code, output)
    result = score_code('import os\nos._exit(125)', TASK, sandbox=VERIFIED)
    assert result['status'] == 'scored' and result['outcome'] == 'runtime_error' and result['passed'] is False


@pytest.mark.parametrize('harness,outcome', [
    ({'phase': 'compile', 'error': 'SyntaxError'}, 'compile_error'),
    ({'phase': 'load', 'error': 'MissingFunction'}, 'missing_function'),
    ({'phase': 'load', 'error': 'Timeout'}, 'timeout'),
    ({'phase': 'cases', 'cases': [{'ok': True, 'value': 4}, {'ok': False, 'error': 'ZeroDivisionError'}, {'ok': True, 'value': 0}]}, 'partial_pass'),
    ({'phase': 'cases', 'cases': [{'ok': True, 'value': 5}, {'ok': True, 'value': 5}, {'ok': True, 'value': 5}]}, 'wrong_answer'),
    ({'phase': 'cases', 'cases': [{'ok': False, 'error': 'Timeout'}] * 3}, 'timeout'),
    ({'phase': 'cases', 'cases': [{'ok': True, 'value': 4}]}, 'partial_pass'),  # truncated case list never passes
])
def test_harness_outcomes(monkeypatch, harness, outcome):
    fake_docker(monkeypatch, 0, json.dumps(harness).encode())
    result = score_code('def f(x): return x', TASK, sandbox=VERIFIED)
    assert result['outcome'] == outcome and result['passed'] is False


def test_unavailable_backend_never_executes():
    backend = UnavailableExecutionBackend('no docker')
    assert backend.status()['status'] == 'blocked'
    assert score_code('import os; os.system("touch /tmp/pwned")', TASK, sandbox=backend)['status'] == 'blocked'


def test_daemon_without_seccomp_is_refused(monkeypatch):
    def run(command, **kwargs):
        if command[1] == 'version':
            return SimpleNamespace(returncode=0, stdout='29.0\n', stderr='')
        if command[1] == 'info':
            return SimpleNamespace(returncode=0, stdout='["name=cgroupns"]|Linux|100', stderr='')
        return SimpleNamespace(returncode=0, stdout='sha256:x', stderr='')
    monkeypatch.setattr(sandbox.subprocess, 'run', run)
    assert DockerExecutionBackend(binary='docker').status()['status'] == 'blocked'


def test_code_extraction():
    assert extract_code('text\n```python\ndef f():\n    return 1\n```\nmore') == 'def f():\n    return 1'
    assert extract_code('```\nx = 1\n```') == 'x = 1'
    assert extract_code('def g(): pass') == 'def g(): pass'
    assert extract_code('```python\ndef f():\n    return 1') == 'def f():\n    return 1'  # unterminated fence


real = discover()
requires_docker = pytest.mark.skipif(real.status()['status'] != 'available',
                                     reason=f"Docker sandbox unavailable: {real.status().get('reason')}")


@requires_docker
@pytest.mark.parametrize('code,outcome', [
    ('def f(x):\n    return x * x', 'passed'),
    ('def f(x):\n    return x + 1', 'wrong_answer'),
    ('def f(x):\n    while True:\n        pass', 'timeout'),
    ('def f(x):\n    a = []\n    while True:\n        a.append(bytearray(10**7))', 'memory_limit'),
    ('def f(x) return x', 'compile_error'),
    ('import os\ndef f(x):\n    while True:\n        os.fork()', 'runtime_error'),
])
def test_real_container_outcomes(code, outcome):
    result = score_code(code, TASK, sandbox=real)
    assert result['status'] == 'scored' and result['outcome'] == outcome


@requires_docker
def test_real_container_has_no_network_no_writable_root_and_is_not_root():
    code = ('import os, socket\n'
            'def f(x):\n'
            '    try:\n        socket.create_connection(("1.1.1.1", 53), timeout=2); net = "open"\n'
            '    except OSError: net = "blocked"\n'
            '    try:\n        open("/etc/feather", "w"); fs = "writable"\n'
            '    except OSError: fs = "read-only"\n'
            '    return [net, fs, os.getuid()]')
    result = score_code(code, {'function': 'f', 'cases': [([0], ['blocked', 'read-only', 65534])]}, sandbox=real)
    assert result['passed'], result


@requires_docker
def test_real_container_exit_tricks_are_runtime_errors():
    for code in ['import os\nos._exit(125)', 'import os\nos._exit(0)', 'import sys\nsys.stdout.write("x")\nimport os\nos._exit(0)']:
        result = score_code(code, TASK, sandbox=real)
        assert result['status'] == 'scored' and result['outcome'] == 'runtime_error', (code, result)


def test_execution_seconds_is_elapsed_time_not_clock_value(monkeypatch):
    fake_docker(monkeypatch, 0, json.dumps({'phase': 'cases', 'cases': [{'ok': True, 'value': 4}] * 3}).encode())
    result = score_code('def f(x): return x', TASK, sandbox=VERIFIED)
    assert 0 <= result['execution']['seconds'] < 60
