import json
import math
import subprocess
from types import SimpleNamespace
import pytest
from backend.evaluation.coding import score_code, _bounded_run, availability

SANDBOX = {'status': 'available', 'binary': 'docker', 'image_id': 'sha256:test'}


def harness_output(*values):
    return json.dumps({'phase': 'cases', 'cases': [{'ok': True, 'value': value} for value in values]}).encode()
from backend.evaluation.runner import evaluate, compare_evaluations
from backend.evaluation.scoring import perplexity, wilson_interval, accuracy_summary
from backend.evaluation.suites import CODING


def test_wilson_and_perplexity():
    assert wilson_interval(0, 0) is None
    assert wilson_interval(0, 10)[0] == pytest.approx(0)
    assert wilson_interval(10, 10)[1] == pytest.approx(1)
    assert wilson_interval(5, 10) == pytest.approx([.236593, .763407], abs=1e-5)
    with pytest.raises(ValueError):
        wilson_interval(2, 1)
    assert perplexity(None)['status'] == 'unsupported'
    assert perplexity([-math.log(2)] * 4)['value'] == pytest.approx(2)
    with pytest.raises(ValueError):
        perplexity([float('nan')])
    assert perplexity([-1000])['status'] == 'overflow'


def test_missing_scores_are_never_passes():
    result = accuracy_summary([{'status': 'unsupported'}, {'status': 'failed'}, {'status': 'scored', 'passed': False}])
    assert result['accuracy'] == 0
    assert result['scored'] == 1
    assert result['failed'] == 1
    assert accuracy_summary([{'status': 'unsupported'}])['accuracy'] is None


def test_docker_unavailable_does_not_execute_code(monkeypatch):
    monkeypatch.setattr('backend.evaluation.sandbox.shutil.which', lambda _: None)
    assert availability()['status'] == 'blocked'
    result = score_code("raise RuntimeError('host execution forbidden')", CODING[0])
    assert result['status'] == 'blocked' and 'no host fallback' in result['reason'].lower()


def test_docker_limits_no_host_mounts_and_cleanup(monkeypatch):
    import backend.evaluation.sandbox as module
    commands = []
    def fake(command, payload, timeout, max_output):
        commands.append(command)
        sent = json.loads(payload)
        assert 'expected' not in sent
        assert 'answer' not in sent and 'cases' not in sent
        return 0, harness_output(-1, 1, -1, 0)
    monkeypatch.setattr(module, '_bounded_run', fake)
    monkeypatch.setattr(module.subprocess, 'run', lambda command, **kwargs: (commands.append(command) or SimpleNamespace(returncode=0, stderr='', stdout='false|2026-11-09T21:00:00Z')))
    result = score_code('def binary_search(a,b): return -1', CODING[0], sandbox=SANDBOX)
    assert result['passed'] and result['outcome'] == 'passed'
    command = commands[0]
    for flag in ['--network', '--read-only', '--cap-drop', '--pids-limit', '--memory', '--memory-swap', '--cpus', '--user',
                 'no-new-privileges', '--pull=never']:
        assert flag in command
    assert command[command.index('--user') + 1] == '65534:65534'
    assert 'none' in command
    assert not any(flag in command for flag in ['-v', '--volume', '--mount', '--privileged'])
    assert commands[-1][1:3] == ['rm', '-f']


def test_docker_timeout_is_failed_score_and_cleans_up(monkeypatch):
    import backend.evaluation.sandbox as module
    calls = []
    def timeout(*args):
        raise subprocess.TimeoutExpired('docker', 5)
    monkeypatch.setattr(module, '_bounded_run', timeout)
    monkeypatch.setattr(module.subprocess, 'run', lambda command, **kwargs: (calls.append(command) or SimpleNamespace(returncode=0, stderr='', stdout='')))
    result = score_code('while True: pass', CODING[0], sandbox=SANDBOX)
    assert result['passed'] is False
    assert result['reason'] == 'Execution timeout' and result['outcome'] == 'timeout'
    assert calls[-1][1:3] == ['rm', '-f']


def test_bounded_output_rejects_flood():
    import sys
    with pytest.raises(ValueError, match='output limit'):
        _bounded_run([sys.executable, '-c', 'print("x" * 100000)'], '', 3)


def test_reasoning_repeated_generation_and_failure_provenance():
    class Backend:
        n = 0
        def models(self): return [{'name': 'model', 'digest': 'digest'}]
        def version(self): return 'test'
        def generate(self, model, prompt, options):
            self.n += 1
            if self.n == 2: raise RuntimeError('OOM')
            return {'output': '12', 'runtime': {}}
    result = evaluate(Backend(), 'model', repeats=2)
    assert len(result['results']) == 6
    assert result['status'] == 'partial'
    assert result['summary']['failed'] == 1
    assert result['summary']['passed'] == 2
    assert result['summary']['scored'] == 5
    assert len(result['suite_sha256']) == 64
    assert result['options']['temperature'] == 0
    assert result['perplexity']['status'] == 'unsupported'


def test_coding_skips_generation_if_no_sandbox(monkeypatch):
    class Backend:
        def models(self): return [{'name':'model'}]
        def version(self): return 'test'
        def generate(self, *args): raise AssertionError('Should not run inference')
    monkeypatch.setattr('backend.evaluation.runner.availability', lambda: {'status':'blocked','reason':'no docker'})
    result = evaluate(Backend(), 'model', suite='coding-small')
    assert result['status'] == 'blocked'
    assert result['summary']['accuracy'] is None


def test_quality_comparison_requires_identity_and_complete_results():
    sample = {'suite_sha256':'s','settings_sha256':'o','repeats':1,'source_sha256':'source','status':'completed',
              'summary':{'accuracy':.5,'accuracy_ci95':[.2,.8]}}
    other = dict(sample)
    result = compare_evaluations(sample, other)
    assert result['accuracy_difference'] == 0
    assert result['difference_interval'] == pytest.approx([-.6,.6])
    assert compare_evaluations(sample, {**other, 'source_sha256':'another'})['status'] == 'incomparable'
    assert compare_evaluations(sample, {**other,'status':'partial'})['status'] == 'insufficient_evidence'


def test_coding_boolean_is_not_integer_correctness(monkeypatch):
    import backend.evaluation.sandbox as module
    monkeypatch.setattr(module, '_bounded_run', lambda *args: (0, harness_output([], [True, False, 3], [False])))
    monkeypatch.setattr(module.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr='', stdout='false|2026-11-09T21:00:00Z'))
    result = score_code('pass', CODING[2], sandbox=SANDBOX)
    assert result['passed'] is False
    assert result['cases_passed'] == 1


def test_bounded_execution_timeout_kills_client():
    import sys
    with pytest.raises(subprocess.TimeoutExpired):
        _bounded_run([sys.executable, '-c', 'import time; time.sleep(5)'], '', .05)


def test_coding_labels_worker_job_for_external_cleanup(monkeypatch):
    import backend.evaluation.sandbox as module
    calls = []
    monkeypatch.setenv('FEATHER_JOB_ID','job-123')
    monkeypatch.setattr(module, '_bounded_run', lambda command, *args: (calls.append(command) or (0, harness_output(-1, 1, -1, 0))))
    monkeypatch.setattr(module.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr='', stdout='false|2026-11-09T21:00:00Z'))
    score_code('pass', CODING[0], sandbox=SANDBOX)
    assert '--label' in calls[0]
    assert 'feather.job=job-123' in calls[0]


def test_external_cleanup_filters_exact_job_and_validates_container_ids(monkeypatch):
    import backend.evaluation.coding as module
    calls = []
    monkeypatch.setattr(module.shutil,'which',lambda _: 'docker')
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0,stdout='abcdef123456\n' if command[1]=='ps' else '',stderr='')
    monkeypatch.setattr(module.subprocess,'run',run)
    assert module.cleanup_job('job-123')
    assert calls[0] == ['docker','ps','-aq','--filter','label=feather.job=job-123']
    assert calls[1] == ['docker','rm','-f','abcdef123456']
    with pytest.raises(ValueError): module.cleanup_job('x --all')
    monkeypatch.setattr(module.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(returncode=1,stdout='',stderr='no daemon'))
    with pytest.raises(RuntimeError,match='cannot be confirmed'): module.cleanup_job('job-123')


def test_evaluation_carries_managed_artifact_lineage():
    class Backend:
        name = 'llama.cpp'
        source_sha256 = 'verified-source'
        def models(self): return [{'name':'managed','artifact_sha256':'artifact'}]
        def version(self): return 'test'
        def generate(self,*args): return {'output':'12'}
    report = evaluate(Backend(),'managed')
    assert report['backend'] == 'llama.cpp'
    assert report['source_sha256'] == 'verified-source'
    assert report['model']['artifact_sha256'] == 'artifact'


def test_wilson_extreme_counts_contain_point_without_rounding_rejection():
    for trials in [1,3,5,10,100]:
        assert wilson_interval(0,trials)[0] == 0.0
        assert wilson_interval(trials,trials)[1] == 1.0
        assert wilson_interval(0,trials)[1] > 0
        assert wilson_interval(trials,trials)[0] < 1
