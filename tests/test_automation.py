import json
import subprocess
from pathlib import Path
import tomllib
import pytest
import yaml
from automation.orchestrator import (Runner, CriticalFailure, load_tasks, eligible_tasks,
                                    review_approved, token_usage, exclusive_lock, invoke)


def manifest(tmp_path, tasks=None):
    tasks = tasks or [{'id': 'first', 'title': 'First', 'role': 'architect', 'depends_on': [],
                       'prompt': 'Write a module', 'acceptance': [['check']]}]
    path = tmp_path / 'tasks.yaml'
    path.write_text(yaml.safe_dump({'version': 1, 'tasks': tasks, 'regression': [['regression']]}))
    return path


def repository(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    subprocess.run(['git', 'config', 'user.name', 'Automation Test'], cwd=tmp_path, check=True)
    subprocess.run(['git', 'config', 'user.email', 'test@example.invalid'], cwd=tmp_path, check=True)
    (tmp_path / '.gitignore').write_text('.feather/\n')
    return tmp_path


def commit(root):
    subprocess.run(['git', 'add', '.'], cwd=root, check=True)
    subprocess.run(['git', 'commit', '-qm', 'Initial'], cwd=root, check=True)


def fake_executor(approve=True, implementation_code=0, interrupt=False):
    def run(argv, root, log, timeout, env):
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text('{"type":"turn.completed","usage":{"input_tokens":12}}\n')
        if argv[0] == 'git':
            return subprocess.run(argv, cwd=root).returncode
        if argv[0] == 'codex':
            if argv[argv.index('--sandbox') + 1] == 'read-only':
                final = Path(argv[argv.index('--output-last-message') + 1])
                final.write_text(json.dumps({'approved': approve, 'findings': [] if approve else ['Defect'], 'summary': 'Reviewed'}))
            else:
                final = Path(argv[argv.index('--output-last-message') + 1])
                final.write_text(json.dumps({'status': 'completed', 'summary': 'Implemented', 'blocker': ''}))
                (root / 'module.py').write_text('value = 1\n')
                if interrupt:
                    raise KeyboardInterrupt
                return implementation_code
        return 0
    return run


def test_dependency_graph_and_cycle_rejection(tmp_path):
    path = manifest(tmp_path, [
        {'id': 'a', 'depends_on': [], 'prompt': 'a', 'acceptance': [['true']]},
        {'id': 'b', 'depends_on': ['a'], 'prompt': 'b', 'acceptance': [['true']]}])
    tasks = load_tasks(path)['tasks']
    state = {'a': {'status': 'pending'}, 'b': {'status': 'pending'}}
    assert [t['id'] for t in eligible_tasks(tasks, state)] == ['a']
    state['a']['status'] = 'completed'
    assert [t['id'] for t in eligible_tasks(tasks, state)] == ['b']
    data = yaml.safe_load(path.read_text())
    data['tasks'][0]['depends_on'] = ['b']
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='cycle'):
        load_tasks(path)


def test_shell_command_strings_are_rejected(tmp_path):
    path = manifest(tmp_path)
    data = yaml.safe_load(path.read_text())
    data['regression'] = ['echo hello; touch unexpected']
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='argv'):
        load_tasks(path)


def test_dirty_tree_refuses_implementation(tmp_path):
    root = repository(tmp_path)
    path = manifest(root)
    commit(root)
    (root / 'unrelated.txt').write_text('preserve')
    runner = Runner(root, path, executor=fake_executor())
    with pytest.raises(CriticalFailure, match='dirty'):
        runner.run(one=True)
    assert not (root / 'module.py').exists()
    assert runner.state['first']['attempts'] == 0


def test_success_requires_review_and_commits_once(tmp_path):
    root = repository(tmp_path)
    path = manifest(root)
    commit(root)
    runner = Runner(root, path, executor=fake_executor())
    runner.run(one=True)
    assert runner.state['first']['status'] == 'completed'
    assert runner.state['first']['attempts'] == 1
    assert runner.state['first']['usage'] == [{'input_tokens': 12}, {'input_tokens': 12}]
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    resumed = Runner(root, path, executor=fake_executor())
    resumed.run()
    assert resumed.state['first']['commit'] == head
    assert resumed.state['first']['attempts'] == 1


def test_review_rejection_never_commits_and_retry_budget_persists(tmp_path):
    root = repository(tmp_path)
    path = manifest(root)
    commit(root)
    original = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root)
    runner = Runner(root, path, executor=fake_executor(approve=False))
    runner.run(one=True)
    assert runner.state['first']['status'] == 'failed'
    assert runner.state['first']['attempts'] == 3
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root) == original
    reopened = Runner(root, path, executor=fake_executor())
    with pytest.raises(ValueError, match='exhausted'):
        reopened.resume('first')


def test_interruption_is_blocked_not_blindly_replayed(tmp_path):
    root = repository(tmp_path)
    path = manifest(root)
    commit(root)
    runner = Runner(root, path, executor=fake_executor(interrupt=True))
    with pytest.raises(KeyboardInterrupt):
        runner.run(one=True)
    assert runner.state['first']['status'] == 'blocked'
    assert runner.state['first']['attempts'] == 1
    reopened = Runner(root, path)
    with pytest.raises(CriticalFailure, match='clean'):
        reopened.resume('first')
    reopened.run()
    assert reopened.state['first']['attempts'] == 1


def test_incomplete_review_or_findings_fail_closed(tmp_path):
    path = tmp_path / 'review.json'
    for payload in [{'approved': True}, {'approved': True, 'findings': ['Defect'], 'summary': 'x'},
                    {'approved': 1, 'findings': [], 'summary': 'x'}, {'approved': True, 'findings': [], 'summary': ''}]:
        path.write_text(json.dumps(payload))
        assert not review_approved(path)
    path.write_text('not-json')
    assert not review_approved(path)


def test_lock_rejects_second_runner(tmp_path):
    with exclusive_lock(tmp_path):
        with pytest.raises(CriticalFailure, match='lock'):
            with exclusive_lock(tmp_path):
                pass


def test_timeout_captured_and_process_stopped(tmp_path):
    import sys
    log = tmp_path / 'timeout.log'
    result = invoke([sys.executable, '-c', 'import time; time.sleep(30)'], tmp_path, log, timeout=1)
    assert result == 124
    metadata = json.loads(log.with_suffix('.log.invocation.json').read_text())
    assert metadata['exit_code'] == 124
    assert metadata['interrupted'] == 'TimeoutExpired'


def test_manifest_change_does_not_reinterpret_saved_state(tmp_path):
    path = manifest(tmp_path)
    runner = Runner(tmp_path, path)
    runner.save()
    path.write_text(path.read_text() + '\n# changed\n')
    with pytest.raises(ValueError, match='manifest changed'):
        Runner(tmp_path, path)


def test_agent_definitions_use_documented_fields():
    root = Path(__file__).resolve().parents[1]
    definitions = list((root / '.codex/agents').glob('*.toml'))
    assert len(definitions) == 7
    for path in definitions:
        config = tomllib.loads(path.read_text())
        assert set(config) == {'name', 'description', 'developer_instructions'}
        assert config['name'] == path.stem


def test_usage_ignores_non_json_diagnostics(tmp_path):
    path = tmp_path / 'usage.log'
    path.write_text('diagnostic\n{"type":"turn.completed","usage":{"output_tokens":9}}\n')
    assert token_usage(path) == [{'output_tokens': 9}]


def test_explicit_blocked_outcome_does_not_retry(tmp_path):
    root = repository(tmp_path)
    path = manifest(root)
    commit(root)
    def blocked(argv, root, log, timeout, env):
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text('')
        final = Path(argv[argv.index('--output-last-message') + 1])
        final.write_text(json.dumps({'status': 'blocked', 'summary': 'Cannot convert', 'blocker': 'Insufficient RAM'}))
        return 0
    runner = Runner(root, path, executor=blocked)
    runner.run(one=True)
    assert runner.state['first']['status'] == 'blocked'
    assert runner.state['first']['attempts'] == 1
    assert runner.state['first']['error'] == 'Insufficient RAM'


def test_adoption_requires_commit_bound_review_and_gates(tmp_path):
    root = repository(tmp_path)
    path = manifest(root)
    commit(root)
    runner = Runner(root, path, executor=fake_executor())
    review = root / '.feather/review.json'
    review.parent.mkdir()
    review.write_text(json.dumps({'task': 'first', 'commit': 'wrong', 'review': {'approved': True, 'findings': [], 'summary': 'Reviewed'}}))
    with pytest.raises(ValueError, match='current HEAD'):
        runner.adopt('first', review)
    record = json.loads(review.read_text())
    record['commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    review.write_text(json.dumps(record))
    runner.adopt('first', review)
    assert runner.state['first']['status'] == 'completed'
    assert runner.state['first']['origin'] == 'externally-reviewed'
    assert runner.state['first']['attempts'] == 0


def test_real_evidence_gate_rejects_missing_artifacts(tmp_path, monkeypatch):
    from automation import validate
    monkeypatch.setattr(validate, 'ROOT', tmp_path)
    (tmp_path / 'experiments').mkdir()
    (tmp_path / 'experiments/quantization-evidence.json').write_text(json.dumps({
        'kind': 'real-run', 'source_precision': 'F16', 'created_by_feather': True,
        'source': {'path': 'missing.gguf', 'sha256': 'a' * 64}}))
    with pytest.raises(ValueError, match='Missing artifact'):
        validate.validate('task02')


def test_shipped_task_manifest_parses():
    root = Path(__file__).resolve().parents[1]
    data = load_tasks(root / 'automation/tasks.yaml')
    assert len(data['tasks']) == 9


def comparison_fixture():
    import copy
    trial = {'warmup': False, 'ttft_seconds': .1, 'prefill_tokens_per_second': 20,
             'decode_tokens_per_second': 10, 'wall_seconds': 1.0}
    config = {'source_sha256': 'a' * 64, 'artifact_sha256': 'b' * 64,
              'file_size_bytes': 100, 'peak_memory_bytes': 200,
              'ttft_seconds': .1, 'prefill_tokens_per_second': 20,
              'decode_tokens_per_second': 10, 'wall_seconds': 1.0,
              'coding_correctness': .8, 'repeats': 5, 'warmups': 1,
              'context_length': 512, 'decoding_settings': {'temperature': 0, 'seed': 42},
              'runtime_version': 'test', 'hardware': {'chip': 'M2', 'architecture': 'arm64',
                 'physical_ram_bytes': 8589934592, 'os': 'test'},
              'raw_trials': [{'warmup': True}] + [copy.deepcopy(trial) for _ in range(5)],
              'perplexity_unsupported_reason': 'Test fixture: runtime has no logits'}
    other = copy.deepcopy(config)
    other['artifact_sha256'] = 'c' * 64
    return {'kind': 'real-run', 'configurations': [config, other]}


@pytest.mark.parametrize('change', ['nan', 'quality-range', 'failed', 'settings', 'warmups', 'hardware'])
def test_comparison_gate_rejects_invalid_or_incomparable_evidence(tmp_path, monkeypatch, change):
    from automation import validate
    monkeypatch.setattr(validate, 'ROOT', tmp_path)
    (tmp_path / 'experiments').mkdir()
    data = comparison_fixture()
    config = data['configurations'][1]
    if change == 'nan':
        config['ttft_seconds'] = float('nan')
    elif change == 'quality-range':
        config['coding_correctness'] = 2
    elif change == 'failed':
        config['raw_trials'][1]['error'] = 'OOM'
    elif change == 'settings':
        config['context_length'] = 1024
    elif change == 'warmups':
        config['raw_trials'][0]['warmup'] = False
    else:
        config['hardware']['chip'] = 'different'
    (tmp_path / 'experiments/comparison-evidence.json').write_text(json.dumps(data))
    with pytest.raises(ValueError):
        validate.validate('task04')


def test_launch_failure_is_logged(tmp_path):
    log = tmp_path / 'missing.log'
    assert invoke(['/does/not/exist'], tmp_path, log, timeout=1) == 127
    assert json.loads(log.with_suffix('.log.invocation.json').read_text())['exit_code'] == 127


def test_external_blocker_preserves_attempts_and_records_committed_evidence(tmp_path):
    root = repository(tmp_path)
    path = manifest(root)
    evidence = root / 'blocker.json'
    evidence.write_text('{"docker":"daemon unavailable"}')
    commit(root)
    runner = Runner(root, path)
    runner.state['first']['attempts'] = 1
    runner.block('first', 'Docker daemon unavailable', evidence)
    entry = runner.state['first']
    assert entry['status'] == 'blocked'
    assert entry['attempts'] == 1
    assert entry['origin'] == 'externally-observed'
    assert entry['blocker_evidence']['path'] == 'blocker.json'
    assert len(entry['blocker_evidence']['sha256']) == 64
    assert Runner(root, path).state['first']['error'] == 'Docker daemon unavailable'
    runner.state['first']['status'] = 'completed'
    with pytest.raises(ValueError, match='completed'):
        runner.block('first', 'Unavailable', evidence)


def test_external_blocker_rejects_dirty_or_untracked_evidence(tmp_path):
    root = repository(tmp_path)
    path = manifest(root)
    commit(root)
    runner = Runner(root, path)
    evidence = root / 'unreviewed.txt'
    evidence.write_text('unreviewed')
    with pytest.raises(CriticalFailure, match='clean'):
        runner.block('first', 'Unavailable', evidence)
    evidence.unlink()
    ignored = root / '.feather/ignored.txt'
    ignored.parent.mkdir()
    ignored.write_text('uncommitted')
    with pytest.raises(ValueError, match='tracked'):
        runner.block('first', 'Unavailable', ignored)


def test_cumulative_attempt_deadline_prevents_next_invocation(tmp_path, monkeypatch):
    import automation.orchestrator as module
    root = repository(tmp_path)
    path = manifest(root)
    commit(root)
    clock = [100.0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    calls = []
    underlying = fake_executor()
    def execute(argv, root, log, timeout, env):
        calls.append((argv, timeout))
        result = underlying(argv, root, log, timeout, env)
        clock[0] += 6
        return result
    runner = Runner(root, path, timeout=5, retries=0, executor=execute)
    runner.run(one=True)
    assert len(calls) == 1
    assert calls[0][1] == 5
    assert runner.state['first']['status'] == 'failed'
    assert 'Cumulative task attempt timeout' in runner.state['first']['error']


def test_remaining_task_budget_clips_later_invocation_timeout(tmp_path, monkeypatch):
    import automation.orchestrator as module
    root = repository(tmp_path)
    path = manifest(root)
    commit(root)
    clock = [100.0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    calls = []
    underlying = fake_executor()
    def execute(argv, root, log, timeout, env):
        calls.append(timeout)
        result = underlying(argv, root, log, timeout, env)
        clock[0] += 1
        return result
    runner = Runner(root, path, timeout=10, retries=0, executor=execute)
    runner.run(one=True)
    assert calls == [10, 9, 8, 7, 6, 5]
    assert runner.state['first']['status'] == 'completed'
