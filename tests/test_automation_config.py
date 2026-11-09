import json
from pathlib import Path
import subprocess
import pytest
import yaml
from automation.codex_config import ConfigError, classify_failure, load_config, resolve_model, selection_path
from automation.orchestrator import CriticalFailure, NonRetryableFailure, Runner
from automation.selftest import run_selftest

# Verbatim event lines captured from codex-cli 0.142.3 on 2026-11-09 (no credentials present).
ENTITLEMENT = ('{"type":"turn.failed","error":{"message":"{\\"type\\":\\"error\\",\\"status\\":400,\\"error\\":{\\"type\\":'
               '\\"invalid_request_error\\",\\"message\\":\\"The \'gpt-6.1-sol\' model is not supported when using Codex with a '
               'ChatGPT account.\\",\\"code\\":\\"model_not_found\\"}}"}}')
OUTDATED = ('{"type":"error","message":"{\\"type\\":\\"error\\",\\"status\\":400,\\"error\\":{\\"type\\":\\"invalid_request_error\\",'
            '\\"message\\":\\"The \'gpt-5.6-luna\' model requires a newer version of Codex. Please upgrade to the latest app or CLI and try again.\\"}}"}')
METADATA_WARNING = ('{"type":"item.completed","item":{"id":"item_0","type":"error","message":"Model metadata for `x` not found. '
                    'Defaulting to fallback metadata; this can degrade performance and cause issues."}}')


def test_classifies_captured_codex_rejections():
    assert classify_failure(ENTITLEMENT)['kind'] == 'model_entitlement'
    assert classify_failure(OUTDATED)['kind'] == 'cli_outdated'
    assert classify_failure('Error: Not logged in. Please log in.')['kind'] == 'authentication'
    assert classify_failure(METADATA_WARNING) is None
    # Agent prose mentioning authentication is not an error event.
    prose = json.dumps({'type': 'item.completed', 'item': {'type': 'agent_message', 'text': '401 unauthorized model_not_found'}})
    assert classify_failure(prose) is None
    assert classify_failure('{"type":"turn.failed","error":{"message":"stream disconnected"}}') is None


def write_config(root, model=None, **extra):
    (root / 'automation').mkdir(exist_ok=True)
    (root / 'automation' / 'config.yaml').write_text(yaml.safe_dump({'version': 1, 'codex': {'model': model}, **extra}))


def test_model_resolution_precedence(tmp_path):
    write_config(tmp_path)
    assert resolve_model(tmp_path, {})['model'] is None
    selection_path(tmp_path).parent.mkdir(parents=True)
    selection_path(tmp_path).write_text(json.dumps({'status': 'accepted', 'model': 'probed-model'}))
    assert resolve_model(tmp_path, {})['model'] == 'probed-model'
    selection_path(tmp_path).write_text(json.dumps({'status': 'rejected', 'model': 'probed-model'}))
    assert resolve_model(tmp_path, {})['model'] is None
    write_config(tmp_path, 'config-model')
    assert resolve_model(tmp_path, {})['source'] == 'automation/config.yaml'
    assert resolve_model(tmp_path, {'FEATHER_CODEX_MODEL': 'env-model'})['model'] == 'env-model'
    with pytest.raises(ConfigError):
        resolve_model(tmp_path, {'FEATHER_CODEX_MODEL': '--dangerously-bypass'})


@pytest.mark.parametrize('extra', [{'task_timeout_seconds': 5}, {'max_attempts': 9}, {'codex': {'model': None, 'sandbox': 'x'}}])
def test_config_validation(tmp_path, extra):
    write_config(tmp_path, **({} if 'codex' in extra else extra))
    if 'codex' in extra:
        (tmp_path / 'automation' / 'config.yaml').write_text(yaml.safe_dump({'version': 1, **extra}))
    with pytest.raises(ConfigError):
        load_config(tmp_path)


def test_shipped_config_is_valid():
    config = load_config(Path(__file__).resolve().parents[1])
    assert config['model'] is None and config['max_attempts'] <= 3


def repository(tmp_path):
    for command in (['git', 'init', '-q'], ['git', 'config', 'user.name', 'T'], ['git', 'config', 'user.email', 't@example.invalid']):
        subprocess.run(command, cwd=tmp_path, check=True)
    (tmp_path / '.gitignore').write_text('.feather/\n')
    (tmp_path / 'tasks.yaml').write_text(yaml.safe_dump({'version': 1, 'regression': [['true']], 'tasks': [
        {'id': 'one', 'depends_on': [], 'prompt': 'p', 'acceptance': [['true']]},
        {'id': 'two', 'depends_on': ['one'], 'prompt': 'p', 'acceptance': [['true']]}]}))
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    subprocess.run(['git', 'commit', '-qm', 'init'], cwd=tmp_path, check=True)
    return tmp_path


def test_model_rejection_blocks_without_retry_and_survives_restart(tmp_path):
    root = repository(tmp_path)
    calls = []

    def executor(argv, cwd, log, timeout, env):
        calls.append(argv)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(ENTITLEMENT + '\n')
        return 1

    runner = Runner(root, root / 'tasks.yaml', retries=2, executor=executor, model='gpt-6.1-sol')
    with pytest.raises(NonRetryableFailure):
        runner.run()
    assert len(calls) == 1, 'entitlement failures must never be retried'
    assert calls[0][calls[0].index('-m') + 1] == 'gpt-6.1-sol'
    entry = runner.state['one']
    assert entry['status'] == 'blocked' and entry['failure_kind'] == 'model_entitlement' and entry['attempts'] == 1
    assert entry['invocations'][0]['exit_code'] == 1 and entry['invocations'][0]['failure_kind'] == 'model_entitlement'
    assert runner.state['two']['status'] == 'pending'
    restarted = Runner(root, root / 'tasks.yaml', retries=2, executor=executor)
    assert restarted.state['one']['status'] == 'blocked'
    assert restarted.state_path.name == 'state.json'  # historical name kept for tasks.yaml
    restarted.run()
    assert len(calls) == 1, 'a blocked predecessor must not let its dependent start'


def test_generic_failure_still_retries_within_budget(tmp_path):
    root = repository(tmp_path)
    calls = []

    def executor(argv, cwd, log, timeout, env):
        calls.append(argv)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text('{"type":"turn.failed","error":{"message":"stream disconnected"}}\n')
        return 1

    runner = Runner(root, root / 'tasks.yaml', retries=1, executor=executor)
    runner.run(one=True)
    assert len(calls) == 2 and '-m' not in calls[0]
    assert runner.state['one']['status'] == 'failed' and runner.state['two']['status'] == 'pending'


def selftest_executor(fail_alpha=False):
    order = []

    def run(argv, root, log, timeout, env):
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text('')
        if argv[0] != 'codex':
            return subprocess.run(argv, cwd=root).returncode
        final = Path(argv[argv.index('--output-last-message') + 1])
        prompt = argv[-1]
        if argv[argv.index('--sandbox') + 1] == 'read-only':
            final.write_text(json.dumps({'approved': True, 'findings': [], 'summary': 'ok'}))
            return 0
        (root / 'selftest').mkdir(exist_ok=True)
        if 'selftest-alpha' in prompt:
            order.append('alpha')
            (root / 'selftest' / 'alpha.txt').write_text('wrong\n' if fail_alpha else 'alpha\n')
        else:
            order.append('beta')
            text = (root / 'selftest' / 'alpha.txt').read_text().strip()[::-1]
            (root / 'selftest' / 'beta.txt').write_text(text + '\n')
        final.write_text(json.dumps({'status': 'completed', 'summary': 'done', 'blocker': ''}))
        return 0
    return run, order


def test_selftest_dependent_task_starts_only_after_predecessor_passes(tmp_path):
    executor, order = selftest_executor()
    result = run_selftest(None, executor=executor, keep=tmp_path)
    assert result['status'] == 'passed' and result['dependent_advanced']
    assert order == ['alpha', 'beta']
    assert result['commits'][:2] == ['Complete selftest-beta: Write beta', 'Complete selftest-alpha: Write alpha']


def test_selftest_failed_predecessor_is_not_completed_and_blocks_dependent(tmp_path):
    executor, order = selftest_executor(fail_alpha=True)
    result = run_selftest(None, executor=executor, keep=tmp_path)
    assert result['status'] == 'failed'
    assert result['tasks']['selftest-alpha']['status'] == 'failed'
    assert result['tasks']['selftest-beta']['status'] == 'pending'
    assert order == ['alpha']


def test_task04_gate_requires_reproducible_cited_study(tmp_path, monkeypatch):
    import automation.validate as validate
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / 'experiments' / 'comparison-evidence-v0.2.json').read_text())
    validate.validate('task04')  # real evidence passes
    for mutate in (lambda d: d.pop('study'), lambda d: d.update(study='../../etc'),
                   lambda d: d['configurations'][0].update(coding_status='partial'),
                   lambda d: d['configurations'][1].update(decode_tokens_per_second=d['configurations'][1]['decode_tokens_per_second'] * 1.01)):
        candidate = json.loads(json.dumps(data))
        mutate(candidate)
        monkeypatch.setattr(validate, 'evidence', lambda name, c=candidate: c)
        with pytest.raises(ValueError):
            validate.validate('task04')
