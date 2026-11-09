import numpy as np
import pytest
import httpx
from backend.quantization.uniform import quantize
from backend.storage.database import Database
from backend.benchmarks.runner import summarize
from backend.backends.ollama import Ollama


def test_quantization_bound_and_zero_channels():
    weights = np.array([[0, 0, 0], [-1, .3, 1]], dtype=np.float32)
    q, scale, metrics = quantize(weights, 4, axis=0)
    assert np.isfinite(scale).all()
    assert np.max(np.abs(weights - q * scale)) <= np.max(scale) / 2 + 1e-7
    assert metrics['mse'] > 0
    assert np.all(q[0] == 0)
    assert quantize(weights, 8)[2]['mse'] < metrics['mse']


@pytest.mark.parametrize('weights,bits', [([float('nan')], 4), ([], 4), ([1], 1)])
def test_invalid_weights(weights, bits):
    with pytest.raises(ValueError):
        quantize(weights, bits)


def test_summary_excludes_warmups_and_failures():
    trials = [{'warmup': True, 'wall_seconds': 100}, {'warmup': False, 'error': 'OOM'},
              {'warmup': False, 'wall_seconds': 1}, {'warmup': False, 'wall_seconds': 3}]
    result = summarize(trials)
    assert result['successful_trials'] == 2
    assert result['failed_trials'] == 1
    assert result['wall_seconds']['median'] == 2
    assert result['wall_seconds']['p95'] == pytest.approx(2.9)
    assert result['ttft_seconds'] is None


def test_trials_persist_individually(tmp_path):
    path = tmp_path / 'db.sqlite'
    db = Database(path)
    db.start('run', {'seed': 42})
    db.save('run', 0, {'error': 'failure'})
    reopened = Database(path)
    assert reopened.report()['trials'] == [{'error': 'failure'}]
    with pytest.raises(ValueError):
        reopened.report('missing')


def test_stream_records_output_and_raw_timings():
    body = '\n'.join(['{"response":"hello","done":false}',
                      '{"response":"","done":true,"eval_count":1,"eval_duration":1000000}'])
    backend = Ollama(transport=httpx.MockTransport(lambda request: httpx.Response(200, text=body)))
    result = backend.generate('model', 'prompt', {})
    assert result['output'] == 'hello'
    assert result['ttft_seconds'] >= 0
    assert result['runtime']['eval_count'] == 1
    backend.close()


def test_incomplete_stream_is_failure():
    backend = Ollama(transport=httpx.MockTransport(lambda request: httpx.Response(200, text='{"response":"partial"}')))
    with pytest.raises(RuntimeError, match='final statistics'):
        backend.generate('model', 'prompt', {})
    backend.close()


def test_runner_preserves_failure_and_continues(tmp_path):
    from backend.benchmarks.runner import run
    class Backend:
        calls = 0
        def models(self):
            return [{'name': 'test', 'digest': 'abc'}]
        def version(self):
            return 'test-version'
        def generate(self, model, prompt, options):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError('out of memory')
            return {'wall_seconds': .5, 'ttft_seconds': .1, 'runtime': {'eval_count': 10, 'eval_duration': 1000000000}}
    db = Database(tmp_path / 'runs.sqlite')
    report = run(Backend(), db, 'test', repeats=3, warmups=1)
    assert len(db.report()['trials']) == 4
    assert report['summary']['failed_trials'] == 1
    assert report['summary']['successful_trials'] == 2
    assert report['summary']['decode_tokens_per_second']['median'] == 10
    assert report['summary']['prefill_tokens_per_second'] is None
    assert len(report['metadata']['prompt_sha256']) == 64
