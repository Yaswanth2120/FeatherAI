import json
from pathlib import Path
import subprocess
import time
import pytest
from fastapi.testclient import TestClient
from backend.api.app import create_app
from backend.api.jobs import Jobs
from backend.storage.database import Database


def wait_terminal(client, identifier):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = client.get(f'/api/jobs/{identifier}').json()
        if job['status'] in {'completed', 'failed', 'cancelled', 'interrupted'}:
            return job
        time.sleep(.05)
    raise AssertionError('Job did not finish')


def test_local_api_validation_and_origin(tmp_path):
    app = create_app(tmp_path / 'data', tmp_path / 'models')
    with TestClient(app) as client:
        assert client.get('/api/health').json()['local_only']
        assert client.get('/api/hardware').json()['physical_ram_bytes'] > 0
        assert client.get('/api/hardware', headers={'host': 'evil.example'}).status_code == 403
        assert client.post('/api/jobs/benchmark', json={'model': 'test'}, headers={'origin': 'https://evil.example'}).status_code == 403
        assert client.post('/api/jobs/benchmark', json={'model': 'x', 'repeats': 100}).status_code == 422
        assert client.post('/api/jobs/benchmark', json={'model': 'x', 'executable': '/bin/sh'}).status_code == 422
        assert client.get('/api/jobs/missing').status_code == 404
        assert client.post('/api/jobs/quantization', json={'source': '/etc/passwd'}).status_code == 422


def test_model_discovery_failure_is_explicit(tmp_path):
    with TestClient(create_app(tmp_path, ollama_url='http://127.0.0.1:1')) as client:
        response = client.get('/api/models')
        assert response.status_code == 503
        assert 'unavailable' in response.json()['detail']


def test_background_failure_persists_across_restart(tmp_path):
    with TestClient(create_app(tmp_path, ollama_url='http://127.0.0.1:1')) as client:
        submitted = client.post('/api/jobs/benchmark', json={'model': 'missing'}).json()
        job = wait_terminal(client, submitted['id'])
        assert job['status'] == 'failed'
        assert job['error']
        assert (tmp_path / 'jobs' / job['id'] / 'worker.log').exists()
    with TestClient(create_app(tmp_path)) as client:
        assert client.get(f'/api/jobs/{job["id"]}').json()['status'] == 'failed'


def test_report_export_and_comparison_uses_saved_data(tmp_path):
    app = create_app(tmp_path)
    db = Database(tmp_path / 'experiments.sqlite3')
    db.start('sample', {'model': {'name': 'test', 'size': 100}, 'prompt_sha256': 'prompt', 'options': {'seed': 42}})
    db.save('sample', 0, {'ordinal': 0, 'warmup': False, 'wall_seconds': 1.5})
    db.connection.close()
    with TestClient(app) as client:
        assert client.get('/api/experiments').json()[0]['summary']['wall_seconds']['median'] == 1.5
        assert client.get('/api/experiments/sample/export?format=json').json()['id'] == 'sample'
        assert 'wall_seconds' in client.get('/api/experiments/sample/export?format=csv').text
        comparison = client.post('/api/comparisons', json={'experiment_ids': ['sample']}).json()
        assert comparison['comparable'] is False
        assert comparison['limitations']
        assert client.get('/api/experiments/missing').status_code == 404


def test_queued_cancellation_and_interrupted_recovery(tmp_path):
    jobs = Jobs(tmp_path, {})
    queued = jobs.add('invalid', {})
    assert jobs.cancel(queued['id'])['status'] == 'cancelled'
    running = jobs.add('invalid', {})
    jobs.update(running['id'], 'running', pid=999999)
    jobs.start()
    try:
        assert jobs.get(running['id'])['status'] == 'interrupted'
    finally:
        jobs.stop()


def test_single_worker_lock(tmp_path):
    first, second = Jobs(tmp_path, {}), Jobs(tmp_path, {})
    first.start()
    try:
        with pytest.raises(RuntimeError, match='Another'):
            second.start()
    finally:
        first.stop()


def test_process_group_cancellation(tmp_path):
    process = subprocess.Popen(['sleep', '30'], start_new_session=True)
    try:
        Jobs.kill(process)
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_quantization_source_symlink_escape(tmp_path):
    models = tmp_path / 'models'
    models.mkdir()
    outside = tmp_path / 'outside.gguf'
    outside.write_bytes(b'not a model')
    (models / 'escape.gguf').symlink_to(outside)
    with TestClient(create_app(tmp_path / 'data', models)) as client:
        assert client.post('/api/jobs/quantization', json={'source': 'escape.gguf'}).status_code == 422


def test_running_job_cancellation_stops_process(tmp_path, monkeypatch):
    original_popen = subprocess.Popen
    def sleeping_worker(command, **kwargs):
        return original_popen(['sleep', '30'], **kwargs)
    monkeypatch.setattr('backend.api.jobs.subprocess.Popen', sleeping_worker)
    jobs = Jobs(tmp_path, {'job_timeout': 10})
    jobs.start()
    try:
        job = jobs.add('benchmark', {})
        deadline = time.monotonic() + 5
        while jobs.get(job['id'])['pid'] is None and time.monotonic() < deadline:
            time.sleep(.02)
        assert jobs.get(job['id'])['status'] == 'running'
        jobs.cancel(job['id'])
        while jobs.get(job['id'])['status'] == 'running' and time.monotonic() < deadline:
            time.sleep(.02)
        assert jobs.get(job['id'])['status'] == 'cancelled'
    finally:
        jobs.stop()


def test_worker_timeout_is_persisted(tmp_path, monkeypatch):
    original_popen = subprocess.Popen
    def sleeping_worker(command, **kwargs):
        return original_popen(['sleep', '30'], **kwargs)
    monkeypatch.setattr('backend.api.jobs.subprocess.Popen', sleeping_worker)
    jobs = Jobs(tmp_path, {'job_timeout': .1})
    jobs.start()
    try:
        job = jobs.add('benchmark', {})
        deadline = time.monotonic() + 5
        while jobs.get(job['id'])['status'] not in {'failed', 'completed'} and time.monotonic() < deadline:
            time.sleep(.02)
        result = jobs.get(job['id'])
        assert result['status'] == 'failed'
        assert result['error'] == 'Job timeout'
    finally:
        jobs.stop()


def test_benchmark_worker_end_to_end_with_fixture_runtime(tmp_path):
    """A fixture HTTP runtime exercises the real process/SQLite/report pipeline."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    class FixtureRuntime(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            result = {'version': 'fixture'} if self.path == '/api/version' else {'models': [{'name': 'fixture-model', 'digest': 'fixture-digest'}]}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200)
            self.end_headers()
            rows = [{'response': 'fixture-output', 'done': False}, {'response': '', 'done': True,
                    'eval_count': 4, 'eval_duration': 1000000000,
                    'prompt_eval_count': 5, 'prompt_eval_duration': 1000000000}]
            self.wfile.write(('\n'.join(json.dumps(row) for row in rows) + '\n').encode())
    server = ThreadingHTTPServer(('127.0.0.1', 0), FixtureRuntime)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with TestClient(create_app(tmp_path, ollama_url=f'http://127.0.0.1:{server.server_port}')) as client:
            job = client.post('/api/jobs/benchmark', json={'model': 'fixture-model', 'repeats': 2, 'warmups': 1}).json()
            result = wait_terminal(client, job['id'])
            assert result['status'] == 'completed', result
            assert result['result']['summary']['successful_trials'] == 2
            reports = client.get('/api/experiments').json()
            assert len(reports) == 1
            assert len(reports[0]['trials']) == 3
            assert reports[0]['summary']['decode_tokens_per_second']['median'] == 4
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_partial_and_unsupported_job_statuses_preserve_evidence():
    assert Jobs.result_status({'kind': 'benchmark'}, {'summary': {'successful_trials': 1, 'failed_trials': 1}}) == 'partial'
    assert Jobs.result_status({'kind': 'benchmark'}, {'summary': {'successful_trials': 0, 'failed_trials': 1}}) == 'failed'
    assert Jobs.result_status({'kind': 'evaluation'}, {'status': 'unsupported'}) == 'unsupported'
    assert Jobs.result_status({'kind': 'evaluation'}, {'status': 'partial'}) == 'partial'


def test_chunked_body_limit_does_not_trust_content_length():
    import asyncio
    from backend.api.limits import BodyLimitMiddleware
    async def scenario():
        called = False
        async def forbidden(scope, receive, send):
            nonlocal called
            called = True
        middleware = BodyLimitMiddleware(forbidden, limit=10)
        incoming = iter([{'type': 'http.request', 'body': b'123456', 'more_body': True},
                         {'type': 'http.request', 'body': b'789012', 'more_body': False}])
        sent = []
        async def receive():
            return next(incoming)
        async def send(message):
            sent.append(message)
        await middleware({'type': 'http', 'headers': [(b'content-length', b'1')]}, receive, send)
        assert not called
        assert sent[0]['status'] == 413
    asyncio.run(scenario())


def test_comparison_requires_matching_runtime_and_distinct_artifacts(tmp_path):
    app = create_app(tmp_path)
    db = Database(tmp_path / 'experiments.sqlite3')
    metadata = {'source_sha256': 'source', 'prompt_sha256': 'prompt', 'options': {'seed': 42},
                'runtime_version': '1', 'backend': 'ollama', 'repeats': 1, 'warmups': 1,
                'settings_sha256': 'fixture-settings-identity', 'backend_settings': {},
                'hardware': {'chip': 'M2', 'architecture': 'arm64', 'os': 'test-os', 'physical_ram_bytes': 8*1024**3}}
    for identifier, version, digest in [('a', '1', 'artifact-a'), ('b', '1', 'artifact-b'), ('c', '2', 'artifact-c')]:
        db.start(identifier, {**metadata, 'runtime_version': version, 'model': {'digest': digest}})
        db.save(identifier, 0, {'warmup': True, 'wall_seconds': 1})
        db.save(identifier, 1, {'warmup': False, 'wall_seconds': 1})
    db.connection.close()
    with TestClient(app) as client:
        assert client.post('/api/comparisons', json={'experiment_ids': ['a', 'b']}).json()['comparable'] is True
        assert client.post('/api/comparisons', json={'experiment_ids': ['a', 'c']}).json()['comparable'] is False
        assert client.post('/api/comparisons', json={'experiment_ids': ['a', 'a']}).json()['comparable'] is False


def test_reasoning_cleanup_does_not_need_docker():
    assert Jobs.cleanup({'kind': 'evaluation', 'payload': {'suite': 'reasoning-small'}}) is None
    assert Jobs.cleanup({'kind': 'evaluation', 'payload': {'suite': 'coding-small'}, 'result': {'status': 'unsupported'}}) is None


def test_cleanup_failure_blocks_startup_instead_of_running_more_code(tmp_path, monkeypatch):
    jobs = Jobs(tmp_path, {})
    job = jobs.add('evaluation', {'suite': 'coding-small'})
    jobs.update(job['id'], 'running')
    monkeypatch.setattr(Jobs, 'cleanup', staticmethod(lambda job: 'Container cleanup not confirmed'))
    with pytest.raises(RuntimeError, match='cleanup'):
        jobs.start()
    assert jobs.get(job['id'])['status'] == 'cleanup_required'
    assert jobs.lock_file is None


def test_evaluation_store_history_and_recommendation_missing_evidence(tmp_path):
    from backend.storage.evaluations import EvaluationStore
    app = create_app(tmp_path)
    EvaluationStore(tmp_path / 'experiments.sqlite3').save({'id': 'evaluation', 'created_at': '2026-11-09', 'status': 'unsupported'})
    db = Database(tmp_path / 'experiments.sqlite3')
    db.start('performance', {'model': {'name': 'fixture'}, 'repeats': 1, 'warmups': 0})
    db.save('performance', 0, {'warmup': False, 'wall_seconds': 1})
    db.connection.close()
    with TestClient(app) as client:
        assert client.get('/api/evaluations').json()[0]['id'] == 'evaluation'
        response = client.post('/api/recommendations', json={'experiment_ids': ['performance'],
                    'available_memory_bytes': 1000000, 'max_memory_bytes': 1000000})
        assert response.status_code == 200
        assert response.json()['status'] == 'insufficient_evidence'


def test_native_job_requires_completed_managed_artifact(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        assert client.post('/api/jobs/benchmark', json={'model': 'native', 'backend': 'llama.cpp'}).status_code == 422
        assert client.post('/api/jobs/benchmark', json={'model': 'native', 'server_executable': '/bin/sh'}).status_code == 422
        assert client.post('/api/jobs/benchmark', json={'model': 'native', 'backend': 'llama.cpp', 'artifact_job_id': '00000000-0000-0000-0000-000000000000'}).status_code == 404


def test_native_worker_verifies_artifact_identity_before_dispatch(tmp_path, monkeypatch):
    from backend.api.worker import execute
    import backend.backends.llama_cpp as native
    import backend.benchmarks.runner as runner
    import backend.core.artifacts as artifacts
    folder = tmp_path / 'artifacts'
    folder.mkdir()
    artifact = folder / 'fixture.gguf'
    artifact.write_bytes(b'fixture-only')
    manager = Jobs(tmp_path, {})
    job = manager.add('quantization', {})
    manager.update(job['id'], 'completed', result={'artifact': {'path': str(artifact), 'sha256': 'expected'}, 'source': {'sha256': 'source'}})
    events = []
    class FixtureBackend:
        def __init__(self, path, **kwargs):
            events.append(kwargs)
        def models(self):
            return [{'name': 'fixture.gguf'}]
        def close(self):
            events.append('closed')
    monkeypatch.setattr(native, 'LlamaCpp', FixtureBackend)
    monkeypatch.setattr(artifacts, 'inspect_gguf', lambda path: {'sha256': 'expected'})
    monkeypatch.setattr(runner, 'run', lambda backend, db, **payload: {'model': payload['model']})
    request = {'id': 'worker', 'kind': 'benchmark', 'root': str(tmp_path),
               'payload': {'model': 'untrusted-display-name', 'backend': 'llama.cpp', 'artifact_job_id': job['id'], 'context': 1024},
               'config': {'server_executable': 'trusted-server'}}
    assert execute(request)['model'] == 'fixture.gguf'
    assert events[0]['executable'] == 'trusted-server'
    assert events[-1] == 'closed'
    monkeypatch.setattr(artifacts, 'inspect_gguf', lambda path: {'sha256': 'changed'})
    with pytest.raises(ValueError, match='changed'):
        execute(request)


def test_comparison_rejects_missing_warmup_rows(tmp_path):
    app = create_app(tmp_path)
    db = Database(tmp_path / 'experiments.sqlite3')
    metadata = {'source_sha256': 'source', 'prompt_sha256': 'prompt', 'options': {'seed': 42},
                'runtime_version': '1', 'backend': 'ollama', 'repeats': 1, 'warmups': 1,
                'settings_sha256': 'fixture-settings-identity', 'backend_settings': {},
                'hardware': {'chip': 'M2', 'architecture': 'arm64', 'os': 'test', 'physical_ram_bytes': 8*1024**3}}
    for identifier in ['a', 'b']:
        db.start(identifier, {**metadata, 'model': {'digest': identifier}})
        db.save(identifier, 0, {'warmup': False, 'wall_seconds': 1})
    db.connection.close()
    with TestClient(app) as client:
        assert client.post('/api/comparisons', json={'experiment_ids': ['a', 'b']}).json()['comparable'] is False


@pytest.mark.parametrize('changed', [{'template': 'changed'}, {'threads': 4}, {'executable_sha256': 'other-binary'}])
def test_comparison_rejects_changed_native_backend_settings(tmp_path, changed):
    app = create_app(tmp_path)
    db = Database(tmp_path / 'experiments.sqlite3')
    settings = {'template': 'chatml-v1', 'threads': 2, 'executable_sha256': 'fixture-binary'}
    metadata = {'source_sha256': 'source', 'prompt_sha256': 'prompt', 'options': {'seed': 42},
                'runtime_version': '1', 'backend': 'llama.cpp', 'repeats': 1, 'warmups': 0,
                'settings_sha256': 'same-claimed-hash', 'backend_settings': settings,
                'hardware': {'chip': 'M2', 'architecture': 'arm64', 'os': 'test', 'physical_ram_bytes': 8*1024**3}}
    for identifier in ['a', 'b']:
        data = {**metadata, 'model': {'artifact_sha256': identifier, 'digest': identifier}}
        if identifier == 'b':
            data['backend_settings'] = {**settings, **changed}
        db.start(identifier, data)
        db.save(identifier, 0, {'warmup': False, 'wall_seconds': 1})
    db.connection.close()
    with TestClient(app) as client:
        assert client.post('/api/comparisons', json={'experiment_ids': ['a', 'b']}).json()['comparable'] is False


def test_platform_status_never_reports_unavailable_as_ready(tmp_path, monkeypatch):
    from backend.evaluation.sandbox import UnavailableExecutionBackend
    import automation.doctor as doctor
    monkeypatch.setattr('backend.evaluation.sandbox.discover', lambda *a, **k: UnavailableExecutionBackend('Docker daemon is not reachable'))
    monkeypatch.setattr(doctor, 'run_checks', lambda root: {'status': 'blocked', 'note': None, 'checks': [
        {'name': 'live_model_probe', 'status': 'fail', 'detail': 'no accepted model', 'action': 'upgrade'}]})
    with TestClient(create_app(tmp_path / 'data', tmp_path / 'models')) as client:
        status = client.get('/api/status').json()
    assert status['sandbox']['status'] == 'blocked' and 'not reachable' in status['sandbox']['reason']
    assert status['automation']['status'] == 'blocked'
    assert status['memory']['physical_ram_bytes'] > 0


def test_study_summary_marks_blocked_and_missing_phases(tmp_path):
    from backend.study.summary import summarize_study
    folder = tmp_path / 'qwen'
    folder.mkdir()
    (folder / 'study.json').write_text(json.dumps({
        'study': 'quantization-study-v0.2', 'source': {'sha256': 'a' * 64, 'size_bytes': 10},
        'phases': {'convert': {'status': 'blocked', 'reason': 'Insufficient available RAM', 'jobs': []},
                   'quality': {'status': 'completed', 'runs': {'coding-v1-F16': {'status': 'blocked', 'summary': {'accuracy': None}}},
                               'comparisons': {}}}}))
    summary = summarize_study(folder / 'study.json')
    assert summary['complete'] is False
    assert summary['phases']['convert'] == 'blocked' and summary['phases']['benchmark'] == 'not_run'
    assert summary['variants']['F16']['quality']['coding-v1']['status'] == 'blocked'
    assert list(summary['variants']) == ['F16'], 'no variant is invented for a blocked conversion'


def test_studies_endpoint_isolates_malformed_study(tmp_path):
    studies = tmp_path / 'studies'
    (studies / 'broken').mkdir(parents=True)
    (studies / 'broken' / 'study.json').write_text('{not json')
    (studies / 'good').mkdir()
    (studies / 'good' / 'study.json').write_text(json.dumps({'source': {'sha256': 'a' * 64}, 'phases': {}}))
    with TestClient(create_app(tmp_path / 'data', tmp_path / 'models', studies_dir=studies)) as client:
        response = client.get('/api/studies')
    assert response.status_code == 200
    by_id = {item['id']: item for item in response.json()}
    assert by_id['broken']['invalid'] is True and by_id['broken']['complete'] is False
    assert by_id['good']['complete'] is False and by_id['good']['phases']['benchmark'] == 'not_run'
