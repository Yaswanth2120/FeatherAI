"""FastAPI application. Use one localhost server process per data directory."""
from __future__ import annotations
from contextlib import asynccontextmanager
import csv
import io
import json
from pathlib import Path
import sqlite3
import sys
import time
from typing import Literal
from urllib.parse import urlsplit
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
import httpx
from pydantic import BaseModel, ConfigDict, Field
from backend.api.jobs import Jobs
from backend.api.limits import BodyLimitMiddleware
from fastapi.staticfiles import StaticFiles
from backend.backends.ollama import Ollama
from backend.benchmarks.runner import summarize
from backend.core.hardware import inspect
from backend.storage.database import Database

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # automation package lives at the repository root


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class BenchmarkRequest(StrictModel):
    backend: Literal['ollama', 'llama.cpp'] = 'ollama'
    artifact_job_id: str | None = Field(default=None, pattern=r'^[0-9a-fA-F-]{36}$')
    model: str = Field(min_length=1, max_length=200)
    repeats: int = Field(default=5, ge=1, le=20)
    warmups: int = Field(default=1, ge=0, le=3)
    context: int = Field(default=1024, ge=128, le=4096)
    tokens: int = Field(default=128, ge=1, le=1024)
    seed: int = Field(default=42, ge=0, le=2**31-1)


class EvaluationRequest(StrictModel):
    backend: Literal['ollama', 'llama.cpp'] = 'ollama'
    artifact_job_id: str | None = Field(default=None, pattern=r'^[0-9a-fA-F-]{36}$')
    model: str = Field(min_length=1, max_length=200)
    suite: Literal['reasoning-small', 'coding-small', 'reasoning-v1', 'coding-v1'] = 'reasoning-small'
    repeats: int = Field(default=1, ge=1, le=5)
    context: int = Field(default=1024, ge=128, le=4096)
    tokens: int = Field(default=128, ge=1, le=1024)
    seed: int = Field(default=42, ge=0, le=2**31-1)


class QuantizationRequest(StrictModel):
    source: str = Field(min_length=1, max_length=4096)
    target: Literal['Q8_0', 'Q6_K', 'Q5_K_M', 'Q4_K_M', 'Q3_K_M'] = 'Q4_K_M'


class ComparisonRequest(StrictModel):
    experiment_ids: list[str] = Field(min_length=1, max_length=10)


class RecommendationRequest(ComparisonRequest):
    available_memory_bytes: int = Field(gt=0)
    max_memory_bytes: int = Field(gt=0)
    min_quality: float = Field(default=0, ge=0, le=1, allow_inf_nan=False)
    max_latency_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    objective: Literal['fastest', 'smallest', 'lowest_latency'] = 'fastest'


def create_app(data_dir: str | Path | None = None, model_root: str | Path | None = None,
               ollama_url='http://127.0.0.1:11434', quantizer='llama-quantize', job_timeout=1800, server_executable='llama-server',
               studies_dir: str | Path | None = None):
    root = Path(data_dir or '.feather').resolve()
    repository_root = Path(__file__).resolve().parents[2]
    models = Path(model_root or 'models').resolve()
    config = {'model_root': str(models), 'ollama_url': ollama_url,
              'quantizer': quantizer, 'job_timeout': job_timeout, 'server_executable': server_executable}
    jobs = Jobs(root, config)
    # Initialize the benchmark database once, closing the setup connection.
    Database(root / 'experiments.sqlite3').connection.close()

    @asynccontextmanager
    async def lifespan(app):
        jobs.start()
        try:
            yield
        finally:
            jobs.stop()

    app = FastAPI(title='Feather AI Local Lab', version='0.2.0', lifespan=lifespan)
    app.state.jobs = jobs
    app.add_middleware(BodyLimitMiddleware)

    @app.middleware('http')
    async def local_only(request: Request, call_next):
        hostname = request.url.hostname
        if hostname not in {'localhost', '127.0.0.1', '::1', 'testserver'}:
            return JSONResponse({'detail': 'Local host required'}, status_code=403)
        origin = request.headers.get('origin')
        # Vite uses a same-origin proxy. Cross-origin browser access is never enabled.
        if origin and origin != f'{request.url.scheme}://{request.url.netloc}':
            return JSONResponse({'detail': 'Cross-origin requests are not allowed'}, status_code=403)
        if request.method in {'POST', 'PUT', 'PATCH'}:
            size = request.headers.get('content-length')
            if size and (not size.isdigit() or int(size) > 65536):
                return JSONResponse({'detail': 'Request too large'}, status_code=413)
        return await call_next(request)

    @app.exception_handler(KeyError)
    async def not_found(request, error):
        return JSONResponse({'detail': str(error)}, status_code=404)

    def report(identifier):
        db = Database(root / 'experiments.sqlite3')
        try:
            result = db.report(identifier)
        except ValueError as error:
            raise HTTPException(404, str(error)) from error
        finally:
            db.connection.close()
        result['summary'] = summarize(result['trials'])
        return result

    @app.get('/api/health')
    def health():
        return {'status': 'ok', 'local_only': True}

    @app.get('/api/hardware')
    def hardware():
        return inspect()

    status_cache = {}

    @app.get('/api/status')
    def platform_status():
        """Automation health, sandbox isolation and memory state. Unavailable is never reported as ready."""
        from backend.diagnostics.memory import system_snapshot
        from backend.evaluation.sandbox import discover
        now = time.monotonic()
        if 'automation' not in status_cache or now - status_cache['automation'][0] > 60:
            try:
                from automation.doctor import run_checks
                result = run_checks(repository_root)
                automation_status = {'status': result['status'], 'note': result.get('note'),
                                     'checks': [{k: c.get(k) for k in ('name', 'status', 'detail', 'action')} for c in result['checks']]}
            except Exception as error:
                automation_status = {'status': 'unknown', 'note': f'{type(error).__name__}: {error}'[:300], 'checks': []}
            status_cache['automation'] = (now, automation_status)
        memory = system_snapshot()
        return {'automation': status_cache['automation'][1], 'sandbox': discover().status(),
                'memory': {k: memory[k] for k in ('physical_ram_bytes', 'available_bytes', 'swap', 'pressure_level')}}

    @app.get('/api/studies')
    def studies():
        from backend.study.summary import summarize_study
        folder = Path(studies_dir) if studies_dir else repository_root / 'experiments' / 'studies'
        results = []
        for path in sorted(folder.glob('*/study.json')) if folder.is_dir() else []:
            try:
                results.append(summarize_study(path))
            except (OSError, ValueError, KeyError, TypeError) as error:
                results.append({'id': path.parent.name, 'complete': False, 'invalid': True,
                                'error': f'{type(error).__name__}: {error}'[:300], 'phases': {}, 'variants': {}})
        return results

    @app.get('/api/models')
    def installed_models():
        backend = Ollama(ollama_url)
        try:
            return backend.models()
        except (httpx.HTTPError, ValueError) as error:
            raise HTTPException(503, f'Ollama unavailable: {error}') from error
        finally:
            backend.close()

    @app.get('/api/jobs')
    def job_list():
        return jobs.list()

    @app.get('/api/jobs/{identifier}')
    def job_detail(identifier: str):
        return jobs.get(identifier)

    @app.post('/api/jobs/{identifier}/cancel')
    def cancel(identifier: str):
        return jobs.cancel(identifier)

    def submit(kind, payload):
        try:
            return jobs.add(kind, payload)
        except ValueError as error:
            raise HTTPException(429, str(error)) from error

    def inference_payload(body):
        payload = body.model_dump()
        if payload['backend'] == 'llama.cpp':
            if not payload['artifact_job_id']:
                raise HTTPException(422, 'Managed llama.cpp requires a completed artifact_job_id')
            job = jobs.get(payload['artifact_job_id'])
            if job['kind'] != 'quantization' or job['status'] != 'completed' or not job['result']:
                raise HTTPException(422, 'Artifact job must be a completed quantization')
        elif payload['artifact_job_id'] is not None:
            raise HTTPException(422, 'artifact_job_id only applies to llama.cpp')
        return payload

    @app.post('/api/jobs/benchmark', status_code=202)
    def benchmark(body: BenchmarkRequest):
        return submit('benchmark', inference_payload(body))

    @app.post('/api/jobs/evaluation', status_code=202)
    def evaluation(body: EvaluationRequest):
        return submit('evaluation', inference_payload(body))

    @app.post('/api/jobs/quantization', status_code=202)
    def quantization(body: QuantizationRequest):
        try:
            source = Path(body.source).expanduser()
            source = (source if source.is_absolute() else models / source).resolve(strict=True)
            if not source.is_relative_to(models) or not source.is_file() or source.suffix.lower() != '.gguf':
                raise ValueError('Source must be a GGUF file inside the configured model directory')
        except (OSError, ValueError) as error:
            raise HTTPException(422, str(error)) from error
        return submit('quantization', {'source': str(source), 'target': body.target})

    @app.get('/api/artifacts')
    def artifacts():
        return [{'job_id': job['id'], **job['result']} for job in jobs.list() if job['kind'] == 'quantization' and job['status'] == 'completed' and job['result']]

    @app.get('/api/experiments')
    def experiments():
        with sqlite3.connect(root / 'experiments.sqlite3') as db:
            identifiers = db.execute('SELECT id FROM experiments ORDER BY rowid DESC LIMIT 200').fetchall()
        return [report(row[0]) for row in identifiers]

    @app.get('/api/evaluations')
    def evaluation_history():
        from backend.storage.evaluations import EvaluationStore
        return EvaluationStore(root / 'experiments.sqlite3').list()

    @app.get('/api/experiments/{identifier}')
    def experiment(identifier: str):
        return report(identifier)

    @app.get('/api/experiments/{identifier}/export')
    def export(identifier: str, format: Literal['json', 'csv'] = 'json'):
        result = report(identifier)
        if format == 'json':
            return Response(json.dumps(result, indent=2), media_type='application/json',
                            headers={'Content-Disposition': f'attachment; filename="experiment-{result["id"]}.json"'})
        stream = io.StringIO()
        fields = ['ordinal', 'warmup', 'wall_seconds', 'ttft_seconds', 'decode_tokens_per_second', 'prefill_tokens_per_second', 'error']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(result['trials'])
        return Response(stream.getvalue(), media_type='text/csv', headers={'Content-Disposition': 'attachment; filename="trials.csv"'})

    @app.post('/api/comparisons')
    def comparisons(body: ComparisonRequest):
        reports = [report(identifier) for identifier in body.experiment_ids]
        def identity(result):
            data = result['metadata']
            hardware = data.get('hardware') or {}
            return json.dumps([data.get('source_sha256'), data.get('prompt_sha256'), data.get('options'),
                data.get('runtime_version'), data.get('backend'), data.get('repeats'), data.get('warmups'),
                data.get('settings_sha256'), data.get('backend_settings'),
                {key: hardware.get(key) for key in ['chip', 'architecture', 'os', 'physical_ram_bytes']}], sort_keys=True)
        identities = [identity(result) for result in reports]
        artifacts = [result['metadata'].get('artifact_sha256') or result['metadata'].get('model', {}).get('digest') for result in reports]
        required = ['source_sha256', 'prompt_sha256', 'options', 'runtime_version', 'backend', 'repeats', 'warmups', 'settings_sha256', 'backend_settings']
        complete = all(all(result['metadata'].get(key) is not None for key in required)
                       and result['metadata'].get('source_sha256')
                       and isinstance(result['metadata'].get('settings_sha256'), str)
                       and bool(result['metadata']['settings_sha256'].strip())
                       and isinstance(result['metadata'].get('backend_settings'), dict)
                       and (result['metadata'].get('backend') != 'llama.cpp' or bool(result['metadata']['backend_settings']))
                       and all(result['metadata'].get('hardware', {}).get(key) is not None for key in ['chip', 'architecture', 'os', 'physical_ram_bytes'])
                       and result['summary']['failed_trials'] == 0
                       and result['summary']['successful_trials'] == result['metadata']['repeats']
                       and len(result['trials']) == result['metadata']['repeats'] + result['metadata']['warmups']
                       and sum(bool(row.get('warmup')) for row in result['trials']) == result['metadata']['warmups']
                       and not any(row.get('error') for row in result['trials']) for result in reports)
        comparable = bool(len(reports) >= 2 and len(set(body.experiment_ids)) == len(reports) and complete
                          and len(set(identities)) == 1 and all(artifacts) and len(set(artifacts)) == len(reports))
        return {'reports': reports, 'comparable': comparable,
                'limitations': [] if comparable else ['Comparability requires distinct verified artifacts from one source, identical prompt/options/backend settings identity/runtime/hardware/warmup/repetition policy and complete measured trials. Imported model digests alone do not establish shared source lineage.']}

    @app.post('/api/recommendations')
    def recommendations(body: RecommendationRequest):
        from backend.optimization.recommender import recommend
        reports = [report(identifier) for identifier in body.experiment_ids]
        from backend.optimization.evidence import configurations_from_reports
        from backend.storage.evaluations import EvaluationStore
        persisted = EvaluationStore(root / 'experiments.sqlite3').list()
        completed = [job['result'] for job in jobs.list()
                     if job['kind'] == 'evaluation' and job['status'] == 'completed' and job['result']]
        evaluations = list({item['id']: item for item in persisted + completed}.values())
        configurations = configurations_from_reports(reports, evaluations)
        try:
            return recommend(configurations, body.available_memory_bytes, body.max_memory_bytes,
                             body.min_quality, body.max_latency_seconds, body.objective)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    frontend_dist = Path(__file__).resolve().parents[2] / 'frontend' / 'dist'
    if frontend_dist.is_dir() and (frontend_dist / 'index.html').is_file():
        app.mount('/', StaticFiles(directory=frontend_dist, html=True), name='dashboard')
    return app


app = create_app()
