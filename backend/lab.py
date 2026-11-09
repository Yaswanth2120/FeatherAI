"""Reproducible sequential source -> quantizations -> generation -> benchmark -> quality study."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import time
from backend.backends.llama_cpp import LlamaCpp
from backend.benchmarks.runner import run
from backend.core.artifacts import inspect_gguf, sha256
from backend.evaluation.runner import evaluate, compare_evaluations
from backend.optimization.evidence import configurations_from_reports
from backend.optimization.recommender import recommend
from backend.quantization.gguf_adapter import convert
from backend.storage.database import Database
from backend.storage.evaluations import EvaluationStore
from backend.core.hardware import inspect

VALIDATION_PROMPT = 'What is 2 + 2? Answer with only the number.'


def study(source, output_dir, quantizer, server, db_path='.feather/experiments.sqlite3', repeats=5):
    source = Path(source).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir()  # Exclusive evidence directory. Never overwrite a prior study.
    source_metadata = inspect_gguf(source)
    if not source_metadata['higher_precision']:
        raise ValueError('Study requires higher-precision source weights')
    source_hash = source_metadata['sha256']
    source_label = 'FP16' if 'F16' in source_metadata['tensor_types'] else 'BF16' if 'BF16' in source_metadata['tensor_types'] else 'FP32'
    state = {'status': 'running', 'source': source_metadata, 'started_at': time.time(),
             'benchmark_reports': [], 'evaluations': [], 'generation': {}, 'conversions': [],
             'limitations': ['Public screening suites are not held-out generalization evidence.',
                             'Process memory is sampled RSS, excluding startup peak and potentially GPU allocations.',
                             'Perplexity is unsupported through this adapter; no held-out token log probabilities are collected.']}
    def save():
        path = output_dir / 'study.json'
        temp = output_dir / 'study.tmp'
        temp.write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
        temp.replace(path)
    save()
    db = Database(db_path)
    quality_store = EvaluationStore(db_path)
    try:
        variants = [(source_label, source)]
        for target in ['Q4_K_M', 'Q8_0']:
            job = convert(source, output_dir / target, target, executable=quantizer)
            variants.append((target, Path(job['artifact']['path'])))
            state['conversions'].append(job)
            save()
        for label, path in variants:
            with LlamaCpp(path, executable=server, context=1024, source_sha256=source_hash,
                          log_path=output_dir / f'{label}-server.log') as backend:
                model = backend.models()[0]['name']
                generation = backend.generate(model, VALIDATION_PROMPT, {'num_ctx':1024, 'num_predict':32, 'seed':42, 'temperature':0})
                if not generation['output'].strip():
                    raise ValueError(f'{label} produced empty validation output')
                state['generation'][label] = generation
                benchmark = run(backend, db, model, repeats=repeats, warmups=1, context=1024, tokens=128, seed=42)
                state['benchmark_reports'].append(benchmark)
                save()
                quality = evaluate(backend, model, suite='reasoning-small', repeats=1, context=1024, tokens=128, seed=42)
                quality_store.save(quality)
                state['evaluations'].append(quality)
                coding = evaluate(backend, model, suite='coding-small', repeats=1, context=1024, tokens=128, seed=42)
                quality_store.save(coding)
                state['evaluations'].append(coding)
                save()
            if label != source_label:
                manifest = output_dir / label / 'manifest.json'
                job = json.loads(manifest.read_text())
                if sha256(path) != job['artifact']['sha256']:
                    raise ValueError('Artifact changed during generation validation')
                job.update(generation_validated=True, generation_validation={
                    'prompt': VALIDATION_PROMPT, 'options': {'context':1024,'tokens':32,'seed':42,'temperature':0},
                    'output': generation['output'], 'baseline_output': state['generation'][source_label]['output'],
                    'runtime_version': benchmark['metadata']['runtime_version'], 'artifact_sha256': job['artifact']['sha256']})
                manifest.write_text(json.dumps(job, indent=2) + '\n')
                state['conversions'] = [job if item['target']==label else item for item in state['conversions']]
                save()
        configs = configurations_from_reports(state['benchmark_reports'], state['evaluations'])
        state['recommendation'] = recommend(configs, inspect()['available_ram_bytes'], max_memory_bytes=1024**3,
                                           min_quality=0, objective='fastest')
        baseline = next(e for e in state['evaluations'] if e['suite']=='reasoning-small')
        state['quality_comparisons'] = [compare_evaluations(baseline, e) for e in state['evaluations'] if e['suite']=='reasoning-small' and e['id']!=baseline['id']]
        state.update(study_status(state))
    except BaseException as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        state['finished_at'] = time.time()
        save()
        db.connection.close()
    return state


def study_status(state):
    coding = [e['status'] for e in state['evaluations'] if e['suite'] == 'coding-small']
    coding_status = ('completed' if coding and all(value == 'completed' for value in coding) else
                     'unsupported' if coding and all(value == 'unsupported' for value in coding) else 'partial')
    benchmark_complete = bool(state['benchmark_reports']) and all(
        b['summary']['failed_trials'] == 0 and b['summary']['successful_trials'] == b['metadata']['repeats'] and
        not any(t.get('error') for t in b['trials']) for b in state['benchmark_reports'])
    reasoning_complete = all(e['status'] == 'completed' for e in state['evaluations'] if e['suite'] == 'reasoning-small')
    return {'coding_evaluation_status': coding_status,
            'status': 'completed_with_limitations' if benchmark_complete and reasoning_complete and coding_status != 'partial' else 'partial'}
