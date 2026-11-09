"""Deterministic screening suites with saved raw responses and explicit missing evidence."""
import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from backend.core.hardware import inspect
from backend.evaluation.coding import availability, score_code
from backend.evaluation.scoring import accuracy_summary, perplexity
from backend.evaluation.suites import CODING, REASONING


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def evaluate(backend, model: str, suite: str = 'reasoning-small', repeats: int = 1,
             seed: int = 42, context: int = 1024, tokens: int = 128) -> dict:
    if suite not in {'reasoning-small', 'coding-small'}:
        raise ValueError('Unknown evaluation suite')
    if not 1 <= repeats <= 20 or not 1 <= context <= 32768 or not 1 <= tokens <= 4096:
        raise ValueError('Invalid evaluation resource limits')
    installed = next((item for item in backend.models() if item['name'] == model), None)
    if installed is None:
        raise ValueError('Model is not installed')
    tasks = REASONING if suite == 'reasoning-small' else CODING
    settings = {'seed': seed, 'temperature': 0, 'num_ctx': context, 'num_predict': tokens}
    sandbox = availability() if suite == 'coding-small' else None
    report = {'id': str(uuid.uuid4()), 'kind': 'evaluation', 'created_at': datetime.now(timezone.utc).isoformat(),
              'model': installed, 'suite': suite, 'split': 'public-screening', 'suite_sha256': _hash(tasks),
              'options': settings, 'settings_sha256': _hash(settings), 'repeats': repeats,
              'backend': getattr(backend, 'name', 'ollama'), 'source_sha256': getattr(backend, 'source_sha256', None),
              'runtime_version': backend.version(), 'hardware': inspect(), 'results': [],
              'perplexity': perplexity(None)}
    for repeat in range(repeats):
        for task in tasks:
            row = {'task_id': task['id'], 'repeat': repeat, 'prompt': task['prompt'], 'prompt_sha256': _hash(task['prompt'])}
            if sandbox and sandbox['status'] != 'available':
                row.update(sandbox)
            else:
                try:
                    raw = backend.generate(model, task['prompt'], settings)
                    row['generation'] = raw
                    if suite == 'coding-small':
                        row.update(score_code(raw['output'], task, sandbox=sandbox))
                    else:
                        # Whitespace-only normalization: explanations and extra answers do not earn credit.
                        answer = re.sub(r'\s+', '', raw['output'].strip())
                        row.update(status='scored', passed=answer == task['answer'], expected=task['answer'])
                except Exception as error:
                    row.update(status='failed', error=f'{type(error).__name__}: {error}')
            report['results'].append(row)
    report['summary'] = accuracy_summary(report['results'])
    report['status'] = ('completed' if report['summary']['scored'] == len(report['results']) else
                        'unsupported' if report['summary']['unsupported'] == len(report['results']) else 'partial')
    return report


def compare_evaluations(baseline: dict, candidate: dict) -> dict:
    for key in ['suite_sha256', 'settings_sha256', 'repeats']:
        if baseline.get(key) is None or baseline.get(key) != candidate.get(key):
            return {'status': 'incomparable', 'reason': f'Mismatched or missing {key}'}
    source_a = baseline.get('source_sha256')
    source_b = candidate.get('source_sha256')
    if not source_a or source_a != source_b:
        return {'status': 'incomparable', 'reason': 'Verified common source_sha256 is required'}
    if baseline.get('status') != 'completed' or candidate.get('status') != 'completed':
        return {'status': 'insufficient_evidence', 'reason': 'Both evaluations must complete all tasks'}
    a, b = baseline['summary'], candidate['summary']
    # Difference bound constructed from marginal Wilson bounds; conservative, not paired bootstrap.
    return {'status': 'completed', 'accuracy_difference': b['accuracy'] - a['accuracy'],
            'difference_interval': [b['accuracy_ci95'][0] - a['accuracy_ci95'][1], b['accuracy_ci95'][1] - a['accuracy_ci95'][0]],
            'interval_method': 'Conservative difference of marginal Wilson 95% intervals; not a calibrated paired confidence interval',
            'baseline': a, 'candidate': b}
