"""Deterministic screening suites with saved raw responses and explicit missing evidence."""
import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from backend.core.hardware import inspect
from backend.evaluation.coding import availability, score_code
from backend.evaluation import answers
from backend.evaluation.datasets import load as load_dataset
from backend.evaluation.scoring import accuracy_summary, category_summary, paired_comparison, perplexity, wilson_interval
from backend.evaluation.suites import CODING, REASONING

DATASET_SUITES = {'reasoning-v1': 'reasoning-v1', 'coding-v1': 'coding-v1'}
DEFAULT_TOKENS = {'reasoning-v1': 256, 'coding-v1': 384}
# Pre-specified (docs/DECISIONS.md, 2026-11-09, before held-out results existed): a category can
# measure quantization loss only if the baseline solves at least this many held-out items.
MIN_BASELINE_CORRECT = 10
MIN_BASELINE_ACCURACY = 0.25


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def evaluate(backend, model: str, suite: str = 'reasoning-small', repeats: int = 1,
             seed: int = 42, context: int = 1024, tokens: int = 128) -> dict:
    if suite in DATASET_SUITES:
        return evaluate_dataset(backend, model, suite, split='heldout', seed=seed, context=context, tokens=tokens)
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
    blocked = all(row.get('status') == 'blocked' for row in report['results'])
    report['status'] = ('completed' if report['summary']['scored'] == len(report['results']) else
                        'blocked' if blocked else
                        'unsupported' if report['summary']['unsupported'] == len(report['results']) else 'partial')
    return report


def _truncated(raw: dict, limit: int) -> bool:
    native = (raw.get('runtime') or {}).get('native') or {}
    if native.get('stop_type') in {'limit', 'length'} or native.get('truncated') is True:
        return True
    count = (raw.get('runtime') or {}).get('eval_count')
    return isinstance(count, int) and count >= limit


def evaluate_dataset(backend, model: str, suite: str = 'reasoning-v1', split: str = 'heldout', seed: int = 42,
                     context: int = 1024, tokens: int | None = None, limit: int | None = None, progress=None) -> dict:
    """Run a frozen, hash-pinned dataset split with deterministic decoding and save every raw response."""
    dataset = load_dataset(DATASET_SUITES[suite], split)
    tokens = tokens or DEFAULT_TOKENS[suite]
    items = dataset['items'][:limit] if limit else dataset['items']
    installed = next((item for item in backend.models() if item['name'] == model), None)
    if installed is None:
        raise ValueError('Model is not installed')
    settings = {'seed': seed, 'temperature': 0, 'num_ctx': context, 'num_predict': tokens}
    sandbox = None
    if suite == 'coding-v1':
        from backend.evaluation.sandbox import discover
        sandbox = discover()
    report = {'id': str(uuid.uuid4()), 'kind': 'evaluation', 'created_at': datetime.now(timezone.utc).isoformat(),
              'model': installed, 'suite': suite, 'split': split, 'dataset_version': dataset['version'],
              'dataset_sha256': dataset['sha256'], 'suite_sha256': _hash([item['id'] for item in items]) + ':' + dataset['sha256'],
              'item_limit': limit, 'options': settings, 'settings_sha256': _hash(settings), 'repeats': 1,
              'backend': getattr(backend, 'name', 'ollama'), 'source_sha256': getattr(backend, 'source_sha256', None),
              'prompt_template': getattr(backend, 'prompt_template', None),
              'runtime_version': backend.version(), 'hardware': inspect(), 'results': [],
              'sandbox': sandbox.status() if sandbox else None, 'perplexity': perplexity(None)}
    blocked = sandbox is not None and sandbox.status()['status'] != 'available'
    for index, item in enumerate(items):
        row = {'task_id': item['id'], 'category': None, 'item_category': item.get('category', 'coding'),
               'prompt_sha256': _hash(item['prompt'])}
        if blocked:
            row.update(status='blocked', reason=sandbox.status()['reason'], category='blocked')
        else:
            try:
                raw = backend.generate(model, item['prompt'], settings)
                truncated = _truncated(raw, tokens)
                row['generation'] = {'output': raw['output'], 'ttft_seconds': raw.get('ttft_seconds'),
                                     'wall_seconds': raw.get('wall_seconds'), 'truncated': truncated,
                                     'eval_count': (raw.get('runtime') or {}).get('eval_count')}
                if suite == 'coding-v1':
                    scored = score_code(raw['output'], {'function': item['function'], 'cases': item['cases']}, sandbox=sandbox)
                    row.update(scored)
                    row['category'] = scored.get('outcome') or scored.get('status')
                    if truncated and scored.get('outcome') in {'compile_error', 'no_code'}:
                        row['category'] = 'truncated'
                else:
                    row.update(answers.score(raw['output'], item['answer'], truncated))
            except Exception as error:  # Recorded per item; never counted as a pass.
                name = type(error).__name__
                row.update(status='failed', category='timeout' if 'Timeout' in name else 'error', error=f'{name}: {error}'[:500])
        report['results'].append(row)
        if progress:
            progress(index + 1, len(items), row)
    report['summary'] = category_summary(report['results'])
    by_category = {}
    for row in report['results']:
        by_category.setdefault(row['item_category'], []).append(row)
    report['by_category'] = {key: category_summary(rows) for key, rows in sorted(by_category.items())}
    statuses = {row['status'] for row in report['results']}
    report['status'] = ('blocked' if statuses == {'blocked'} else 'completed' if statuses == {'scored'} else 'partial')
    return report


def compare_dataset_evaluations(baseline: dict, candidate: dict) -> dict:
    """Paired comparison per category, with baseline-validity labelling (no gate is applied)."""
    for key in ['dataset_sha256', 'suite_sha256', 'settings_sha256', 'split', 'prompt_template']:
        if baseline.get(key) != candidate.get(key):
            return {'status': 'incomparable', 'reason': f'Mismatched {key}'}
    if not baseline.get('source_sha256') or baseline.get('source_sha256') != candidate.get('source_sha256'):
        return {'status': 'incomparable', 'reason': 'Verified common source_sha256 is required'}
    if baseline.get('status') != 'completed' or candidate.get('status') != 'completed':
        return {'status': 'insufficient_evidence', 'reason': 'Both evaluations must score every item'}
    base = {row['task_id']: row for row in baseline['results']}
    cand = {row['task_id']: row for row in candidate['results']}
    if base.keys() != cand.keys():
        return {'status': 'incomparable', 'reason': 'Different item sets'}
    groups = {'overall': list(base)}
    for task_id, row in base.items():
        groups.setdefault(row['item_category'], []).append(task_id)
    result = {'status': 'completed', 'categories': {}}
    for name, ids in groups.items():
        a = [bool(base[i]['passed']) for i in ids]
        b = [bool(cand[i]['passed']) for i in ids]
        comparison = paired_comparison(a, b)
        informative = sum(a) >= MIN_BASELINE_CORRECT and sum(a) / len(a) >= MIN_BASELINE_ACCURACY
        comparison.update(baseline_correct=sum(a), informative=informative,
                          validity=('informative' if informative else
                                    f'uninformative: baseline solved {sum(a)}/{len(a)} (< {MIN_BASELINE_CORRECT} and {int(MIN_BASELINE_ACCURACY*100)}%)'),
                          evidence=('insufficient' if not informative else
                                    'no detectable change' if comparison['difference_ci95'][0] <= 0 <= comparison['difference_ci95'][1]
                                    else 'detectable change'))
        result['categories'][name] = comparison
    return result


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
