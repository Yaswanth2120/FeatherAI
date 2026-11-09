"""Small-sample uncertainty and optional token-probability scoring."""
import math
from typing import Iterable


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float] | None:
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError('Invalid binomial counts')
    if trials == 0:
        return None
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    # Floating point cancellation near p=0/1 must never exclude the point estimate.
    return [max(0.0, min(p, center - half)), min(1.0, max(p, center + half))]


def perplexity(token_log_probabilities: Iterable[float] | None) -> dict:
    if token_log_probabilities is None:
        return {'status': 'unsupported', 'reason': 'Backend did not provide held-out token log probabilities'}
    values = list(token_log_probabilities)
    if not values or any(not isinstance(x, (int, float)) or not math.isfinite(x) or x > 0 for x in values):
        raise ValueError('Expected nonempty finite nonpositive token log probabilities')
    mean_nll = -sum(values) / len(values)
    try:
        value = math.exp(mean_nll)
    except OverflowError:
        return {'status': 'overflow', 'mean_nll': mean_nll, 'tokens': len(values)}
    return {'status': 'completed', 'value': value, 'mean_nll': mean_nll, 'tokens': len(values)}


def accuracy_summary(rows: list[dict]) -> dict:
    scored = [row for row in rows if row.get('status') == 'scored']
    passed = sum(bool(row['passed']) for row in scored)
    return {'scored': len(scored), 'passed': passed,
            'accuracy': passed / len(scored) if scored else None,
            'accuracy_ci95': wilson_interval(passed, len(scored)),
            'unsupported': sum(row.get('status') in {'unsupported', 'blocked'} for row in rows),
            'failed': sum(row.get('status') == 'failed' for row in rows),
            'uncertainty_note': 'Wilson interval is descriptive for this small public suite; repeated deterministic tasks are not independent evidence.'}


def paired_comparison(baseline: list[bool], candidate: list[bool], resamples: int = 10000, seed: int = 20261109) -> dict:
    """Candidate minus baseline accuracy on the same items.

    95% percentile interval from a paired bootstrap over items, and an exact two-sided McNemar
    test on discordant pairs. Items are the resampling unit because each item is scored once
    per variant under deterministic decoding.
    """
    import numpy as np
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError('Paired comparison needs equal-length nonempty outcome lists')
    a = np.asarray(baseline, dtype=float)
    b = np.asarray(candidate, dtype=float)
    differences = b - a
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(a), size=(resamples, len(a)))
    boot = differences[indices].mean(axis=1)
    only_baseline = int(((a == 1) & (b == 0)).sum())
    only_candidate = int(((a == 0) & (b == 1)).sum())
    discordant = only_baseline + only_candidate
    if discordant:
        k = min(only_baseline, only_candidate)
        tail = sum(math.comb(discordant, i) for i in range(k + 1)) / 2 ** discordant
        p_value = min(1.0, 2 * tail)
    else:
        p_value = 1.0
    return {'items': len(a), 'baseline_accuracy': float(a.mean()), 'candidate_accuracy': float(b.mean()),
            'difference': float(differences.mean()),
            'difference_ci95': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            'ci_method': f'paired bootstrap over items, {resamples} resamples, seed {seed}',
            'baseline_only_correct': only_baseline, 'candidate_only_correct': only_candidate,
            'mcnemar_exact_p': p_value}


def category_summary(rows: list[dict]) -> dict:
    scored = [row for row in rows if row.get('status') == 'scored']
    passed = sum(bool(row['passed']) for row in scored)
    categories = {}
    for row in rows:
        key = row.get('category') or row.get('outcome') or row.get('status')
        categories[key] = categories.get(key, 0) + 1
    lenient = [row for row in scored if 'lenient_passed' in row]
    complete = bool(rows) and len(scored) == len(rows)
    # Accuracy is reported only when every item was scored; otherwise the denominator would hide failures.
    return {'items': len(rows), 'scored': len(scored), 'passed': passed, 'complete': complete,
            'accuracy': passed / len(scored) if complete else None,
            'accuracy_ci95': wilson_interval(passed, len(scored)) if complete else None,
            'accuracy_over_scored_items': passed / len(scored) if scored else None,
            'outcomes': categories,
            'not_scored': len(rows) - len(scored),
            'lenient_accuracy': (sum(r['lenient_passed'] for r in lenient) / len(lenient)) if lenient else None}
