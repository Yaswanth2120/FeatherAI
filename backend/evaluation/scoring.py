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
            'unsupported': sum(row.get('status') == 'unsupported' for row in rows),
            'failed': sum(row.get('status') == 'failed' for row in rows),
            'uncertainty_note': 'Wilson interval is descriptive for this small public suite; repeated deterministic tasks are not independent evidence.'}
