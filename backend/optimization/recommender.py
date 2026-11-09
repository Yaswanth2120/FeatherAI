"""Deterministic ranking with explicit rejection reasons and no imputed metrics."""
import math

METRICS = ('peak_memory_bytes', 'quality', 'latency_seconds', 'decode_tokens_per_second', 'file_size_bytes')
IDENTITY = ('source_sha256', 'workload_sha256', 'settings_sha256')
OBJECTIVES = {'fastest': ('decode_tokens_per_second', True), 'smallest': ('file_size_bytes', False),
              'lowest_latency': ('latency_seconds', False), 'highest_quality': ('quality', True)}


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def recommend(configurations: list[dict], available_memory_bytes: int, max_memory_bytes: int,
              min_quality: float = 0.0, max_latency_seconds: float | None = None,
              objective: str = 'fastest') -> dict:
    if not _finite(available_memory_bytes) or available_memory_bytes <= 0 or not _finite(max_memory_bytes) or max_memory_bytes <= 0:
        raise ValueError('Memory constraints must be finite and positive')
    if not _finite(min_quality) or not 0 <= min_quality <= 1:
        raise ValueError('Quality constraint must be between 0 and 1')
    if max_latency_seconds is not None and (not _finite(max_latency_seconds) or max_latency_seconds <= 0):
        raise ValueError('Latency constraint must be finite and positive')
    if objective not in OBJECTIVES:
        raise ValueError('Unknown optimization objective')
    if not isinstance(configurations, list) or any(not isinstance(c, dict) for c in configurations):
        raise ValueError('Configurations must be a list of objects')
    limit = min(available_memory_bytes, max_memory_bytes)
    result = {'status': 'insufficient_evidence', 'objective': objective, 'eligible': [], 'rejected': [],
              'memory_limit_bytes': limit, 'limitations': [
                  'Only supplied measurements are considered; no metric is inferred from quantization width.',
                  'Measured peak memory is workload-specific and does not guarantee fit for larger contexts.',
                  'Sampled process RSS is not total GPU or unified-memory allocation; unsampled peaks may be missed.',
                  'Public screening quality and repeated deterministic trials do not establish held-out generalization.']}
    if min_quality == 0:
        result['limitations'].append('Minimum quality is zero: ranking is diagnostic only and does not establish useful output quality.')
    usable = []
    ids = [c.get('id') for c in configurations]
    for config in configurations:
        reasons = []
        identifier = config.get('id')
        if not isinstance(identifier, str) or not identifier or ids.count(identifier) != 1:
            reasons.append('Missing or duplicate configuration id')
        if any(not isinstance(config.get(key), str) or not config[key].strip() for key in IDENTITY):
            reasons.append('Missing source, workload or decoding-settings identity')
        for key in METRICS:
            value = config.get(key)
            if not _finite(value) or value < 0 or (key in {'peak_memory_bytes', 'decode_tokens_per_second', 'file_size_bytes'} and value == 0):
                reasons.append(f'Missing or invalid measured {key}')
        if _finite(config.get('quality')) and not 0 <= config['quality'] <= 1:
            reasons.append('Quality is outside [0, 1]')
        for key in ['measured_trials', 'quality_trials']:
            value = config.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                reasons.append(f'Missing positive {key}')
        ci = config.get('quality_ci')
        if ci is not None and (not isinstance(ci, (list, tuple)) or len(ci) != 2 or
                               any(not _finite(v) for v in ci) or not 0 <= ci[0] <= ci[1] <= 1 or
                               not _finite(config.get('quality')) or not ci[0] <= config['quality'] <= ci[1]):
            reasons.append('Invalid quality confidence interval')
        if reasons:
            result['rejected'].append({'id': identifier, 'reasons': reasons, 'evidence_missing': True})
        else:
            usable.append(config)
    groups = {tuple(c[key] for key in IDENTITY) for c in usable}
    if len(groups) > 1:
        result['reason'] = 'Configurations have different source, workload or settings; select one comparable experiment group'
        result['rejected'].extend({'id': c['id'], 'reasons': ['Incomparable experiment identity'], 'evidence_missing': True} for c in usable)
        return result
    for config in usable:
        reasons = []
        if config['peak_memory_bytes'] > limit:
            reasons.append('Measured peak memory exceeds available or allowed memory')
        quality_bound = config['quality_ci'][0] if config.get('quality_ci') is not None else config['quality']
        if quality_bound < min_quality:
            reasons.append('Measured quality or lower confidence bound is below the requirement')
        if max_latency_seconds is not None and config['latency_seconds'] > max_latency_seconds:
            reasons.append('Measured latency exceeds the requirement')
        if reasons:
            result['rejected'].append({'id': config['id'], 'reasons': reasons, 'evidence_missing': False})
        else:
            item = dict(config)
            item['reasons'] = [f'Measured peak memory {config["peak_memory_bytes"]} <= {limit} bytes',
                               f'Quality constraint uses {"lower confidence bound" if config.get("quality_ci") is not None else "point estimate"} {quality_bound}',
                               f'Ranked by measured {OBJECTIVES[objective][0]}']
            item['confidence'] = {'measured_trials': config['measured_trials'], 'quality_trials': config['quality_trials'],
                                  'quality_ci95': config.get('quality_ci'),
                                  'level': 'limited' if min(config['measured_trials'], config['quality_trials']) < 5 or config.get('quality_ci') is None else 'reported_interval'}
            result['eligible'].append(item)
    key, descending = OBJECTIVES[objective]
    result['eligible'].sort(key=lambda c: ((-c[key] if descending else c[key]), c['id']))
    for index, config in enumerate(result['eligible'], start=1):
        config['rank'] = index
    if result['eligible']:
        result.update(status='completed', selected_id=result['eligible'][0]['id'])
    elif usable:
        result.update(status='no_eligible_configuration', reason='No fully measured configuration satisfies every constraint')
    else:
        result['reason'] = 'No complete measured configurations are available'
    return result
