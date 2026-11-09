"""Join persisted performance and quality evidence only through verified artifact lineage."""
from __future__ import annotations
import hashlib
import json
from backend.benchmarks.runner import summarize
from backend.evaluation.scoring import accuracy_summary


def _signature(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _quality_complete(report: dict) -> bool:
    rows = report.get('results')
    return (report.get('status') == 'completed' and isinstance(rows, list) and bool(rows) and
            all(isinstance(row, dict) and row.get('status') == 'scored' and isinstance(row.get('passed'), bool) for row in rows) and
            bool(report.get('suite_sha256')) and isinstance(report.get('options'), dict) and bool(report['options']))


def configurations_from_reports(benchmarks: list[dict], evaluations: list[dict],
                                suite: str = 'reasoning-small', split: str | None = None) -> list[dict]:
    """Return recommendation inputs, retaining None for any missing measured evidence.

    The newest completed evaluation in the requested public screening suite is selected
    deterministically by (created_at, id), after matching artifact and source hashes.
    Incomplete benchmark trials are retained as insufficient evidence, never ranked.
    """
    configurations = []
    for report in benchmarks:
        metadata = report.get('metadata', {})
        model = metadata.get('model', {})
        source = metadata.get('source_sha256')
        artifact = model.get('artifact_sha256')
        matches = [item for item in evaluations if _quality_complete(item) and item.get('suite') == suite and
                   (split is None or item.get('split') == split) and
                   source and artifact and item.get('source_sha256') == source and
                   item.get('model', {}).get('artifact_sha256') == artifact]
        quality = max(matches, key=lambda item: (str(item.get('created_at', '')), str(item.get('id', '')))) if matches else None
        rows = report.get('trials', [])
        summary = summarize(rows)
        settings = metadata.get('settings_sha256')
        prompt = metadata.get('prompt_sha256')
        quality_summary = accuracy_summary(quality['results']) if quality else {}
        expected = metadata.get('repeats')
        warmups = metadata.get('warmups')
        complete = (isinstance(expected, int) and not isinstance(expected, bool) and expected > 0 and
                    isinstance(warmups, int) and not isinstance(warmups, bool) and warmups >= 0 and
                    summary['successful_trials'] == expected and summary['failed_trials'] == 0 and
                    len(rows) == expected + warmups and not any(row.get('error') for row in rows))
        workload_signature = (_signature({'benchmark_prompt_sha256': prompt, 'evaluation_suite_sha256': quality['suite_sha256']})
                              if prompt and quality else None)
        combined_settings = (_signature({'benchmark_settings_sha256': settings, 'evaluation_options': quality['options'],
                                          'evaluation_runtime_version': quality.get('runtime_version'),
                                          'evaluation_repeats': quality.get('repeats')})
                             if settings and quality else None)
        configurations.append({
            'id': report.get('id'), 'source_sha256': source, 'artifact_sha256': artifact,
            'workload_sha256': workload_signature, 'settings_sha256': combined_settings,
            'peak_memory_bytes': metadata.get('peak_memory_bytes'),
            'file_size_bytes': model.get('size'),
            'latency_seconds': (summary.get('wall_seconds') or {}).get('median'),
            'decode_tokens_per_second': (summary.get('decode_tokens_per_second') or {}).get('median'),
            'measured_trials': summary['successful_trials'] if complete else 0,
            'quality': quality_summary.get('accuracy'), 'quality_ci': quality_summary.get('accuracy_ci95'),
            'quality_trials': quality_summary.get('scored', 0),
            'evidence': {'benchmark_id': report.get('id'), 'evaluation_id': quality.get('id') if quality else None,
                         'evaluation_suite': suite, 'evaluation_split': quality.get('split') if quality else None,
                         'benchmark_complete': complete,
                         'memory_metric': metadata.get('memory_metric'),
                         'memory_note': metadata.get('memory_note'),
                         'join_policy': 'Exact artifact_sha256 and verified source_sha256; model display names are never joined'},
        })
    return configurations
