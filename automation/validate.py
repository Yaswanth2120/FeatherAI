"""Fail-closed task acceptance gates; real-run evidence is never inferred from fixtures."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def evidence(name):
    path = ROOT / 'experiments' / name
    require(path.is_file(), f'Missing real-run evidence: {path}')
    result = json.loads(path.read_text())
    require(result.get('kind') == 'real-run', 'Evidence must explicitly identify a real run')
    return result


def verify_artifact(artifact):
    require(isinstance(artifact.get('sha256'), str) and re.fullmatch(r'[0-9a-f]{64}', artifact['sha256']), 'Expected SHA-256 hex digest')
    path = Path(artifact['path'])
    if not path.is_absolute():
        path = ROOT / path
    require(path.is_file(), f'Missing artifact: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    require(digest.hexdigest() == artifact['sha256'], f'Artifact hash mismatch: {path}')


def validate_workflow(data, record=None):
    workflow_path = (ROOT / str(data.get('workflow', ''))).resolve()
    e2e_root = (ROOT / 'experiments' / 'e2e').resolve()
    require(workflow_path.is_relative_to(e2e_root) and workflow_path.is_file(), 'Workflow record must be inside experiments/e2e/')
    record = json.loads(workflow_path.read_text()) if record is None else record
    folder = workflow_path.parent
    wid = record.get('workflow_id')
    require(record.get('kind') == 'real-run' and record.get('status') == 'completed' and wid == data.get('workflow_id') == folder.name,
            'Workflow identity or status mismatch')
    stages = record.get('stages', {})
    names = ['import', 'preflight', 'quantization', 'generation_validation', 'benchmark', 'evaluation', 'recommendation', 'export']
    for name in names:
        require(stages.get(name, {}).get('workflow_id') == wid and stages[name].get('status') == 'completed', f'Stage {name} missing or incomplete')
    constraints_file = ROOT / record['constraints_file']
    require(hashlib.sha256(constraints_file.read_bytes()).hexdigest() == record['constraints_sha256'], 'Recommendation constraints changed after the run')
    committed = subprocess.run(['git', 'log', '-1', '--format=%cI', '--', record['constraints_file']], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    from datetime import datetime
    require(bool(committed) and datetime.fromisoformat(committed) < datetime.fromisoformat(record['started_at']),
            'Constraints must be committed before the workflow started')
    constraints = json.loads(constraints_file.read_text())
    source = stages['import']['source']
    verify_artifact({'path': source['path'], 'sha256': source['sha256']})
    require(stages['preflight']['sandbox'].get('status') == 'available', 'Sandbox was not available')
    artifacts = stages['quantization']['artifacts']
    require(len(artifacts) >= 2 and len({a['artifact_sha256'] for a in artifacts.values()}) == len(artifacts), 'Need distinct quantized artifacts')
    for target, item in artifacts.items():
        require(item['status'] == 'completed' and item['source_sha256'] == source['sha256'], f'{target} not created from the imported source')
        require(item['artifact_sha256'] != source['sha256'], f'{target} artifact equals source')
        verify_artifact({'path': item['artifact_path'], 'sha256': item['artifact_sha256']})
        validation = stages['generation_validation']['results'][target]
        require(validation['artifact_sha256'] == item['artifact_sha256'] and validation['nonempty'], f'{target} generation not validated')
        bench = stages['benchmark']['results'][target]
        require(bench['status'] == 'completed' and bench['artifact_sha256'] == item['artifact_sha256'] and bench['source_sha256'] == source['sha256'],
                f'{target} benchmark lineage mismatch')
        summary = bench['summary']
        require(summary['successful_trials'] == record['settings']['benchmark_repeats'] and summary['failed_trials'] == 0, f'{target} benchmark incomplete')
        evaluation = stages['evaluation']['results'][target]
        require(evaluation['status'] == 'completed' and evaluation['artifact_sha256'] == item['artifact_sha256'] and
                evaluation['source_sha256'] == source['sha256'], f'{target} evaluation lineage mismatch')
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from backend.evaluation.datasets import SHA256, load
        heldout = len(load('coding-v1', 'heldout')['items'])
        require(evaluation['dataset_sha256'] == SHA256['coding-v1'] and evaluation['split'] == 'heldout', f'{target} wrong dataset')
        require((evaluation.get('sandbox') or {}).get('status') == 'available', f'{target} coding was not sandboxed')
        require(evaluation['summary']['scored'] == evaluation['summary']['items'] == heldout, f'{target} coding items not all scored')
    settings = {stages['benchmark']['results'][t]['settings_sha256'] for t in artifacts}
    require(None not in settings, 'Benchmark settings identity missing')
    recommendation = stages['recommendation']
    request = recommendation['request']
    for key in ['max_memory_bytes', 'min_quality', 'max_latency_seconds', 'objective', 'quality_suite']:
        require(request[key] == constraints[key], f'Recommendation constraint {key} differs from the pre-registered value')
    require(sorted(request['experiment_ids']) == sorted(stages['benchmark']['results'][t]['experiment_id'] for t in artifacts),
            'Recommendation must use exactly this workflow\'s benchmarks')
    require(recommendation['outcome'] in constraints['expected_valid_outcomes'] and recommendation['http_status'] == 200, 'Invalid recommendation outcome')
    saved = json.loads((folder / 'recommendation.json').read_text())
    require(saved['request'] == request and saved['response'].get('status') == recommendation['outcome'], 'Saved recommendation does not match the stage record')
    files = stages['export']['files']
    for target in artifacts:
        for name in [f'exports/benchmark-{target}.json', f'exports/benchmark-{target}.csv', f'exports/evaluation-{target}.json']:
            require(name in files and files[name]['http_status'] == 200, f'Missing export {name}')
    for name, item in files.items():
        path = folder / name
        require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], f'Export {name} missing or altered')


def validate(identifier):
    if identifier == 'task01':
        require((ROOT / 'docs/AUDIT.md').is_file(), 'Repository audit document is missing')
    elif identifier == 'task02':
        data = evidence('quantization-evidence.json')
        require(data.get('source_precision') in {'F32', 'F16', 'BF16'}, 'Higher-precision source required')
        require(data.get('created_by_feather') is True, 'Artifact must be created by Feather')
        verify_artifact(data['source'])
        verify_artifact(data['output'])
        require(data['source']['sha256'] != data['output']['sha256'], 'Source and output must differ')
        require(data.get('generation_validated') is True, 'Real generation validation is missing')
        require(bool(data.get('baseline_output', '').strip()) and bool(data.get('quantized_output', '').strip()), 'Baseline and quantized generation outputs required')
    elif identifier == 'task04':
        # v0.2 evidence supersedes the v0.1 file, which is kept unmodified as history; the checks are identical.
        data = evidence('comparison-evidence-v0.2.json' if (ROOT / 'experiments' / 'comparison-evidence-v0.2.json').is_file()
                        else 'comparison-evidence.json')
        configs = data.get('configurations', [])
        require(len(configs) >= 2, 'At least two configurations required')
        require(len({c['source_sha256'] for c in configs}) == 1, 'Configurations must share exact source')
        require(len({c['artifact_sha256'] for c in configs}) >= 2, 'Configurations must use distinct artifacts')
        # Evidence must cite an exported study and be exactly reproducible from it (no exemption).
        require(isinstance(data.get('study'), str) and bool(data['study']), 'Evidence must cite its exported study directory')
        studies = (ROOT / 'experiments' / 'studies').resolve()
        study_dir = (ROOT / data['study']).resolve()
        require(study_dir.is_relative_to(studies) and (study_dir / 'study.json').is_file(), 'Cited study must be inside experiments/studies/')
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from backend.study.evidence import build
        rebuilt = json.loads(json.dumps(build(study_dir)))
        rebuilt['study'] = data['study']  # the recorded path is repository-relative; every measured field must match
        require(rebuilt == data, 'Comparison evidence does not match its exported study')
        require(all(c.get('coding_status') == 'completed' for c in configs), 'Coding evaluation must have completed for every configuration')
        reference = configs[0]
        for field in ['context_length', 'decoding_settings', 'runtime_version', 'repeats', 'warmups']:
            require(all(config.get(field) == reference.get(field) for config in configs), f'Incomparable experiment setting: {field}')
        for config in configs:
            for field in ['file_size_bytes', 'peak_memory_bytes', 'ttft_seconds', 'prefill_tokens_per_second',
                          'decode_tokens_per_second', 'wall_seconds', 'coding_correctness', 'repeats', 'warmups',
                          'context_length', 'decoding_settings', 'runtime_version', 'hardware', 'raw_trials']:
                require(field in config and config[field] is not None, f'Missing measured/configuration field: {field}')
            for field in ['source_sha256', 'artifact_sha256']:
                require(isinstance(config[field], str) and re.fullmatch(r'[0-9a-f]{64}', config[field]), f'Invalid hash: {field}')
            for field in ['file_size_bytes', 'peak_memory_bytes', 'ttft_seconds', 'prefill_tokens_per_second', 'decode_tokens_per_second', 'wall_seconds']:
                value = config[field]
                require(type(value) in {int, float} and math.isfinite(value) and value > 0, f'Invalid measured metric: {field}')
            require(type(config['coding_correctness']) in {int, float} and 0 <= config['coding_correctness'] <= 1, 'Coding correctness must be a fraction from 0 to 1')
            require(type(config['repeats']) is int and type(config['warmups']) is int, 'Trial counts must be integers')
            require(config['repeats'] >= 5 and config['warmups'] >= 1, 'Require five measured trials after warmup')
            require(type(config['context_length']) is int and config['context_length'] > 0, 'Invalid context length')
            require(isinstance(config['decoding_settings'], dict) and bool(config['decoding_settings']), 'Missing decoding settings')
            require(isinstance(config['runtime_version'], str) and bool(config['runtime_version']), 'Missing runtime version')
            require(isinstance(config['hardware'], dict), 'Invalid hardware metadata')
            for field in ['chip', 'architecture', 'physical_ram_bytes', 'os']:
                require(config['hardware'].get(field) is not None and config['hardware'].get(field) == reference['hardware'].get(field), f'Incomparable hardware: {field}')
            require(isinstance(config['raw_trials'], list), 'Raw trials must be a list')
            measured = [trial for trial in config['raw_trials'] if trial.get('warmup') is False and not trial.get('error')]
            warmups = [trial for trial in config['raw_trials'] if trial.get('warmup') is True and not trial.get('error')]
            require(len(measured) == config['repeats'], 'Successful raw measured trials do not match requested repetitions')
            require(len(warmups) == config['warmups'], 'Successful raw warmups do not match warmup policy')
            require(not any(trial.get('error') for trial in config['raw_trials']), 'Comparison contains failed trials; rerun a complete controlled experiment')
            for trial in measured:
                for field in ['ttft_seconds', 'prefill_tokens_per_second', 'decode_tokens_per_second', 'wall_seconds']:
                    value = trial.get(field)
                    require(type(value) in {int, float} and math.isfinite(value) and value > 0, f'Missing/invalid raw measured metric: {field}')
            if config.get('perplexity') is not None:
                value = config['perplexity']
                require(type(value) in {int, float} and math.isfinite(value) and value > 0, 'Invalid perplexity')
            require(config.get('perplexity') is not None or config.get('perplexity_unsupported_reason'), 'Perplexity needs measurement or unsupported reason')
    elif identifier == 'task08':
        # v0.2: one traceable workflow is mandatory. The v0.1 flag-only file (no coding stage) is history.
        data = evidence('e2e-evidence-v0.2.json')
        require(data.get('status') == 'completed', 'End-to-end workflow did not complete')
        require(all(data.get(stage) is True for stage in ['imported', 'quantized', 'generation_validated',
                    'benchmarked', 'evaluated', 'coding_evaluated', 'recommended', 'exported']), 'Incomplete real end-to-end workflow')
        validate_workflow(data)
        tests = subprocess.run(['uv', 'run', 'pytest', '-q', 'tests/test_api.py', '-k',
                                'cancellation or interrupted or failure or timeout or restart'], cwd=ROOT)
        require(tests.returncode == 0, 'Interrupted-job and failure regression tests must pass')
        validate('task02')
        validate('task04')
    elif identifier == 'task07':
        subprocess.run(['npm', 'run', 'build'], cwd=ROOT / 'frontend', check=True)
    elif identifier == 'task09':
        for filename in ['README.md', 'docs/FINAL_REPORT.md', 'docs/PROGRESS.md', 'docs/DECISIONS.md']:
            require((ROOT / filename).is_file(), f'Missing documentation: {filename}')
        validate('task08')
    elif identifier in {'task03', 'task05', 'task06'}:
        names = {'task03': 'test_evaluation', 'task05': 'test_recommend', 'task06': 'test_api'}
        matches = list((ROOT / 'tests').glob(names[identifier] + '*.py'))
        require(bool(matches), f'Missing focused tests: {names[identifier]}*.py')
        subprocess.run(['uv', 'run', 'pytest', *[str(path) for path in matches]], cwd=ROOT, check=True)
    else:
        raise ValueError(f'Unknown task: {identifier}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    args = parser.parse_args()
    try:
        validate(args.task)
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        print(f'Acceptance failed: {error}', file=sys.stderr)
        sys.exit(1)
