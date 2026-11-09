"""Managed llama.cpp conversion. A successful conversion is not an inference validation."""
import json
import shutil
import subprocess
import time
from pathlib import Path
import psutil
from backend.core.artifacts import inspect_gguf, sha256

TARGETS = ('Q8_0', 'Q6_K', 'Q5_K_M', 'Q4_K_M', 'Q3_K_M')
TIME = '/usr/bin/time'


def admission_estimate(metadata: dict) -> dict:
    """Conservative pre-flight memory estimate for llama-quantize (not a measurement).

    v0.2 rule (docs/DECISIONS.md): 1.5 x source size + 256 MiB. Measured llama-quantize peak max RSS
    was 1.01-1.11 x source size (experiments/memory/quantize-smollm.json); the v0.1 rule
    (2 x source + 256 MiB) was an unmeasured guess.
    """
    return {'bytes': int(metadata['size_bytes'] * 1.5) + 256 * 1024**2,
            'rule': '1.5 x source_size + 256 MiB (v0.2, measured basis: experiments/memory/quantize-smollm.json)'}


def convert(source, output_dir, target='Q4_K_M', executable='llama-quantize', timeout=1800, measure_memory=True):
    if target not in TARGETS or timeout <= 0:
        raise ValueError('Unsupported target or invalid timeout')
    source = Path(source).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    binary = shutil.which(str(executable))
    if binary is None:
        raise ValueError('llama-quantize is not installed; provide --executable /path/to/llama-quantize')
    metadata = inspect_gguf(source)
    if not metadata['higher_precision']:
        raise ValueError('Source must contain only F32/F16/BF16 tensors; requantization is not supported')
    # Conservative admission estimate, not a guarantee of peak memory.
    estimate = admission_estimate(metadata)
    required = estimate['bytes']
    available = psutil.virtual_memory().available
    if required > available:
        raise ValueError(f'Insufficient available RAM for quantization stage: admission estimate {required} bytes '
                         f'({estimate["rule"]}) exceeds available {available} bytes. Close other applications, use a '
                         f'smaller source, or provide measured evidence for a revised rule; the guard is not bypassed.')
    if not output_dir.parent.is_dir():
        raise ValueError('Output parent directory must already exist')
    if shutil.disk_usage(output_dir.parent).free < metadata['size_bytes'] + 64 * 1024**2:
        raise ValueError('Insufficient free disk space')
    output_dir.mkdir()  # Exclusive job directory: never overwrite an artifact or logs.
    output = output_dir / f'model-{target}.gguf'
    command = [binary, str(source), str(output), target]
    job = {'status': 'running', 'started_at': time.time(), 'source': metadata, 'target': target,
           'command': command, 'executable_sha256': sha256(binary),
           'memory_admission_estimate_bytes': required, 'memory_admission_rule': estimate['rule'],
           'available_ram_bytes': available, 'generation_validated': False}
    manifest = output_dir / 'manifest.json'

    def save():
        temp = output_dir / 'manifest.tmp'
        temp.write_text(json.dumps(job, indent=2) + '\n')
        temp.replace(manifest)

    save()
    try:
        from backend.diagnostics.memory import measure_command
        if measure_memory and Path(TIME).exists():
            measured = measure_command(command, output_dir / 'conversion.log', timeout=timeout)
            returncode = measured['exit_code']
            job['memory'] = {'stage': 'quantization', 'metric_source': '/usr/bin/time -l (whole llama-quantize process)',
                             **measured['time_l'], 'sampled_peak_rss_bytes': measured['sampled_peak_rss_bytes'],
                             'wall_seconds': measured['wall_seconds'], 'system_delta': measured['system_delta'],
                             'pressure_before': measured['system_before']['pressure_level'],
                             'swap_used_before_bytes': (measured['system_before'].get('swap') or {}).get('used_bytes')}
        else:
            with (output_dir / 'conversion.log').open('w') as log:
                returncode = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
        if returncode:
            raise ValueError(f'llama-quantize exited with code {returncode}; inspect {output_dir / "conversion.log"}')
        artifact = inspect_gguf(output)
        if artifact['tensor_count'] != metadata['tensor_count'] or artifact['parameter_count'] != metadata['parameter_count']:
            raise ValueError('Output tensor or parameter count differs from source')
        if {k: v['shape'] for k, v in artifact['tensors'].items()} != {k: v['shape'] for k, v in metadata['tensors'].items()}:
            raise ValueError('Output tensor names or shapes differ from source')
        if artifact['architecture'] != metadata['architecture'] or artifact['tokenizer_sha256'] != metadata['tokenizer_sha256']:
            raise ValueError('Output architecture or tokenizer differs from source')
        if artifact['higher_precision']:
            raise ValueError('Output contains no quantized tensors')
        if sha256(source) != metadata['sha256']:
            raise ValueError('Source changed during conversion')
        job.update(status='converted', artifact=artifact,
                   storage_reduction_fraction=1 - artifact['size_bytes'] / metadata['size_bytes'])
    except BaseException as error:
        job.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        job['finished_at'] = time.time()
        save()
    return job
