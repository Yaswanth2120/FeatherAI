"""Managed llama.cpp conversion. A successful conversion is not an inference validation."""
import json
import shutil
import subprocess
import time
from pathlib import Path
import psutil
from backend.core.artifacts import inspect_gguf, sha256

TARGETS = ('Q8_0', 'Q6_K', 'Q5_K_M', 'Q4_K_M', 'Q3_K_M')


def convert(source, output_dir, target='Q4_K_M', executable='llama-quantize', timeout=1800):
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
    # Conservative admission estimate, not a guarantee of peak runtime memory.
    required = metadata['size_bytes'] * 2 + 256 * 1024**2
    available = psutil.virtual_memory().available
    if required > available:
        raise ValueError(f'Insufficient available RAM: admission estimate {required} bytes, available {available}')
    if not output_dir.parent.is_dir():
        raise ValueError('Output parent directory must already exist')
    if shutil.disk_usage(output_dir.parent).free < metadata['size_bytes'] + 64 * 1024**2:
        raise ValueError('Insufficient free disk space')
    output_dir.mkdir()  # Exclusive job directory: never overwrite an artifact or logs.
    output = output_dir / f'model-{target}.gguf'
    command = [binary, str(source), str(output), target]
    job = {'status': 'running', 'started_at': time.time(), 'source': metadata, 'target': target,
           'command': command, 'executable_sha256': sha256(binary),
           'memory_admission_estimate_bytes': required, 'available_ram_bytes': available,
           'generation_validated': False}
    manifest = output_dir / 'manifest.json'

    def save():
        temp = output_dir / 'manifest.tmp'
        temp.write_text(json.dumps(job, indent=2) + '\n')
        temp.replace(manifest)

    save()
    try:
        with (output_dir / 'conversion.log').open('w') as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        if process.returncode:
            raise ValueError(f'llama-quantize exited with code {process.returncode}; inspect {output_dir / "conversion.log"}')
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
