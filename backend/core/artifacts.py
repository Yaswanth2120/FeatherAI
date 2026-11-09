"""Inspect GGUF structure and record actual tensor types, without loading weights into RAM."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from gguf import GGUFReader


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_gguf(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError('Expected a GGUF file')
    try:
        reader = GGUFReader(path)
        if not reader.tensors:
            raise ValueError('GGUF contains no tensors')
        architecture = reader.get_field('general.architecture')
        tokenizer = hashlib.sha256()
        for key, field in sorted(reader.fields.items()):
            if key.startswith('tokenizer.'):
                canonical = json.dumps({'key': key, 'types': [int(t) for t in field.types], 'value': field.contents()}, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
                tokenizer.update(len(canonical.encode()).to_bytes(8, 'little'))
                tokenizer.update(canonical.encode())
        types = Counter(t.tensor_type.name for t in reader.tensors)
        for tensor in reader.tensors:
            if tensor.data_offset + tensor.n_bytes > path.stat().st_size:
                raise ValueError(f'Truncated tensor: {tensor.name}')
        return {'path': str(path), 'sha256': sha256(path), 'size_bytes': path.stat().st_size,
                'architecture': architecture.contents() if architecture else None, 'tokenizer_sha256': tokenizer.hexdigest(),
                'tensor_count': len(reader.tensors), 'tensor_types': dict(types),
                'tensors': {t.name: {'shape': [int(x) for x in t.shape], 'type': t.tensor_type.name} for t in reader.tensors},
                'parameter_count': sum(int(t.n_elements) for t in reader.tensors),
                'higher_precision': all(t in {'F32', 'F16', 'BF16'} for t in types)}
    except (KeyError, IndexError, TypeError, OverflowError) as error:
        raise ValueError(f'Invalid GGUF: {error}') from error
