import json
import subprocess
from types import SimpleNamespace
import numpy as np
import pytest
from gguf import GGUFWriter, GGMLQuantizationType
from gguf.quants import quantize
from backend.core.artifacts import inspect_gguf, sha256
from backend.quantization.gguf_adapter import convert


def write_model(path, quantized=False):
    writer = GGUFWriter(path, 'llama')
    weights = np.linspace(-1, 1, 1024, dtype=np.float32).reshape(32, 32)
    if quantized:
        writer.add_tensor('test.weight', quantize(weights, GGMLQuantizationType.Q8_0), raw_dtype=GGMLQuantizationType.Q8_0)
    else:
        writer.add_tensor('test.weight', weights)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_artifact_inspects_tensor_types_and_hash(tmp_path):
    source = tmp_path / 'source.gguf'
    write_model(source)
    data = inspect_gguf(source)
    assert data['higher_precision']
    assert data['parameter_count'] == 1024
    assert data['tensor_types'] == {'F32': 1}
    assert data['sha256'] == sha256(source)


def test_rejects_requantization(tmp_path):
    source = tmp_path / 'source.gguf'
    write_model(source, True)
    with pytest.raises(ValueError, match='requantization'):
        convert(source, tmp_path / 'job', executable='/usr/bin/true')
    assert not (tmp_path / 'job').exists()


def test_conversion_failure_persists_manifest_and_logs(tmp_path):
    source = tmp_path / 'source.gguf'
    write_model(source)
    job = tmp_path / 'job'
    with pytest.raises(ValueError, match='exited'):
        convert(source, job, executable='/usr/bin/false')
    manifest = json.loads((job / 'manifest.json').read_text())
    assert manifest['status'] == 'failed'
    assert not manifest['generation_validated']
    assert (job / 'conversion.log').exists()


def test_conversion_validates_output_and_lineage(tmp_path, monkeypatch):
    source = tmp_path / 'source.gguf'
    write_model(source)
    def fake_quantizer(command, **kwargs):
        write_model(command[2], True)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess, 'run', fake_quantizer)
    result = convert(source, tmp_path / 'job', target='Q8_0', executable='/usr/bin/true')
    assert result['status'] == 'converted'
    assert result['artifact']['tensor_types'] == {'Q8_0': 1}
    assert result['storage_reduction_fraction'] > 0
    assert result['source']['sha256'] != result['artifact']['sha256']
    with pytest.raises(FileExistsError):
        convert(source, tmp_path / 'job', executable='/usr/bin/true')


def test_rejects_insufficient_memory(tmp_path, monkeypatch):
    import psutil
    source = tmp_path / 'source.gguf'
    write_model(source)
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=1))
    with pytest.raises(ValueError, match='RAM'):
        convert(source, tmp_path / 'job', executable='/usr/bin/true')


def test_timeout_persists_failure(tmp_path, monkeypatch):
    source = tmp_path / 'source.gguf'
    write_model(source)
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs['timeout'])
    monkeypatch.setattr(subprocess, 'run', timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        convert(source, tmp_path / 'job', executable='/usr/bin/true')
    assert json.loads((tmp_path / 'job/manifest.json').read_text())['status'] == 'failed'


def test_tokenizer_fingerprint_preserves_token_boundaries(tmp_path):
    from backend.core.artifacts import inspect_gguf
    paths=[]
    for name,tokens in [('a',['ab','c']),('b',['a','bc'])]:
        path=tmp_path/f'{name}.gguf'
        writer=GGUFWriter(path,'llama')
        writer.add_token_list(tokens)
        writer.add_tensor('test.weight',np.ones((32,32),dtype=np.float32))
        writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
        paths.append(path)
    assert inspect_gguf(paths[0])['tokenizer_sha256'] != inspect_gguf(paths[1])['tokenizer_sha256']
