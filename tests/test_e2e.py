"""task08 gate probes against the committed real workflow evidence (local artifacts required)."""
import copy
import json
from pathlib import Path
import pytest
import automation.validate as validate
from backend.e2e import summary_evidence

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'experiments' / 'e2e-evidence-v0.2.json'
data = json.loads(EVIDENCE.read_text()) if EVIDENCE.is_file() else None
record = json.loads((ROOT / data['workflow']).read_text()) if data else None
local = bool(record) and Path(record['stages']['import']['source']['path']).is_file() and all(
    Path(a['artifact_path']).is_file() for a in record['stages']['quantization']['artifacts'].values())
requires_local = pytest.mark.skipif(not local, reason='Workflow model artifacts are local-only (ignored by Git)')


@requires_local
def test_real_workflow_passes_gate():
    validate.validate_workflow(data)
    assert summary_evidence((ROOT / data['workflow']).parent) == data


@requires_local
@pytest.mark.parametrize('mutate,message', [
    (lambda r: r['stages']['recommendation']['request'].update(min_quality=0.0), 'pre-registered'),
    (lambda r: r['stages']['benchmark']['results']['Q8_0'].update(artifact_sha256='0' * 64), 'lineage'),
    (lambda r: r['stages']['evaluation']['results']['Q4_K_M']['sandbox'].update(status='blocked'), 'sandboxed'),
    (lambda r: r['stages']['export'].update(workflow_id='other'), 'Stage export'),
    (lambda r: r['stages']['evaluation']['results']['Q8_0']['summary'].update(scored=29), 'not all scored'),
    (lambda r: r['stages']['quantization']['artifacts']['Q4_K_M'].update(source_sha256='1' * 64), 'imported source'),
    (lambda r: r.update(status='failed'), 'identity or status'),
])
def test_gate_rejects_tampered_workflow(mutate, message):
    tampered = copy.deepcopy(record)
    mutate(tampered)
    with pytest.raises(ValueError, match=message):
        validate.validate_workflow(data, tampered)


def test_gate_rejects_workflow_outside_e2e_directory():
    with pytest.raises(ValueError, match='inside experiments/e2e'):
        validate.validate_workflow({'workflow': 'experiments/studies/smollm2-135m/study.json'})
