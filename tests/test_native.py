import json
import os
from types import SimpleNamespace
import httpx
import pytest
from backend.backends.llama_cpp import LlamaCpp
from backend.benchmarks.system_monitor import ProcessMonitor
from backend.storage.evaluations import EvaluationStore


def adapter(body):
    b = LlamaCpp.__new__(LlamaCpp)
    b.artifact = {'path': '/model/source.gguf', 'size_bytes': 50, 'sha256': 'a'*64, 'tensor_types': {'F16': 1}, 'parameter_count': 32}
    b.context = 1024
    b.client = httpx.Client(base_url='http://localhost', transport=httpx.MockTransport(lambda request: httpx.Response(200, text=body)))
    return b


def test_native_stream_keeps_raw_timings_and_first_chunk():
    body='\n'.join(['data: '+json.dumps({'content':'answer','stop':False}), 'data: '+json.dumps({'content':'','stop':True,'timings':{'predicted_n':5,'predicted_ms':200,'prompt_n':10,'prompt_ms':20}})])
    b=adapter(body)
    try:
        result=b.generate('source.gguf','question',{'num_ctx':1024})
        assert result['output']=='answer'
        assert result['runtime']['eval_duration']==200000000
        assert result['ttft_seconds']>=0
    finally:
        b.client.close()


def test_native_rejects_incomplete_stream():
    b=adapter('data: {"content":"partial","stop":false}')
    try:
        with pytest.raises(RuntimeError,match='Incomplete'):
            b.generate('source.gguf','question',{'num_ctx':1024})
    finally:
        b.client.close()


def test_native_requires_owned_model_and_context():
    b=adapter('')
    try:
        with pytest.raises(ValueError):
            b.generate('other.gguf','question',{})
        with pytest.raises(ValueError):
            b.generate('source.gguf','question',{'num_ctx':2048})
    finally:
        b.client.close()


def test_process_monitor_records_owned_rss():
    with ProcessMonitor(os.getpid()) as monitor:
        pass
    assert monitor.peak>0
    assert monitor.samples>=2


def test_evaluation_store_survives_reopen(tmp_path):
    path=tmp_path/'data.sqlite'
    store=EvaluationStore(path)
    store.save({'id':'a','created_at':'2026-11-09','status':'unsupported'})
    assert EvaluationStore(path).list()[0]['status']=='unsupported'


def test_native_close_reaps_process_and_closes_client_and_log():
    b=adapter('')
    class Process:
        stopped=False
        waited=False
        def poll(self): return None
        def terminate(self): self.stopped=True
        def wait(self, timeout): self.waited=True
    b.process=Process()
    b.log=SimpleNamespace(close=lambda: None)
    b.close()
    assert b.process.stopped and b.process.waited and b.client.is_closed


def test_study_does_not_label_failed_benchmarks_or_partial_quality_complete():
    from backend.lab import study_status
    good={'metadata':{'repeats':1},'summary':{'successful_trials':1,'failed_trials':0},'trials':[{'warmup':False}]}
    assert study_status({'benchmark_reports':[good], 'evaluations':[{'suite':'reasoning-small','status':'completed'},{'suite':'coding-small','status':'unsupported'}]})['status']=='completed_with_limitations'
    state={'benchmark_reports':[good], 'evaluations':[{'suite':'coding-small','status':'partial'}]}
    assert study_status(state)=={'coding_evaluation_status':'partial','status':'partial'}
    bad={'metadata':{'repeats':1},'summary':{'successful_trials':0,'failed_trials':1},'trials':[{'error':'OOM'}]}
    assert study_status({'benchmark_reports':[bad], 'evaluations':[{'suite':'coding-small','status':'unsupported'}]})['status']=='partial'
