import pytest
from backend.optimization.recommender import recommend


def candidate(identifier='q4', **overrides):
    return dict({'id':identifier, 'source_sha256':'source', 'workload_sha256':'suite', 'settings_sha256':'settings',
                 'peak_memory_bytes':1000, 'quality':.8, 'latency_seconds':.5, 'decode_tokens_per_second':40,
                 'file_size_bytes':500, 'measured_trials':5, 'quality_trials':10}, **overrides)


def test_ranking_constraints_and_deterministic_ties():
    configurations = [candidate('q8',decode_tokens_per_second=30), candidate('q4'), candidate('q3',quality=.2)]
    result = recommend(configurations, 2000, 1500, min_quality=.7)
    assert result['selected_id'] == 'q4'
    assert [c['id'] for c in result['eligible']] == ['q4','q8']
    assert result['rejected'][0]['id'] == 'q3'
    assert recommend([candidate('b'),candidate('a')],2000,1500)['selected_id'] == 'a'


def test_constraints_apply_available_memory_and_ci_lower_bound():
    result = recommend([candidate(quality_ci=[.5,.95])],2000,1500,min_quality=.7)
    assert result['status'] == 'no_eligible_configuration'
    assert recommend([candidate()],900,1500)['status'] == 'no_eligible_configuration'
    assert recommend([candidate()],2000,1500,max_latency_seconds=.1)['status'] == 'no_eligible_configuration'


def test_no_fabrication_for_missing_memory_or_nan():
    assert recommend([],2000,1500)['status'] == 'insufficient_evidence'
    for key,value in [('peak_memory_bytes',None),('quality',float('nan')),('decode_tokens_per_second',float('inf')),('file_size_bytes',0),('quality_trials',0)]:
        data = candidate()
        data[key] = value
        result = recommend([data],2000,1500)
        assert result['status'] == 'insufficient_evidence'
        assert result['rejected'][0]['evidence_missing']


def test_different_source_or_workload_never_compared():
    for key in ['source_sha256','workload_sha256','settings_sha256']:
        result = recommend([candidate('q4'), candidate('q8', **{key:'other'})],2000,1500)
        assert result['status'] == 'insufficient_evidence'
        assert not result['eligible']


def test_all_objectives_and_validation():
    data = [candidate('a'), candidate('b',file_size_bytes=100, latency_seconds=.1,quality=.9)]
    for objective in ['smallest','lowest_latency','highest_quality']:
        assert recommend(data,2000,1500,objective=objective)['selected_id'] == 'b'
    for kwargs in [{'min_quality':2},{'available_memory_bytes':0},{'objective':'made-up'}, {'max_latency_seconds':float('nan')}]:
        arguments = {'available_memory_bytes':2000,'max_memory_bytes':1500, **kwargs}
        with pytest.raises(ValueError):
            recommend(data, **arguments)
    assert recommend([candidate(),candidate()],2000,1500)['status'] == 'insufficient_evidence'


def evidence_pair():
    benchmark = {'id':'bench', 'metadata': {'model':{'artifact_sha256':'artifact','size':100}, 'source_sha256':'source',
                 'prompt_sha256':'prompt', 'settings_sha256':'settings', 'peak_memory_bytes':1000,'repeats':2,'warmups':1},
                 'trials':[{'warmup':True,'wall_seconds':10},
                           {'warmup':False,'wall_seconds':.5,'decode_tokens_per_second':40},
                           {'warmup':False,'wall_seconds':.7,'decode_tokens_per_second':30}]}
    quality = {'id':'eval','status':'completed','created_at':'2026-11-09','suite':'reasoning-small',
               'model':{'artifact_sha256':'artifact'}, 'source_sha256':'source', 'suite_sha256':'suite',
               'options':{'seed':42,'temperature':0},'runtime_version':'1','repeats':1,
               'results':[{'status':'scored','passed':True},{'status':'scored','passed':False}]}
    return benchmark, quality


def test_evidence_join_has_actual_quality_uncertainty_and_metrics():
    from backend.optimization.evidence import configurations_from_reports
    benchmark,quality = evidence_pair()
    [config] = configurations_from_reports([benchmark],[quality])
    assert config['quality'] == .5
    assert config['quality_ci'][0] < .5 < config['quality_ci'][1]
    assert config['latency_seconds'] == pytest.approx(.6)
    assert config['decode_tokens_per_second'] == 35
    assert config['measured_trials'] == 2
    assert config['quality_trials'] == 2
    assert config['evidence']['evaluation_id'] == 'eval'
    assert recommend([config],2000,1500)['status'] == 'completed'


def test_evidence_never_joins_by_model_name_or_incomplete_quality():
    from backend.optimization.evidence import configurations_from_reports
    benchmark,quality = evidence_pair()
    for changes in [{'source_sha256':'other'}, {'model':{'artifact_sha256':'other'}}, {'status':'partial'},
                    {'results':[{'status':'failed'}]}]:
        [config] = configurations_from_reports([benchmark],[dict(quality,**changes)])
        assert config['quality'] is None
        assert config['quality_trials'] == 0
        assert recommend([config],2000,1500)['status'] == 'insufficient_evidence'


def test_evidence_rejects_interrupted_benchmark_and_selects_newest_quality():
    from backend.optimization.evidence import configurations_from_reports
    benchmark,quality = evidence_pair()
    benchmark['trials'].pop()
    [config] = configurations_from_reports([benchmark],[quality,dict(quality,id='later',created_at='2026-11-10')])
    assert config['measured_trials'] == 0
    assert config['evidence']['evaluation_id'] == 'later'
    assert config['evidence']['benchmark_complete'] is False


def test_evidence_identity_covers_quality_settings_and_suite():
    from backend.optimization.evidence import configurations_from_reports
    benchmark,quality = evidence_pair()
    [a] = configurations_from_reports([benchmark],[quality])
    [b] = configurations_from_reports([benchmark],[dict(quality,options={'seed':43})])
    assert a['settings_sha256'] != b['settings_sha256']
    [c] = configurations_from_reports([benchmark],[dict(quality,suite_sha256='other')])
    assert a['workload_sha256'] != c['workload_sha256']


def test_zero_quality_remains_rankable_only_as_explicit_diagnostic():
    from backend.evaluation.scoring import wilson_interval
    result = recommend([candidate(quality=0,quality_ci=wilson_interval(0,3))],2000,1500,min_quality=0)
    assert result['status'] == 'completed'
    assert any('diagnostic' in warning for warning in result['limitations'])
    assert any('GPU' in warning for warning in result['limitations'])


def test_evidence_keeps_memory_measurement_provenance():
    from backend.optimization.evidence import configurations_from_reports
    benchmark,quality = evidence_pair()
    benchmark['metadata']['memory_metric'] = 'sampled_process_rss'
    benchmark['metadata']['memory_note'] = 'GPU allocation is not measured'
    [config] = configurations_from_reports([benchmark],[quality])
    assert config['evidence']['memory_metric'] == 'sampled_process_rss'
    assert config['evidence']['memory_note'] == 'GPU allocation is not measured'
