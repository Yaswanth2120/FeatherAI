from collections import Counter
import numpy as np
import pytest
from backend.diagnostics.memory import delta, parse_footprint, parse_swapusage, parse_time_l, parse_vm_stat
from backend.study.benchmark import analyze, round_orders, session_flags
from backend.study.perplexity import parse as parse_ppl
from backend.study.server import buffer_sizes, metal_evidence
from backend.study.stats import describe, median_ci, ratio_ci
import re

TIME_L = """        1.01 real         0.55 user         0.18 sys
           513556480  maximum resident set size
                   0  average shared memory size
               32718  page reclaims
                3198  page faults
                5833  involuntary context switches
         11946873769  instructions retired
           492946488  peak memory footprint
"""


def test_parsers():
    parsed = parse_time_l(TIME_L)
    assert parsed['max_rss_bytes'] == 513556480 and parsed['peak_memory_footprint_bytes'] == 492946488
    assert parsed['real_seconds'] == 1.01
    assert parse_footprint('sleep [1]: 64-bit    Footprint: 800 KB (16384 bytes per page)') == 800 * 1024
    assert parse_footprint('x Footprint: 1.5 GB') == int(1.5 * 1024 ** 3)
    assert parse_swapusage('vm.swapusage: total = 3072.00M  used = 2603.44M  free = 468.56M  (encrypted)')['used_bytes'] == int(2603.44 * 2**20)
    vm = parse_vm_stat('Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:  5756.\nPageins:  10.\nSwapouts: 3.\n')
    assert vm['page_size'] == 16384 and vm['pages']['pageins'] == 10 and vm['pages']['swapouts'] == 3


def test_delta_and_flags():
    before = {'time': 0, 'swap': {'used_bytes': 0}, 'pressure_level': 1, 'vm_stat': {'page_size': 16384, 'pages': {'pageins': 1}}}
    after = {'time': 5, 'swap': {'used_bytes': 100 * 2**20}, 'pressure_level': 1, 'vm_stat': {'page_size': 16384, 'pages': {'pageins': 4}}}
    d = delta(before, after)
    assert d['swap_used_delta_bytes'] == 100 * 2**20 and d['pageins'] == 3
    record = {'system_delta': d, 'system_before': {'pressure_level': 1}, 'system_after': {'pressure_level': 1},
              'metal': {'verified': True}, 'docker_vm_running': False}
    assert session_flags(record) == ['memory_pressure:swap_growth']
    record['system_delta'] = {**d, 'swap_used_delta_bytes': 0}
    record['system_after'] = {'pressure_level': 2}
    record['metal'] = {'verified': False}
    assert session_flags(record) == ['memory_pressure:level', 'metal_unverified']


def test_round_orders_balance_variants_within_rounds():
    orders = round_orders(['F16', 'Q8_0', 'Q4_K_M'], 6, 20261109)
    assert all(sorted(o) == ['F16', 'Q4_K_M', 'Q8_0'] for o in orders)
    assert len({tuple(o) for o in orders}) == 6, 'six rounds cover every permutation once'
    assert round_orders(['a', 'b', 'c'], 5, 1) == round_orders(['a', 'b', 'c'], 5, 1)
    first_position = Counter(o[0] for o in orders)
    assert set(first_position.values()) == {2}


def test_describe_and_cluster_bootstrap():
    values = [10.0, 11.0, 12.0, 13.0, 100.0]
    d = describe(values)
    assert d['median'] == 12 and d['min'] == 10 and d['max'] == 100 and d['n'] == 5
    assert d['sd'] == pytest.approx(np.std(values, ddof=1))
    groups = [[100, 101, 99, 100], [102, 103, 101, 102], [98, 99, 97, 98]]
    low, high = median_ci(groups, resamples=2000)
    assert low <= 100.5 <= high and high - low < 10
    single = median_ci([[5, 5, 5]], resamples=200)
    assert single == [5, 5]
    ratio = ratio_ci([[200, 200], [202, 198]], [[100, 100], [101, 99]], resamples=2000)
    assert ratio['ratio'] == pytest.approx(2.0) and ratio['ci95'][0] <= 2.0 <= ratio['ci95'][1]
    with pytest.raises(ValueError):
        median_ci([[]])


def session(variant, values, flags=(), error=None, first=False, load=1.0, peak=100):
    trials = [{'warmup': True, 'decode_tokens_per_second': 1.0}] + [
        {'warmup': False, 'decode_tokens_per_second': v, 'prefill_tokens_per_second': v * 10, 'ttft_seconds': .1, 'wall_seconds': 1}
        for v in values]
    return {'variant': variant, 'trials': trials, 'flags': list(flags), 'error': error, 'first_load_in_study': first,
            'server': {'load_seconds': load, 'time_l': {'peak_memory_footprint_bytes': peak, 'max_rss_bytes': peak + 1}}}


def test_analysis_excludes_warmups_failures_and_flagged_sessions_in_sensitivity():
    sessions = [session('F16', [100, 101], first=True, load=3), session('F16', [99, 100]),
                session('Q4', [150, 151], first=True), session('Q4', [10, 10], flags=['memory_pressure:level']),
                session('Q4', [], error='crash')]
    result = analyze(sessions, ['F16', 'Q4'])
    primary, sensitivity = result['primary_all_sessions'], result['sensitivity_unflagged_sessions']
    assert primary['F16']['decode_tokens_per_second']['n'] == 4
    assert primary['Q4']['decode_tokens_per_second']['n'] == 4 and primary['Q4']['failed_sessions'] == 1
    assert sensitivity['Q4']['decode_tokens_per_second']['n'] == 2
    assert sensitivity['Q4']['decode_tokens_per_second']['median'] == 150.5
    assert 'ratio_to_F16' in primary['Q4']['decode_tokens_per_second']
    assert primary['F16']['load_seconds_first_in_study']['median'] == 3
    assert primary['F16']['load_seconds_repeat']['median'] == 1


def test_server_log_evidence():
    log = ('ggml_metal_device_init: GPU name:   MTL0 (Apple M2)\nload_tensors: offloaded 25/25 layers to GPU\n'
           'load_tensors:   CPU_Mapped model buffer size =    54.00 MiB\nload_tensors:  MTL0 model buffer size =   256.63 MiB\n'
           'llama_kv_cache:       MTL0 KV buffer size =    45.00 MiB\n'
           'llama_kv_cache: size =   45.00 MiB (  2048 cells,  30 layers,  1/1 seqs), K (f16):   22.50 MiB, V (f16):   22.50 MiB\n')
    evidence = metal_evidence(log, re.search(r"offloaded (\d+)/(\d+) layers to GPU", log))
    assert evidence == {'device': 'Apple M2', 'offloaded_layers': [25, 25], 'verified': True}
    sizes = buffer_sizes(log)
    assert sizes['model_MTL0'] == 256.63 and sizes['kv_MTL0'] == 45.0 and sizes['kv_cells'] == 2048 and sizes['kv_type_k'] == 'f16'
    assert metal_evidence('no gpu', None)['verified'] is False


def test_perplexity_parser():
    text = ('Final estimate: PPL = 3.1234 +/- 0.05678\n'
            '====== Perplexity statistics ======\nMean PPL(Q)                   :   3.200000 ±   0.060000\n'
            'Mean PPL(base)                :   3.123400 ±   0.056780\nMean PPL(Q)/PPL(base)         :   1.024500 ±   0.002000\n'
            '====== KL divergence statistics ======\nMean    KLD:   0.012345 ±   0.000500\nMaximum KLD:   2.500000\n'
            '99.0%   KLD:   0.150000\nMedian  KLD:   0.004000\n====== Token probability statistics ======\n'
            'RMS Δp    :  3.210 ±  0.050 %\nSame top p: 95.123 ± 0.200 %\n')
    parsed = parse_ppl(text)
    assert parsed['ppl'] == 3.1234 and parsed['mean_kld'] == 0.012345 and parsed['same_top_p'] == 95.123
    assert parsed['ppl_ratio'] == 1.0245 and parsed['kld_p99'] == 0.15 and parsed['rms_delta_p'] == 3.21


def test_server_stop_after_rejected_start_records_unstarted(tmp_path, monkeypatch):
    from backend.study import server as module
    monkeypatch.setattr(module, 'sha256', lambda path: 'x')
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: type('R', (), {'stdout': 'version: test', 'stderr': ''})())
    monkeypatch.setattr(module, 'system_snapshot', lambda: {'available_bytes': 1})
    session = module.ServerSession({'path': '/x.gguf', 'size_bytes': 10**9}, '/bin/echo', tmp_path / 'log')
    with pytest.raises(MemoryError, match='rejected before start'):
        session.start()
    assert session.stop()['started'] is False


def test_incomplete_evaluation_reports_no_accuracy():
    from backend.evaluation.scoring import category_summary
    rows = [{'status': 'scored', 'passed': True}, {'status': 'failed'}]
    summary = category_summary(rows)
    assert summary['accuracy'] is None and summary['accuracy_ci95'] is None and summary['complete'] is False
    assert summary['accuracy_over_scored_items'] == 1.0
    assert category_summary([{'status': 'scored', 'passed': True}])['accuracy'] == 1.0


def test_context_manager_never_starts_a_started_session(tmp_path, monkeypatch):
    from backend.study import server as module
    monkeypatch.setattr(module, 'sha256', lambda path: 'x')
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: type('R', (), {'stdout': 'v', 'stderr': ''})())
    session = module.ServerSession({'path': '/x.gguf', 'size_bytes': 1}, '/bin/echo', tmp_path / 'log')
    starts = []
    def fake_start():
        starts.append(1)
        session.wrapper = object()
        return session
    session.start = fake_start
    session.record = {'done': True}  # avoid stop() in __exit__
    session.start()
    with session as entered:
        assert entered is session
    assert len(starts) == 1


def test_benchmark_summary_labels_method_and_only_clusters_sessions_v2():
    from backend.benchmarks.runner import summarize
    trials = [{'warmup': True, 'session': 1}] + [
        {'warmup': False, 'session': s, 'decode_tokens_per_second': v, 'wall_seconds': 1, 'ttft_seconds': .1, 'prefill_tokens_per_second': 9}
        for s, values in [(1, [100, 101]), (2, [110, 111])] for v in values]
    legacy = summarize(trials)
    assert legacy['method'] == 'legacy-sampled-rss-v1' and 'median_ci95' not in legacy['decode_tokens_per_second']
    v2 = summarize(trials, {'method': 'sessions-v2'})
    assert v2['method'] == 'sessions-v2' and v2['decode_tokens_per_second']['sessions_used'] == 2
    low, high = v2['decode_tokens_per_second']['median_ci95']
    assert low <= v2['decode_tokens_per_second']['median'] <= high


def test_server_session_stays_in_callers_process_group_and_uses_path_lookup(tmp_path, monkeypatch):
    from backend.study import server as module
    monkeypatch.setattr(module, 'sha256', lambda path: 'x')
    real_run = module.subprocess.run
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: type('R', (), {'stdout': 'v', 'stderr': ''})())
    # A bare name is looked up on PATH, never in the current directory.
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'llama-server').write_text('#!/bin/sh\n')
    monkeypatch.setattr(module.shutil, 'which', lambda name: None)
    with pytest.raises(ValueError, match='not found'):
        module.ServerSession({'path': '/x.gguf', 'size_bytes': 1}, 'llama-server', tmp_path / 'log')
    monkeypatch.setattr(module.shutil, 'which', lambda name: '/usr/local/bin/llama-server')
    session = module.ServerSession({'path': '/x.gguf', 'size_bytes': 1}, 'llama-server', tmp_path / 'log')
    assert session.binary == '/usr/local/bin/llama-server'
    # The server must not get its own session: API job cancellation kills the worker's process group.
    captured = {}
    class Stop(Exception):
        pass
    def popen(argv, **kwargs):
        captured.update(kwargs)
        raise Stop
    monkeypatch.setattr(module.subprocess, 'Popen', popen)
    monkeypatch.setattr(module, 'system_snapshot', lambda: {'available_bytes': 10**12})
    with pytest.raises(Stop):
        session.start()
    assert not captured.get('start_new_session') and 'preexec_fn' not in captured


def test_holm_and_coding_v2_analysis_rules():
    from backend.study.coding_v2_run import holm, analyze
    adjusted = holm({'a': 0.01, 'b': 0.04})
    assert adjusted['a']['holm_adjusted_p'] == pytest.approx(0.02) and adjusted['a']['reject_at_family_0.05']
    assert adjusted['b']['holm_adjusted_p'] == pytest.approx(0.04) and adjusted['b']['reject_at_family_0.05']
    assert holm({'a': 0.03, 'b': 0.04})['a']['reject_at_family_0.05'] is False
    rows = lambda passes: {'results': [{'task_id': str(i), 'passed': p} for i, p in enumerate(passes)]}
    same = [True] * 150 + [False] * 210
    result = analyze({'F16': rows(same), 'Q8_0': rows(same), 'Q4_K_M': rows([False] * 40 + same[40:])})
    assert result['baseline_informative'] and result['comparisons']['Q8_0']['equivalence_label'] == 'equivalent within +/-10 pp'
    q4 = result['comparisons']['Q4_K_M']
    assert q4['difference'] == pytest.approx(-40 / 360) and q4['reject_at_family_0.05'] and q4['equivalence_label'] == 'not shown equivalent'
