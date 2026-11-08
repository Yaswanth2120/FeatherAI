from pathlib import Path

import pytest
import yaml

from harness import gates, sysinfo
from harness.llama_server import build_cmd, check_metal, parse_sse_events

FX = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).resolve().parent.parent


def fx(name):
    return (FX / name).read_text()


# ---------------------------------------------------------------- parsers (real fixtures)
def test_swapusage():
    s = sysinfo.parse_swapusage(fx("swapusage.txt"))
    assert set(s) == {"total_mb", "used_mb", "free_mb"}
    assert s["total_mb"] == pytest.approx(s["used_mb"] + s["free_mb"], abs=0.02)


def test_vm_stat():
    v = sysinfo.parse_vm_stat(fx("vm_stat.txt"))
    assert v["page_size"] == 16384
    for k in ("pageins", "pageouts", "swapins", "swapouts", "pages_free",
              "pages_inactive", "pages_speculative", "translation_faults"):
        assert isinstance(v[k], int)
    assert sysinfo.available_bytes(v) > 0


def test_therm_ok_fixture():
    assert sysinfo.parse_pmset_therm(fx("pmset_therm_ok.txt"))["ok"] is True


@pytest.mark.parametrize("text", ["", "Thermal warning level set to 1.\n",
                                  "Note: No thermal warning level has been recorded\nCPU_Speed_Limit = 70\n"])
def test_therm_fails_closed(text):
    assert sysinfo.parse_pmset_therm(text)["ok"] is False


def test_batt_and_lpm():
    b = sysinfo.parse_pmset_batt(fx("pmset_batt_battery.txt"))
    assert b == {"ac_power": False, "battery_power": True}
    assert sysinfo.parse_pmset_batt("Now drawing from 'AC Power'\n")["ac_power"] is True
    assert sysinfo.parse_lowpowermode(fx("pmset_g_lpm1.txt")) == 1
    assert sysinfo.parse_lowpowermode(" lowpowermode         0\n") == 0
    assert sysinfo.parse_lowpowermode("nothing\n") is None


def test_time_l_mixed_with_log():
    t = sysinfo.parse_time_l("srv: some log line\n12 garbage\n" + fx("time_l.txt"))
    assert t["peak_footprint_bytes"] == 852160
    assert t["max_rss_bytes"] == 1163264
    assert t["page_faults"] == 1 and t["page_reclaims"] == 211
    assert t["real_s"] == pytest.approx(0.12)
    with pytest.raises(ValueError):
        sysinfo.parse_time_l("no report here")


# ---------------------------------------------------------------- llama-server helpers
def test_sse():
    lines = [b'data: {"content":"a","stop":false}\n', b"\n",
             b'data: {"content":"","stop":true,"timings":{"predicted_n":2}}\n']
    evs = list(parse_sse_events(lines))
    assert len(evs) == 2 and evs[1]["timings"]["predicted_n"] == 2


def test_check_metal():
    log = (FX / "server_lv4_excerpt.txt").read_text()
    m = check_metal(log)
    assert m["ok"] and m["metal_device"] == "Apple M2" and m["offloaded_layers"] == [7, 7]
    assert m["embed_library"] == 1 and m["mtl0_buffers_mib"]["model"] == 12.56
    assert not check_metal("load_tensors: offloaded 0/29 layers to GPU")["ok"]
    assert not check_metal("offloaded 29/29 layers to GPU")["ok"]  # no Metal device line


def test_build_cmd_modes():
    cfg = yaml.safe_load((ROOT / "configs/bench.yaml").read_text())
    nm = build_cmd(cfg, Path("/m.gguf"), "no_mmap")
    mm = build_cmd(cfg, Path("/m.gguf"), "mmap")
    assert nm[:2] == ["/usr/bin/time", "-l"]
    assert nm[-2:] == ["--load-mode", "none"] and mm[-2:] == ["--load-mode", "mmap"]
    assert "--cache-ram" in nm and "--no-cache-prompt" in nm
    assert nm[nm.index("--fit") + 1] == "off" and nm[nm.index("-lv") + 1] == "4"
    with pytest.raises(ValueError):
        build_cmd(cfg, Path("/m.gguf"), "other")


# ---------------------------------------------------------------- gates
def thr(**over):
    t = yaml.safe_load((ROOT / "configs/thresholds.yaml").read_text())
    t["validity_gates"].update(over)
    return t


def snap(swap_used=10.0, ac=True, lpm=0, therm_ok=True, avail=6 * 2**30, pageins=100):
    return {"swap": {"used_mb": swap_used}, "power": {"ac_power": ac}, "lowpowermode": lpm,
            "therm": {"ok": therm_ok, "unexpected_lines": []}, "available_bytes": avail,
            "vm_stat": {"pageins": pageins, "pageouts": 0, "swapins": 0, "swapouts": 0}}


def test_thresholds_file_values():
    g = thr()["validity_gates"]
    assert g["swap_used_at_start_max_mb"] == 256
    assert g["swap_growth_during_run_max_mb"] == 64
    assert g["cooldown_between_runs_s"] == 300
    assert g["free_memory_headroom_mb"] == 1024
    l2 = thr()["level2_criteria"]
    assert l2["memory_reduction_min_frac"] == 0.25
    assert l2["decode_tps_min_frac_of_baseline"] == 0.90
    assert l2["baseline"]["ctx"] == 2048 and l2["baseline"]["quant"] == "Q4_K_M"
    assert l2["kl_divergence"]["margin"] == 0.02
    assert l2["accuracy"]["point_loss_max_pp"] == 2 and l2["accuracy"]["ci_upper_loss_max_pp"] == 4


def test_pre_run_clean():
    assert gates.pre_run_failures(snap(), thr(free_memory_headroom_mb=512), 2**30, []) == []


def test_pre_run_each_gate():
    t = thr(free_memory_headroom_mb=512)
    assert any("AC" in f for f in gates.pre_run_failures(snap(ac=False), t, None, []))
    assert any("Low Power" in f for f in gates.pre_run_failures(snap(lpm=1), t, None, []))
    assert any("thermal" in f for f in gates.pre_run_failures(snap(therm_ok=False), t, None, []))
    assert any("swap" in f for f in gates.pre_run_failures(snap(swap_used=256.1), t, None, []))
    assert gates.pre_run_failures(snap(swap_used=256.0), t, None, []) == []
    assert any("available" in f for f in gates.pre_run_failures(snap(avail=2**30), t, 2**30, []))
    assert any("VS Code" in f for f in gates.pre_run_failures(snap(), t, None, ["VS Code"]))


def test_pre_run_null_headroom_refuses():
    assert any("null" in f for f in gates.pre_run_failures(
        snap(), thr(free_memory_headroom_mb=None), 2**30, []))


def _server(pre_pi, load_pi, warm_pi, meas_pi, t_load=10.0, t_warm=12.0, t_meas=42.0, **over):
    """Server stub whose phase snapshots carry cumulative vm_stat pageins."""
    ph = {"after_load": {"t_monotonic": t_load, "snap": snap(pageins=pre_pi + load_pi)},
          "after_warmup": {"t_monotonic": t_warm, "snap": snap(pageins=pre_pi + load_pi + warm_pi)},
          "after_measured": {"t_monotonic": t_meas,
                             "snap": snap(pageins=pre_pi + load_pi + warm_pi + meas_pi)}}
    return {"error": None, "metal": {"ok": True}, "time_l": {"peak_footprint_bytes": 1},
            "phase_snapshots": ph} | over


def test_load_pageins_do_not_contaminate():
    t = thr(pagein_rate_contaminated_pages_per_s=10.0)
    srv = _server(0, load_pi=200_000, warm_pi=50, meas_pi=150)       # 150 / 30 s = 5 /s
    v = gates.post_run_validity(snap(pageins=0), snap(pageins=200_210), srv, t, 45.0,
                                t_pre=0.0, t_post=45.0)
    assert v["valid"] and v["paging_contaminated"] is False
    assert v["load_pageins"] == 200_000
    assert v["measured_window_pageins"] == 150 and v["measured_window_s"] == 30.0
    assert v["pagein_rate_pages_per_s"] == 5.0
    assert v["phase_paging"]["shutdown"]["pageins"] == 10
    assert set(v["phase_paging"]) == {"load", "warmup", "measured", "shutdown"}


def test_measured_window_contaminated():
    t = thr(pagein_rate_contaminated_pages_per_s=10.0)
    srv = _server(0, load_pi=0, warm_pi=0, meas_pi=301)              # 301 / 30 s > 10 /s
    v = gates.post_run_validity(snap(pageins=0), snap(pageins=301), srv, t, 45.0, 0.0, 45.0)
    assert v["valid"] and v["paging_contaminated"] is True


def test_post_run_swap_and_failures():
    t = thr(pagein_rate_contaminated_pages_per_s=10.0)
    srv = _server(0, 0, 0, 0)
    assert gates.post_run_validity(snap(swap_used=0), snap(swap_used=64), srv, t, 45, 0, 45)["valid"]
    assert not gates.post_run_validity(snap(swap_used=0), snap(swap_used=64.1), srv, t, 45, 0, 45)["valid"]
    v = gates.post_run_validity(snap(), snap(), srv | {"metal": {"ok": False}}, t, 45, 0, 45)
    assert not v["valid"]
    # request failed before the measured window completed -> invalid, rate unknown
    srv2 = _server(0, 0, 0, 0)
    del srv2["phase_snapshots"]["after_measured"]
    v = gates.post_run_validity(snap(), snap(), srv2, t, 45, 0, 45)
    assert not v["valid"] and v["pagein_rate_pages_per_s"] is None and v["paging_contaminated"] is None
    # uncalibrated threshold -> flag unknown, not silently False
    assert gates.post_run_validity(snap(), snap(), srv, thr(), 45, 0, 45)["paging_contaminated"] is None


def test_pilot_window_uses_whole_window():
    t = thr(pagein_rate_contaminated_pages_per_s=10.0)
    v = gates.post_run_validity(snap(pageins=0), snap(pageins=600), None, t, 300.0)
    assert v["pagein_rate_pages_per_s"] == 2.0 and v["phase_paging"] is None


def test_token_check():
    import hashlib
    from harness.cli import token_check, expected_tokens
    exp = expected_tokens("llama3-bpe")
    assert exp["A0"]["tokens"] > 0
    prompts = [{"id": "A0", "prompt": __import__("json").loads(
        (ROOT / "configs/prompt_sets/A.jsonl").read_text().splitlines()[0])["prompt"]}]
    ok = token_check({"warmup": [], "requests": [{"prompt_id": "A0",
                                                  "prompt_n": exp["A0"]["tokens"]}]},
                     prompts, "llama3-bpe")
    assert ok["mismatches"] == []
    bad = token_check({"warmup": [], "requests": [{"prompt_id": "A0", "prompt_n": 1}]},
                      prompts, "llama3-bpe")
    assert bad["mismatches"] == ["A0"]
    stale = token_check({"warmup": [], "requests": [{"prompt_id": "A0",
                                                     "prompt_n": exp["A0"]["tokens"]}]},
                        [{"id": "A0", "prompt": "edited text"}], "llama3-bpe")
    assert stale["mismatches"] == ["A0"] and stale["per_prompt"][0]["count_stale"]


def test_bench_ctx_and_kv():
    cfg = yaml.safe_load((ROOT / "configs/bench.yaml").read_text())
    assert cfg["llama_server"]["ctx_size"] == 2048
    cmd = build_cmd(cfg, Path("/m.gguf"), "no_mmap")
    assert cmd[cmd.index("-c") + 1] == "2048"
    assert cmd[cmd.index("-ctk") + 1] == "f16" and cmd[cmd.index("-ctv") + 1] == "f16"


def test_find_nulls():
    assert gates.find_nulls({"a": None, "b": {"c": None, "d": 1}}) == ["a", "b.c"]
