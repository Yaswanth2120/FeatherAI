import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("gx", ROOT / "configs/eval/gsm8k_extract.py")
gx = importlib.util.module_from_spec(spec)
_dwb, sys.dont_write_bytecode = sys.dont_write_bytecode, True   # keep configs/eval/ clean
spec.loader.exec_module(gx)
sys.dont_write_bytecode = _dwb


D = Decimal


# ---------------------------------------------------------------- gold parsing ("#### N")
@pytest.mark.parametrize("field,want", [
    ("Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n48+24 = <<48+24=72>>72\n#### 72", "72"),
    ("... #### 1,234", "1234"),
    ("... #### -5", "-5"),
    ("... #### 0.5", "0.5"),
])
def test_gold(field, want):
    assert gx.gold_answer(field) == D(want)


def test_gold_missing_raises():
    with pytest.raises(ValueError):
        gx.gold_answer("no marker here")


# ---------------------------------------------------------------- strict extraction fixtures
@pytest.mark.parametrize("completion,want", [
    ("Answer: $1,234", "1234"),
    ("Answer: 1,234", "1234"),
    ("Answer: 18.0", "18"),
    ("Answer: 18", "18"),
    ("Answer: 18.", "18"),                     # trailing period
    ("Answer: -5", "-5"),
    ("Answer: -$5", "-5"),
    ("Answer: $-5", "-5"),
    ("Answer: 1/2", "0.5"),                    # fraction evaluated exactly
    ("Answer: 3 / 4", "0.75"),
    ("Answer: .5", "0.5"),
    ("Answer: 10", "10"),
    ("Answer: 18 apples", "18"),
    ("**Answer:** 18", "18"),                  # Markdown emphasis
    ("Answer: **18**", "18"),
    ("answer: 18", "18"),                      # case-insensitive tag
    ("Answer: 3\nAnswer: 4", "4"),             # two Answer lines: last wins
    ("work... Answer: 7\nLet me double-check.\nAnswer: 9.", "9"),
])
def test_strict_extract(completion, want):
    assert gx.extract_strict(completion) == D(want)


@pytest.mark.parametrize("completion", [
    "Answer:",                                 # no number
    "Answer: none",
    "Answer: eighteen",
    "Answer: 18\nActually I'm not sure.\nAnswer: unsure",   # final Answer has no number
    "Answer: 1/0",                             # division by zero
    "The total is 18.",                        # no Answer tag at all
])
def test_strict_no_prediction(completion):
    assert gx.extract_strict(completion) is None


def test_equivalent_forms_compare_equal():
    assert gx.normalize("18.0") == gx.normalize("18") == gx.normalize("18.")
    assert gx.normalize("1/2") == gx.normalize("0.5")
    assert gx.normalize("-0") == gx.normalize("0")
    assert gx.fmt(gx.normalize("10")) == "10"   # no '1E+1' in logs


# ---------------------------------------------------------------- per-item scoring + summary
def test_score_rules():
    g = D("18")
    s = gx.score("16-3-4=9, 9*2=18\nAnswer: 18", g, truncated=False)
    assert s["strict_correct"] and not s["format_failure"]
    s = gx.score("... so 18", g, truncated=False)              # format miss
    assert not s["strict_correct"] and s["flexible_correct"] and s["format_failure"]
    s = gx.score("Answer: 18 and then the model kept going", g, truncated=True)
    assert not s["strict_correct"] and not s["flexible_correct"] and not s["format_failure"]
    s = gx.score("Answer: unsure", g, truncated=False)          # tag present, no number
    assert s["format_failure"] and s["answer_tag_present"]
    s = gx.score("Answer: 17", g, truncated=False)              # reasoning miss, format fine
    assert not s["strict_correct"] and not s["format_failure"]


def test_summarize_splits_loss():
    g = D("18")
    items = [gx.score("Answer: 18", g, False),                  # correct
             gx.score("Answer: 17", g, False),                  # reasoning error
             gx.score("so the total is 18", g, False),          # format failure, lenient right
             gx.score("Answer: 18 blah", g, True)]              # truncated
    sm = gx.summarize(items)
    assert sm == {"n": 4, "strict_acc": 0.25, "flexible_acc": 0.5, "truncation_rate": 0.25,
                  "format_failure_rate": 0.25, "answer_tag_rate": 0.75}


# ---------------------------------------------------------------- MMLU scoring + token mass
def _mmlu():
    spec = importlib.util.spec_from_file_location("ms", ROOT / "configs/eval/mmlu_score.py")
    m = importlib.util.module_from_spec(spec)
    _d, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    spec.loader.exec_module(m)
    sys.dont_write_bytecode = _d
    return m


def test_mmlu_letter_mass():
    import numpy as np
    ms = _mmlu()
    ids = {"A": 32, "B": 33, "C": 34, "D": 35}
    v = 128256
    logits = np.zeros(v); logits[34] = 20.0                     # confident "C"
    r = ms.score_item(logits, ids, "C")
    assert r["pred"] == "C" and r["correct"] and r["letter_mass"] > 0.99
    logits = np.zeros(v); logits[1000] = 30.0; logits[32] = 1.0  # mass on a non-letter token
    r = ms.score_item(logits, ids, "A")
    assert r["pred"] == "A" and r["letter_mass"] < 1e-9         # right letter, prompt failed
    flat = ms.score_item(np.zeros(v), ids, "B")
    assert abs(flat["letter_mass"] - 4 / v) < 1e-12
    sm = ms.summarize([ms.score_item(np.zeros(v), ids, "A"), r])
    assert sm["n"] == 2 and sm["acc"] == 1.0 and sm["letter_mass_min"] < 1e-9


def test_prompt_templates_have_slots():
    g = (ROOT / "configs/eval/gsm8k_prompt.txt").read_text()
    assert g.count("{question}") == 1 and "Answer: <number>" in g
    m = (ROOT / "configs/eval/mmlu_prompt.txt").read_text()
    for slot in ("{subject}", "{question}", "{A}", "{B}", "{C}", "{D}"):
        assert m.count(slot) == 1


def test_mmlu_token_map():
    t = yaml.safe_load((ROOT / "configs/eval/mmlu_tokens.yaml").read_text())
    assert t["letters"] == {"A": 32, "B": 33, "C": 34, "D": 35}
    assert len(set(t["letters"].values())) == 4


def test_protocol_hashes_match_prereg():
    from harness.sysinfo import sha256_file
    thr = yaml.safe_load((ROOT / "configs/thresholds.yaml").read_text())
    hashes = thr["protocol_files_sha256"]
    assert set(hashes) == {str(p.relative_to(ROOT)) for p in (ROOT / "configs/eval").iterdir()
                           if p.is_file()}
    for rel, h in hashes.items():
        assert sha256_file(ROOT / rel) == h, rel
