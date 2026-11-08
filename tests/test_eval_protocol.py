import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("gx", ROOT / "configs/eval/gsm8k_extract.py")
gx = importlib.util.module_from_spec(spec)
_dwb, sys.dont_write_bytecode = sys.dont_write_bytecode, True   # keep configs/eval/ clean
spec.loader.exec_module(gx)
sys.dont_write_bytecode = _dwb


def test_gold():
    assert gx.gold_answer("so 3+4=7\n#### 7") == Decimal("7")
    assert gx.gold_answer("#### 1,234") == Decimal("1234")


def test_strict_last_answer_line_wins():
    c = "Step 1... Answer: 5\nWait, recheck.\nAnswer: $1,250.00"
    assert gx.extract_strict(c) == Decimal("1250")
    assert gx.extract_strict("the result is 42") is None


def test_score_rules():
    g = Decimal("18")
    assert gx.score("16-3-4=9, 9*2=18\nAnswer: 18", g, truncated=False)["strict_correct"]
    s = gx.score("... so 18", g, truncated=False)
    assert not s["strict_correct"] and s["flexible_correct"]           # format miss: strict wrong
    t = gx.score("Answer: 18 and then the model kept going", g, truncated=True)
    assert not t["strict_correct"] and not t["flexible_correct"]       # truncation = incorrect
    assert gx.score("Answer: -3.50", Decimal("-3.5"), False)["strict_correct"]


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
