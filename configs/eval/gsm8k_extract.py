"""GSM8K answer extraction and scoring. Frozen by prereg-v1 (sha256 in thresholds.yaml).

Rules
- Final answer = the LAST occurrence of "Answer:" (case-insensitive) in the
  completion. The number must follow it, optionally wrapped in Markdown
  emphasis (*, _), a currency sign ($), and/or a sign (-$5 or $-5).
  If the last "Answer:" has no parseable number, there is no prediction, even
  if an earlier "Answer:" line had one (the model's final word counts).
- The prediction is the FIRST number immediately after the last "Answer:";
  anything after it is ignored ("Answer: 5 + 3 = 8" -> 5, "Answer: 18 apples"
  -> 18, "Answer: 25%" -> 25). Any word before the number is a format failure
  ("Answer: about 18" -> no prediction).
- Numbers: thousands commas allowed; trailing period ignored; 18.0 == 18.
  A simple fraction a/b is evaluated exactly (1/2 == 0.5); b == 0 is no number.
- Truncated completions (hit max_new_tokens without end-of-generation or a stop
  sequence) are incorrect whatever they contain.

Scores per item
- strict_correct (GATED): not truncated and the final-Answer value == gold.
- flexible_correct (reported, never gated): not truncated and the last number
  anywhere in the completion == gold.
- truncated, format_failure (not truncated and no parseable final Answer):
  reported, never gated.
"""
import re
from decimal import Decimal, InvalidOperation

_NUM = r"-?\$?-?(?:\d[\d,]*)?\.?\d+(?:\s*/\s*\d+)?"
ANSWER_TAG_RE = re.compile(r"answer\s*:", re.IGNORECASE)
AFTER_TAG_RE = re.compile(r"[\s*_]*(" + _NUM + r")")
NUMBER_RE = re.compile(_NUM)
GOLD_RE = re.compile(r"####\s*(" + _NUM + r")")


def normalize(num: str) -> Decimal | None:
    s = num.replace(",", "").replace("$", "").replace(" ", "").strip().rstrip(".")
    neg = s.count("-") % 2 == 1
    s = s.replace("-", "")
    try:
        if "/" in s:
            a, b = s.split("/", 1)
            if Decimal(b) == 0:
                return None
            v = Decimal(a) / Decimal(b)
        else:
            v = Decimal(s)
    except (InvalidOperation, ValueError):
        return None
    v = -v if neg else v
    return v.normalize() + 0  # "+ 0" turns -0 into 0


def fmt(d: Decimal | None) -> str | None:
    return None if d is None else format(d, "f")


def gold_answer(gsm8k_answer_field: str) -> Decimal:
    m = GOLD_RE.search(gsm8k_answer_field)
    if not m:
        raise ValueError("gold answer has no '#### <number>'")
    v = normalize(m.group(1))
    if v is None:
        raise ValueError(f"unparseable gold answer {m.group(1)!r}")
    return v


def extract_strict(completion: str) -> Decimal | None:
    tags = list(ANSWER_TAG_RE.finditer(completion))
    if not tags:
        return None
    m = AFTER_TAG_RE.match(completion, tags[-1].end())
    return normalize(m.group(1)) if m else None


def has_answer_tag(completion: str) -> bool:
    return ANSWER_TAG_RE.search(completion) is not None


def extract_flexible(completion: str) -> Decimal | None:
    ms = NUMBER_RE.findall(completion)
    return normalize(ms[-1]) if ms else None


def score(completion: str, gold: Decimal, truncated: bool) -> dict:
    strict = None if truncated else extract_strict(completion)
    flex = None if truncated else extract_flexible(completion)
    return {"truncated": truncated,
            "answer_tag_present": has_answer_tag(completion),
            "format_failure": (not truncated) and strict is None,
            "strict_pred": fmt(strict),
            "strict_correct": strict is not None and strict == gold,
            "flexible_pred": fmt(flex),
            "flexible_correct": flex is not None and flex == gold}


def summarize(scores: list[dict]) -> dict:
    """Per-config columns. strict_acc is gated; the rest are reported only."""
    n = len(scores)
    if n == 0:
        raise ValueError("no scores")
    rate = lambda k: sum(bool(s[k]) for s in scores) / n
    return {"n": n,
            "strict_acc": rate("strict_correct"),          # GATED
            "flexible_acc": rate("flexible_correct"),      # reported
            "truncation_rate": rate("truncated"),          # reported
            "format_failure_rate": rate("format_failure"), # reported
            "answer_tag_rate": rate("answer_tag_present")} # reported
