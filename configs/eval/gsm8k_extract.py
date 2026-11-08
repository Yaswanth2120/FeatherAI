"""GSM8K answer extraction and scoring. Frozen by prereg-v1 (sha256 in thresholds.yaml).

Primary (gated): strict. The LAST line matching ANSWER_RE in the completion;
no match counts as incorrect. Truncated completions (hit max_new_tokens
without an end-of-generation token or stop sequence) count as incorrect,
whatever they contain.
Secondary (reported, never gated): flexible, the last number in the completion.
"""
import re
from decimal import Decimal, InvalidOperation

ANSWER_RE = re.compile(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)")
NUMBER_RE = re.compile(r"-?[\d,]*\.?\d+")
GOLD_RE = re.compile(r"####\s*(-?[\d,]*\.?\d+)")


def normalize(num: str) -> Decimal | None:
    s = num.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return Decimal(s).normalize()
    except InvalidOperation:
        return None


def gold_answer(gsm8k_answer_field: str) -> Decimal:
    m = GOLD_RE.search(gsm8k_answer_field)
    if not m:
        raise ValueError("gold answer has no '#### <number>'")
    return normalize(m.group(1))


def extract_strict(completion: str) -> Decimal | None:
    ms = ANSWER_RE.findall(completion)
    return normalize(ms[-1]) if ms else None


def extract_flexible(completion: str) -> Decimal | None:
    ms = NUMBER_RE.findall(completion)
    return normalize(ms[-1]) if ms else None


def score(completion: str, gold: Decimal, truncated: bool) -> dict:
    strict = None if truncated else extract_strict(completion)
    flex = None if truncated else extract_flexible(completion)
    return {"truncated": truncated,
            "strict_pred": None if strict is None else str(strict),
            "strict_correct": strict is not None and strict == gold,
            "flexible_pred": None if flex is None else str(flex),
            "flexible_correct": flex is not None and flex == gold}
