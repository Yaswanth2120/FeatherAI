"""Answer extraction for integer-answer reasoning items.

Strict rule (gated): the integer after the LAST "Answer:" tag. Markdown emphasis, a leading
"$", thousands separators and a trailing ".0" are tolerated; any other content before the
number (for example "about 18") or a non-integer value is a format failure.
Lenient rule (reported only, never used for pass/fail): the last integer anywhere in the text.
"""
from __future__ import annotations

import re

TAG = re.compile(r"answer\s*[:：]", re.I)
AFTER_TAG = re.compile(r"[\s*_`]*\$?\s*(-?\d[\d,]*)(\.\d+)?")
ANY_INT = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _to_int(digits: str, fraction: str | None) -> int | None:
    if fraction and set(fraction[1:]) != {"0"}:
        return None
    cleaned = digits.replace(",", "")
    if not re.fullmatch(r"-?\d+", cleaned):
        return None
    return int(cleaned)


def extract(text: str) -> dict:
    text = text or ""
    tags = list(TAG.finditer(text))
    strict = None
    if tags:
        match = AFTER_TAG.match(text, tags[-1].end())
        if match:
            strict = _to_int(match.group(1), match.group(2))
    lenient = None
    for token in reversed(ANY_INT.findall(text)):
        digits, _, fraction = token.partition(".")
        value = _to_int(digits, "." + fraction if fraction else None)
        if value is not None:
            lenient = value
            break
    return {"strict": strict, "lenient": lenient, "answer_tag": bool(tags)}


def score(output: str, expected: int, truncated: bool) -> dict:
    found = extract(output)
    if found["strict"] is not None:
        category = "correct" if found["strict"] == expected else "incorrect"
    elif truncated:
        category = "truncated"
    else:
        category = "invalid_format"
    return {"status": "scored", "passed": category == "correct", "category": category, "expected": expected,
            "extracted": found["strict"], "lenient_extracted": found["lenient"],
            "lenient_passed": found["lenient"] == expected, "answer_tag": found["answer_tag"], "truncated": truncated}
