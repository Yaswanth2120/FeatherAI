"""coding-v2: MBPP (sanitized) problems with hidden, exactly parsed unit tests.

Pre-registered in docs/DECISIONS.md ("coding-v2 design"). Inclusion: every test is
`assert <name>(<literal args>) == <literal>`, there are no test imports, and one function name.
Expected values are stored in a type-preserving canonical encoding and compared with Python `==`
(MBPP's own semantics). Reference solutions are executed only inside the Docker sandbox at freeze
time; problems whose reference fails its own tests are excluded and counted.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import random

RAW = Path(__file__).resolve().parent / "third_party" / "mbpp" / "sanitized-mbpp.json"
RAW_SHA256 = "ca95deaa9a01ef0a6f439f88bcf0dd3db3563d22f22aad6cae04ebb9a8d8c8e9"
SOURCE = {"repository": "google-research/google-research", "path": "mbpp/sanitized-mbpp.json",
          "commit": "f82046ba5aabbbb427dbfd38a254d26bff08b533", "sha256": RAW_SHA256}
SPLIT_SEED = 20261110
SCREENING_SIZE = 40
PROMPT = ("{text} The function must be named `{name}` and take {count} positional argument{plural}. "
          "Use only the Python standard library. Respond with the complete function in a single Python code block "
          "and no explanation.")


def canon(value, depth=0):
    """Host copy of the harness encoding (must match sandbox.HARNESS `canon`)."""
    if depth > 50:
        raise ValueError("too deep")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [canon(x, depth + 1) for x in value]
    if isinstance(value, tuple):
        return {"__tuple__": [canon(x, depth + 1) for x in value]}
    if isinstance(value, (set, frozenset)):
        return {"__set__": sorted((canon(x, depth + 1) for x in value), key=lambda x: json.dumps(x, sort_keys=True))}
    if isinstance(value, dict):
        return {"__dict__": sorted(([canon(k, depth + 1), canon(v, depth + 1)] for k, v in value.items()),
                                   key=lambda x: json.dumps(x, sort_keys=True))}
    raise TypeError(type(value).__name__)


def parse_test(text: str):
    node = ast.parse(text).body[0]
    if not isinstance(node, ast.Assert):
        return None, "not an assert"
    test = node.test
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)):
        return None, "not a single == comparison"
    call = test.left
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and not call.keywords):
        return None, "left side is not a plain function call"
    try:
        args = [ast.literal_eval(a) for a in call.args]
        expected = ast.literal_eval(test.comparators[0])
    except (ValueError, SyntaxError, TypeError):
        return None, "non-literal argument or expected value"
    return (call.func.id, args, expected), None


def candidates() -> tuple[list[dict], dict]:
    raw = RAW.read_bytes()
    if hashlib.sha256(raw).hexdigest() != RAW_SHA256:
        raise ValueError("Vendored MBPP file does not match its pinned SHA-256")
    excluded: dict[str, list[int]] = {}
    items = []
    for problem in json.loads(raw):
        reason = None
        if problem["test_imports"]:
            reason = "tests need imports"
        parsed = []
        for test in problem["test_list"]:
            result, why = parse_test(test)
            if result is None:
                reason = reason or why
            parsed.append(result)
        if reason is None and len({p[0] for p in parsed}) != 1:
            reason = "tests call more than one function"
        if reason is None:
            try:
                # Arguments and expected values both use the type-preserving encoding (decoded in the sandbox).
                cases = [[[canon(a) for a in p[1]], canon(p[2])] for p in parsed]
            except (TypeError, ValueError):
                reason = "expected value not encodable"
        if reason:
            excluded.setdefault(reason, []).append(problem["task_id"])
            continue
        name, count = parsed[0][0], len(parsed[0][1])
        items.append({"id": f"mbpp-{problem['task_id']}", "task_id": problem["task_id"], "function": name,
                      "comparison": "python-eq", "cases": cases, "reference": problem["code"],
                      "prompt": PROMPT.format(text=problem["prompt"].strip(), name=name, count=count,
                                              plural="" if count == 1 else "s")})
    return items, excluded


def build(reference_results: dict[str, bool]) -> dict:
    """``reference_results`` maps item id -> whether the MBPP reference passed all parsed tests in the sandbox."""
    items, excluded = candidates()
    kept = []
    for item in items:
        if reference_results.get(item["id"]) is not True:
            excluded.setdefault("reference solution failed its own parsed tests in the sandbox", []).append(item["task_id"])
            continue
        kept.append({k: v for k, v in item.items() if k != "reference"})
    order = list(range(len(kept)))
    random.Random(SPLIT_SEED).shuffle(order)
    screening = set(order[:SCREENING_SIZE])
    for index, item in enumerate(kept):
        item["split"] = "screening" if index in screening else "heldout"
    return {"name": "coding", "version": "v2", "source": SOURCE, "split_seed": SPLIT_SEED, "items": kept,
            "excluded": {k: sorted(v) for k, v in sorted(excluded.items())},
            "notes": "MBPP sanitized; hidden tests parsed exactly; Python == semantics via type-preserving encoding; "
                     "references verified in the Docker sandbox at freeze time. Public benchmark: possible training contamination."}
