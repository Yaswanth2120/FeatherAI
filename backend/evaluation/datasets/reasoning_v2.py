"""Reasoning suite v2 (replication set for the Q8_0 reasoning signal; plan in docs/DECISIONS.md).

Same generator code and four categories as reasoning-v1, new seed, larger held-out split
(150 per category = 600) plus 10 screening per category. Every question must differ from all
reasoning-v1 questions, so no v1 item is reused.
"""
from __future__ import annotations

import random

from backend.evaluation.datasets import reasoning_v1

SEED = 20261110
PER_CATEGORY = {"screening": 10, "heldout": 150}


def build() -> dict:
    v1_questions = {item["question"] for item in reasoning_v1.build()["items"]}
    items = []
    for category in reasoning_v1.CATEGORIES:
        rng = random.Random(f"{SEED}-{category}")
        seen = set(v1_questions)
        for split, count in PER_CATEGORY.items():
            made = attempts = 0
            while made < count:
                attempts += 1
                if attempts > 200_000:
                    raise RuntimeError(f"Could not generate enough unique {category} items")
                question, answer = reasoning_v1.GENERATORS[category](rng)
                if question in seen:
                    continue
                seen.add(question)
                made += 1
                items.append({"id": f"{category}-{split}-{made:03d}", "category": category, "split": split,
                              "question": question, "prompt": reasoning_v1.PROMPT.format(question=question), "answer": int(answer)})
    return {"name": "reasoning", "version": "v2", "seed": SEED, "answer_format": "Answer: <integer>", "items": items,
            "notes": "Replication set: same generators/categories as v1, new seed, no question shared with v1."}
