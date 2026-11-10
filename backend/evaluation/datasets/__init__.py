"""Versioned, hash-pinned evaluation datasets."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Recorded when the datasets were frozen (2026-11-09), before any model saw them.
SHA256 = {
    "reasoning-v1": "02e5f81bd76d0bc2159de994ef1ec7d08d1f411662e610aadfe0ce49c9f7d5dd",
    "coding-v1": "e41bfe10854cd6fa6e8ad69b9e56d34c73da66e6b48ec9dd86d5e74565d531bb",
    # Frozen 2026-11-10 from MBPP sanitized (pinned) after sandboxed reference checks; before any model ran on it.
    "coding-v2": "37e35bd02721dfd6354e1ae5565e53caad6a4464bdef84b7ba8081892aa606e2",
}


def load(name: str, split: str | None = None) -> dict:
    if name not in SHA256:
        raise ValueError(f"Unknown dataset {name!r}; available: {sorted(SHA256)}")
    path = HERE / f"{name}.json"
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SHA256[name]:
        raise ValueError(f"Dataset {name} hash mismatch ({digest}); frozen datasets must not be edited")
    data = json.loads(raw)
    if split is not None:
        if split not in {"screening", "heldout"}:
            raise ValueError("split must be 'screening' or 'heldout'")
        data = {**data, "items": [item for item in data["items"] if item["split"] == split], "split": split}
    data["sha256"] = digest
    return data
