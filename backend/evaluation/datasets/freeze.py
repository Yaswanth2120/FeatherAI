"""Write frozen dataset files once. Existing files are never overwritten."""
import hashlib
import json
from pathlib import Path

from backend.evaluation.datasets import coding_v1, reasoning_v1

HERE = Path(__file__).resolve().parent


def main():
    for module, name in [(reasoning_v1, "reasoning-v1.json"), (coding_v1, "coding-v1.json")]:
        path = HERE / name
        text = json.dumps(module.build(), indent=1, sort_keys=True) + "\n"
        if path.exists():
            status = "unchanged" if path.read_text() == text else "DIFFERS from generator (not overwritten)"
        else:
            path.write_text(text)
            status = "written"
        print(name, status, hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
