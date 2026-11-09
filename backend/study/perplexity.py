"""Perplexity and KL divergence against base-model logits with llama-perplexity."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess
import sys

CORPUS_FILES = ["textwrap.py", "heapq.py", "bisect.py", "fractions.py", "statistics.py", "difflib.py", "string.py",
                "shlex.py", "colorsys.py", "graphlib.py", "functools.py", "contextlib.py"]


def build_corpus(out: Path) -> dict:
    """Concatenate a fixed list of CPython standard-library sources (code-domain text)."""
    library = Path(sys.base_prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    parts = []
    for name in CORPUS_FILES:
        parts.append(f"# ---- {name} ----\n" + (library / name).read_text())
    text = "\n".join(parts)
    out.write_text(text)
    return {"path": str(out), "sha256": hashlib.sha256(text.encode()).hexdigest(), "bytes": len(text.encode()),
            "files": CORPUS_FILES, "python": sys.version.split()[0], "library": str(library)}


def parse(text: str) -> dict:
    result = {}
    final = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", text)
    if final:
        result["ppl"], result["ppl_se"] = float(final.group(1)), float(final.group(2))
    for key, pattern in {"mean_kld": r"Mean\s+KLD:\s+([-\d.]+)\s+±\s+([\d.]+)",
                         "same_top_p": r"Same top p:\s+([\d.]+)\s+±\s+([\d.]+)\s*%",
                         "mean_ppl_q": r"Mean\s+PPL\(Q\)\s*:\s+([\d.]+)\s+±\s+([\d.]+)",
                         "mean_ppl_base": r"Mean\s+PPL\(base\)\s*:\s+([\d.]+)\s+±\s+([\d.]+)",
                         "ppl_ratio": r"Mean\s+PPL\(Q\)/PPL\(base\)\s*:\s+([\d.]+)\s+±\s+([\d.]+)",
                         "rms_delta_p": r"RMS Δp\s*:\s+([\d.]+)\s+±\s+([\d.]+)\s*%"}.items():
        match = re.search(pattern, text)
        if match:
            result[key], result[key + "_se"] = float(match.group(1)), float(match.group(2))
    for key, pattern in {"kld_p99": r"99\.0%\s+KLD:\s+([\d.]+)", "kld_median": r"Median\s+KLD:\s+([-\d.]+)",
                         "kld_max": r"Maximum KLD:\s+([\d.]+)"}.items():
        match = re.search(pattern, text)
        if match:
            result[key] = float(match.group(1))
    return result


def run(binary: str, model: str, corpus: Path, log: Path, base_logits: Path, base: bool, context=512, chunks=20) -> dict:
    command = [binary, "-m", model, "-f", str(corpus), "-c", str(context), "--chunks", str(chunks), "-b", str(context)]
    command += ["--kl-divergence-base", str(base_logits)] + ([] if base else ["--kl-divergence"])
    with log.open("w") as stream:
        code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=3600).returncode
    text = log.read_text(errors="replace")
    return {"command": command, "exit_code": code, **parse(text),
            "note": "± values are llama-perplexity standard errors (per-token), not confidence intervals over texts."}
