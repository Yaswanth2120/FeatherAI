"""MMLU scoring and token-mass diagnostic. Frozen by prereg-v1 (sha256 in thresholds.yaml).

Input: the full-vocabulary logits at the first assistant position (one forward
pass) and the letter token IDs from configs/eval/mmlu_tokens.yaml.

- prediction (GATED via accuracy): argmax over the four letter logits only.
- letter_mass (reported, never gated): softmax probability over the FULL
  vocabulary summed over the four letter tokens. Low mass means the prompt
  failed to elicit a letter answer, not that the model got the question wrong.
"""
import numpy as np

LETTERS = ("A", "B", "C", "D")


def score_item(logits, letter_ids: dict, gold: str) -> dict:
    x = np.asarray(logits, dtype=np.float64)
    ids = [letter_ids[k] for k in LETTERS]
    m = x.max()
    log_z = m + np.log(np.exp(x - m).sum())           # stable log-sum-exp
    letter_logp = x[ids] - log_z
    mass = float(np.exp(letter_logp).sum())
    pred = LETTERS[int(np.argmax(x[ids]))]
    return {"pred": pred, "correct": pred == gold, "letter_mass": mass,
            "letter_probs_full_vocab": dict(zip(LETTERS, map(float, np.exp(letter_logp))))}


def summarize(items: list[dict]) -> dict:
    n = len(items)
    if n == 0:
        raise ValueError("no items")
    mass = np.array([i["letter_mass"] for i in items])
    return {"n": n,
            "acc": sum(i["correct"] for i in items) / n,              # GATED
            "letter_mass_mean": float(mass.mean()),                   # reported
            "letter_mass_min": float(mass.min()),                     # reported
            "letter_mass_p05": float(np.percentile(mass, 5)),         # reported
            "letter_mass_median": float(np.median(mass))}             # reported
