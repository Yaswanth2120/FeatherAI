"""Descriptive statistics and session-clustered bootstrap intervals for benchmark trials."""
from __future__ import annotations

import numpy as np

RESAMPLES = 10_000
SEED = 20261109


def describe(values: list[float]) -> dict | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {"n": int(array.size), "median": float(np.median(array)), "mean": float(array.mean()),
            "sd": float(array.std(ddof=1)) if array.size > 1 else None, "p95": float(np.percentile(array, 95)),
            "min": float(array.min()), "max": float(array.max())}


def cluster_bootstrap_medians(groups: list[list[float]], resamples: int = RESAMPLES, seed: int = SEED) -> np.ndarray:
    """Two-stage bootstrap: resample sessions with replacement, then trials within each chosen session.

    Trials in one server session share load state, so they are not independent; sessions are the
    primary resampling unit.
    """
    groups = [np.asarray(g, dtype=float) for g in groups if len(g)]
    if not groups:
        raise ValueError("No measured trials")
    rng = np.random.default_rng(seed)
    medians = np.empty(resamples)
    for r in range(resamples):
        chosen = rng.integers(0, len(groups), size=len(groups))
        pooled = np.concatenate([groups[i][rng.integers(0, len(groups[i]), size=len(groups[i]))] for i in chosen])
        medians[r] = np.median(pooled)
    return medians


def median_ci(groups: list[list[float]], resamples: int = RESAMPLES, seed: int = SEED) -> list[float]:
    boot = cluster_bootstrap_medians(groups, resamples, seed)
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def ratio_ci(candidate: list[list[float]], baseline: list[list[float]], resamples: int = RESAMPLES, seed: int = SEED) -> dict:
    """Ratio of medians (candidate / baseline) with independent cluster bootstraps per variant."""
    a = cluster_bootstrap_medians(candidate, resamples, seed)
    b = cluster_bootstrap_medians(baseline, resamples, seed + 1)
    ratio = a / b
    point = float(np.median(np.concatenate([np.asarray(g, float) for g in candidate])) /
                  np.median(np.concatenate([np.asarray(g, float) for g in baseline])))
    return {"ratio": point, "ci95": [float(np.percentile(ratio, 2.5)), float(np.percentile(ratio, 97.5))],
            "method": f"independent two-stage cluster bootstraps, {resamples} resamples, seeds {seed}/{seed + 1}"}
