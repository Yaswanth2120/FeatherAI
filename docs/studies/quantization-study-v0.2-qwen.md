# Quantization study v0.2: Qwen2.5-Coder-0.5B-Instruct (pre-registered primary model)

*Run 2026-11-09/10 on an Apple M2 with 8 GiB. Every number is measured and copied from `experiments/studies/qwen2.5-coder-0.5b-clean/`. The protocol is the one in `docs/DECISIONS.md` ("Quantization study v0.2"), unchanged. The earlier blocked attempt stays preserved in `experiments/studies/qwen2.5-coder-0.5b/`.*

## 1. Conditions and provenance

**Admission.** Neither guard was changed.
- Available memory was about 2.33 GB at memory-pressure level 1, with Docker stopped, for the conversion and benchmark phases.
- That is enough for the quantization rule (1.5 × source + 256 MiB = 2 067 MiB) and the runtime rule (1 720 MiB for F16).
- The quality phases ran with the Docker VM running (2 GiB). Measured available memory was then 1 734–1 832 MiB, so the F16 runtime guard still passed.

**Code.** Repository commit `2b81883`, clean working tree (empty diff, no untracked files). llama.cpp `9d340ec3`. VS Code was open.

**Source.** Official `Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF`, revision `ebb20151`: F16 GGUF, 1 266 425 728 bytes, SHA-256 `bb69b303…`, 630 167 424 parameters.

**Variants created by Feather:**

| Variant | Bytes | SHA-256 (prefix) | llama-quantize peak RSS |
|---|---:|---|---:|
| Q8_0 | 675 710 848 | `0d93a22a` | 1 002 MiB (0.83 × source) |
| Q4_K_M | 491 400 064 | `3dc9c14a` | 1 003 MiB (0.83 × source) |

Neither conversion came close to the 1.5 × rule.

**Generation validation.** All three variants answered "What is 2 + 2?" with `4`. Metal offloaded every layer in all sessions.

## 2. Performance

5 order-balanced rounds; 15 sessions; 20 measured requests per variant, each exactly 128 tokens. No failed requests.

| Metric | F16 | Q8_0 | Q4_K_M |
|---|---:|---:|---:|
| Decode tok/s, median [95% CI] | 84.8 [84.0, 85.2] | 124.4 [123.3, 124.7] | 149.7 [144.7, 150.1] |
| Decode, mean ± SD (min) | 84.5 ± 1.0 (81.0) | 123.8 ± 1.4 (119.3) | 146.8 ± 5.6 (132.5) |
| Decode ratio to F16 [95% CI] | — | **1.468** [1.453, 1.478] | **1.766** [1.704, 1.779] |
| Prefill tok/s, median | 2 889 | 2 761 | 2 374 |
| Prefill ratio to F16 [95% CI] | — | 0.956 [0.953, 0.958] | 0.822 [0.817, 0.824] |
| TTFT, median | 41.9 ms | 43.7 ms | 50.8 ms |
| Peak memory footprint, median | **1 590 MiB** | **899 MiB** | **721 MiB** |
| Load s, first in study / repeat median | 1.25 / 1.27 | 0.63 / 0.81 | 0.63 / 0.66 |
| Sessions flagged for memory pressure | 3 of 5 | 2 of 5 | 1 of 5 |

**Sensitivity analysis (unflagged sessions only).** Decode medians: F16 84.8 (n = 8), Q8_0 124.4 (n = 12), Q4_K_M 149.7 (n = 16). The ratios are 1.467 and 1.766, matching the primary analysis.

**Memory breakdown from the server log (MiB):**

| Variant | Weights (Metal + CPU) | KV cache (2048 cells, f16) | Compute (Metal + CPU) |
|---|---|---:|---|
| F16 | 942.4 + 259.7 | 24.0 | 37.5 + 9.0 |
| Q8_0 | 500.8 + 137.9 | 24.0 | 37.5 + 9.0 |
| Q4_K_M | 373.7 + 89.3 | 24.0 | 37.5 + 9.0 |

Quantization speeds up decoding much more on this 0.5B model (×1.47 and ×1.77) than on SmolLM2-135M (×1.14 and ×1.19). Prompt processing is slower for the quantized variants on both models.

## 3. Quality (held-out; every item scored; coding in the Docker sandbox)

| Suite | F16 | Q8_0 | Q4_K_M |
|---|---|---|---|
| reasoning-v1, 160 items: strict pass (Wilson 95%) | 47 (29.4%) [22.9, 36.8] | 41 (25.6%) [19.5, 32.9] | 45 (28.1%) [21.7, 35.5] |
| coding-v1, 30 items: pass (Wilson 95%) | 21 (70.0%) [52.1, 83.3] | 21 (70.0%) [52.1, 83.3] | 19 (63.3%) [45.5, 78.1] |

**Baseline validity (pre-registered rule: F16 must solve at least 10 items and at least 25%).**
- Informative: reasoning overall (47/160), arithmetic (29/40), programming concepts (17/40), and coding (21/30).
- Uninformative: multi-step (0/40) and algorithmic (1/40).

**Paired comparisons against F16** (paired bootstrap 95% CI; exact McNemar):

| Comparison | Δ | 95% CI | McNemar p | Pre-registered label |
|---|---:|---|---:|---|
| reasoning overall, Q8_0 | −3.75 pp | [−7.5, −0.6] | 0.070 | detectable change |
| reasoning programming concepts, Q8_0 | −12.5 pp | [−22.5, −2.5] | 0.062 | detectable change |
| reasoning arithmetic, Q8_0 | −2.5 pp | [−10, +5] | 1.0 | no detectable change |
| reasoning overall, Q4_K_M | −1.25 pp | [−7.5, +5.0] | 0.845 | no detectable change |
| coding, Q8_0 | 0 pp | [0, 0] | 1.0 | no detectable change |
| coding, Q4_K_M | −6.7 pp | [−20, +6.7] | 0.625 | no detectable change |

**How to read the two Q8_0 "detectable" labels.** They follow the pre-registered rule (the bootstrap CI excludes 0), but the evidence is weak:
- The exact McNemar test does not reach p < 0.05.
- The percentile bootstrap is anti-conservative when there are few discordant pairs.
- 12 paired comparisons were made, with no correction for multiplicity.
- Q8_0 is almost identical to F16 at the token level: mean KLD 0.00025, with the same top token 99.2% of the time.

Small output differences at temperature 0 can still flip individual answers, but this result is not evidence that Q8_0 reliably degrades reasoning. It needs a separate, pre-registered reasoning replication on new held-out reasoning items (planned in `docs/DECISIONS.md`) before anyone relies on it. The coding suites cannot test a reasoning effect.

**Exploratory.** Q4_K_M ran out of tokens more often on reasoning items (32 of 160, vs 24 for F16).

The screening dry run (F16, before any held-out output) scored 8/40 on reasoning and 5/10 on coding. The prompts were not changed afterwards.

## 4. Perplexity and KL divergence (20 chunks × 512 tokens, CPython stdlib corpus)

| | F16 | Q8_0 | Q4_K_M |
|---|---:|---:|---:|
| PPL | 5.5410 ± 0.1961 | 5.5444 | 5.6284 |
| PPL(Q)/PPL(F16) | — | 1.0010 | 1.0162 |
| Mean KLD vs F16 | reference | 0.000245 | 0.012147 |
| 99th-percentile KLD | — | 0.0018 | 0.0998 |
| Same top token | — | 99.16% | 95.04% |

## 5. Findings

1. **Q4_K_M** is 1.77× faster at decoding and uses 55% less peak memory than F16 (721 vs 1 590 MiB). No quality loss was detectable at these sample sizes. Its distribution shift (KLD 0.012) is about 50× Q8_0's.
2. **Q8_0** is 1.47× faster, uses 43% less memory, and is nearly lossless at the token level. Its borderline reasoning difference needs confirmation (see §3).
3. **Coding with 30 problems cannot detect losses under about 15–20 pp.** The frozen `coding-v2` suite (360 held-out problems from MBPP) is sized to detect about 10 pp; see the DECISIONS power analysis.

## 6. Limitations

- One model, on one machine, with VS Code open.
- The Docker VM was running during the quality phases. That does not affect the quality results, but it was a condition of the run.
- The reasoning categories "multi-step" and "algorithmic" are uninformative for this model.
- Qwen2.5-Coder may have seen public benchmarks during training. reasoning-v1 is generated by this project; coding-v1 is original.
- TTFT includes localhost HTTP overhead. "First load in study" is not a true cold load.

## 7. coding-v2 (pre-registered analysis; this measures coding only)

The run followed the plan in `docs/DECISIONS.md` ("coding-v2 run on Qwen2.5-Coder-0.5B"). Evidence is in `experiments/studies/qwen2.5-coder-0.5b-clean/coding-v2/`.

**Completeness and provenance.**
- Every run scored every item: screening-F16 40/40; held-out F16, Q8_0 and Q4_K_M each 360/360. No item failed infrastructure or was blocked.
- The dataset SHA-256 is `37e35bd0…` in every run. Artifact hashes are re-verified: F16 `bb69b303`, Q8_0 `0d93a22a`, Q4_K_M `3dc9c14a`. The sandbox was available throughout.
- The run started at clean commit `30397f7`. The previous Claude Code session ended during the Q8_0 held-out run, before any partial Q8_0 result was written. The runner resumed from Q8_0 at clean commit `02e53ec`, which only added the frozen reasoning-v2 file and its hash entry; the coding-v2 code was unchanged. The resumption is recorded in `run.json`.

**Exclusions.** These were made at freeze time, before any model ran:
- 26 MBPP problems were excluded by the parsing rules: 13 need test imports, 6 have non-literal arguments or expected values, 6 are not a single `==` comparison, and 1 has an expected value that cannot be encoded.
- 1 problem (mbpp-123) was excluded because its reference solution failed its own parsed tests in the sandbox.

**Pass rates (360 held-out problems; Wilson 95%):**

| Variant | Passed | Pass rate [95% CI] | Outcome counts |
|---|---:|---|---|
| F16 | 138 | 38.3% [33.5, 43.5] | 28 partial, 79 wrong answer, 93 runtime error, 18 truncated, 2 missing function, 1 compile error, 1 timeout |
| Q8_0 | 137 | 38.1% [33.2, 43.2] | 28 partial, 76 wrong answer, 96 runtime error, 19 truncated, 2 missing function, 2 timeouts |
| Q4_K_M | 135 | 37.5% [32.7, 42.6] | 26 partial, 88 wrong answer, 94 runtime error, 12 truncated, 4 missing function, 1 compile error |

**Baseline validity:** F16 solved 138/360, so the comparison is informative.

**Paired comparisons against F16:**

| Variant | Δ pass rate | 95% CI | 90% CI | Pairs F16-only / variant-only | Exact McNemar p | Holm-adjusted p | Pre-registered result |
|---|---:|---|---|---|---:|---:|---|
| Q8_0 | −0.28 pp | [−1.39, +0.83] | [−1.39, +0.83] | 3 / 2 | 1.00 | 1.00 | No difference detected; **equivalent within ±10 pp** |
| Q4_K_M | −0.83 pp | [−3.89, +2.22] | [−3.33, +1.67] | 17 / 14 | 0.72 | 1.00 | No difference detected; **equivalent within ±10 pp** |

For Q8_0 the 95% and 90% intervals are identical because only 5 problems differ between the variants, so the bootstrap distribution is very coarse.

**What this does and does not show.**
- For coding on this suite, both quantized variants are **equivalent to F16 within the pre-registered ±10 pp margin**. Both 90% intervals also lie well inside a tighter ±5 pp, but that margin was not pre-registered for coding and is noted only descriptively.
- This says nothing about the Q8_0 *reasoning* signal from §3. That is tested only by the separate `reasoning-v2` replication, which has not been run.
- MBPP is a public benchmark that Qwen2.5-Coder may have seen in training. Absolute pass rates are therefore not capability claims; the paired variant-vs-F16 differences are the result.
- "Runtime error" is the largest failure category for every variant. Those are failures in the generated code (exceptions inside the sandbox), not infrastructure failures.
