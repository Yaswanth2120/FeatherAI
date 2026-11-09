# Quantization study v0.2

*Run 2026-11-09 on one Apple M2 with 8 GiB of memory. Every number below is measured and is copied from `experiments/studies/`. Where a measurement is missing, the report says why.*

## 1. Research question

How does weight quantization change speed, memory and output quality on this M2 laptop? Variants are F16 → Q8_0 → Q4_K_M, all created by Feather from one higher-precision source and served by the same llama.cpp build.

## 2. Outcome in brief

| Study | Status |
|---|---|
| **Qwen2.5-Coder-0.5B-Instruct** (pre-registered primary model) | **BLOCKED.** Both memory guards rejected it before it started; nothing was converted or run. See §8. |
| **SmolLM2-135M-Instruct** (fallback, full protocol) | **Complete.** Every phase ran under the pre-registered protocol. |

**Confirmatory findings (SmolLM2-135M, this machine, these conditions):**
- **Speed:**
  - Q4_K_M decodes 1.19× faster than F16, 95% CI [1.11, 1.27].
  - Q8_0 decodes 1.14× faster, 95% CI [1.10, 1.23].
  - Prompt processing is *slower* for quantized variants: 0.84× for Q4_K_M and 0.94× for Q8_0.
- **Memory:** median peak process footprint is 472 MiB (F16), 301 MiB (Q8_0) and 264 MiB (Q4_K_M).
- **Distribution shift:** mean KLD vs F16 is 0.0012 for Q8_0 and 0.0445 for Q4_K_M, a 37× larger shift. The share of tokens where the variant's top token matches F16's is 98.0% and 89.3%.
- **Task quality:** no detectable change in coding pass rate.
  - Q8_0 vs F16 is −10 pp [−20, 0] and Q4_K_M vs F16 is 0 pp [−16.7, +16.7], on 30 problems. The intervals are wide.
  - Reasoning quality could not be assessed: the F16 baseline solved 3/160 held-out items, so every reasoning category is labelled uninformative.

## 3. Environment

| | |
|---|---|
| **Hardware** | Apple M2, 8 GiB unified memory (8 589 934 592 bytes), macOS 26.7.3 (arm64) |
| **Runtime** | llama.cpp `9d340ec341e079d2c28afd50871afce358cdbf48` (0.9.0-dev, Release, Metal, embedded library), built locally under `.feather/tools/llama.cpp` |
| **Repository at run time** | `1f6f8f1`. The pre-registration was committed before any study output existed. |
| **Measurement conditions** | Run by the agent while VS Code, Chrome and other user applications were open. See the table below. |

| Phase | Docker VM | Memory pressure |
|---|---|---|
| Conversion and benchmark | stopped | `kern.memorystatus_vm_pressure_level` was 2 (warning) in most sessions; system swap in use was about 3.2 GB throughout |
| Quality (coding needs the sandbox) | running | not separately flagged |

## 4. Model identity and provenance

**Source:** `SmolLM2-135M-Instruct-f16.gguf` from `bartowski/SmolLM2-135M-Instruct-GGUF`, revision `09816acd5d99df7be770d85ea30822623dab342c`.
- SHA-256 `f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57`, 270 885 952 bytes.
- Tensor types F16 and F32 only; architecture `llama`.

**Variants:** Feather created these with `llama-quantize` at the same commit, under the measured-memory admission guard. They are not downloaded quantizations.

| Variant | Bytes | Storage reduction | SHA-256 (prefix) | Quantization peak RSS | Quantization peak footprint |
|---|---:|---:|---|---:|---:|
| F16 (source) | 270 885 952 | — | `f535f83e` | — | — |
| Q8_0 | 144 811 072 | 46.5% | `a2c82248` | 286.9 MiB | 241.5 MiB |
| Q4_K_M | 105 454 144 | 61.1% | `76ab8f09` | 287.6 MiB | 242.2 MiB |

Lineage checks run on each conversion:
- Output tensor names and shapes, architecture, tokenizer hash and parameter count match the source.
- Hashes were rechecked before every phase.
- Each variant produced non-empty output for the generation-validation prompt, with Metal offload of 31/31 layers verified from the server log.

## 5. Methods

These follow the pre-registration in `docs/DECISIONS.md` ("Quantization study v0.2").

**Server.** llama-server with:
- `--load-mode none`, so weights are anonymous memory and are counted in the footprint
- `-c 2048 -np 1 -t 4 --cache-ram 0 --fit off --no-warmup -lv 4`
- fixed ChatML template, temperature 0, seed 42, `cache_prompt: false`

**Benchmark.**
- 5 rounds × 3 variants = 15 server sessions. The order in each round came from a seeded shuffle of the 6 permutations (seed 20261109); 5 distinct orders were used.
- Each session ran 1 warm-up, then 4 measured requests, with a 30 s cooldown between sessions.
- Each request used a fixed 121-token prompt (`backend/study/prompts/benchmark.txt`, SHA-256 in `study.json`) with `n_predict 128` and `ignore_eos`. Every measured request produced exactly 128 tokens.
- Speed and timing sources:
  - Decode and prefill rates come from the server's own timings.
  - TTFT and wall time are client wall-clock values over localhost HTTP.
  - Load latency runs from process spawn to `/health` = 200.
- Memory sources:
  - Peak footprint and max RSS come from `/usr/bin/time -l` over the whole server process, including startup.
  - Stage snapshots come from `/usr/bin/footprint`.

**Statistics.**
- Per-variant median, mean, SD, p95, minimum and maximum.
- 95% CI for the median from a two-stage cluster bootstrap: resample sessions, then trials within each session. 10 000 resamples, seed 20261109. Trials within one session are not treated as independent.
- Variant/F16 ratios of medians, with CIs from independent bootstraps.

**Quality.**
- Datasets: hash-pinned `reasoning-v1` held-out split (160 integer-answer items, 4 categories × 40, answers computed by a generator) and `coding-v1` held-out split (30 Python problems with hidden unit tests).
- Settings: context 2048, temperature 0, seed 42; `n_predict` 256 for reasoning, 384 for coding.
- Reasoning scoring is strict: the integer after the last `Answer:`. A lenient score is reported but never used for pass/fail.
- Coding scoring ran inside the Docker sandbox: pinned image, no network, read-only root, no capabilities, uid 65534, and memory, CPU and pid limits.
- Comparisons are paired: bootstrap over items (10 000 resamples) plus exact McNemar.
- Pre-registered baseline-validity rule: a category is informative only if F16 solves at least 10 items and at least 25% of them.

**Perplexity and KLD.**
- `llama-perplexity -c 512 --chunks 20` on a concatenation of 12 CPython 3.12.9 standard-library modules (SHA-256 `68640572d8e7…`, 322 575 bytes).
- KLD is measured against F16 base logits.
- This corpus is code-domain text the model may have seen in training. It supports relative comparisons between variants, not an absolute capability claim.

## 6. Results

All values come from `experiments/studies/smollm2-135m/` (raw files: `benchmark/trials.csv`, `benchmark/sessions.jsonl`, `quality/*.json`, `perplexity/*.log`; flat summaries: `summary-*.csv`).

### 6.1 Performance (primary analysis, all 15 sessions; n = 20 measured trials per variant)

| Metric | F16 | Q8_0 | Q4_K_M |
|---|---:|---:|---:|
| Decode tok/s, median [95% CI] | 220.8 [207.1, 225.8] | 252.6 [244.9, 256.4] | 262.8 [246.2, 268.5] |
| Decode, mean ± SD | 217.5 ± 8.6 | 249.4 ± 11.7 | 242.8 ± 61.6 † |
| Decode, min / p95 / max | 201.0 / 226.4 / 228.6 | 217.6 / 259.8 / 260.4 | 0.34 † / 272.5 / 273.7 |
| Decode ratio to F16 [95% CI] | — | 1.144 [1.104, 1.226] | 1.190 [1.114, 1.273] |
| Prefill tok/s, median [95% CI] | 7499 [7447, 7570] | 7026 [6731, 7150] | 6277 [5912, 6348] |
| Prefill ratio to F16 [95% CI] | — | 0.937 [0.898, 0.956] | 0.837 [0.788, 0.849] |
| TTFT ms, median [95% CI] | 17.2 [17.0, 17.3] | 18.3 [18.0, 19.2] | 20.5 [20.3, 21.5] |
| End-to-end request s, median [95% CI] | 0.597 [0.584, 0.635] | 0.525 [0.518, 0.542] | 0.509 [0.497, 0.541] |
| Peak memory footprint, median (range) | 472.1 MiB (470.8–473.2) | 300.5 MiB (300.2–302.0) | 264.2 MiB (263.3–264.8) |
| Max RSS, median | 484.2 MiB | 337.4 MiB | 304.8 MiB |
| Load s, first in study | 0.632 | 0.181 | 0.440 |
| Load s, repeat sessions, median (range) | 0.751 (0.619–1.185) | 0.564 (0.549–0.570) | 0.458 (0.405–0.477) |
| Failed requests / sessions | 0 / 0 | 0 / 0 | 0 / 0 |

† **Q4_K_M outliers.** Session 4, request 4 decoded at 0.34 tok/s; session 15, request 4 at 166.9 tok/s. Both requests completed with 128 tokens and no error, so the protocol keeps them. The median and its CI are robust to them; the mean and SD are not.

**Session-level variability (exploratory).** Per-session median decode tok/s:

| Variant | Session medians |
|---|---|
| F16 | 205.1, 218.8, 222.7, 218.6, 226.0 |
| Q8_0 | 234.3, 256.7, 252.6, 244.9, 258.0 |
| Q4_K_M | 247.8, 270.0, 263.8, 254.2, 263.1 |

The first round was the slowest session for every variant.

**Memory breakdown (exploratory, from the server log, MiB).**

| Variant | Model weights (Metal) | Model weights (CPU) | KV cache (2048 cells, f16 K/V) | Compute (Metal + CPU) |
|---|---:|---:|---:|---:|
| F16 | 256.6 | 54.0 | 45.0 | 15.5 + 6.5 |
| Q8_0 | 136.4 | 28.7 | 45.0 | 15.5 + 6.5 |
| Q4_K_M | 98.9 | 28.7 | 45.0 | 15.5 + 6.5 |

Footprint snapshots after load, after warm-up and after the measured requests are 471/472/472 MiB (F16), 299/301/301 (Q8_0) and 263/265/235 (Q4_K_M). In every variant, about 99% of the peak footprint is reached at load time; prefill and generation add almost nothing at this context size.

**Sensitivity analysis (pre-registered: exclude sessions flagged for memory pressure).**
- All 5 F16 sessions and all 5 Q4_K_M sessions were flagged: memory-pressure level ≥ 2, plus swap growth over 64 MiB in two F16 sessions.
- 3 of 5 Q8_0 sessions were flagged.
- So the sensitivity analysis can only be computed for Q8_0: 2 sessions, 8 trials, median 244.9 tok/s [219.5, 252.3]. No unflagged comparison between variants is possible. The memory-pressure state comes from the whole machine, which other applications share.

### 6.2 Quality (held-out splits; every item ran; nothing missing)

| Suite | F16 | Q8_0 | Q4_K_M |
|---|---|---|---|
| reasoning-v1 strict, 160 items | 3 (1.9%) [0.6, 5.4] | 3 (1.9%) [0.6, 5.4] | 2 (1.3%) [0.3, 4.4] |
| reasoning-v1 outcomes (correct / incorrect / bad format / truncated) | 3 / 22 / 112 / 23 | 3 / 20 / 116 / 21 | 2 / 10 / 100 / 48 |
| reasoning-v1 lenient (reported only) | 8.8% | 10.6% | 6.9% |
| coding-v1, 30 problems, pass (Wilson 95%) | 11 (36.7%) [21.9, 54.5] | 8 (26.7%) [14.2, 44.4] | 11 (36.7%) [21.9, 54.5] |
| coding-v1 outcomes | 11 passed, 13 partial, 3 wrong, 2 runtime errors, 1 truncated | 8 passed, 17 partial, 2 wrong, 1 runtime error, 2 truncated | 11 passed, 14 partial, 5 wrong |

**Paired comparisons against F16** (difference in pass rate, paired bootstrap 95% CI, exact McNemar p):

| Comparison | Δ | 95% CI | p | Baseline validity | Evidence |
|---|---:|---|---:|---|---|
| coding Q8_0 | −10.0 pp | [−20.0, 0.0] | 0.25 | informative (F16 11/30) | no detectable change |
| coding Q4_K_M | 0.0 pp | [−16.7, +16.7] | 1.00 | informative | no detectable change |
| reasoning Q8_0, overall | 0.0 pp | [−1.9, +1.9] | 1.00 | **uninformative** (F16 3/160) | insufficient |
| reasoning Q4_K_M, overall | −0.6 pp | [−3.1, +1.9] | 1.00 | **uninformative** | insufficient |

Every reasoning category is uninformative: F16 solved 2/40 arithmetic, 0/40 multi-step, 0/40 programming-concepts and 1/40 algorithmic items.

**Exploratory observation.** Q4_K_M ran out of its token limit more often than the other variants (48 of 160 reasoning items vs 23 for F16). That is consistent with longer or repetitive generations, but it is not a pre-registered outcome.

The screening-split dry run (F16 only, before any held-out output) gave reasoning 0/40 and coding 2/10. The prompts were not changed afterwards.

### 6.3 Perplexity and KL divergence (20 chunks × 512 tokens)

| | F16 | Q8_0 | Q4_K_M |
|---|---:|---:|---:|
| PPL | 7.6022 ± 0.2771 | 7.6076 ± 0.2772 ‡ | 7.9029 ± 0.2854 ‡ |
| PPL(Q)/PPL(base) | — | 1.000725 ± 0.00074 | 1.039573 ± 0.00495 |
| Mean KLD | reference | 0.001178 ± 0.000027 | 0.044508 ± 0.00082 |
| 99th-percentile KLD | — | 0.0065 | 0.2847 |
| Same top token | — | 98.02% | 89.33% |

The ± values are llama-perplexity's per-token standard errors, not confidence intervals over texts. ‡ The quantized variants report "Mean PPL(Q)" from the KLD run over the same chunks.

## 7. Deviations and corrections (recorded, not hidden)

1. **Model.** The pre-registered primary model, Qwen2.5-Coder-0.5B, was blocked by both memory guards (§8). The complete protocol ran on the SmolLM2-135M fallback instead, which v0.1 had already identified as weak. No model was substituted silently, and neither guard was relaxed.
2. **Sensitivity analysis.** It is almost empty because the machine stayed at memory-pressure level 2 (§6.1). The primary analysis is unchanged.
3. **Provenance correction.** `study.json` first recorded `toolchain_commit` as the repository commit (`1f6f8f1`) because of a path bug in `run.py`. Every server session's `runtime_version` reports `commit 9d340ec`, and the llama.cpp checkout HEAD is `9d340ec341e0…`. The field was corrected and the correction is recorded under `provenance_corrections`. No measurement depends on this field.
4. **Code provenance.** At run time, the `backend/study/` code was uncommitted, on top of `1f6f8f1`. Post-run changes and review fixes are listed in Amendment 1 of `docs/DECISIONS.md`. None of them changes a recorded result.
5. **Order design.** Five rounds used 5 of the 6 permutations. This follows from the pre-registered "seeded shuffle, cycled" design and is not a deviation, but it means F16 was first in a round once while each other variant was first twice.

## 8. Qwen2.5-Coder-0.5B: memory-failure diagnosis

**Source.** Official `Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF` revision `ebb20151`: `qwen2.5-coder-0.5b-instruct-fp16.gguf`, 1 266 425 728 bytes, SHA-256 `bb69b303…`, F16/F32 tensors, 630 167 424 parameters.

| Stage | Result |
|---|---|
| Download and storage | OK: file present, about 81 GB of disk free |
| Source loading (GGUF header and tensor inspection) | OK: reads metadata only, without loading weights |
| Conversion and quantization | **Rejected by the pre-flight guard.** The estimate is 1.5 × source + 256 MiB = 2 168 074 048 bytes. Available memory was 1 615 937 536 bytes at the recorded rejection, and ranged from about 1.46 to 2.11 GB across observations today. `llama-quantize` never ran. |
| Runtime initialization | **Rejected by the pre-flight guard.** The estimate is artifact + 512 MiB = 1 803 296 640 bytes; 1 460 060 160 bytes were available. |
| Prompt prefill and token generation | Not reached |

**The v0.1 "Qwen exceeded the memory guard" report** came from a pre-flight estimate (2 × source + 256 MiB), not from an observed out-of-memory event.

**Measured basis for the v0.2 rule** (`experiments/memory/quantize-smollm.json`, plus the SmolLM2 study conversions):
- `llama-quantize` peak max RSS was 1.01–1.11 × source size.
- Peak footprint was 0.92–0.94 × source size.
- No conversion exceeded the 1.5× rule.

**Runtime memory on SmolLM2:**
- Peak footprint was F16 file size + 214 MiB at context 2048.
- KV cache was 45 MiB at that context.

**What would let Qwen run:**
- Available memory of about 2.1 GB or more at the moment of admission. That would mean running the study from Terminal with VS Code, browsers and Docker closed.
- No guard was weakened, and no application was closed to force admission.

**The 1.5B model was not attempted.** It would need a roughly 3.1 GB download, which requires your approval. Its F16 weights alone would exceed the memory available today.

## 9. Limitations

- Results cover one small model, on one machine, under background load and memory pressure. They are not a general claim about quantization.
- Quality sample sizes are small: 30 coding problems give CIs about ±17 pp wide. Reasoning cannot measure quantization loss for this model.
- TTFT includes localhost HTTP overhead. "First load in study" is not a true cold load, because the page cache was not purged (that needs sudo).
- Memory footprint is a per-process measure. Swap and memory pressure are system-wide.
- Docker isolates generated code but is not an absolute security boundary.

## 10. Reproduce

```sh
cmake --build .feather/tools/llama.cpp/build --target llama-quantize llama-server llama-perplexity
uv run feather sandbox status   # must report "available" for coding
uv run feather quant-study --source models/SmolLM2-135M-Instruct-f16.gguf --out .feather/study-<new> \
    --phases convert,validate,dryrun,benchmark,quality,perplexity --export experiments/studies/<new>
```

For cleaner conditions, close VS Code, browsers and Docker, and run the conversion and benchmark phases from Terminal. Then start Docker and run `--phases dryrun,quality,perplexity` against the same `--out` directory; phases resume. Running the same commands with the Qwen source attempts the primary study; the guards decide admission.
