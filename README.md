# Feather AI

A local-first laboratory for quantizing language models, measuring performance, evaluating quality and selecting tested configurations. Python/FastAPI backend, React/TypeScript dashboard, SQLite storage. One project root; application code is in `backend/`.

**Current status (v0.2):**
- **Working:** coding evaluation runs in a hardened Docker sandbox, and the reasoning and coding datasets are hash-pinned.
- **Complete study:** the pre-registered quantization study is complete for SmolLM2-135M ([study](docs/studies/quantization-study-v0.2.md)).
- **End-to-end:** one traceable SmolLM2 workflow went through the API (task08 complete); its recommendation truthfully found no configuration that meets the pre-registered 70% coding requirement.
- **Qwen2.5-Coder-0.5B study:** complete under clean conditions ([report](docs/studies/quantization-study-v0.2-qwen.md)). Q4_K_M decodes 1.77× faster than F16 with 55% less peak memory; no detectable quality loss at these sample sizes.
- **coding-v2 (360 held-out MBPP problems):** on Qwen, Q8_0 and Q4_K_M preserved coding performance within the pre-registered ±10 pp margin (−0.28 pp and −0.83 pp vs F16). This is a coding-only result; MBPP is public, so contamination is possible.
- **Live Codex automation:** verified with codex-cli 0.167.0 (`experiments/automation-selftest-live.json`).
- **Qwen end-to-end:** completed on retry (workflow `5cf6580c`, passes the task08 gate). The recommendation is `no_eligible_configuration`: coding lower bounds 0.52 and 0.46 are below the pre-registered 0.70. The first attempt (`e14e38ca`) was rejected by the memory guard and is preserved.
- **Deferred:** the optional 1.5B experiment.

See [the final report](docs/FINAL_REPORT.md).

## Measured results

SmolLM2-135M-Instruct on an Apple M2 with 8 GiB RAM, using the same F16 source and Feather-created quantizations:

| Variant | Decode tok/s, median [95% CI] | Median peak process footprint | Coding pass rate (30 held-out problems) | Mean KLD vs F16 |
| --- | ---: | ---: | ---: | ---: |
| F16 (source) | 220.8 [207.1, 225.8] | 472.1 MiB | 36.7% | Reference |
| Q8_0 | 252.6 [244.9, 256.4] | 300.5 MiB | 26.7% | 0.0012 |
| Q4_K_M | 262.8 [246.2, 268.5] | 264.2 MiB | 36.7% | 0.0445 |

These are measurements from this machine, not hardware-independent promises. The study used 15 server sessions and 20 measured requests per variant; 13 sessions were flagged for memory pressure. Coding comparisons have wide confidence intervals (30 problems), so no quality difference was detected. The F16 reasoning baseline solved only 3/160 held-out items, making reasoning degradation uninformative. The fallback-model switch and other protocol deviations are documented; see the [full study and limitations](docs/studies/quantization-study-v0.2.md) and [decision record](docs/DECISIONS.md).

## Install and open

The native study targets **macOS on Apple Silicon**. Requires Git, Python 3.12+, [uv](https://docs.astral.sh/uv/), Node 20.19+ and npm. Native llama.cpp builds also require CMake and Xcode Command Line Tools. Ollama is optional for installed-runtime inference. llama.cpp is required for local GGUF conversion/native inference; Docker is required for generated-code scoring.

```sh
git clone https://github.com/Yaswanth2120/FeatherAI.git
cd FeatherAI
uv sync --locked
npm --prefix frontend ci
npm --prefix frontend run build
uv run feather serve
```

Open **http://127.0.0.1:8000**. The same localhost server serves API and dashboard; API docs are at `/docs`. Stop with Ctrl-C. No cloud account is required to use the local lab; starting the server does not publish a service.

A fresh clone includes source code and committed reports, but **not model weights, compiled llama.cpp tools, runtime databases or ignored study working files**. You can open the dashboard before installing native tools; model discovery needs a running Ollama service. To enable GGUF conversion and native inference, build the pinned toolchain below, then start the server with:

```sh
uv run feather serve --quantizer .feather/tools/llama.cpp/build/bin/llama-quantize --server .feather/tools/llama.cpp/build/bin/llama-server
```

To rebuild the recorded toolchain on a fresh clone:

```sh
mkdir -p .feather/tools
git clone https://github.com/ggml-org/llama.cpp.git .feather/tools/llama.cpp
git -C .feather/tools/llama.cpp checkout 9d340ec341e079d2c28afd50871afce358cdbf48
cmake -S .feather/tools/llama.cpp -B .feather/tools/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build .feather/tools/llama.cpp/build --target llama-quantize llama-server llama-perplexity -j 2
```

## Download the study source weights

Run from the cloned repository root. This explicit download is approximately **271 MB (258 MiB)**. It imports the higher-precision F16 source from [bartowski/SmolLM2-135M-Instruct-GGUF](https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF), pinned to revision `09816acd5d99df7be770d85ea30822623dab342c`. Feather creates Q8/Q4 itself; this does not download a prequantized variant.

```sh
mkdir -p models
curl --fail --location --retry 3 \
  'https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/09816acd5d99df7be770d85ea30822623dab342c/SmolLM2-135M-Instruct-f16.gguf' \
  --output models/SmolLM2-135M-Instruct-f16.gguf.partial &&
printf '%s  %s\n' \
  'f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57' \
  'models/SmolLM2-135M-Instruct-f16.gguf.partial' | shasum -a 256 --check &&
mv models/SmolLM2-135M-Instruct-f16.gguf.partial models/SmolLM2-135M-Instruct-f16.gguf
```

Proceed only after the checksum reports `OK`. The partial file is promoted to the model filename only after verification. Source provenance and the full expected hash are also recorded in [experiments/quantization-evidence.json](experiments/quantization-evidence.json). Weights remain gitignored. See the upstream model repository for its license and terms.

A fresh clone can read the committed results, but acceptance checks that re-hash local source/output files also require the corresponding weights and generated artifacts; downloading the F16 source alone does not restore ignored study outputs.

## CLI

```sh
uv run feather hardware
uv run feather models list
uv run feather benchmark run --model qwen3:1.7b --repeats 5 --warmups 1 --context 1024 --tokens 128
uv run feather benchmark report --latest
uv run feather benchmark report --latest --output report.json
uv run feather benchmark report --latest --format csv --output trials.csv
uv run feather artifact models/SmolLM2-135M-Instruct-f16.gguf
mkdir -p .feather   # ignored working directory; keeps generated manifests/logs out of Git
uv run feather quantize --source models/SmolLM2-135M-Instruct-f16.gguf --output-dir .feather/my-q4 --target Q4_K_M --executable .feather/tools/llama.cpp/build/bin/llama-quantize
```

The model name must exactly match `models list`; Feather does not silently download models. Global `--db` and `--ollama-url` flags precede the command. The standalone benchmark CLI uses Ollama. The `study` command and dashboard additionally support owned native llama.cpp inference.

A complete sequential source/precision study:

```sh
uv run feather study --source models/SmolLM2-135M-Instruct-f16.gguf --output-dir .feather/new-study --quantizer .feather/tools/llama.cpp/build/bin/llama-quantize --server .feather/tools/llama.cpp/build/bin/llama-server --repeats 5
```

The output directory must be new; its parent must exist. The source must contain genuine F32/F16/BF16 weights. Downloads are explicit: the tested FP16 source repository, pinned revision, URL and local SHA-256 appear in `experiments/quantization-evidence.json`. Model weights are ignored by Git. The command creates Q4/Q8 itself, validates nonempty generation against a baseline, records repeated performance, scores reasoning, attempts isolated coding, computes recommendations and exports raw evidence. Unsupported coding/perplexity remain explicit; a completed study with limitations is not a complete MVP acceptance result.

### v0.2 commands

```sh
uv run feather sandbox status                       # is isolated coding execution available? (blocked => no code runs)
uv run feather sandbox prepare                      # explicit pull of the pinned python:3.12-alpine digest (~20 MB)
uv run feather sandbox selftest --output experiments/sandbox-selftest.json
uv run feather automation doctor                    # offline preflight; add --probe to test account/model access
uv run feather e2e --source models/SmolLM2-135M-Instruct-f16.gguf   # one traceable API workflow (task08)
uv run feather quant-study --source models/SmolLM2-135M-Instruct-f16.gguf --out .feather/study-smollm2-local \
    --phases convert,validate,dryrun,benchmark,quality,perplexity --export experiments/studies/smollm2-local
```

`quant-study` follows the protocol pre-registered in [docs/DECISIONS.md](docs/DECISIONS.md):
- Feather creates the Q8_0 and Q4_K_M variants itself, with a measured-memory admission guard.
- The benchmark runs 5 order-balanced rounds of server sessions under `/usr/bin/time -l`, giving 20 measured trials per variant, with bootstrap CIs.
- Quality is measured on the hash-pinned `reasoning-v1` and `coding-v1` held-out splits, with paired comparisons.
- PPL and KLD are computed with `llama-perplexity`.

Phases are resumable. For the cleanest numbers, run the study from Terminal with VS Code, browsers and Docker closed, except for the quality phase, which needs Docker for coding. The toolchain build above includes `llama-perplexity` for PPL/KLD.

## Dashboard workflow

1. **Overview:** actual hardware, job counts, saved benchmark chart.
2. **Model Library:** installed Ollama models, plus artifacts returned by completed API quantization jobs when available; study-only artifacts are not automatically imported.
3. **Quantization Studio:** enter an FP16 GGUF filename relative to `models/` (e.g. `SmolLM2-135M-Instruct-f16.gguf`) and target precision. RAM/disk admission runs before conversion.
4. **Benchmark & Evaluation:** choose an installed model or completed quantization artifact. Native jobs validate artifact path/hash before loading. Heavy jobs are sequential; inspect, cancel and review persisted results. Choose legacy three-item screening suites or the hash-pinned reasoning-v1 and coding-v1 held-out suites. Saved results identify their suite and split.
5. **Comparison Reports:** select saved runs, inspect comparability, export JSON/CSV and apply memory/quality constraints. Choose the quality evidence suite; the dashboard defaults to coding-v1 held-out evidence and a 70% minimum quality requirement. Missing evidence returns insufficient evidence; the measured SmolLM2 configurations do not meet that quality requirement. A 0% threshold is diagnostic only.

For frontend development, `npm --prefix frontend run dev` serves a localhost Vite proxy. Production uses the same-origin FastAPI static mount. See [architecture and API usage](docs/ARCHITECTURE.md).

## Checks and autonomous development

```sh
uv run pytest
npm --prefix frontend run build
# With a populated local API running at port 8000:
npm --prefix frontend test
uv run python automation/orchestrator.py --dry-run
uv run python automation/orchestrator.py --status
```

The browser test uses installed Google Chrome by default; see `frontend/playwright.config.ts`. It exercises real saved local data and exports, not production mock data. CI runs Python tests and the frontend build. Agent roles and bounded development automation are documented in [automation/README.md](automation/README.md).

Reports retain raw timing fields, prompts, outputs, failures, seeds, context, hashes, runtime build, hardware and uncertainty. Review exported content before sharing. System RAM is distinguished from sampled process RSS; GPU allocations and unsampled/startup peaks are not fully captured. [Methodology and limitations](docs/METHODOLOGY.md) explain what each number means.

## License

Original Feather AI code and documentation are available under the [MIT License](LICENSE). Third-party dependencies, model weights and third-party dataset/corpus material retain their own licenses; the MIT license does not replace those terms.
