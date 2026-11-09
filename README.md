# Feather AI

A local-first laboratory for quantizing language models, measuring performance, evaluating quality and selecting tested configurations. Python/FastAPI backend, React/TypeScript dashboard, SQLite storage. One project root; application code is in `backend/`.

**Current status (v0.2):**
- **Working:** coding evaluation runs in a hardened Docker sandbox, and the reasoning and coding datasets are hash-pinned.
- **Complete study:** the pre-registered quantization study is complete for SmolLM2-135M ([study](docs/studies/quantization-study-v0.2.md)).
- **Blocked:** the primary Qwen2.5-Coder-0.5B study (memory guards), live Codex automation (CLI/account model availability), and full end-to-end acceptance (task08).

See [the final report](docs/FINAL_REPORT.md).

## Install and open

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), Node 20.19+ and npm. Ollama is optional for installed-runtime inference. llama.cpp is required for local GGUF conversion/native inference; Docker is required for generated-code scoring.

```sh
cd /Users/yaswanth/Desktop/FeatherAI
uv sync --locked
npm --prefix frontend ci
npm --prefix frontend run build
uv run feather serve
```

Open **http://127.0.0.1:8000**. The same localhost server serves API and dashboard; API docs are at `/docs`. Stop with Ctrl-C. No cloud account is required, and nothing is published.

This checkout already has a local llama.cpp build under ignored `.feather/tools/llama.cpp`. To enable native jobs with it:

```sh
uv run feather serve --quantizer .feather/tools/llama.cpp/build/bin/llama-quantize --server .feather/tools/llama.cpp/build/bin/llama-server
```

To rebuild the recorded toolchain on a fresh clone:

```sh
mkdir -p .feather/tools
git clone https://github.com/ggml-org/llama.cpp.git .feather/tools/llama.cpp
git -C .feather/tools/llama.cpp checkout 9d340ec341e079d2c28afd50871afce358cdbf48
cmake -S .feather/tools/llama.cpp -B .feather/tools/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build .feather/tools/llama.cpp/build --target llama-quantize llama-server -j 2
```

## CLI

```sh
uv run feather hardware
uv run feather models list
uv run feather benchmark run --model qwen3:1.7b --repeats 5 --warmups 1 --context 1024 --tokens 128
uv run feather benchmark report --latest
uv run feather benchmark report --latest --output report.json
uv run feather benchmark report --latest --format csv --output trials.csv
uv run feather artifact models/source-f16.gguf
uv run feather quantize --source models/source-f16.gguf --output-dir experiments/my-q4 --target Q4_K_M --executable /path/to/llama-quantize
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
uv run feather automation doctor [--probe]          # Codex automation preflight
uv run feather quant-study --source models/<source-f16>.gguf --out .feather/study-<name> \
    --phases convert,validate,dryrun,benchmark,quality,perplexity --export experiments/studies/<name>
```

`quant-study` follows the protocol pre-registered in [docs/DECISIONS.md](docs/DECISIONS.md):
- Feather creates the Q8_0 and Q4_K_M variants itself, with a measured-memory admission guard.
- The benchmark runs 5 order-balanced rounds of server sessions under `/usr/bin/time -l`, giving 20 measured trials per variant, with bootstrap CIs.
- Quality is measured on the hash-pinned `reasoning-v1` and `coding-v1` held-out splits, with paired comparisons.
- PPL and KLD are computed with `llama-perplexity`.

Phases are resumable. For the cleanest numbers, run the study from Terminal with VS Code, browsers and Docker closed, except for the quality phase, which needs Docker for coding. The tools directory also needs `llama-perplexity`:

```sh
cmake --build .feather/tools/llama.cpp/build --target llama-perplexity
```

## Dashboard workflow

1. **Overview:** actual hardware, job counts, saved benchmark chart.
2. **Model Library:** installed Ollama models and Feather-created artifacts.
3. **Quantization Studio:** enter an FP16 GGUF filename relative to `models/` (e.g. `SmolLM2-135M-Instruct-f16.gguf`) and target precision. RAM/disk admission runs before conversion.
4. **Benchmark & Evaluation:** choose an installed model or completed quantization artifact. Native jobs validate artifact path/hash before loading. Heavy jobs are sequential; inspect, cancel and review persisted results. Reasoning scores use public screening tasks, not a held-out benchmark.
5. **Comparison Reports:** select saved runs, inspect comparability, export JSON/CSV and apply memory/quality constraints. Missing evidence returns insufficient evidence. Default 70% minimum quality rejects the tested model; a 0% threshold is diagnostic only.

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
