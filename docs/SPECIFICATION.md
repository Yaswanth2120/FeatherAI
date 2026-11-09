# Feather AI

Local LLM Quantization, Benchmarking & Optimization Platform

## Feather AI

Making large language models lighter, faster, and more accessible.

Open-source · Local-first · Apple Silicon · ML systems engineering

# 4

Development phases

# 8

Planned weeks

# $0

Required cloud spend

The eight-week schedule is a target for an MVP, not a guaranteed delivery time. Cloud spending is optional for larger-model experiments.

## 1. What exactly are we building?

Feather AI is a local-first engineering platform that measures, quantizes, evaluates, and optimizes open-weight language models for specific hardware.

Imagine someone has a MacBook with 8 GB of RAM. They want to run a coding model but don't know:

- Which quantization format will fit?
- How much accuracy will be lost?
- How fast will the model respond?
- Will 3-bit quantization actually outperform 4-bit?
- Can the machine run it without heavy swapping?
- Which model configuration offers the best trade-off?

Feather AI will answer those questions using actual measurements rather than model marketing or guesses.

It will also let users quantize compatible models themselves.

### The complete workflow

1\. Import a model

Local file, supported model repository, or installed runtime

Quantize

Generate smaller weight representations

Benchmark

Measure speed, memory, latency

3\. Evaluate output quality

Code tests, reasoning, perplexity, reference comparisons

4\. Analyze and recommend

Interactive dashboard, quality trade-offs, hardware-fit recommendation

Final artifact

Tested model configuration + benchmark report + reproducibility metadata

An important design distinction: an imported quantized model can be benchmarked immediately, while creating a new quantized artifact requires suitable original or higher-precision weights. Feather AI must never pretend it quantized a model when it merely downloaded an existing quantization.

## 2. What will the finished application look like?

Here is a conceptual preview of the dashboard we will build. The numbers are illustrative, not measurements from your Mac.

## Feather AI

Local Engine

DEVICE

MacBook Air M2 · 8 GB

MODEL FAMILY

Qwen 2.5 Coder 1.5B

Quantization comparison

FP16Q8Q4Q3

Model file

# 1 GB

Generation speed

# 42 tok/s

Coding pass rate

# 74%

Peak memory

# 2 GB

Generation speed by quantization

0 tok/s15 tok/s30 tok/s45 tok/sFP16Q8Q4Q3

Illustrative values, not real benchmark results

Illustrative result

Q4 has the highest measured throughput in this fictional dataset. Q8 preserves more coding accuracy. The best configuration depends on the user's constraints.

Interactive product mockup. Select a precision to inspect sample results. All displayed metrics are invented solely to demonstrate Feather AI's intended interface.

The actual application will have five main pages: Overview, Model Library, Quantization Studio, Benchmark & Evaluation, and Comparison Reports.

## 3. The four development phases

Phase 1

Benchmarking Engine

Weeks 1–2

### Phase 1: Build the measurement foundation

Before optimizing anything, we need to measure how the models perform.

We will build a Python application capable of communicating with local model runtimes, executing standardized test prompts, and recording the results.

What we implement

1. Hardware detection: Apple Silicon chip, physical RAM, OS, runtime versions.
2. Model registry: model name, architecture, parameter count, quantization, tokenizer, size, source.
3. Inference runner: initial Ollama adapter, then llama.cpp and MLX adapters.
4. Benchmark scheduler: warm-up executions, repeated trials, configurable context lengths.
5. Monitoring: process memory, macOS memory pressure, swapping indicators where available.
6. Results persistence: SQLite database with JSON/CSV exports.

Ollama exposes token counts and nanosecond timing fields through its API, which gives us a useful starting measurement interface.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+1



### What we measure

| Metric              | Meaning                                            |
| ------------------- | -------------------------------------------------- |
| File size           | Storage occupied by model artifact                 |
| Peak memory         | Maximum measured runtime memory                    |
| Load latency        | Time needed to load model                          |
| Time to first token | Delay before streaming first output token          |
| Prefill throughput  | Prompt processing speed                            |
| Decode throughput   | Generated tokens per second                        |
| End-to-end latency  | Total time taken for a request                     |
| Thermal behavior    | Performance changes during sustained runs          |
| Swap activity       | Whether macOS starts relying heavily on SSD memory |

We will distinguish runtime memory from whole-system memory, and hot-model performance from cold-start performance.

### Example terminal experience

```
feather models list

feather benchmark run \
  --model qwen2.5-coder:1.5b \
  --backend ollama \
  --suite coding-small \
  --repeats 5

feather benchmark report --latest
```

These are proposed Feather AI commands, not existing commands that will work before development.

Phase 1 result

By the end of this phase, you should be able to compare two quantized models, identify their runtime cost, and reproduce the measurements on the same machine.

Deliverable: working benchmark CLI, database, reports, and automated tests.

### Phase 1 completion criteria

- Run at least 3 supported model configurations.
- Complete 5 measured trials per configuration after warm-up.
- Save raw timing data, not just averages.
- Report median and p95 latency, plus throughput distribution.
- Record runtime versions, hardware, prompt hashes, seed and context settings.
- Handle model failures or memory errors without losing previously recorded results.

Phase 2

Quantization Engine

Weeks 3–4

### Phase 2: Actually compress model weights

This is the central engineering component of Feather AI.

We need to understand quantization mathematically before relying on existing libraries.

The basic process is:

Original floating-point weights

Higher numerical precision

Calculate scale and quantization parameters

Map weights to lower-precision values

Reconstruct approximations and measure error

Quantized weights + metadata

A simple symmetric quantization scheme can be expressed as:

\\[ s=\frac{\max(|W|)}{2^{b-1}-1} \\]

\\[ W_q=\operatorname{clip}\left(\operatorname{round}\left(\frac{W}{s}\right),-2^{b-1}+1,2^{b-1}-1\right) \\]

Here \\(W\\) represents the original weights, \\(b\\) the bit width, and \\(s\\) the scale. This is a teaching implementation, not the exact algorithm used by every GGUF quantization format.

What we implement

- A NumPy-based quantizer for toy tensors.
- Per-tensor and per-channel scaling.
- Weight-only quantization and dequantization.
- Reconstruction-error measurements such as MSE and maximum error.
- An MLX-LM adapter for supported models.
- A llama.cpp/GGUF adapter for compatible architectures.
- Artifact validation, hashes and lineage tracking.

MLX-LM supports conversion and quantization with options for bit width and quantization group size. llama.cpp supports quantizing higher-precision GGUF files, including optional importance-matrix guidance.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub+1



### Important hardware constraint

Your M2 with 8 GB cannot safely quantize arbitrary 7B–14B source models.

For our first real end-to-end quantization, we will use a 0.5B or 1.5B model and verify that the source weights and conversion process fit available memory. Larger experiments can later run on a 64 GB Mac or rented hardware.

### Expected results

We should produce multiple precision variants of the same source model, each with:

- Verified model file and metadata.
- Quantization settings and conversion logs.
- File-size reduction compared with source weights.
- Numerical reconstruction error where measurable.
- Compatibility and inference tests.

Phase 2 result

You can import supported higher-precision weights, generate a new 4-bit model, run it locally, and prove precisely how it differs from the source.

Deliverable: genuine quantization engine with validated model artifacts.

Completion requires at least one successfully quantized language model, tested through generation, and one numerical unit test demonstrating expected quantization error on controlled data.

Phase 3

Quality Evaluation Engine

Weeks 5–6

### Phase 3: Determine what quality we actually lose

Smaller files are meaningless if the compressed model stops solving problems correctly.

Feather AI must measure quality, not infer it from bit width.

Our evaluation engine will test the model across several tasks.

| Evaluation suite        | Example                     | Scoring                           |
| ----------------------- | --------------------------- | --------------------------------- |
| Coding                  | Implement binary search     | Hidden unit tests                 |
| Debugging               | Fix a broken function       | Tests passed                      |
| Algorithms              | Solve small DSA tasks       | Correctness and complexity checks |
| Reasoning               | Mathematical word problems  | Exact or normalized answer        |
| Language modeling       | Predict held-out text       | Perplexity                        |
| Distribution comparison | Compare token probabilities | KL divergence, where supported    |

For coding, evaluation must use isolated subprocesses or containers with strict time, filesystem, and resource limits. Generated code should never execute unrestricted on the developer's machine.

We will use separate calibration, screening, and held-out evaluation datasets. This prevents tuning a quantization to the very questions later used to judge it.

### What a comparison will tell us

Consider these hypothetical outcomes:

| Metric              | FP16   | Q8     | Q4     |
| ------------------- | ------ | ------ | ------ |
| Coding tests passed | 80%    | 79%    | 75%    |
| Perplexity          | 7.20   | 7.22   | 7.55   |
| Model file          | 3.1 GB | 1.6 GB | 1.0 GB |

We would then know that Q4 offers substantial storage savings but loses five percentage points of coding accuracy in this hypothetical test.

The actual objective is to quantify the trade-off, not to prove that quantization is always beneficial.

### Evaluation reliability

We will:

- Use the same source model and tokenizer across precision variants.
- Fix the prompt templates, context, generation limits and decoding settings.
- Record seeds and control stochastic sampling where possible.
- Measure differences across multiple tasks.
- Report confidence intervals for accuracy differences.
- Keep benchmark data separate from tuning data.
- Track missing or failed runs explicitly instead of treating them as successes.

Phase 3 result

Feather AI can make evidence-backed statements such as:

"On this hardware and evaluation suite, Q4 reduces model storage by X%, changes coding pass rate by Y percentage points, and changes median decode throughput by Z%."

Deliverable: accuracy and quality evaluation framework with reproducible results.

Completion requires a held-out evaluation suite, correct scoring, and a report containing both performance and quality uncertainty.

Phase 4

Dashboard & Optimization

Weeks 7–8

### Phase 4: Turn the engineering tools into a usable product

So far, Feather AI will mostly be a CLI and Python backend.

Phase 4 converts it into an interactive local web application with a REST API.

Main screens

Overview

Detect the host machine, installed models, completed experiments, resource usage, and recent results.

Model Library

Inspect imported models, file formats, quantization methods, model lineage, sizes and compatibility.

Quantization Studio

Select an original model, supported target precision, output directory and quantization settings. Track job progress and failures.

Benchmark & Evaluation

Launch controlled experiments. Visualize speed, memory, coding accuracy, perplexity and run-to-run variation.

Comparison Reports

Compare configurations, inspect Pareto trade-offs, apply hardware and accuracy constraints, and export reproducible reports.

### How the recommendation engine works

The user specifies:

- Maximum allowed memory, for example 5 GB.
- Minimum acceptable coding pass rate, for example 70%.
- Desired objective, such as fastest generation or smallest file.
- Supported model family and workload.

Feather AI filters out configurations that fail the constraints, then ranks those that remain using measured performance.

It should also display insufficient evidence when results are incomplete. A fake optimization recommendation is worse than admitting no reliable measurement exists.

Phase 4 result

A developer can open Feather AI in a browser, choose a model, create quantizations, benchmark them, inspect quality differences, and receive a hardware-specific recommendation.

Deliverable: working local application, API, documentation, tests and exportable reports.

## 4. Final system architecture

Feather AI Dashboard

React + TypeScript

Backend API

Python + FastAPI + Job Manager

Quantization

NumPy, MLX, GGUF

Benchmarks

Latency, speed, memory

Evaluation

Quality, code, PPL

Optimizer

Constraint-based ranking

Inference & Model Adapters

Ollama · llama.cpp · MLX-LM

Local storage

SQLite · Model files · Experiment artifacts · Reports

### Technology decisions

| Component                  | Selection                   |
| -------------------------- | --------------------------- |
| Core programming language  | Python 3.12+                |
| Quantization fundamentals  | NumPy                       |
| Apple Silicon quantization | MLX / MLX-LM                |
| GGUF support               | llama.cpp                   |
| Baseline inference         | Ollama                      |
| Backend                    | FastAPI                     |
| Frontend                   | React, TypeScript           |
| Database                   | SQLite                      |
| Charts                     | Recharts                    |
| Testing                    | pytest, Playwright          |
| Packaging                  | Python CLI and local web UI |
| Version control            | Git + GitHub                |
| CI                         | GitHub Actions              |

We will add infrastructure only when it solves a real problem. No Kubernetes, Kafka, microservices, or other distributed-system machinery in version 1. A single-user application on an 8 GB Mac does not need an entire data center's organizational problems.

## 5. Repository structure

```
feather-ai/
├── README.md
├── pyproject.toml
├── .github/
│   └── workflows/
├── configs/
│   ├── models.yaml
│   ├── benchmarks.yaml
│   └── thresholds.yaml
├── feather/
│   ├── cli/
│   ├── api/
│   ├── core/
│   │   ├── hardware.py
│   │   ├── models.py
│   │   └── artifacts.py
│   ├── backends/
│   │   ├── ollama.py
│   │   ├── llama_cpp.py
│   │   └── mlx.py
│   ├── quantization/
│   │   ├── uniform.py
│   │   ├── mlx_adapter.py
│   │   └── gguf_adapter.py
│   ├── benchmarks/
│   │   ├── runner.py
│   │   ├── metrics.py
│   │   └── system_monitor.py
│   ├── evaluation/
│   │   ├── coding.py
│   │   ├── perplexity.py
│   │   └── scoring.py
│   ├── optimization/
│   │   └── recommender.py
│   └── storage/
│       ├── database.py
│       └── repositories.py
├── frontend/
│   ├── src/
│   └── package.json
├── datasets/
│   └── manifests/
├── experiments/
├── tests/
└── docs/
```

Each major experiment will store model identifiers, hashes, quantization configuration, hardware information, software versions, prompts, raw results, and any errors. Without this information, reproducing an experiment becomes guesswork.

## 6. What models will we use?

For your M2 8 GB, I would select Qwen 2.5 Coder 1.5B as the primary research model.

It has accessible Q4, Q8, and FP16 versions, with published downloads of approximately 986 MB, 1.6 GB, and 3.1 GB respectively. This makes it much better suited to our experiments than beginning with Qwen 3.5 9B.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://registry.ollama.ai\&sz=32)

registry.ollama.ai+1



| Model               | Feather AI purpose                         |
| ------------------- | ------------------------------------------ |
| Qwen 2.5 Coder 0.5B | First quantization and low-memory testing  |
| Qwen 2.5 Coder 1.5B | Primary precision-versus-quality study     |
| Qwen 2.5 Coder 3B   | Benchmark scaling and hardware-fit testing |
| Qwen 3.5 4B         | Optional later general-model comparison    |
| 7B–9B models        | Future higher-memory experiments           |

In Phase 3, we must not directly treat differences between different model sizes as quantization losses. Those comparisons are separate experiments.

## 7. What results should we expect from Feather AI?

The objective is not to claim that Feather AI makes every model faster or preserves every answer.

Here are the actual outcomes the project should provide.

Result 1: Measurable model compression

Expected

Starting with compatible higher-precision weights, Feather AI should generate smaller model artifacts and report actual storage reductions.

Result 2: Hardware-specific performance data

Expected

We should learn which configurations perform best on your M2 under controlled context and memory conditions.

Result 3: Quantified accuracy differences

Expected

Feather AI should measure changes in model behavior. It cannot guarantee that low-bit output stays identical to FP16.

Result 4: Faster inference

Not guaranteed

Quantization can improve speed through reduced memory traffic, but particular formats may run slower due to dequantization overhead or inefficient kernels.

Result 5: An optimization recommendation

Expected

Feather AI should choose among tested, supported configurations according to explicit constraints, not invent benchmark scores for untested models.

### Final project acceptance criteria

Feather AI v1.0 completion checklist

0/10

Hardware and model inspection work correctly

Three or more model configurations benchmark successfully

Warm-up runs and repeated measurements are implemented

At least one original model is quantized by Feather AI

Original and quantized model artifacts are validated

Coding accuracy and perplexity evaluation work

Runs retain reproducibility metadata and uncertainty estimates

Dashboard displays real saved experiment data

Recommendation engine applies RAM and accuracy constraints

Tests, documentation, and one reproducible case study are published

Project targets. This checklist is for planning and manual progress tracking; nothing is marked complete yet.

## 8. What happens after the four phases?

The first four phases produce Feather AI v1.0, a working optimization and evaluation platform.

Further development can introduce three more advanced capabilities.

| Future release | Feature                       | Engineering challenge                               |
| -------------- | ----------------------------- | --------------------------------------------------- |
| v1.1           | Mixed-precision quantization  | Choose precision per layer or tensor                |
| v1.2           | Automatic quantization search | Optimize quality under RAM and latency budgets      |
| v2.0           | Local inference gateway       | Route requests between quantized models dynamically |

For example, Feather AI might eventually determine that selected sensitive layers benefit from 8-bit weights while others can use 4-bit weights, then generate and validate that configuration.

That is a more ambitious research problem than simple uniform quantization, and it should be attempted only after the measurement system is trustworthy.

## 9. Exact starting point

Here is the setup we need on your Mac.

| Requirement                  | Purpose                  |
| ---------------------------- | ------------------------ |
| Python 3.12+                 | Main development         |
| Git                          | Version control          |
| Ollama                       | Initial local inference  |
| Xcode Command Line Tools     | Native build tools       |
| VS Code                      | Editor                   |
| NumPy, pytest, psutil, httpx | Initial Python stack     |
| SQLite                       | Local experiment storage |

The first model download:

```
ollama pull qwen2.5-coder:1.5b
```

The first engineering milestone is intentionally narrow:

Build a Python program that sends a coding prompt to this model, records its output and inference statistics, runs it five times, and saves a reproducible benchmark report.

We should complete that before writing a dashboard.

## 10. What will Feather AI demonstrate in your portfolio?

By the end of this project, you should have concrete evidence of experience with:

- LLM internals and numerical representations.
- Model quantization and inference runtimes.
- Apple Silicon performance profiling.
- Benchmark design and statistical evaluation.
- Python backend and API development.
- Model artifact management and reproducibility.
- Frontend dashboards and data visualization.
- Performance engineering under resource constraints.

Your eventual GitHub project should include its architecture, a clear installation guide, executable tests, real benchmark results, and a technical report explaining the conclusions. Report failures and negative results too. Those are often more informative than conveniently perfect speedup charts.

### Final project definition

> Feather AI is an open-source, local-first LLM optimization laboratory that enables developers to quantize compatible open-weight models, benchmark inference performance, evaluate quality degradation, and select the most suitable model configuration for their hardware.

Version 1.0 will be considered complete when all four phases work together on your M2 Mac, from importing a model to producing a validated quantization and reproducible recommendation.

That is our scope. A real engineering platform with defensible measurements, not merely a polished interface around somebody else's quantized models.