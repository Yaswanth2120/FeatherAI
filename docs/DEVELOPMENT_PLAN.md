# Development plan

1. Audit and preserve the tested baseline.
2. Add bounded resumable orchestration and documented project agent roles.
3. Build a local pinned llama.cpp toolchain and quantize genuine Qwen 2.5 Coder 0.5B FP16 GGUF weights, sequentially. Record source revision, hashes, tools, logs, generation and baseline outputs. Attempt smaller model if memory admission fails; never bypass evidence gates.
4. Independently build isolated quality scoring and evidence-based recommendations.
5. Integrate process-based, persisted FastAPI jobs with localhost and path constraints.
6. Build the five-page React/TypeScript dashboard using actual API data.
7. Benchmark comparable variants, exercise evaluation and exports, review security/failure handling, run full checks and commit verified increments.
8. Report all acceptance criteria, measurements, commit references and exact remaining blockers.

Heavy runs are serialized. Docker absence blocks arbitrary generated-code evaluation but not deterministic reasoning evaluation, API work, frontend work or statistical unit tests.
