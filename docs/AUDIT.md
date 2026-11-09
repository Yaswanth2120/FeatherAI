# Repository audit — 2026-11-09

Root: `/Users/yaswanth/Desktop/FeatherAI`. No nested project or backup folder.

Baseline: 15 tests passed under Python 3.12.9 on Apple M2 / 8 GiB unified memory. Codex CLI 0.142.3 provides `exec`, `exec resume`, `exec review`, JSON events, output schemas and sandbox selection. Agent definitions follow official standalone project TOML documentation.

Existing modules: core hardware and GGUF inspection; Ollama streaming adapter; repeated benchmark runner; SQLite experiments/trials; symmetric NumPy quantizer; managed llama-quantize conversion; argparse CLI. Commands: hardware, models list, benchmark run/report, artifact, quantize.

Baseline dependencies: NumPy, httpx, psutil, gguf; pytest development tests. Existing report experiments/m2-qwen3-smoke.json contains five successful measured trials after one warmup on qwen3:1.7b (32 generated-token budget). It is a smoke benchmark, not a precision comparison or quality evaluation.

Git baseline contained uncommitted replacement of old harness, tests, configs and reports plus new backend/tests/docs. These changes were produced in this session's earlier requested work and are preserved. No remote push is authorized.

Verified gaps: no real language-model conversion or generation validation; no isolated coding evaluation, recommendations, API, dashboard, orchestration or end-to-end precision study. llama-quantize and llama-cli were absent from PATH. Docker executable exists but daemon is unavailable. Existing benchmark reports explicitly lack runtime peak memory. Available RAM was approximately 2 GiB before unloading the installed Ollama model.
