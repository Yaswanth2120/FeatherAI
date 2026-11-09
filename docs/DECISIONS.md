# Design decisions

- Preserve a single repository with Python under backend; terminal command remains feather.
- Reuse the current Ollama benchmark and SQLite modules rather than replace them.
- Build llama.cpp inside ignored .feather/tools rather than install machine-wide packages. Pin and record its source commit.
- Prefer official Qwen 0.5B FP16 GGUF source for the first conversion; downloading higher-precision GGUF is source import, not a downloaded quantized result.
- Heavy inference and conversion run sequentially on the 8 GiB host.
- Generated Python code requires a restricted Docker container. Docker daemon absence is reported as unsupported, with no host-execution fallback.
- Autonomous Codex execution uses documented exec JSON mode and structured independent review. Local commits are permitted; remote publication is not.
- Dashboard and API stay local. Sites hosting registration/deployment is explicitly excluded by the user's no-publishing instruction.

Sources: https://learn.chatgpt.com/docs/agent-configuration/subagents and https://learn.chatgpt.com/docs/non-interactive-mode; llama.cpp quantization usage https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md.
