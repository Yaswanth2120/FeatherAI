# Remaining work

Implemented: real GGUF conversion and native generation; warmups/repeated measurements; source/artifact/tokenizer provenance; public reasoning and Docker-isolated coding tooling; confidence intervals; deterministic recommendations; localhost FastAPI jobs; five-page React dashboard; bounded resumable development automation; Python/API/browser tests and real saved case-study evidence.

Blocked acceptance: Docker daemon unavailable, so real coding accuracy is unmeasured. Full task04 coding comparison and task08 complete end-to-end acceptance remain blocked. Qwen0.5B conversion exceeded available-RAM admission; smaller SmolLM2 fallback succeeded.

Next: provision the local isolated Docker evaluator within memory limits; run coding against FP16/Q4/Q8; add disjoint held-out datasets; collect supported perplexity; improve useful task quality; add MLX conversion; measure Metal allocations/startup peak and sustained thermal/swap behavior. Re-run the original v1.0 acceptance checklist only after those measurements exist.
