# Measurement and quantization methodology

## Source and conversion

The Qwen 2.5 Coder 0.5B official FP16 GGUF source was downloaded at a pinned revision. The memory admission guard rejected conversion: 2,801,286,912 bytes estimated vs 1,807,138,816 available. This is a conservative estimate, not proof the converter could never run. The guard was not bypassed.

The smaller fallback was SmolLM2-135M-Instruct FP16 from bartowski/SmolLM2-135M-Instruct-GGUF revision 09816acd5d99df7be770d85ea30822623dab342c. Its higher-precision GGUF is source import; Q4_K_M and Q8_0 artifacts were generated locally by Feather using llama.cpp 9d340ec341e079d2c28afd50871afce358cdbf48. No downloaded prequantized model is counted as a Feather conversion. Model license/provenance remain upstream; weights are not redistributed in Git.

Preflight validates floating-point tensor types and available memory/disk. Conversion preserves logs and atomic job status. Output checks include source stability, tensor names/shapes/counts, architecture, canonical length-delimited tokenizer fingerprints, actual quantized tensor types and full hashes. Generation validation loads the exact output and requires nonempty text. Nonempty generation establishes runtime compatibility, not correctness. The NumPy symmetric quantizer is a teaching implementation using int8 storage, not a packed language-model format.

## Performance

Final case study: three variants of the exact same source/tokenizer; native localhost server; ChatML-v1 template; context1024, temperature0, seed42, max128 generated tokens, prompt-cache disabled, one server slot, two CPU threads, default Metal offloading, one warmup plus five measured requests. Server build/executable/template/settings identity is retained. Warmups and failed trials are excluded from summaries; completeness is required for comparisons. Each trial commits independently. The tested coding prompt is a timing workload, not a correctness score.

TTFT is client-observed arrival of the first nonempty response chunk. Wall time is client elapsed request time. Prefill/decode token counts and nanosecond-equivalent timings come from the runtime; raw native timing objects are preserved. Median/p95 and min/max are descriptive over five trials. No throughput confidence interval or thermal stability claim is made. Prompt cache is explicitly disabled for native runs. Server load/startup is excluded from the hot-model benchmark. Native first requests after validation are not cold starts.

Memory is peak sampled process-tree resident set size during measured requests at 50ms intervals. It excludes startup peak and may omit GPU/unified-memory allocations or short peaks. It is not total machine RAM usage or a guarantee of model fit. Hardware metadata includes system RAM and swap snapshots, separately labeled. No sustained thermal benchmark is implemented.

## Quality and recommendations

Reasoning uses three deterministic public screening questions and strict whitespace-normalized exact answers. All three variants scored0/3; explanatory answers do not earn credit even if they contain an answer. Wilson95% intervals are descriptive for the tiny suite (0 to approximately0.5615). Repeating deterministic questions does not create independent evidence. Difference bounds are conservative marginal-interval differences, not calibrated paired confidence intervals. No useful reasoning quality is established by this run.

Coding scoring requires a restricted Docker container with no network/host mounts, read-only root, capability drop, process/memory/CPU/file-output limits and bounded wall time. The daemon was unavailable, so every coding result is unsupported rather than zero or success. Generated code never ran directly on the Mac. Perplexity from supplied held-out token log probabilities is implemented, but these runtime adapters do not collect that evidence; PPL is explicitly unsupported. Separate calibration and held-out language-model corpora are not yet implemented.

Recommendations join completed raw reports by exact artifact/source hashes, then require common workload and combined performance/quality settings. Unknown metrics are never filled. Memory and latency constraints use measured values; quality thresholds use the lower available confidence bound. With min_quality0 the ranking is diagnostic speed ranking only. The default70% quality requirement yields no eligible tested configuration. For useful deployment recommendations, add a larger held-out suite, isolated coding results and fuller memory accounting.

## Reproduction limits

The committed experiments JSON preserves raw evidence; weights, local SQLite state, toolchain binaries and node_modules are ignored. Source/output hash gates need those local weights to exist. Performance will vary with OS, thermal state, memory pressure and concurrent apps. Archived data alone does not rerun a model or constitute a full MVP pass.
