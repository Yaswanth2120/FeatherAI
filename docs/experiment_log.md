# FeatherAI experiment log

Append-only. Every protocol amendment, tag, invalid or rerun attempt, and deviation goes here with date and reason.

| Date | Event | Details |
|---|---|---|
| 2026-11-08 | Harness built (Task 2) | llama.cpp v0.9.0 @ cd959bae99ff; uv.lock (mlx 0.32.0, mlx-lm 0.31.3) |
| 2026-11-08 | Pre-tag extractor fixes | Fixture tests found: `-$5` unparsed; `1/2` read as 1; a final `Answer:` with no number fell back to an earlier line; Markdown `**18**` / `**Answer:**` and lowercase `answer:` treated as format failures. Fixed before any tag and before any model output. |
| | prereg-v1 | pending: page-in limit from idle pilot |
