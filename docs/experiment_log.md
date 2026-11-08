# FeatherAI experiment log

Append-only. Every protocol amendment, tag, invalid or rerun attempt, and deviation goes here with date and reason.

| Date | Event | Details |
|---|---|---|
| 2026-11-08 | Harness built (Task 2) | llama.cpp v0.9.0 @ cd959bae99ff; uv.lock (mlx 0.32.0, mlx-lm 0.31.3) |
| 2026-11-08 | Pre-tag extractor fixes | Fixture tests found: `-$5` unparsed; `1/2` read as 1; a final `Answer:` with no number fell back to an earlier line; Markdown `**18**` / `**Answer:**` and lowercase `answer:` treated as format failures. Fixed before any tag and before any model output. |
| 2026-11-08 | First-number rule made explicit | Prediction = first number after the last `Answer:` (`5 + 3 = 8` -> 5; `18 apples` -> 18; `25%` -> 25); a word before the number (`about 18`) = format failure. Behaviour unchanged, now documented and tested. Protocol refinement closed by the user. |
| 2026-11-08 16:03 | Pilot attempt 1: REJECTED | results/sessions/20261108-160346. Started 3.6 min after boot; windows 0-9 peaked at 2045 pages/s (~959 MB in 30 s), windows 10-19 at <=94 pages/s. The rule would have given a 4090 pages/s limit, too loose. Kept on file, never used. |
| 2026-11-08 | Protocol amended before tag | User added post-boot settle gate: min_uptime_before_session_s = 900 (15 min, about 1.7x the single observed quiet point of 8-9 min). Applies to the pilot and every measured run. Rule for the limit unchanged. Pilot now records top processes per window (diagnostic only). |
| 2026-11-08 | Finding: Desktop is iCloud-synced | ~/Desktop (project location) syncs via iCloud Drive (FXICloudDriveDesktop=1); fileproviderd ~54% CPU observed at 27 min uptime. Upload reads would appear as page-ins. Pending user decision on relocating the project. |
| | prereg-v1 | pending: page-in limit from idle pilot |
