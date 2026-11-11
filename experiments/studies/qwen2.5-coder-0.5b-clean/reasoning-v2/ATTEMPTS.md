# reasoning-v2 launch attempts

`run.json` records the start and every restart automatically. These are the attempts behind those records (times in UTC, 2026-11-11):

| Time | Commit | Launched by | Outcome |
|---|---|---|---|
| 15:56:38 (`started`) | `acc33d8` | Terminal (cannot be attributed further from logs) | Crashed with `KeyError: 'reasoning-v2'` before any generation. The suite was missing from the runner registry; fixed in `de4ccf9`. |
| 15:56:51 (resumption 1) | `acc33d8` | Terminal (cannot be attributed further from logs) | Same `KeyError`; no output. |
| 15:58:53 (resumption 2) | `de4ccf9` | Background waiter, after it measured 1 821 MiB available | The F16 runtime guard rejected it: 1 802 928 128 bytes available vs 1 803 296 640 required. Memory fell between the waiter's check and the start. No output. |
| 16:07:52 (resumption 3) | `de4ccf9` | The user's Terminal run | Completed every run: screening-F16 40/40, held-out F16/Q8_0/Q4_K_M 600/600 each, no infrastructure failures. |

Every result file comes from the 16:07:52 run at clean commit `de4ccf9` (`provenance` in each `run.json` entry). No partial output from a failed attempt exists. No item was excluded.
