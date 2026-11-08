# Pilot attempt 1: REJECTED

Rejected: post-boot background activity; protocol amended before tag.

- Started 3.6 min after boot (boot 16:00:07, session 16:03:46, 2026-11-08).
- Gates all passed (AC, Low Power Mode off, thermal clean, swap 0 MB).
- Windows 0-9 max 2045 pages/s (window 7: ~959 MB paged in over 30 s); windows 10-19 max 94 pages/s.
- Pre-declared rule would give limit = 4090 pages/s, too loose to detect paging on 8 GB.
- User decision 2026-11-08: add a 15-min uptime settle gate (min_uptime_before_session_s: 900)
  and rerun the pilot. This file is kept as attempt 1 and is never used for the limit
  (scripts/derive_pagein_limit.py also refuses it: no uptime recorded).
