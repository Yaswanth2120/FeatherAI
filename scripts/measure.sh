#!/bin/bash
# FeatherAI measurement entrypoint. Run from Terminal.app with VS Code
# (and Claude Code) closed, on AC power, Low Power Mode off, after a fresh reboot.
#
#   scripts/measure.sh check                     # show environment + gate status, run nothing
#   scripts/measure.sh pilot [windows] [secs]    # idle-harness pilot (no model)
#   scripts/measure.sh matrix configs/matrix.yaml
#
# One session per boot: `pilot` and `matrix` each start a session, and the
# harness refuses a second session on the same kern.boottime.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ "${TERM_PROGRAM:-}" = "vscode" ]; then
  echo "ABORT: running inside VS Code's terminal. Use Terminal.app with VS Code closed." >&2
  exit 2
fi
if [ "${TERM_PROGRAM:-}" != "Apple_Terminal" ]; then
  echo "WARNING: TERM_PROGRAM=${TERM_PROGRAM:-unset} (expected Apple_Terminal)." >&2
fi

PY=(uv run --frozen --no-sync python -m harness.cli)
cmd="${1:-}"; shift || true
case "$cmd" in
  check)
    exec "${PY[@]}" check ;;
  pilot)
    SESSION="$("${PY[@]}" session-start)"
    echo "session: $SESSION"
    # caffeinate -i: keep the Mac from idle-sleeping during the session.
    exec caffeinate -i "${PY[@]}" pilot --session "$SESSION" \
      --windows "${1:-5}" --window-s "${2:-300}" ;;
  matrix)
    [ -f "${1:-}" ] || { echo "usage: $0 matrix <matrix.yaml>" >&2; exit 64; }
    "${PY[@]}" check >/dev/null || { echo "ABORT: gates failing; run '$0 check'" >&2; exit 2; }
    SESSION="$("${PY[@]}" session-start)"
    echo "session: $SESSION"
    exec caffeinate -i "${PY[@]}" matrix --session "$SESSION" --matrix "$1" ;;
  *)
    sed -n '2,10p' "$0"; exit 64 ;;
esac
