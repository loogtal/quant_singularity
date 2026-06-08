#!/usr/bin/env bash
# ── Quant Singularity watchdog ────────────────────────────────────────────────
# Keeps main.py running 24/7.  Restarts automatically on crash with exponential
# back-off (5s → 10s → 20s … up to 300s) to avoid hammering the exchange on
# repeated failures.
#
# Usage:
#   chmod +x scripts/start.sh
#   ./scripts/start.sh            # foreground (Ctrl-C to stop)
#   nohup ./scripts/start.sh &    # background (check logs/watchdog.log)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$DIR/.venv/bin/python"
MAIN="$DIR/main.py"
LOG_DIR="$DIR/storage/logs"
WATCHDOG_LOG="$LOG_DIR/watchdog.log"

mkdir -p "$LOG_DIR"

log() {
    local ts
    ts="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "[$ts] $*" | tee -a "$WATCHDOG_LOG"
}

log "=== Quant Singularity watchdog starting ==="
log "Python: $PYTHON"
log "Main:   $MAIN"

BACKOFF=5
MAX_BACKOFF=300
CONSECUTIVE_FAILS=0

while true; do
    log "Starting main.py ..."
    START_TS=$(date +%s)

    # Run with live trading log file (append, never truncate)
    # Use setsid to give a new session so stdin/stdout redirects work cleanly
    if setsid "$PYTHON" "$MAIN" >> "$LOG_DIR/trading.log" 2>&1; then
        EXIT_CODE=0
    else
        EXIT_CODE=$?
    fi

    END_TS=$(date +%s)
    UPTIME=$(( END_TS - START_TS ))

    if [ "$EXIT_CODE" -eq 0 ]; then
        log "main.py exited cleanly (uptime ${UPTIME}s) — restarting in ${BACKOFF}s"
        CONSECUTIVE_FAILS=0
        BACKOFF=5
    else
        CONSECUTIVE_FAILS=$(( CONSECUTIVE_FAILS + 1 ))
        log "main.py CRASHED (exit=$EXIT_CODE uptime=${UPTIME}s fails=${CONSECUTIVE_FAILS}) — restarting in ${BACKOFF}s"

        # Double back-off on repeated crashes (cap at MAX_BACKOFF)
        BACKOFF=$(( BACKOFF * 2 ))
        if [ "$BACKOFF" -gt "$MAX_BACKOFF" ]; then
            BACKOFF=$MAX_BACKOFF
        fi

        # After 5 consecutive crashes send a Telegram alert if possible
        if [ "$CONSECUTIVE_FAILS" -ge 5 ]; then
            "$PYTHON" - <<'PYEOF' 2>/dev/null || true
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if '__file__' in dir() else '.')
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass
try:
    from monitoring.alerting import Alerting
    Alerting().halt("main.py crashed 5+ times — watchdog waiting 300s before retry")
except Exception:
    pass
PYEOF
        fi
    fi

    sleep "$BACKOFF"
done
