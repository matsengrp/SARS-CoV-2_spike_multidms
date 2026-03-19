#!/usr/bin/env bash
# Check remote git status and tmux session for a named run.
#
# Usage: scripts/remote-status.sh [--host HOST] [run_name]
set -euo pipefail

# Parse --host override
HOST_OVERRIDE=""
if [ "${1:-}" = "--host" ]; then
    HOST_OVERRIDE="$2"
    shift 2
fi

RUN_NAME="${1:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load remote config (with override)
RC_ARGS=""
if [ -n "$HOST_OVERRIDE" ]; then
    RC_ARGS="host=${HOST_OVERRIDE}"
fi
eval "$(python3 "$SCRIPT_DIR/remote_config.py" $RC_ARGS)"

echo "==> Remote git status ($host:$remote_dir):"
ssh "$host" "cd $remote_dir && git log --oneline -3 && echo && git status -s"

echo ""

if [ -n "$RUN_NAME" ]; then
    TMUX_SESSION="smk-${RUN_NAME}"
    echo "==> Checking tmux session: $TMUX_SESSION"
    if ssh "$host" "tmux has-session -t '$TMUX_SESSION' 2>/dev/null"; then
        echo "    Session ACTIVE"
        echo "    Attach: ssh $host -t 'tmux attach -t $TMUX_SESSION'"
    else
        echo "    Session NOT found (pipeline may have finished)"
    fi

    echo ""
    echo "==> Run directory:"
    ssh "$host" "ls -la $remote_dir/runs/$RUN_NAME/ 2>/dev/null || echo '    (not yet created)'"
else
    echo "==> Active tmux sessions:"
    ssh "$host" "tmux list-sessions 2>/dev/null | grep '^smk-' || echo '    (none)'"
fi
