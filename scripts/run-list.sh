#!/usr/bin/env bash
# List available runs (local and optionally remote).
#
# Usage: scripts/run-list.sh [--remote]
set -euo pipefail

SHOW_REMOTE=false
if [ "${1:-}" = "--remote" ]; then
    SHOW_REMOTE=true
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Current active run
CURRENT=""
if [ -L "$PROJECT_DIR/runs/current" ]; then
    CURRENT="$(readlink "$PROJECT_DIR/runs/current")"
fi

echo "==> Local runs:"
if [ -d "$PROJECT_DIR/runs" ]; then
    for run_dir in "$PROJECT_DIR/runs"/*/; do
        [ -d "$run_dir" ] || continue
        name="$(basename "$run_dir")"
        [ "$name" = "current" ] && continue

        marker=""
        if [ "$name" = "$CURRENT" ]; then
            marker=" (active)"
        fi

        meta="$run_dir/_meta.yaml"
        if [ -f "$meta" ]; then
            started=$(grep 'started:' "$meta" 2>/dev/null | head -1 | sed 's/started: *//')
            git_sha=$(grep 'git_sha:' "$meta" 2>/dev/null | head -1 | sed 's/git_sha: *//' | cut -c1-7)
            echo "  $name${marker}  [${started:-?} | ${git_sha:-?}]"
        else
            echo "  $name${marker}"
        fi
    done
else
    echo "  (no runs directory)"
fi

if [ "$SHOW_REMOTE" = true ]; then
    eval "$(python3 "$SCRIPT_DIR/remote_config.py")"

    echo ""
    echo "==> Remote runs ($host):"
    ssh "$host" "ls -1 $remote_dir/runs/ 2>/dev/null | grep -v '^current$' || echo '  (none)'"
fi
