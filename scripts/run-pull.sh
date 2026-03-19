#!/usr/bin/env bash
# Pull a named run from the remote server via rsync.
#
# Usage: scripts/run-pull.sh <run_name> [--include-pkl]
#
# By default, .pkl files are excluded (they're large and not needed locally
# for notebook exploration). Pass --include-pkl to include them.
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <run_name> [--include-pkl]"
    exit 1
fi

RUN_NAME="$1"
INCLUDE_PKL=false
if [ "${2:-}" = "--include-pkl" ]; then
    INCLUDE_PKL=true
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load remote config
eval "$(python3 "$SCRIPT_DIR/remote_config.py")"

LOCAL_DIR="$PROJECT_DIR/runs/$RUN_NAME/"
REMOTE_PATH="$host:$remote_dir/runs/$RUN_NAME/"

RSYNC_ARGS=(-avz --progress)
if [ "$INCLUDE_PKL" = false ]; then
    RSYNC_ARGS+=(--exclude='*.pkl')
fi

echo "==> Pulling run '$RUN_NAME' from $host..."
mkdir -p "$LOCAL_DIR"
rsync "${RSYNC_ARGS[@]}" "$REMOTE_PATH" "$LOCAL_DIR"

echo "==> Done. Run available at: runs/$RUN_NAME/"
echo "    Use: pixi run run-use $RUN_NAME"
