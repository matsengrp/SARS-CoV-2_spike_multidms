#!/usr/bin/env bash
# Pull a named run from the remote server via rsync.
#
# Usage: scripts/run-pull.sh <run_name> [--exclude-pkl]
#
# By default, all files including .pkl are pulled. Pass --exclude-pkl
# to skip large pickle files.
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <run_name> [--exclude-pkl]"
    exit 1
fi

RUN_NAME="$1"
EXCLUDE_PKL=false
if [ "${2:-}" = "--exclude-pkl" ]; then
    EXCLUDE_PKL=true
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load remote config
eval "$(python3 "$SCRIPT_DIR/remote_config.py")"

LOCAL_DIR="$PROJECT_DIR/runs/$RUN_NAME/"
REMOTE_PATH="$host:$remote_dir/runs/$RUN_NAME/"

RSYNC_ARGS=(-avz --progress)
if [ "$EXCLUDE_PKL" = true ]; then
    RSYNC_ARGS+=(--exclude='*.pkl')
fi

echo "==> Pulling run '$RUN_NAME' from $host..."
mkdir -p "$LOCAL_DIR"
rsync "${RSYNC_ARGS[@]}" "$REMOTE_PATH" "$LOCAL_DIR"

echo "==> Done. Run available at: runs/$RUN_NAME/"
echo "    Use: pixi run run-use $RUN_NAME"
