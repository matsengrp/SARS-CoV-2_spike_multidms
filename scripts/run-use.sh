#!/usr/bin/env bash
# Set a named run as the active run by creating a runs/current symlink.
#
# Usage: scripts/run-use.sh <run_name>
#
# After running this, interactive notebooks will automatically read from
# the named run via the runs/current symlink resolution in load_config().
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <run_name>"
    echo ""
    echo "Available runs:"
    ls -1 runs/ 2>/dev/null | grep -v '^current$' || echo "  (none)"
    exit 1
fi

RUN_NAME="$1"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RUN_DIR="$PROJECT_DIR/runs/$RUN_NAME"

if [ ! -d "$RUN_DIR" ]; then
    echo "Error: Run directory not found: runs/$RUN_NAME"
    echo ""
    echo "Available runs:"
    ls -1 "$PROJECT_DIR/runs/" 2>/dev/null | grep -v '^current$' || echo "  (none)"
    exit 1
fi

# Create or update symlink
ln -sfn "$RUN_NAME" "$PROJECT_DIR/runs/current"

echo "==> Active run set to: $RUN_NAME"
echo "    Symlink: runs/current -> $RUN_NAME"
echo "    Notebooks will now read from: runs/$RUN_NAME/"
