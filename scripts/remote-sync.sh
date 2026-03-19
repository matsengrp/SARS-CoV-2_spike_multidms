#!/usr/bin/env bash
# Sync local code to remote server via git push + SSH pull.
#
# Usage: scripts/remote-sync.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load remote config
eval "$(python3 "$SCRIPT_DIR/remote_config.py")"

BRANCH="${branch:-main}"

echo "==> Pushing local changes to origin/$BRANCH..."
cd "$PROJECT_DIR"
git push origin "$BRANCH"

echo "==> Pulling on remote ($host)..."
ssh "$host" "cd $remote_dir && git fetch origin && git checkout $BRANCH && git pull origin $BRANCH"

echo "==> Sync complete."
