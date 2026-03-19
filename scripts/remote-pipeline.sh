#!/usr/bin/env bash
# Sync code and launch pipeline on remote server in a tmux session.
#
# Usage: scripts/remote-pipeline.sh <run_name> [extra snakemake args...]
#
# Example:
#   scripts/remote-pipeline.sh sigmoid-v2
#   scripts/remote-pipeline.sh sigmoid-v2 --config profile=test
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <run_name> [extra snakemake args...]"
    exit 1
fi

RUN_NAME="$1"
shift

# Collect extra args, merging any --config values into a single --config flag
# with run_name to avoid Snakemake's --config replacement behavior.
CONFIG_VALS="run_name=${RUN_NAME}"
OTHER_ARGS=""
while [ $# -gt 0 ]; do
    if [ "$1" = "--config" ]; then
        shift
        # Consume all key=value pairs until next flag
        while [ $# -gt 0 ] && [[ "$1" != -* ]]; do
            CONFIG_VALS="${CONFIG_VALS} $1"
            shift
        done
    else
        OTHER_ARGS="${OTHER_ARGS} $1"
        shift
    fi
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Sync first
"$SCRIPT_DIR/remote-sync.sh"

# Load remote config
eval "$(python3 "$SCRIPT_DIR/remote_config.py")"

PIXI_ENV="${pixi_env:-cuda}"
TMUX_SESSION="smk-${RUN_NAME}"

echo "==> Launching pipeline on $host (tmux: $TMUX_SESSION)..."

# Build the remote command — single --config flag with all config values merged
REMOTE_CMD="export PATH=\$HOME/.pixi/bin:\$PATH && cd ${remote_dir} && pixi run -e ${PIXI_ENV} snakemake --configfile config/config.yaml --config ${CONFIG_VALS}${OTHER_ARGS} -j8"

# Use tmux send-keys to avoid nested quoting issues
if ssh "$host" "tmux has-session -t ${TMUX_SESSION} 2>/dev/null"; then
    echo "Error: tmux session '${TMUX_SESSION}' already exists on $host"
    echo "    Attach: ssh $host -t \"tmux attach -t ${TMUX_SESSION}\""
    exit 1
fi

ssh "$host" "tmux new-session -d -s ${TMUX_SESSION} && tmux send-keys -t ${TMUX_SESSION} '${REMOTE_CMD}' Enter"

echo "==> Pipeline launched in tmux session: $TMUX_SESSION"
echo "    Attach: ssh $host -t \"tmux attach -t $TMUX_SESSION\""
echo "    Status: scripts/remote-status.sh $RUN_NAME"
