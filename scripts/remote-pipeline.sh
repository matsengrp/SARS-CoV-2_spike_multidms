#!/usr/bin/env bash
# Sync code and launch pipeline on remote server in a tmux session.
#
# Usage: scripts/remote-pipeline.sh [--host HOST] <run_name> [extra snakemake args...]
#
# Example:
#   scripts/remote-pipeline.sh sigmoid-v2
#   scripts/remote-pipeline.sh --host orca03 sigmoid-v2 --config profile=test
set -euo pipefail

# Parse --host override (must come before run_name)
HOST_OVERRIDE=""
if [ "${1:-}" = "--host" ]; then
    HOST_OVERRIDE="$2"
    shift 2
fi

if [ $# -lt 1 ]; then
    echo "Usage: $0 [--host HOST] <run_name> [extra snakemake args...]"
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

# Build remote_config.py override args
RC_ARGS=""
if [ -n "$HOST_OVERRIDE" ]; then
    RC_ARGS="host=${HOST_OVERRIDE}"
fi

# Sync first (pass host override)
"$SCRIPT_DIR/remote-sync.sh" $RC_ARGS

# Load remote config (with override)
eval "$(python3 "$SCRIPT_DIR/remote_config.py" $RC_ARGS)"

PIXI_ENV="${pixi_env:-cuda}"
TMUX_SESSION="smk-${RUN_NAME}"

# Determine available GPU count from config for Snakemake resource scheduling.
# gpu_ids in config.yaml controls which GPUs fit_models uses; we tell Snakemake
# how many GPU "slots" are available so it can schedule jobs accordingly.
N_GPUS=$(python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('$SCRIPT_DIR/../config/config.yaml'))
# Check for profile override
profile_val = '$CONFIG_VALS'
for kv in profile_val.split():
    if kv.startswith('profile='):
        p = kv.split('=',1)[1]
        try:
            override = yaml.safe_load(open(f'$SCRIPT_DIR/../config/profile_{p}.yaml'))
            if override and 'gpu_ids' in override:
                cfg['gpu_ids'] = override['gpu_ids']
        except FileNotFoundError:
            pass
ids = cfg.get('gpu_ids') or []
print(len(ids) if ids else 1)
")
RESOURCE_ARGS="--resources gpu=${N_GPUS}"

echo "==> Launching pipeline on $host (tmux: $TMUX_SESSION, gpus: $N_GPUS)..."

# Build the remote command — single --config flag with all config values merged
REMOTE_CMD="export PATH=\$HOME/.pixi/bin:\$PATH && cd ${remote_dir} && pixi run -e ${PIXI_ENV} snakemake --configfile config/config.yaml --config ${CONFIG_VALS} ${RESOURCE_ARGS}${OTHER_ARGS} -j8"

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
