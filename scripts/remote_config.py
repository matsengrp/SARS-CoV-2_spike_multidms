#!/usr/bin/env python3
"""Read remote configuration from ~/.config/spike-multidms/remote.yaml.

Uses a simple key:value parser (no PyYAML dependency) so this works with
the system Python outside of pixi.
"""

import os
import sys

CONFIG_PATH = os.path.expanduser("~/.config/spike-multidms/remote.yaml")

REQUIRED_KEYS = ["host", "remote_dir"]
DEFAULTS = {
    "branch": "main",
    "pixi_env": "cuda",
}


def load_remote_config():
    """Load and validate remote configuration.

    Expected format (simple YAML subset)::

        host: user@gpu-server
        remote_dir: /home/user/projects/SARS-CoV-2_spike_multidms
        branch: main          # optional, default: main
        pixi_env: cuda        # optional, default: cuda
    """
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: Remote config not found at {CONFIG_PATH}", file=sys.stderr)
        print(
            "\nCreate it with:\n"
            f"  mkdir -p {os.path.dirname(CONFIG_PATH)}\n"
            f"  cat > {CONFIG_PATH} << 'EOF'\n"
            "  host: user@gpu-server\n"
            "  remote_dir: /path/to/SARS-CoV-2_spike_multidms\n"
            "  EOF",
            file=sys.stderr,
        )
        sys.exit(1)

    cfg = {}
    with open(CONFIG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Handle inline comments
            if " #" in line:
                line = line[: line.index(" #")]
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            cfg[key.strip()] = value.strip()

    for key in REQUIRED_KEYS:
        if key not in cfg:
            print(
                f"Error: Missing required key '{key}' in {CONFIG_PATH}",
                file=sys.stderr,
            )
            sys.exit(1)

    for key, default in DEFAULTS.items():
        cfg.setdefault(key, default)

    return cfg


if __name__ == "__main__":
    cfg = load_remote_config()
    # Output as shell-friendly key=value pairs
    for key, value in cfg.items():
        print(f"{key}={value}")
