# Add remote pipeline execution and named-run artifact management

## Summary

Add pixi tasks for syncing code to a GPU server, running the Snakemake pipeline in tmux, and pulling results back locally — all via simple commands like `pixi run remote-pipeline RUN=sigmoid-v2`. Introduce a **named-run** system that stores each experimental run's artifacts in a separate directory outside the git-tracked tree, making it trivial to switch between runs for interactive notebook exploration.

## Background

- The pipeline currently runs locally via `pixi run test` / `pixi run production`, writing results to git-ignored paths like `results/simulation/` and `results/spike_analysis/`.
- There is no built-in way to run the pipeline on a remote GPU server, sync code, or retrieve results.
- The `dasm2-experiments` repo has a proven Makefile-based pattern for `remote-sync`, `remote-tmux`, and `artifacts-pull` that we can adapt.
- Notebooks read their I/O paths from `config["simulation"]["output_dir"]` and `config["spike"]["output_dir"]` via `load_config()`. This means we can redirect where results are written/read by overriding `output_dir` in a config profile — no notebook changes needed.
- The biggest gap is **managing multiple experimental runs**: if you run the pipeline with different configs (e.g., different fusion regularization, different subsample fractions), you want to keep each run's results intact and be able to explore any of them interactively.

## Proposed Approach

### Named runs via config profiles

Each experimental run gets a **nickname** (e.g., `sigmoid-v2`, `high-reg`, `baseline`). Results live in `runs/<nickname>/` which is git-ignored. This is achieved by:

1. A new config profile `config/profile_<nickname>.yaml` that sets `output_dir` overrides
2. OR, more ergonomically, a **single `run_name` config key** that `load_config()` uses to prefix output directories automatically

The second approach is cleaner — it avoids creating a profile per run and keeps the existing profile system for its intended purpose (test vs production parameters).

**Key design decisions:**

- **`run_name` config key redirects output**: When `run_name` is set (e.g., via `snakemake --config run_name=sigmoid-v2`), output directories become `runs/sigmoid-v2/simulation/` and `runs/sigmoid-v2/spike_analysis/` etc. When unset, behavior is unchanged (results go to `results/` as today).
- **User-level remote config**: Remote host/dir stored in `~/.config/spike-multidms/remote.yaml`, following the dasm2 pattern but using XDG conventions.
- **pixi tasks, not Makefile**: Keeps the project single-toolchain. Shell scripts under `scripts/` handle the multi-step logic; pixi tasks are thin wrappers.
- **Symlink for interactive use**: `runs/current -> runs/<nickname>` symlink makes it easy to point notebooks at a specific run without editing config.

### Run directory layout

```
runs/                          # git-ignored
  sigmoid-v2/                  # one named run
    simulation/                # mirrors results/simulation/
    spike_analysis/            # mirrors results/spike_analysis/
    html/                      # mirrors results/html/
    supplemental/              # mirrors results/supplemental/
    _meta.yaml                 # auto-generated: timestamp, git SHA, config snapshot
  high-reg/
    ...
  current -> sigmoid-v2        # symlink to "active" run
```

## User Interface

### Remote config (one-time setup)

```yaml
# ~/.config/spike-multidms/remote.yaml
remote_host: rhino03           # or any SSH alias
remote_dir: /fh/fast/matsen_e/user/dre/SARS-CoV-2_spike_multidms
```

### Pixi task commands

```bash
# ---- Remote operations ----

# Sync code to remote (push current branch via git)
pixi run remote-sync

# Run full pipeline on remote GPU server in tmux (with a named run)
pixi run remote-pipeline RUN=sigmoid-v2

# Run with additional snakemake/config overrides
pixi run remote-pipeline RUN=high-reg EXTRA="--config spike.fitting.maxiter=50"

# Check remote status (git + tmux)
pixi run remote-status

# ---- Artifact sync ----

# Pull a named run's results from remote to local
pixi run run-pull RUN=sigmoid-v2

# Pull only simulation results for a run
pixi run run-pull RUN=sigmoid-v2 DIR=simulation

# List available runs (local and/or remote)
pixi run run-list
pixi run run-list --remote

# Set the "current" symlink to a run for interactive notebook use
pixi run run-use sigmoid-v2

# ---- Interactive notebook exploration ----

# After `run-use sigmoid-v2`, open any notebook interactively:
# It will read from runs/sigmoid-v2/ instead of results/
pixi run jupyter notebook notebooks/spike/spike_07_shifted_mutations.ipynb
```

### How notebooks find the right run

The `load_config()` function is extended with one new parameter: `run_name`. The resolution order:

1. **Papermill parameter** `-p run_name sigmoid-v2` (pipeline execution)
2. **Environment variable** `MULTIDMS_RUN` (for interactive use; set by `run-use`)
3. **`runs/current` symlink** (fallback for interactive use)
4. **Default**: `results/` (unchanged, for backward compatibility)

When a `run_name` is active, `output_dir` values from config are rewritten:
- `results/simulation` -> `runs/<run_name>/simulation`
- `results/spike_analysis` -> `runs/<run_name>/spike_analysis`

This means **no notebook code changes** — they already read from `output_dir`.

## Proposed Changes

### 1. Remote config reader

**File:** `scripts/remote_config.py` (new)

A small utility that reads `~/.config/spike-multidms/remote.yaml` and prints values, called by shell scripts. No external dependencies beyond PyYAML (already a dependency).

```python
"""Read remote configuration for pixi tasks."""
import os
import sys
import yaml

CONFIG_PATH = os.path.expanduser("~/.config/spike-multidms/remote.yaml")

def get_remote_config():
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: Create {CONFIG_PATH} with remote_host and remote_dir", file=sys.stderr)
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    for key in ("remote_host", "remote_dir"):
        if key not in cfg:
            print(f"Error: {key} not set in {CONFIG_PATH}", file=sys.stderr)
            sys.exit(1)
    return cfg

if __name__ == "__main__":
    cfg = get_remote_config()
    key = sys.argv[1] if len(sys.argv) > 1 else None
    if key:
        print(cfg[key])
    else:
        for k, v in cfg.items():
            print(f"{k}={v}")
```

### 2. Shell scripts for remote operations

**File:** `scripts/remote-sync.sh` (new)

```bash
#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST=$(python scripts/remote_config.py remote_host)
REMOTE_DIR=$(python scripts/remote_config.py remote_dir)
BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Require clean working tree
if [ -n "$(git status --porcelain)" ]; then
    echo "Error: uncommitted changes. Commit or stash first."
    exit 1
fi

git push origin "$BRANCH"
ssh -A "$REMOTE_HOST" "cd $REMOTE_DIR && git fetch origin && git checkout $BRANCH && git pull --ff-only origin $BRANCH"
echo "Remote synced to $BRANCH"
```

**File:** `scripts/remote-pipeline.sh` (new)

```bash
#!/usr/bin/env bash
set -euo pipefail

RUN="${RUN:?Error: set RUN=<name>}"
EXTRA="${EXTRA:-}"

# Sync code first
bash scripts/remote-sync.sh

REMOTE_HOST=$(python scripts/remote_config.py remote_host)
REMOTE_DIR=$(python scripts/remote_config.py remote_dir)

CMD="cd $REMOTE_DIR && pixi run -e cuda snakemake --configfile config/config.yaml --config run_name=$RUN $EXTRA -j8"

echo "Running in tmux on $REMOTE_HOST: $CMD"
ssh "$REMOTE_HOST" "tmux has-session -t spike 2>/dev/null || tmux new-session -d -s spike; \
    tmux new-window -t spike: '$CMD; echo \"=== Run $RUN complete ===\"; exec \$SHELL'"
echo "Attached to tmux session 'spike' on $REMOTE_HOST"
echo "  ssh $REMOTE_HOST -t 'tmux attach -t spike'   # to monitor"
```

**File:** `scripts/run-pull.sh` (new)

```bash
#!/usr/bin/env bash
set -euo pipefail

RUN="${RUN:?Error: set RUN=<name>}"
DIR="${DIR:-}"  # optional subdirectory filter

REMOTE_HOST=$(python scripts/remote_config.py remote_host)
REMOTE_DIR=$(python scripts/remote_config.py remote_dir)

SRC="$REMOTE_HOST:$REMOTE_DIR/runs/$RUN/$DIR"
DST="runs/$RUN/$DIR"
mkdir -p "$DST"

echo "Pulling $SRC -> $DST"
rsync -avz --progress --exclude='.DS_Store' --exclude='*.pkl' --omit-dir-times "$SRC/" "$DST/"
echo "Done. Use 'pixi run run-use $RUN' to activate for interactive notebooks."
```

Note: `--exclude='*.pkl'` by default because pickle files are large and non-portable (different JAX versions). This can be overridden with an `INCLUDE_PKL=1` flag if needed.

**File:** `scripts/run-use.sh` (new)

```bash
#!/usr/bin/env bash
set -euo pipefail

RUN="${1:?Usage: run-use <run-name>}"

if [ ! -d "runs/$RUN" ]; then
    echo "Error: runs/$RUN does not exist. Available runs:"
    ls -1 runs/ 2>/dev/null || echo "  (none)"
    exit 1
fi

ln -sfn "$RUN" runs/current
echo "Active run: $RUN"
echo "Notebooks will read from runs/$RUN/"
```

### 3. Extend `load_config()` to support named runs

**File:** `notebooks/_common.py` (existing — modify `load_config`)

```python
def load_config(config_path="config/config.yaml", profile=None, run_name=None):
    """Load config with optional profile overrides and run-name redirection.

    When run_name is set (via parameter, MULTIDMS_RUN env var, or
    runs/current symlink), output_dir values are rewritten to
    runs/<run_name>/<subdir>.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if profile:
        profile_path = f"config/profile_{profile}.yaml"
        with open(profile_path) as f:
            overrides = yaml.safe_load(f)
        if overrides:
            config = deep_merge(config, overrides)

    # Resolve run name: explicit param > env var > current symlink > None
    run = run_name or os.environ.get("MULTIDMS_RUN")
    if not run and os.path.islink("runs/current"):
        run = os.path.basename(os.readlink("runs/current"))

    if run:
        for section in ("simulation", "spike"):
            if section in config and "output_dir" in config[section]:
                default_dir = config[section]["output_dir"]
                # results/simulation -> runs/<run>/simulation
                subdir = default_dir.replace("results/", "", 1)
                config[section]["output_dir"] = f"runs/{run}/{subdir}"

    return config
```

Also modify the `Snakefile` to apply the same `run_name` -> `output_dir` rewriting, so Snakemake rules write to the correct location. The Snakefile already has the `config` dict available after profile merging; add the same rewrite block there.

### 4. Snakefile changes

**File:** `workflow/Snakefile` (existing — add after profile merge block)

```python
# Named-run output redirection
run_name = config.get("run_name", "")
if run_name:
    for section in ("simulation", "spike"):
        if section in config and "output_dir" in config[section]:
            default_dir = config[section]["output_dir"]
            subdir = default_dir.replace("results/", "", 1)
            config[section]["output_dir"] = f"runs/{run_name}/{subdir}"
    # Also redirect HTML output
    # (HTML paths in rules are hardcoded; will need to be parameterized)
```

**Note:** The HTML output paths in rules are currently hardcoded (`results/html/...`). These should be refactored to derive from config as well. This is the one area that needs rule-level changes.

### 5. pixi.toml task additions

**File:** `pixi.toml` (existing — add tasks)

```toml
[tasks]
# ... existing tasks ...

# Remote operations
remote-sync = "bash scripts/remote-sync.sh"
remote-status = { cmd = "bash scripts/remote-status.sh" }
remote-pipeline = { cmd = "bash scripts/remote-pipeline.sh" }

# Named run management
run-pull = { cmd = "bash scripts/run-pull.sh" }
run-use = { cmd = "bash scripts/run-use.sh" }
run-list = { cmd = "bash scripts/run-list.sh" }
```

### 6. .gitignore addition

```
runs/
```

### 7. Run metadata

**File:** `scripts/write-run-meta.sh` (new) — called at the start of each pipeline run

Writes `runs/<name>/_meta.yaml`:
```yaml
run_name: sigmoid-v2
created: 2026-03-18T14:30:00
git_sha: 22c6747
git_branch: main
config_snapshot: { ... full merged config ... }
```

This makes runs self-documenting — you can always trace back what config produced a given set of results.

## Implementation Details

### How the output_dir rewriting works end-to-end

1. **Pipeline (headless):** `snakemake --config run_name=sigmoid-v2` -> Snakefile rewrites `config["simulation"]["output_dir"]` to `runs/sigmoid-v2/simulation/` -> papermill passes config to notebooks -> notebooks read/write from `runs/sigmoid-v2/`.

2. **Interactive (local):** User runs `pixi run run-use sigmoid-v2` -> creates `runs/current` symlink -> user opens notebook in Jupyter -> `load_config()` sees the symlink -> sets `output_dir` to `runs/sigmoid-v2/simulation/` -> notebook reads the remote-pulled CSVs/figures.

3. **Default (no run):** No `run_name`, no env var, no symlink -> `output_dir` stays as `results/simulation/` -> fully backward compatible.

### HTML output path parameterization

Currently, rules hardcode paths like `results/html/simulation/sim_01_data_simulation.html`. These need to become dynamic based on `config`. The simplest approach:

```python
# In Snakefile, after config merge:
HTML_BASE = f"runs/{run_name}/html" if run_name else "results/html"
SIM_OUT = config.get("simulation", {}).get("output_dir", "results/simulation")
SPIKE_OUT = config.get("spike", {}).get("output_dir", "results/spike_analysis")
```

Then rules reference `HTML_BASE`, `SIM_OUT`, `SPIKE_OUT` instead of hardcoded paths.

### Pickle file handling

`.pkl` files (fitted models) are large and tied to specific JAX/equinox versions. The default `run-pull` excludes them. For cases where you need pickles locally:

```bash
pixi run run-pull RUN=sigmoid-v2 INCLUDE_PKL=1
```

Alternatively, downstream notebooks that only need CSVs/PDFs (visualization, evaluation) work fine without pickles.

## Testing Strategy

- **Unit test `load_config` with `run_name`:** Verify output_dir rewriting for both simulation and spike sections. Verify fallback chain (param > env > symlink > default).
- **Integration test with Snakemake:** `pixi run snakemake --config profile=test run_name=test-run -j4` should produce outputs under `runs/test-run/` instead of `results/`.
- **Remote script dry-run:** Each script should support a `DRY_RUN=1` mode that prints what it would do.
- Edge cases:
  - `run_name` with special characters (restrict to `[a-zA-Z0-9_-]`)
  - `runs/current` symlink pointing to nonexistent directory
  - Remote server not reachable (clear error messages)
  - Running without `~/.config/spike-multidms/remote.yaml` (non-remote tasks should still work)

## Documentation Updates

- [ ] README: Add "Remote execution" and "Named runs" sections
- [ ] `config/README.md` or inline comments: Document `run_name` config key
- [ ] CLAUDE.md: Update pipeline section with new commands

## Dependencies

No new dependencies. Uses PyYAML (existing), rsync (system), ssh (system), tmux (remote system).

## Alternatives Considered

### Alternative 1: Config profiles per run

Create `config/profile_sigmoid-v2.yaml` with `output_dir` overrides for each run.

**Why not chosen:** Proliferates config files. Couples run naming to config profiles (which serve a different purpose: test vs production parameters). A run should be able to use *any* profile.

### Alternative 2: Makefile instead of pixi tasks

Follow the dasm2 pattern exactly with a Makefile.

**Why not chosen:** This repo is pixi-native. Adding Make would be a second build system. Pixi tasks with shell scripts achieve the same thing while keeping a single entry point (`pixi run`).

### Alternative 3: Store runs on a shared filesystem only (no local sync)

Mount `/fh/fast/...` via SSHFS and work directly.

**Why not chosen:** SSHFS is slow for interactive Jupyter. Local copies with rsync give a much better experience. The run metadata + pull workflow also works for collaborators who don't have filesystem access.

### Alternative 4: Use Snakemake's `--directory` flag

Snakemake's `--directory` changes the working directory for all rules.

**Why not chosen:** This changes *all* paths including inputs (data/), not just outputs. Would break rules that read from `data/`. The `output_dir` rewriting approach is more surgical.

## Success Criteria

- [ ] `pixi run remote-pipeline RUN=test-run` syncs code and starts pipeline in tmux on remote
- [ ] `pixi run run-pull RUN=test-run` rsyncs results to `runs/test-run/` locally
- [ ] `pixi run run-use test-run` + opening a notebook interactively reads from `runs/test-run/`
- [ ] Default behavior (no `run_name`) is completely unchanged
- [ ] Existing `pixi run test` and `pixi run production` work as before
- [ ] Run metadata (`_meta.yaml`) records git SHA and config for reproducibility

## Future Work

- **`run-diff`**: Compare outputs between two named runs (e.g., diff CSVs, overlay plots)
- **`run-archive` / `run-clean`**: Archive old runs to compressed tarballs, clean up disk
- **Snakemake remote executor**: Use Snakemake's built-in SSH/SLURM executor instead of manual tmux (more robust for cluster environments)
- **`run-resume`**: Re-run only failed/incomplete rules for a named run on the remote
- **Dashboard**: Auto-generate an HTML index of all runs with metadata and key figures
