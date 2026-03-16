# SARS-CoV-2 Spike Multidms Pipeline Constitution

## Core Principles

### I. Reproducibility First
Every analysis result must be reproducible from raw data to final figures with a single command (`snakemake`). No manual notebook execution, no hidden state. The Snakemake DAG is the source of truth for execution order and data dependencies.

### II. Config-Driven Parameters
All tunable parameters (model hyperparameters, data paths, fitting options, figure aesthetics) live in YAML config files, never hardcoded in notebooks. Profile-based overrides (test vs. production) enable fast iteration without code changes.

### III. Modular Notebooks with Clear I/O
Each notebook has a single responsibility with explicitly declared file inputs and outputs. Notebooks communicate only through files on disk — never through shared in-memory state. Snakemake rules enforce these contracts.

### IV. Preserve Manuscript Compatibility
Output paths under `results/spike_analysis/` and `results/simulation_validation/` must remain stable — the PNAS manuscript (`multidms-tex/`) references these paths. Data directory structure (`data/{Delta,Omicron_BA1,Omicron_BA2}/`) is unchanged.

### V. Test Profiles for Fast Iteration
A `profile_test.yaml` overlay must allow the full pipeline to complete in minutes on a laptop (reduced iterations, smaller parameter grids, fewer variants). Production runs use full parameters. The same code path executes in both profiles.

### VI. Simplicity Over Cleverness
Prefer explicit `papermill` + `jupyter nbconvert` shell commands in Snakemake rules over Snakemake's built-in `notebook:` directive. Each rule is readable without Snakemake expertise. Avoid unnecessary abstractions — three similar rules are better than a premature macro.

## Technology Stack

- **Orchestration**: Snakemake (≥8.0)
- **Notebooks**: Jupyter + papermill for parameterized execution
- **Output**: `jupyter nbconvert --to html` for documentation
- **Environment**: pixi (replacing requirements.txt)
- **Language**: Python ≥3.11
- **Core library**: multidms (editable install from sibling directory)
- **Key deps**: JAX, equinox, jaxopt, pandas, matplotlib, seaborn, altair, scipy, biopython

## Development Workflow

- Each notebook has a parameters cell (tagged `parameters`) with `config_path` and `profile` variables
- Notebooks load config via a shared `_common.py` utility module
- Snakemake rules declare input/output files matching notebook I/O
- GitHub Actions CI runs the test profile on each push
- GitHub Pages hosts executed HTML notebooks with a generated index

## Governance

This constitution guides the pipeline refactor. Deviations require explicit justification in the relevant GitHub issue. The specification and plan documents elaborate on these principles.

**Version**: 1.0.0 | **Ratified**: 2026-03-16 | **Last Amended**: 2026-03-16
