# SARS-CoV-2 Spike multidms analysis

Analysis of SARS-CoV-2 spike homologs as seen in our manuscript
_Jointly modeling deep mutational scans identifies shifted mutational effects among SARS-CoV-2 spike homologs_.

Browse the executed notebooks on [GitHub Pages](https://matsengrp.github.io/SARS-CoV-2_spike_multidms/).

## Quick start

```bash
# Clone with multidms sibling
git clone https://github.com/matsengrp/SARS-CoV-2_spike_multidms.git
git clone https://github.com/matsengrp/multidms.git

# Install environment
cd SARS-CoV-2_spike_multidms
pixi install

# Run test profile (~5 min, CPU only)
pixi run test

# Run production pipeline (requires GPU, hours)
pixi run production
```

## Pipeline overview

The analysis is a [Snakemake](https://snakemake.readthedocs.io/) pipeline that executes 18 parameterized Jupyter notebooks via [papermill](https://papermill.readthedocs.io/), then converts each to HTML with `nbconvert`.

All tunable parameters live in `config/config.yaml`. A scaled-down test profile (`config/profile_test.yaml`) enables fast iteration on a laptop.

### Simulation validation (4 notebooks)

| Notebook | Description |
|----------|-------------|
| `sim_01_data_simulation` | Synthetic DMS data generation for two homologs |
| `sim_02_model_fitting` | Fit models across fusion regularization grid |
| `sim_03_evaluation` | Convergence, accuracy, sparsity, cross-validation |
| `sim_04_visualization` | Final selection plots and GE landscape |

### Spike empirical analysis (12 notebooks)

| Notebook | Description |
|----------|-------------|
| `spike_01_data_loading` | Load and filter functional score data |
| `spike_02_exploratory_stats` | Replicate correlation and barcode stats |
| `spike_03_fit_models` | Joint model fitting (GPU-intensive) |
| `spike_04_model_evaluation` | Convergence, sparsity, mutation export |
| `spike_05_cross_validation` | 80/20 CV with shrinkage trace plots |
| `spike_06_global_epistasis` | Sigmoid GE landscape and predictions |
| `spike_07_shifted_mutations` | Interactive Altair chart and heatmaps |
| `spike_08_naive_comparison` | Joint vs independent-condition fitting |
| `spike_09_linear_comparison` | Sigmoid vs linear GE comparison |
| `spike_10_validation` | Viral titer fold-change validation |
| `spike_11_reference_sensitivity` | Reference choice robustness |
| `spike_12_sparsity_correlation` | Shift sparsity and correlation |

### Supplemental (2 notebooks)

| Notebook | Description |
|----------|-------------|
| `supplemental_sim_figures` | Publication-quality simulation figures |
| `supplemental_structure` | Structural analysis of shifted mutations |

## Project structure

```
├── pixi.toml                    # Environment (replaces requirements.txt)
├── config/
│   ├── config.yaml              # Production parameters
│   └── profile_test.yaml        # Test profile overlay
├── workflow/
│   ├── Snakefile                # Pipeline entry point
│   └── rules/                   # Rule files per pipeline
├── notebooks/
│   ├── _common.py               # Shared utilities
│   ├── simulation/              # 4 simulation notebooks
│   ├── spike/                   # 12 spike analysis notebooks
│   └── supplemental/            # 2 supplemental notebooks
├── data/                        # Input data (unchanged)
├── results/                     # Pipeline outputs (gitignored)
└── archive/                     # Original monolithic notebooks
```

## Configuration

Edit `config/config.yaml` to change parameters. Override for testing:

```bash
snakemake --snakefile workflow/Snakefile --config profile=test -j4
```

## Key outputs

After a pipeline run, key results are in `results/spike_analysis/`:
- `mutations_df.csv` — mutation parameters and phenotype effects
- `training_functional_scores.csv` — curated training data
- `full_models.pkl` — fitted model collection
