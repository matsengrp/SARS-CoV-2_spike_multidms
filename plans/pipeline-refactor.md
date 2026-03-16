# Implementation Plan: Snakemake + Papermill Pipeline Refactor

**Date**: 2026-03-16 | **Spec**: `specs/pipeline-refactor.md`

## Summary

Refactor two monolithic Jupyter notebooks (spike-analysis.ipynb at 64MB/130 cells and simulation_validation.ipynb at 24MB/174 cells) into 18 modular, parameterized notebooks orchestrated by Snakemake. Each notebook receives config via papermill parameter injection, executes independently, and communicates with other notebooks only through files on disk.

## Technical Context

**Language/Version**: Python ≥3.11
**Primary Dependencies**: Snakemake ≥8.0, papermill, jupyter nbconvert, multidms (editable), JAX, pandas, matplotlib, seaborn, altair
**Storage**: File-based (CSV, pickle, PDF/PNG, HTML)
**Testing**: Snakemake dry-run + test profile execution; no unit tests (notebooks are the tests)
**Target Platform**: macOS (M-series), Linux (CI), CPU + optional GPU
**Project Type**: Reproducible analysis pipeline
**Performance Goals**: Test profile <5 min on laptop; production ~hours with GPU
**Constraints**: Preserve `results/` output paths for manuscript compatibility

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Reproducibility First | ✅ | Snakemake DAG ensures reproducibility |
| II. Config-Driven | ✅ | All params in YAML, papermill injection |
| III. Modular Notebooks | ✅ | 18 notebooks with explicit I/O |
| IV. Preserve Manuscript Compat | ✅ | Output paths unchanged |
| V. Test Profiles | ✅ | profile_test.yaml with reduced params |
| VI. Simplicity | ✅ | Explicit papermill commands, no magic |

## Project Structure

```text
SARS-CoV-2_spike_multidms/
├── pixi.toml                        # Environment definition (replaces requirements.txt)
├── workflow/
│   ├── Snakefile                    # Main workflow entry point
│   └── rules/
│       ├── simulation.smk           # Simulation pipeline rules (4 rules)
│       ├── spike.smk                # Spike analysis rules (12 rules)
│       ├── supplemental.smk         # Supplemental analysis rules (2 rules)
│       └── pages.smk                # HTML documentation generation
├── config/
│   ├── config.yaml                  # Production parameters
│   └── profile_test.yaml            # Test profile overlay
├── notebooks/
│   ├── _common.py                   # Shared utilities (config loading, deep merge)
│   ├── simulation/
│   │   ├── sim_01_data_simulation.ipynb
│   │   ├── sim_02_model_fitting.ipynb
│   │   ├── sim_03_evaluation.ipynb
│   │   └── sim_04_visualization.ipynb
│   ├── spike/
│   │   ├── spike_01_data_loading.ipynb
│   │   ├── spike_02_exploratory_stats.ipynb
│   │   ├── spike_03_fit_models.ipynb
│   │   ├── spike_04_model_evaluation.ipynb
│   │   ├── spike_05_cross_validation.ipynb
│   │   ├── spike_06_global_epistasis.ipynb
│   │   ├── spike_07_shifted_mutations.ipynb
│   │   ├── spike_08_naive_comparison.ipynb
│   │   ├── spike_09_linear_comparison.ipynb
│   │   ├── spike_10_validation.ipynb
│   │   ├── spike_11_reference_sensitivity.ipynb
│   │   └── spike_12_sparsity_correlation.ipynb
│   └── supplemental/
│       ├── supplemental_sim_figures.ipynb
│       └── supplemental_structure.ipynb
├── data/                            # UNCHANGED — existing data layout
│   ├── Delta/
│   ├── Omicron_BA1/
│   ├── Omicron_BA2/
│   ├── viral_titers.csv
│   └── spike_validation_data.csv
├── results/                         # UNCHANGED paths — Snakemake outputs here
│   ├── spike_analysis/
│   ├── simulation_validation/
│   └── docs/                        # NEW — HTML notebooks + index
├── .github/workflows/
│   └── pipeline.yml                 # CI: run test profile
├── specs/                           # Spec-kit artifacts
├── plans/                           # Spec-kit artifacts
└── archive/                         # Old monolithic notebooks moved here
    ├── spike-analysis.ipynb
    └── simulation_validation.ipynb
```

## Implementation Phases

### Phase 0: Foundation (Issues 1–3)

**Goal**: Working environment and scaffold — pipeline runs but does nothing interesting yet.

**Issue 1: pixi.toml + environment setup**
- Create `pixi.toml` with all dependencies from requirements.txt + analysis_requirements.txt
- Add snakemake, papermill, nbconvert
- Pin multidms as editable install from sibling dir
- Remove requirements.txt and analysis_requirements.txt
- Verify: `pixi shell` succeeds, `python -c "import multidms"` works

**Issue 2: Config system**
- Create `config/config.yaml` with all parameters extracted from spike-analysis.ipynb and simulation_validation.ipynb
- Create `config/profile_test.yaml` with reduced parameters
- Create `notebooks/_common.py` with `load_config(config_path, profile)` function using deep merge
- Verify: `python -c "from notebooks._common import load_config; c = load_config('config/config.yaml', 'test')"` works

**Issue 3: Snakemake scaffold**
- Create `workflow/Snakefile` with configfile directive and profile merge logic
- Create empty rule files: `workflow/rules/{simulation,spike,supplemental,pages}.smk`
- Add `rule all:` targeting all final outputs
- Verify: `snakemake -n` completes (dry run, all rules show as pending)

### Phase 1: Simulation Pipeline (Issues 4–7)

**Goal**: Complete simulation validation pipeline running end-to-end.

**Issue 4: sim_01_data_simulation.ipynb**
- Extract cells 1–75 from simulation_validation.ipynb (simulation + library + phenotype + counts)
- Add parameters cell with `config_path`, `profile`
- Use _common.py to load config
- Outputs: `simulated_muteffects.csv`, `simulated_bottleneck_cbf.csv`, `simulated_functional_scores.csv`
- Add Snakemake rule in `workflow/rules/simulation.smk`
- Verify: `snakemake results/simulation_validation/simulated_functional_scores.csv --config profile=test`

**Issue 5: sim_02_model_fitting.ipynb**
- Extract cells ~111–125 (Data encoding + fit_models)
- Inputs: simulated_functional_scores.csv
- Outputs: `fit_collection.pkl`
- Add Snakemake rule
- Verify: test profile runs in <2 min

**Issue 6: sim_03_evaluation.ipynb**
- Extract cells ~126–160 (all evaluation metrics)
- Inputs: fit_collection.pkl, simulated_muteffects.csv
- Outputs: 5 evaluation CSV files
- Add Snakemake rule

**Issue 7: sim_04_visualization.ipynb**
- Extract cells ~161–174 (final plots)
- Inputs: evaluation CSVs, fit_collection.pkl
- Outputs: HTML notebook (figures rendered in-notebook)
- Add Snakemake rule with nbconvert step

### Phase 2: Spike Analysis Pipeline (Issues 8–19)

**Goal**: Complete spike analysis pipeline. Issues 8–19 can proceed in parallel with Phase 1 but have internal sequential dependencies.

**Issue 8: spike_01_data_loading.ipynb**
- Extract cells ~16–35 (data loading, condition ID construction, functional score aggregation)
- Outputs: `training_functional_scores.csv`

**Issue 9: spike_02_exploratory_stats.ipynb**
- Extract cells ~36–45 (replicate correlation, barcode stats)
- Inputs: training_functional_scores.csv
- Outputs: `replicate_functional_score_correlation_scatter.{pdf,png}`

**Issue 10: spike_03_fit_models.ipynb**
- Extract cells ~46–58 (Data encoding + model fitting)
- This is the most computationally expensive step
- Inputs: training_functional_scores.csv
- Outputs: `full_models.pkl`
- Snakemake rule should request GPU resources when available

**Issue 11: spike_04_model_evaluation.ipynb**
- Extract cells ~59–65 (model selection, mutations_df export)
- Inputs: full_models.pkl
- Outputs: `mutations_df.csv`, convergence plots

**Issue 12: spike_05_cross_validation.ipynb**
- Extract cells ~66–71 (CV splitting, fitting, trace plots)
- Inputs: training_functional_scores.csv
- Outputs: `cv_models.pkl`, shrinkage trace plots

**Issue 13: spike_06_global_epistasis.ipynb**
- Extract cells ~72–80 (sigmoid fit analysis)
- Inputs: full_models.pkl
- Outputs: `global_epistasis_and_prediction_correlations.{pdf,png}`

**Issue 14: spike_07_shifted_mutations.ipynb**
- Extract cells ~81–91 (interactive chart + heatmap)
- Inputs: full_models.pkl, mutations_df.csv
- Outputs: shift heatmaps, `interactive_shift_chart.html`

**Issue 15: spike_08_naive_comparison.ipynb**
- Extract cells ~92–103 (independent-condition fits, comparison)
- Inputs: training_functional_scores.csv, full_models.pkl
- Outputs: `shift_distribution_correlation_naive.{pdf,png}`

**Issue 16: spike_09_linear_comparison.ipynb**
- Extract cells ~104–112 (linear vs sigmoid comparison)
- Inputs: training_functional_scores.csv
- Outputs: `shrinkage_analysis_linear_models.{pdf,png}`

**Issue 17: spike_10_validation.ipynb**
- Extract cells ~113–116 (experimental validation)
- Inputs: full_models.pkl, `data/viral_titers.csv`, `data/spike_validation_data.csv`
- Outputs: `validation_titer_fold_change.{pdf,png}`

**Issue 18: spike_11_reference_sensitivity.ipynb**
- Extract cells ~117–123 (alternative reference fits)
- Inputs: training_functional_scores.csv
- Outputs: `reference_model_comparison_params_scatter.{pdf,png}`

**Issue 19: spike_12_sparsity_correlation.ipynb**
- Extract cells ~124–129 (sparsity CDF + shift correlations)
- Inputs: full_models.pkl, mutations_df.csv
- Outputs: sparsity lineplot, shift correlation plots

### Phase 3: Supplemental + Documentation (Issues 20–23)

**Issue 20: supplemental_sim_figures.ipynb**
- Adapt `simulation_manuscript_figures.ipynb` to read from pipeline outputs
- Inputs: simulation results CSVs

**Issue 21: supplemental_structure.ipynb**
- Adapt `spike-structure-analysis.ipynb` to read from pipeline outputs
- Inputs: mutations_df.csv

**Issue 22: GitHub Pages generation**
- Create `workflow/rules/pages.smk` with index generation rule
- Template generates `results/docs/index.html` linking all HTML notebooks
- Add `.github/workflows/pipeline.yml` for CI (test profile) and Pages deployment

**Issue 23: Archive and cleanup**
- Move `spike-analysis.ipynb` and `simulation_validation.ipynb` to `archive/`
- Remove `requirements.txt`, `analysis_requirements.txt`
- Update `README.md` with new usage instructions
- Update `.gitignore` for new layout (ignore executed notebooks, keep source notebooks)

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Notebook tech | Jupyter + papermill | Manuscript uses Jupyter; marimo adds migration risk |
| Execution method | Explicit `papermill` + `nbconvert` in shell | Transparent, debuggable, no Snakemake notebook magic |
| Sim decomposition | 4 notebooks (vs marimo's 9) | Fewer boundaries = less I/O overhead; sim is sequential |
| Spike decomposition | 12 notebooks (vs marimo's 11) | Split global epistasis from model evaluation for clarity |
| Config format | YAML with profile overlay | Simple, standard, matches marimo fork pattern |
| Environment | pixi | Already used at workspace level; superior to requirements.txt |
| CI | Test profile only | Production runs are too expensive for CI |
| Output paths | Unchanged | Manuscript compatibility (Principle IV) |

## Dependency Graph

```
                    ┌─ sim_01 ─► sim_02 ─► sim_03 ─► sim_04
                    │
Foundation ─────────┤
(pixi, config,      │                        ┌─► spike_06 (GE)
 Snakefile)         │                        ├─► spike_07 (shifts)
                    │            spike_04 ───┼─► spike_08 (naive)
                    │           (eval)       ├─► spike_10 (validation)
                    └─ spike_01 ─► spike_03 ─┤   └─► spike_12 (sparsity)
                       (data)     (fit)      │
                         │                   └─► spike_11 (ref sensitivity)
                         ├─► spike_02 (stats)
                         ├─► spike_05 (CV)
                         └─► spike_09 (linear)
```

Note: spike_08, spike_09, spike_11, spike_12 need training data AND/OR models — check spec table for exact inputs.
