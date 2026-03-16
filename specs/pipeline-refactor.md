# Feature Specification: Snakemake + Papermill Pipeline Refactor

**Created**: 2026-03-16
**Status**: Draft
**Input**: Refactor monolithic Jupyter notebooks into a reproducible, config-driven Snakemake pipeline

## User Scenarios & Testing

### User Story 1 — Run Full Pipeline End-to-End (Priority: P1)

A researcher clones the repo, runs `pixi shell && snakemake`, and the entire analysis executes from raw data to final figures and HTML documentation — no manual notebook interaction required.

**Why this priority**: This is the core value proposition. Without end-to-end reproducibility, nothing else matters.

**Independent Test**: Clone to a clean environment, run `snakemake --profile test`, verify all expected outputs exist under `results/`.

**Acceptance Scenarios**:

1. **Given** a clean clone with pixi installed, **When** `pixi shell && snakemake` is run, **Then** all CSV outputs, pickle files, PDF/PNG figures, and HTML notebooks are produced under `results/`.
2. **Given** the pipeline has already run, **When** no inputs have changed and `snakemake` is re-run, **Then** nothing re-executes (Snakemake caching).
3. **Given** one input data file is updated, **When** `snakemake` is run, **Then** only the affected downstream notebooks re-execute.

---

### User Story 2 — Fast Test Profile (Priority: P1)

A developer runs the pipeline in test mode to verify correctness in minutes on a laptop, without GPU.

**Why this priority**: Without fast iteration, development is impractical. Ties with P1 — must ship together.

**Independent Test**: Run `snakemake --config profile=test`, verify it completes in <5 minutes and produces structurally identical outputs (same files, possibly different values).

**Acceptance Scenarios**:

1. **Given** `config/profile_test.yaml` exists with reduced parameters, **When** `snakemake --config profile=test` is run, **Then** all notebooks execute with reduced iterations/grid and complete in <5 minutes.
2. **Given** the test profile, **When** comparing output file names to production, **Then** the same set of files is produced.

---

### User Story 3 — Modify a Single Analysis Step (Priority: P2)

A researcher wants to change the lasso regularization grid. They edit `config/config.yaml`, re-run `snakemake`, and only the affected model fitting and downstream notebooks re-execute.

**Why this priority**: Config-driven parameterization is the second most important feature after reproducibility.

**Independent Test**: Change one config value, run snakemake, verify only downstream rules re-execute.

**Acceptance Scenarios**:

1. **Given** the pipeline has run, **When** `fusionreg_values` is changed in config.yaml, **Then** `snakemake` re-executes model fitting and all downstream notebooks but not data loading.
2. **Given** a notebook is parameterized, **When** it is opened in Jupyter interactively, **Then** it can be run standalone with default config path for debugging.

---

### User Story 4 — Browse Results as HTML Documentation (Priority: P2)

A collaborator visits the GitHub Pages site and can browse all executed notebooks as HTML, with a generated index page.

**Why this priority**: Communicating results is essential for the PNAS manuscript review process.

**Independent Test**: After pipeline runs, open `results/docs/index.html` and verify all notebooks are linked and render correctly.

**Acceptance Scenarios**:

1. **Given** all notebooks have executed, **When** the pages rule runs, **Then** `results/docs/` contains an `index.html` linking to all HTML-converted notebooks.
2. **Given** a GitHub Actions workflow, **When** a push to main occurs, **Then** GitHub Pages is updated with the latest HTML notebooks.

---

### User Story 5 — Add a New Analysis Notebook (Priority: P3)

A developer wants to add a new analysis step (e.g., a new comparison or validation). They create a notebook, add a Snakemake rule, and it integrates into the DAG.

**Why this priority**: Extensibility matters but is not blocking for the initial refactor.

**Independent Test**: Add a stub notebook with a single cell, add a Snakemake rule, verify it executes as part of the pipeline.

**Acceptance Scenarios**:

1. **Given** the pipeline scaffold exists, **When** a new notebook and rule are added following the documented pattern, **Then** it executes in the correct position in the DAG.

---

### Edge Cases

- What happens when a notebook fails mid-execution? → Snakemake marks the rule as failed; partial outputs are not persisted (use `--shadow` or temp files).
- What happens when multidms is updated? → Re-run pipeline; all model fitting re-executes since the editable install changes.
- What happens on a machine without GPU? → Pipeline runs on CPU (JAX falls back); test profile is designed for CPU-only.
- What happens if a data file is missing? → Snakemake reports a missing input error before any execution.

## Requirements

### Functional Requirements

- **FR-001**: Pipeline MUST execute all analysis steps via `snakemake` with no manual intervention
- **FR-002**: All tunable parameters MUST be read from `config/config.yaml` via papermill parameter injection
- **FR-003**: A test profile (`config/profile_test.yaml`) MUST allow full pipeline completion in <5 minutes on a laptop
- **FR-004**: Each notebook MUST have a tagged `parameters` cell with `config_path` and `profile` variables
- **FR-005**: Notebook execution MUST use explicit `papermill` commands in Snakemake shell blocks
- **FR-006**: Each executed notebook MUST be converted to HTML via `jupyter nbconvert`
- **FR-007**: Output file paths under `results/spike_analysis/` and `results/simulation_validation/` MUST remain unchanged from current layout
- **FR-008**: A shared `notebooks/_common.py` utility MUST provide config loading with profile-based deep merge
- **FR-009**: `pixi.toml` MUST replace `requirements.txt` and `analysis_requirements.txt` as the environment definition
- **FR-010**: GitHub Actions CI MUST run the test profile on push to main
- **FR-011**: GitHub Pages MUST be generated from executed HTML notebooks with an auto-generated index

### Notebook Decomposition

The two monolithic notebooks decompose as follows:

#### Spike Analysis Pipeline (from `spike-analysis.ipynb`)

| # | Notebook | Inputs | Outputs |
|---|----------|--------|---------|
| 01 | `spike_01_data_loading.ipynb` | `data/{homolog}/functional_selections.csv`, `data/{homolog}/*_func_scores.csv` | `results/spike_analysis/training_functional_scores.csv` |
| 02 | `spike_02_exploratory_stats.ipynb` | training_functional_scores.csv | `results/spike_analysis/replicate_functional_score_correlation_scatter.{pdf,png}` |
| 03 | `spike_03_fit_models.ipynb` | training_functional_scores.csv, config (fusionreg grid, fit params) | `results/spike_analysis/full_models.pkl` |
| 04 | `spike_04_model_evaluation.ipynb` | full_models.pkl | `results/spike_analysis/mutations_df.csv`, convergence plots |
| 05 | `spike_05_cross_validation.ipynb` | training_functional_scores.csv, config | `results/spike_analysis/cv_models.pkl`, shrinkage trace plots |
| 06 | `spike_06_global_epistasis.ipynb` | full_models.pkl | `results/spike_analysis/global_epistasis_and_prediction_correlations.{pdf,png}` |
| 07 | `spike_07_shifted_mutations.ipynb` | full_models.pkl, mutations_df.csv | shift heatmaps, interactive HTML chart |
| 08 | `spike_08_naive_comparison.ipynb` | training_functional_scores.csv, full_models.pkl | `results/spike_analysis/shift_distribution_correlation_naive.{pdf,png}` |
| 09 | `spike_09_linear_comparison.ipynb` | training_functional_scores.csv, config | `results/spike_analysis/shrinkage_analysis_linear_models.{pdf,png}` |
| 10 | `spike_10_validation.ipynb` | full_models.pkl, `data/viral_titers.csv`, `data/spike_validation_data.csv` | `results/spike_analysis/validation_titer_fold_change.{pdf,png}` |
| 11 | `spike_11_reference_sensitivity.ipynb` | training_functional_scores.csv, config | `results/spike_analysis/reference_model_comparison_params_scatter.{pdf,png}` |
| 12 | `spike_12_sparsity_correlation.ipynb` | full_models.pkl, mutations_df.csv | sparsity lineplot, shift correlation plots |

#### Simulation Validation Pipeline (from `simulation_validation.ipynb`)

| # | Notebook | Inputs | Outputs |
|---|----------|--------|---------|
| 01 | `sim_01_data_simulation.ipynb` | config (gene length, mutation params) | `results/simulation_validation/simulated_muteffects.csv`, `simulated_bottleneck_cbf.csv`, `simulated_functional_scores.csv` |
| 02 | `sim_02_model_fitting.ipynb` | simulated_functional_scores.csv, config | `results/simulation_validation/fit_collection.pkl` |
| 03 | `sim_03_evaluation.ipynb` | fit_collection.pkl, simulated_muteffects.csv | `model_vs_truth_beta_shift.csv`, `fit_sparsity.csv`, `library_replicate_correlation.csv`, `model_vs_truth_variant_phenotype.csv`, `cross_validation_loss.csv` |
| 04 | `sim_04_visualization.ipynb` | All evaluation CSVs, fit_collection.pkl | Figures (in-notebook, rendered to HTML) |

#### Supplemental Notebooks

| # | Notebook | Inputs | Outputs |
|---|----------|--------|---------|
| S1 | `supplemental_sim_figures.ipynb` | simulation results CSVs | Manuscript-ready simulation figures |
| S2 | `supplemental_structure.ipynb` | mutations_df.csv | Structure analysis figures |

### Key Entities

- **Config**: YAML dictionary of all pipeline parameters, loaded with profile overlay support
- **Notebook**: Jupyter `.ipynb` file with a papermill `parameters` cell; each has declared file I/O
- **Snakemake Rule**: Maps one notebook to its inputs, outputs, and shell command
- **Profile**: Named YAML overlay (e.g., `test`, `production`) that deep-merges over base config

## Success Criteria

### Measurable Outcomes

- **SC-001**: `snakemake -n` (dry run) completes without errors, showing the full DAG
- **SC-002**: Test profile completes in <5 minutes on a MacBook (M-series, CPU-only)
- **SC-003**: Production profile produces byte-identical CSV outputs compared to current `results/` contents (within floating-point tolerance for model outputs)
- **SC-004**: All 18 notebooks (12 spike + 4 sim + 2 supplemental) execute successfully
- **SC-005**: `results/docs/index.html` links to all HTML notebooks
- **SC-006**: GitHub Actions CI passes on the test profile
- **SC-007**: No hardcoded parameters remain in notebook code — all configurable values come from config YAML
