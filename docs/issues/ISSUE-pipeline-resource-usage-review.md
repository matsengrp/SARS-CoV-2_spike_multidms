# Review and profile pipeline resource usage for GPU/CPU fitting

## Summary

The current pipeline's GPU resource management has been patched incrementally to work around transient multi-GPU failures, silent exception handling in `multidms`, and JAX memory pre-allocation issues. Before investing further in GPU infrastructure, we need systematic profiling to determine whether multi-GPU fitting actually improves throughput — and whether CPU multiprocessing might be simpler and equally fast.

## Background

### Current state

The pipeline runs Jupyter notebooks via Snakemake + papermill on remote GPU servers (orca01–05, 4x L40S 45GB each). GPU scheduling is controlled by:

- **`config.gpu_ids`**: Specifies which physical GPUs are available (e.g., `[0,1,2,3]`)
- **Snakemake `--resources gpu=N`**: Limits concurrent GPU jobs
- **`GPU_FIT` / `GPU_LOAD`**: Fitting notebooks claim all GPUs; analysis notebooks claim 1

The `multidms` library's `fit_models()` distributes model fits across GPUs using `ThreadPoolExecutor` with one thread per GPU and a semaphore pattern ensuring one model per GPU at a time.

### Known issues encountered during production runs

1. **JAX memory pre-allocation OOM**: Multiple notebook processes running simultaneously each tried to pre-allocate 75% of GPU memory. Fixed with `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

2. **Silent multi-GPU fit failures**: `_fit_models_gpu()` in `multidms/model_collection.py` (line ~238) catches all exceptions with bare `except Exception: result = None` — no logging, no traceback. In production, 6/8 fits failed silently on 4 GPUs but all 8 succeeded sequentially on 1 GPU.

3. **`stack_fit_models` bug with `failures="tolerate"`**: When fits return `None`, `stack_fit_models()` crashes with `AttributeError: 'NoneType' object has no attribute 'to_frame'` because it doesn't filter `None` results before calling `pd.concat`.

4. **`robust_fit_models` workaround**: The pipeline now catches `ModelCollectionFitError` and retries all fits sequentially on a single GPU. This works but means **production fitting effectively uses 1 GPU**, wasting the other 3.

5. **CUDA_VISIBLE_DEVICES applied globally**: All notebook processes (including CPU-only jobs like spike_01, spike_02) see all 4 GPUs via `CUDA_VISIBLE_DEVICES=0,1,2,3`, causing unnecessary JAX/CUDA initialization overhead.

6. **No per-job GPU isolation**: Analysis notebooks (spike_06, spike_07, spike_10) each claim `gpu=1` but all see all 4 GPUs. Snakemake could run 4 analysis jobs in parallel on 4 separate GPUs, but without per-job `CUDA_VISIBLE_DEVICES` pinning, they may contend on the same device.

### Where the code lives

| Component | Location | Responsibility |
|-----------|----------|----------------|
| GPU scheduling | `workflow/Snakefile` (GPU_FIT, GPU_LOAD, JAX_ENV) | Snakemake resource allocation |
| Rule GPU claims | `workflow/rules/{simulation,spike,supplemental}.smk` | Per-notebook GPU demand |
| Multi-GPU fitting | `multidms/model_collection.py` (_fit_models_gpu) | ThreadPoolExecutor + semaphores |
| Exception handling (GPU) | `multidms/model_collection.py` (`_fit_models_gpu`, line 238) | Silent `except Exception: result = None` |
| Exception handling (CPU) | `multidms/model_collection.py` (`_fit_fun`, line 165) | Same silent failure pattern on CPU multiprocessing path |
| Retry workaround | `notebooks/_common.py` (robust_fit_models) | Catch + retry on 1 GPU |
| Config | `config/config.yaml` (gpu_ids, n_processes) | User-facing GPU/CPU selection |

## Proposed experiments

This is a profiling and exploration task, not an implementation proposal. The goal is to collect data that informs future optimization decisions.

Each experiment uses a standalone benchmark script that loads the pipeline's existing intermediate files directly — no need to run the full Snakemake pipeline.

### Experiment 1: GPU vs CPU fitting throughput

**Question**: Is GPU fitting actually faster than CPU multiprocessing for our workload?

**Setup**: Write `scripts/benchmark_fitting.py` that:

1. Loads `runs/prod-v1/spike_analysis/training_functional_scores.csv` (~689k rows)
2. Aggregates into a `multidms.Data` object (same as spike_03 cell-6)
3. Calls `fit_models()` with a single fusionreg value (e.g., `4.0`) and 1 dataset (1 model fit total), then scales to the full grid (4 fusionreg x 2 replicates = 8 fits)

**Configurations to benchmark** (run each 3x for variance):

| Label | `gpu_ids` | `n_processes` | Expected behavior |
|-------|-----------|---------------|-------------------|
| `cpu-1` | `None` | `1` | Sequential CPU baseline |
| `cpu-4` | `None` | `4` | 4 CPU processes via multiprocessing.spawn |
| `cpu-8` | `None` | `8` | 8 CPU processes |
| `gpu-1` | `[0]` | `1` | Single GPU, sequential |
| `gpu-4` | `[0,1,2,3]` | `1` | 4 GPUs, ThreadPoolExecutor |

**Metrics to collect**:

| Metric | How to measure | Why it matters |
|--------|---------------|----------------|
| **Wall time (total)** | `time.perf_counter()` around `fit_models()` | Primary throughput metric |
| **Wall time (first fit only)** | Time 1 fit separately before the batch | Isolates JIT compilation overhead |
| **Peak GPU memory per device** | `jax.devices()[i].memory_stats()` or `nvidia-smi --query-gpu=memory.used --format=csv -l 1` in background | Determines if multiple fits can share a GPU |
| **Peak RSS (CPU memory)** | `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` | Matters for CPU multiprocessing on shared servers |
| **Convergence (final loss)** | `model.training_loss["total"]` for each fit | Sanity check — results should be identical across configs |

**Prerequisites**: Before running, the `stack_fit_models` bug in `multidms` must be patched (filter `None` before `pd.concat`), or the benchmark must use `failures="error"` with a try/except wrapper — see `robust_fit_models` in `notebooks/_common.py` for the pattern.

**Running on orca servers**:
```bash
# From local machine, sync code then run in tmux on a GPU server:
pixi run remote-sync
ssh orca03 "cd /fh/fast/matsen_e/shared/multidms/SARS-CoV-2_spike_multidms && \
    tmux new-session -d -s benchmark && \
    tmux send-keys -t benchmark \
    'XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0,1,2,3 \
     ~/.pixi/bin/pixi run -e cuda python3 scripts/benchmark_fitting.py 2>&1 | \
     tee runs/benchmark_fitting.log' Enter"
```

**Script skeleton**:

```python
#!/usr/bin/env python3
"""Benchmark fit_models() across GPU/CPU configurations.

Run on an orca server with available GPUs:
    XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0,1,2,3 \
        pixi run -e cuda python3 scripts/benchmark_fitting.py
"""
import time, json, resource, sys, platform
import pandas as pd
import multidms
from notebooks._common import robust_fit_models

SPIKE_DATA = "runs/prod-v1/spike_analysis/training_functional_scores.csv"
SIM_DATA = "runs/prod-v1/simulation/simulated_func_scores.csv"
RESULTS_FILE = "runs/benchmark_fitting_results.json"


def build_spike_datasets(csv_path, reference="Omicron_BA1", subsample_frac=None):
    """Load spike training data. Columns: condition, aa_substitutions, func_score, replicate."""
    func_score_df = pd.read_csv(csv_path).fillna({"aa_substitutions": ""})
    if subsample_frac is not None:
        func_score_df = (
            pd.concat([
                g.sample(frac=subsample_frac, random_state=0)
                for _, g in func_score_df.groupby(["condition", "replicate"])
            ]).reset_index(drop=True)
        )
    datasets = []
    for rep, fsdf in func_score_df.groupby("replicate"):
        df_agg = (
            fsdf.groupby(["condition", "aa_substitutions"], dropna=False)
            .agg({"func_score": "mean"}).reset_index()
        )
        df_agg["aa_substitutions"] = df_agg["aa_substitutions"].fillna("")
        data = multidms.Data(
            df_agg, alphabet=multidms.AAS_WITHSTOP_WITHGAP,
            reference=reference, assert_site_integrity=False,
            verbose=False, name=f"rep-{rep}",
        )
        datasets.append(data)
    return datasets


def build_sim_datasets(csv_path, reference="h1"):
    """Load simulation training data. Columns: library, homolog, aa_substitutions, func_score_type, func_score."""
    func_score_df = pd.read_csv(csv_path).fillna({"aa_substitutions": ""})
    # Only use observed_phenotype measurement type (exclude enrichment-based)
    func_score_df = func_score_df[~func_score_df["func_score_type"].str.contains("enrichment")]
    datasets = []
    for (library, measurement), fs_df in func_score_df.groupby(["library", "func_score_type"]):
        # Map simulation columns to multidms expectations
        df = fs_df.rename(columns={"homolog": "condition"})[
            ["condition", "aa_substitutions", "func_score"]
        ].copy()
        df["aa_substitutions"] = df["aa_substitutions"].fillna("")
        data = multidms.Data(
            df, reference=reference, alphabet=multidms.AAS_WITHSTOP_WITHGAP,
            verbose=False, name=f"{library}_{measurement}",
        )
        datasets.append(data)
    return datasets


def benchmark(datasets, gpu_ids, n_processes, fusionreg_values, maxiter, label):
    """Run fit_models and collect timing/memory metrics."""
    fit_params = {
        "fusionreg": fusionreg_values,
        "dataset": datasets,
        "ge_type": ["Sigmoid"],
        "l2reg": [1e-4],
        "beta0_ridge": [0],
        "maxiter": [maxiter],
        "tol": [1e-6],
        "warmstart": [False],
        "loss_kwargs": [{"δ": 1.0}],
        "ge_kwargs": [{"tol": 1e-5, "maxiter": 1000, "maxls": 40, "jit": True, "verbose": False}],
        "cal_kwargs": [{"tol": 1e-4, "maxiter": 1000, "maxls": 40, "jit": True, "verbose": False}],
        "beta_clip_range": [(-10, 10)],
    }

    # Warm-up: single fit to JIT-compile (maxiter=3)
    warm_params = {**fit_params, "fusionreg": [fusionreg_values[0]], "dataset": [datasets[0]], "maxiter": [3]}
    t0 = time.perf_counter()
    robust_fit_models(warm_params, gpu_ids=gpu_ids, n_processes=n_processes)
    jit_time = time.perf_counter() - t0

    # Full benchmark — uses robust_fit_models to handle multi-GPU failures
    t0 = time.perf_counter()
    n_fit, n_failed, models = robust_fit_models(
        fit_params, gpu_ids=gpu_ids, n_processes=n_processes
    )
    wall_time = time.perf_counter() - t0

    # RSS: ru_maxrss is in KB on Linux, bytes on macOS
    rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_mb = rss_raw / 1024 if platform.system() == "Linux" else rss_raw / (1024 * 1024)

    losses = [row.model.training_loss["total"] for _, row in models.iterrows() if row.model is not None]

    return {
        "label": label,
        "gpu_ids": gpu_ids,
        "n_processes": n_processes,
        "n_fits": len(fusionreg_values) * len(datasets),
        "n_succeeded": n_fit,
        "n_failed": n_failed,
        "jit_warmup_s": round(jit_time, 1),
        "wall_time_s": round(wall_time, 1),
        "peak_rss_mb": round(peak_rss_mb, 1),
        "mean_loss": round(sum(losses) / len(losses), 2) if losses else None,
    }


if __name__ == "__main__":
    fusionreg_values = [0.0, 2.0, 4.0, 8.0]

    configs = [
        ("cpu-1", None, 1),
        ("cpu-4", None, 4),
        ("cpu-8", None, 8),
        ("gpu-1", [0], 1),
        ("gpu-4", [0, 1, 2, 3], 1),
    ]

    results = []

    # Spike (full)
    print("=== Loading spike datasets ===", flush=True)
    spike_ds = build_spike_datasets(SPIKE_DATA)
    for label, gpu_ids, n_proc in configs:
        print(f"\n=== spike-full / {label} ===", flush=True)
        r = benchmark(spike_ds, gpu_ids, n_proc, fusionreg_values, maxiter=75, label=f"spike-full/{label}")
        results.append(r)
        print(json.dumps(r, indent=2), flush=True)

    # Spike (subsampled 1%)
    print("\n=== Loading spike datasets (1% subsample) ===", flush=True)
    spike_sub_ds = build_spike_datasets(SPIKE_DATA, subsample_frac=0.01)
    for label, gpu_ids, n_proc in [("cpu-1", None, 1), ("cpu-4", None, 4), ("gpu-1", [0], 1)]:
        print(f"\n=== spike-1pct / {label} ===", flush=True)
        r = benchmark(spike_sub_ds, gpu_ids, n_proc, fusionreg_values, maxiter=75, label=f"spike-1pct/{label}")
        results.append(r)
        print(json.dumps(r, indent=2), flush=True)

    # Simulation
    print("\n=== Loading simulation datasets ===", flush=True)
    sim_ds = build_sim_datasets(SIM_DATA)
    for label, gpu_ids, n_proc in [("cpu-1", None, 1), ("cpu-4", None, 4), ("gpu-1", [0], 1)]:
        print(f"\n=== sim / {label} ===", flush=True)
        r = benchmark(sim_ds, gpu_ids, n_proc, fusionreg_values, maxiter=75, label=f"sim/{label}")
        results.append(r)
        print(json.dumps(r, indent=2), flush=True)

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")
```

**Expected outcome**: A table like:

| Config | JIT warmup | Wall time (8 fits) | Fits OK | Peak RSS |
|--------|-----------|-------------------|---------|----------|
| cpu-1 | — | ? | 8/8 | ? |
| cpu-4 | — | ? | 8/8 | ? |
| gpu-1 | ?s | ? | 8/8 | ? |
| gpu-4 | ?s | ? | ?/8 | ? |

### Experiment 2: Characterize multi-GPU failure mode

**Question**: Why do fits fail on multi-GPU but succeed on single GPU?

**Setup**: Temporarily patch `_fit_models_gpu()` in the local `multidms` install to log exceptions instead of swallowing them:

```python
# In multidms/model_collection.py, line ~236-239, change:
try:
    result = fit_one_model(**kwargs)
except Exception:
    result = None

# To:
try:
    result = fit_one_model(**kwargs)
except Exception:
    logger.exception(f"Model {idx} FAILED on GPU {gpu_ids[gpu_idx]}")
    result = None
```

Then run Experiment 1's `gpu-4` config and collect the tracebacks. Categorize failures as:

| Category | Indicator in traceback |
|----------|----------------------|
| CUDA OOM | `RESOURCE_EXHAUSTED` or `OUT_OF_MEMORY` |
| JIT compilation race | `XLA compilation failed` or `ptxas` errors |
| Device context conflict | `INVALID_ARGUMENT` device mismatch |
| Numerical divergence | `NaN`, `inf`, or convergence failures |

**Metrics**:

| Metric | How to measure |
|--------|---------------|
| Failure rate | `n_failed / n_total` across 10 repeated runs |
| Failure consistency | Do the same parameter sets always fail, or is it random? |
| GPU memory at failure | `nvidia-smi` snapshot at time of exception |

### Experiment 3: JIT compilation overhead

**Question**: How much time is spent on JIT compilation vs actual fitting?

**Setup**: Run a single fit twice — first cold (no cache), then warm (cached):

```python
import jax
import multidms

jax.clear_caches()  # Force cold start

# Build a single dataset + fit_params with 1 fusionreg, 1 dataset
single_params = { ... }  # same as benchmark but 1 fit only

# Cold fit (includes JIT compilation)
t0 = time.perf_counter()
multidms.model_collection.fit_models(single_params, gpu_ids=[0])
cold_time = time.perf_counter() - t0

# Warm fit (same computation graph, JIT cached in-process)
t0 = time.perf_counter()
multidms.model_collection.fit_models(single_params, gpu_ids=[0])
warm_time = time.perf_counter() - t0

jit_overhead = cold_time - warm_time
```

**Metrics**:

| Metric | Meaning |
|--------|---------|
| Cold fit time | Total time including JIT compilation |
| Warm fit time | Pure fitting time (JIT cached) |
| JIT overhead | `cold - warm`; the cost of first compilation |
| JIT overhead % | `(cold - warm) / cold * 100` |

Also measure whether `JAX_COMPILATION_CACHE_DIR` (persistent on-disk cache) eliminates cold-start cost across separate Python processes.

### Experiment 4: GPU memory footprint per fit

**Question**: How much GPU memory does a single model fit actually need?

**Setup**: Monitor `nvidia-smi` in a background loop while running a single fit:

```bash
# Background monitor
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv -l 1 > gpu_memory_trace.csv &
MONITOR_PID=$!

# Run single fit
python3 -c "
import multidms
# ... load data, fit one model with maxiter=75 ...
"

kill $MONITOR_PID
```

**Metrics**:

| Metric | How to measure | Why it matters |
|--------|---------------|----------------|
| Peak GPU memory (1 fit) | Max `memory.used` during single fit | Determines how many fits can share a GPU |
| Peak GPU memory (4 fits on 4 GPUs) | Per-GPU max during `gpu-4` run | Check for memory leaks or cross-device allocation |
| GPU utilization % during fit | Average `utilization.gpu` | Low utilization suggests GPU is not the bottleneck |
| Idle GPU memory (JAX import only) | Memory after `import jax; jax.devices()` | Baseline overhead of having JAX see GPUs |

**Key ratio**: If peak memory for 1 fit is <11 GB (45 GB / 4), then 4 fits could theoretically share 1 GPU. If it's >22 GB, even 2 fits per GPU would OOM.

### Experiment 5: Simulation vs spike scaling

**Question**: Does dataset size change the GPU vs CPU tradeoff?

**Setup**: The benchmark script (Experiment 1) already includes all three dataset sizes. It handles the different column schemas:

- **Spike data** has columns: `condition`, `aa_substitutions`, `func_score`, `replicate` — groups by `replicate`
- **Simulation data** has columns: `library`, `homolog`, `aa_substitutions`, `func_score_type`, `func_score` — rename `homolog` → `condition`, group by `(library, func_score_type)`, use `reference="h1"`
- **Subsampled spike**: Use `subsample_frac=0.01` — done manually before `multidms.Data()` construction via `df.sample(frac=0.01)` per condition/replicate group (not a multidms parameter)

| Dataset | File | Rows | Variants after aggregation |
|---------|------|------|---------------------------|
| Spike (full) | `runs/prod-v1/spike_analysis/training_functional_scores.csv` | ~689k | ~160k |
| Simulation | `runs/prod-v1/simulation/simulated_func_scores.csv` | ~350k | measure during benchmark |
| Spike (subsampled 1%) | Same spike file, subsampled | ~6.9k | ~1.6k |

Run `cpu-1`, `cpu-4`, `gpu-1` for each (skip `cpu-8` and `gpu-4` to save time — the full spike run already covers those). This reveals whether GPU speedup depends on dataset size — small datasets may not saturate GPU compute, making CPU competitive.

**Metrics**: Same as Experiment 1 (wall time, peak memory, loss).

## Decision framework

Use the profiling results to choose one of these strategies:

| Scenario | Evidence | Action |
|----------|----------|--------|
| **GPU is 3x+ faster than CPU** | `gpu-1` wall time << `cpu-4` wall time | Invest in fixing multi-GPU reliability in `multidms` |
| **GPU ≈ CPU** | `gpu-1` ≈ `cpu-4` wall time | Drop GPU complexity, use `n_processes` for all fitting |
| **Multi-GPU is reliable** | `gpu-4` failure rate < 5% across 10 runs | Remove `robust_fit_models` workaround, use `gpu-4` directly |
| **Multi-GPU is fragile** | `gpu-4` failure rate > 20% | Either fix threading in `multidms` or use `gpu-1` only |
| **JIT overhead dominates** | JIT overhead > 50% of total time | Implement persistent compilation cache |
| **Small datasets don't benefit from GPU** | Simulation `gpu-1` ≈ `cpu-1` | Use CPU for simulation, GPU for spike only |

## `multidms` library issues to file regardless of profiling outcome

These are bugs/improvements that should be addressed independent of the GPU vs CPU decision:

- [ ] **Log exceptions in `_fit_models_gpu` and `_fit_fun`**: Both the GPU path (line 238) and CPU multiprocessing path (line 165) silently swallow all exceptions with `except Exception: result = None`. Replace with `logger.exception()` in both
- [ ] **Fix `stack_fit_models` for None values**: Filter `None` before `pd.concat` so `failures="tolerate"` actually works
- [ ] **Add `failures="retry"` mode**: Automatically retry failed fits sequentially (what `robust_fit_models` does externally)

## Success criteria

- [ ] Benchmark script `scripts/benchmark_fitting.py` written and runnable on orca servers
- [ ] Profiling data collected for all 5 experiments
- [ ] Results summarized in a table/plot comparing GPU vs CPU throughput
- [ ] Root cause identified for multi-GPU transient failures
- [ ] Recommendation written: GPU, CPU, or hybrid strategy for production
- [ ] `multidms` library bug fixes filed as separate issues

## Future work

Depending on profiling results, potential follow-up issues:

- **If CPU is competitive**: Remove GPU complexity, use `n_processes` for parallelism, simplify Snakemake scheduling
- **If GPU is faster but multi-GPU is fragile**: Fix `multidms` threading, or switch to `multiprocessing.spawn` for GPU isolation
- **If per-job GPU isolation helps**: Implement dynamic `CUDA_VISIBLE_DEVICES` assignment in Snakemake rules
- **If JAX overhead dominates**: Evaluate persistent JAX compilation cache (`JAX_COMPILATION_CACHE_DIR`), or pre-warming JIT in a setup step
- **If GPU memory is small per fit**: Run multiple fits per GPU (increase semaphore count in `_fit_models_gpu`)
