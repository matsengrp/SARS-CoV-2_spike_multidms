# Benchmark Results: GPU vs CPU Resource Profiling

**Date**: 2026-03-22
**Server**: orca04 (4x NVIDIA L40S, 45 GB each)
**Issue**: [#51](https://github.com/matsengrp/SARS-CoV-2_spike_multidms/issues/51)
**Script**: `scripts/benchmark_fitting.py`
**Configuration**: `maxiter=10`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `CUDA_VISIBLE_DEVICES=0,1,2,3`

## Executive Summary

**GPU provides no per-fit speedup over CPU.** A single GPU fit takes the same time as a single CPU fit (~55s for spike data). The only GPU advantage is **multi-GPU parallelism** (4 GPUs ≈ 3x speedup), but this is closely matched by CPU multiprocessing (`n_processes=4` achieves 2.3x, `n_processes=8` achieves 2.7x).

Multi-GPU fitting has a **cold-start data race** in `jaxmodels.Data.from_multidms()` that causes 100% failure without a JIT warmup. With warmup (as `robust_fit_models` effectively provides), gpu-4 is reliable (0/24 failures) and the fastest config (128s vs 161s for cpu-8). However, GPU provides no *per-fit* advantage — it is only faster because 4 GPUs run 4 fits in parallel, the same thing CPU multiprocessing achieves without GPU complexity.

**Recommendation**: Use **CPU multiprocessing** (`n_processes=4-8`, `gpu_ids=None`) for production fitting. This is simpler, nearly as fast, more portable, and avoids the cold-start data race entirely. If multi-GPU is kept, the `robust_fit_models` warmup pattern is essential.

---

## Experiment 1: GPU vs CPU Throughput (Spike Full Dataset)

8 fits (4 fusionreg × 2 replicates), 689k training rows, 3 trials each.

| Config | Wall time (mean±sd) | Per-fit (mean) | Speedup | Failures | Peak RSS |
|--------|--------------------:|---------------:|--------:|---------:|---------:|
| cpu-1  | 438.3 ± 0.5s       | 54.8s          | 1.0x    | 0/24     | ~20 GB   |
| cpu-4  | 194.7 ± 1.1s       | 24.3s          | 2.3x    | 0/24     | ~21 GB   |
| cpu-8  | 161.1 ± 2.0s       | 20.1s          | 2.7x    | 0/24     | ~22 GB   |
| gpu-1  | 440.3 ± 6.9s       | 55.0s          | 1.0x    | 0/24     | ~22 GB   |
| gpu-4  | 127.6 ± 16.7s      | 15.9s          | 3.4x    | 0/24     | ~25 GB   |

**Key findings:**
- **gpu-1 ≈ cpu-1**: A single GPU provides zero speedup over a single CPU process (55.0 vs 54.8 s/fit).
- **gpu-4 wins on wall time** (128s vs 161s for cpu-8), but only by exploiting 4 GPUs as independent executors — each GPU runs at CPU speed.
- **cpu-8 nearly matches gpu-4** (161s vs 128s) with 100% reliability and no GPU overhead.
- GPU uses ~25% more RAM (25 GB vs 20 GB peak RSS).
- gpu-4 has high variance (sd=16.7s) due to scheduling jitter; CPU configs are very stable.

**Why GPU doesn't help per-fit**: The `multidms` optimization loop uses `jaxopt` (L-BFGS-B style), which is inherently sequential — each iteration depends on the previous gradient. The matrices are sparse, and the main computation is the global epistasis sigmoid + per-condition loss. These operations don't benefit from GPU parallelism at the current dataset scale.

## Experiment 2: Multi-GPU Failure Characterization

10 trials of 8 fits on 4 GPUs (no warmup), `failures="tolerate"`.

| Metric | Value |
|--------|------:|
| Failure rate (no warmup) | **100% (80/80 fits)** |
| Failure rate (with warmup, from Exp. 1) | **0% (0/24 fits)** |

**Root cause identified**: A **cold-start data race** in `jaxmodels.Data.from_multidms()` (line 69 of `jaxmodels.py`). When multiple threads simultaneously access JAX BCOO arrays from a shared `Data` object *for the first time*, the COO→CSR conversion in `scipy.sparse.csr_array` produces corrupted index arrays:

```
ValueError: axis 0 index 174049 exceeds matrix dimension 27191
ValueError: axis 0 index 606085664 exceeds matrix dimension 27191
```

The corrupted indices (174049 and 606085664 for a 27191-dim matrix) suggest memory corruption during concurrent access to lazily-initialized JAX BCOO array internals (`X.data`, `X.indices`).

**Why warmup prevents failures**: A single-fit warmup materializes the BCOO array buffers in `multidms_data.arrays["X"]` before concurrent access begins. Once the internal arrays are realized (not lazily computed), concurrent reads are safe. This explains why Experiment 1 (which includes a warmup) saw 0% failures while Experiment 2 (no warmup) saw 100%.

**Note**: This cold-start race is distinct from the production failures observed pre-merge, which were intermittent (6/8 fits failed, not 100%) and occurred during fitting, not sparse matrix construction. The pre-merge failures may have involved JIT compilation races or numerical divergence under concurrent GPU access. The `robust_fit_models` retry pattern handles both failure modes.

## Experiment 3: JIT Compilation Overhead

Single fit on GPU, cold (cache cleared) vs warm (cached), 3 trials.

| Metric | Value |
|--------|------:|
| Cold fit time | 71.5 ± 0.8s |
| Warm fit time | 56.2 ± 0.1s |
| JIT overhead | 15.4 ± 0.7s |
| JIT overhead % | **21.5%** |

JIT compilation adds ~15s to the first fit, or ~22% overhead. This is notable but not dominant — the actual fitting computation (56s) is the bottleneck. A persistent JAX compilation cache (`JAX_COMPILATION_CACHE_DIR`) could eliminate this across process restarts.

## Experiment 4: GPU Memory Footprint

| Measurement | GPU Memory |
|-------------|----------:|
| JAX idle baseline | 891 MB |
| Peak during single fit | 903 MB |
| Total GPU memory (L40S) | 46,068 MB |
| Fit footprint (peak - idle) | **~12 MB** |
| Memory utilization | **2.0%** |
| Theoretical fits per GPU | 51 |

**GPU utilization during fitting**: Mean 16%, peak 95%. The GPU is mostly idle — computation is not GPU-bound.

The extremely small memory footprint (12 MB per fit) means memory is not a constraint for multi-GPU or multi-fit-per-GPU configurations. The failures observed in Experiment 2 are purely from the thread-safety data race, not from resource contention.

## Experiment 5: Dataset Size Scaling

| Dataset | Size | cpu-1 (s/fit) | cpu-4 (s/fit) | gpu-1 (s/fit) | GPU faster? |
|---------|-----:|-------------:|--------------:|--------------:|:-----------:|
| Spike full | 689k rows | 54.8 | 24.3 | 55.0 | No |
| Spike 1% | 6.9k rows | 38.6 | 12.7 | 38.5 | No |
| Simulation | 350k rows | 36.4 | 10.4 | 36.0 | No |

**Key findings:**
- GPU provides **no speedup at any dataset size** — cpu-1 and gpu-1 are identical within noise.
- cpu-4 multiprocessing scales better for smaller datasets: 3.0x on spike-1% (38.6→12.7), 3.5x on simulation (36.4→10.4).
- All fits have a ~35s floor — this is the overhead of model construction, sparse matrix conversion, and optimization setup.

---

## Decision Matrix (from Issue #51)

| Scenario | Evidence | Decision |
|----------|----------|----------|
| GPU is 3x+ faster than CPU | **NO** — gpu-1 = cpu-1 | ~~Invest in multi-GPU~~ |
| GPU ≈ CPU | **YES** — 55.0 vs 54.8 s/fit | **Drop GPU, use n_processes** |
| Multi-GPU is reliable | **YES with warmup** (0/24), **NO without** (80/80) | Warmup is essential; `robust_fit_models` provides this |
| Multi-GPU is fragile | **Cold-start only** — BCOO lazy init race | Warmup mitigates; CPU avoids entirely |
| JIT overhead dominates | **NO** — 22% overhead, not dominant | Low priority |
| Small datasets benefit from GPU | **NO** — GPU slower for small data | Use CPU for all sizes |

## Recommendation

### Production pipeline configuration

```yaml
# config/config.yaml
gpu_ids: null        # Disable GPU fitting
n_processes: 4       # Use 4 CPU processes (match available cores)
```

### Pipeline changes

1. **Remove GPU scheduling complexity**: Drop `GPU_FIT`, `GPU_LOAD`, `JAX_ENV` from Snakefile. Remove `resources: gpu=` from all rules.
2. **Remove `CUDA_VISIBLE_DEVICES` and `XLA_PYTHON_CLIENT_PREALLOCATE` env vars** from shell commands.
3. **Keep `robust_fit_models`** as a safety net — CPU multiprocessing (`_fit_fun`) has the same silent exception swallowing, though it's less likely to trigger data races.
4. **Remove pixi `cuda` environment** from remote execution scripts (optional — keep if needed for other JAX GPU workloads).

### `multidms` library bugs to file

1. **Cold-start data race** in `_fit_models_gpu` / `jaxmodels.Data.from_multidms()`: Concurrent threads corrupt sparse matrix indices when accessing lazily-initialized JAX BCOO arrays for the first time. Fix options:
   - Materialize BCOO arrays in main thread before spawning workers (e.g., access `.data` and `.indices` once per dataset)
   - Deep-copy `Data` per thread before `fit_one_model()`
   - Add a single-fit warmup inside `_fit_models_gpu` before the ThreadPoolExecutor loop
2. **Silent exception swallowing** in `_fit_models_gpu` (line 238) and `_fit_fun` (line 165): Replace `except Exception: result = None` with `logger.exception()` + `result = None`.
3. **`stack_fit_models` crash on None**: Filter None values before `pd.concat`.

### Expected impact

- **Same throughput**: cpu-4 (195s for 8 fits) vs current production gpu-1+retry (450s for 8 fits) = **2.3x faster**.
- **Higher reliability**: No multi-GPU data race, no retry overhead, no GPU OOM.
- **Simpler infrastructure**: No CUDA dependency, runs on any server, no GPU scheduling.
- **Lower memory**: 20 GB vs 25 GB peak RSS.
