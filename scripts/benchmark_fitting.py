#!/usr/bin/env python3
"""Benchmark fit_models() across GPU/CPU configurations.

Covers all 5 experiments from issue #51:
  1. GPU vs CPU fitting throughput
  2. Multi-GPU failure characterization (via patched logging)
  3. JIT compilation overhead (cold vs warm fits)
  4. GPU memory footprint per fit (via nvidia-smi background monitor)
  5. Simulation vs spike dataset size scaling

Run on an orca server with available GPUs::

    XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0,1,2,3 \
        pixi run -e cuda python3 scripts/benchmark_fitting.py

Or run specific experiments::

    pixi run -e cuda python3 scripts/benchmark_fitting.py --experiments 1,3
    pixi run -e cuda python3 scripts/benchmark_fitting.py --experiments 2 --repeats 10

Use --help for full options.
"""

import argparse
import json
import logging
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
import traceback

import pandas as pd

# ---------------------------------------------------------------------------
# Logging — Experiment 2 patches multidms to log exceptions rather than
# swallowing them silently. We configure logging early so those messages
# are captured.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPIKE_DATA = "runs/prod-v1/spike_analysis/training_functional_scores.csv"
SIM_DATA = "runs/prod-v1/simulation/simulated_func_scores.csv"
RESULTS_DIR = "runs/benchmark"

# ---------------------------------------------------------------------------
# Data loaders — mirror the exact logic in spike_03 and sim_02 notebooks
# ---------------------------------------------------------------------------

def build_spike_datasets(csv_path, reference="Omicron_BA1", subsample_frac=None):
    """Load spike training data, one multidms.Data per replicate."""
    import multidms

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
            .agg({"func_score": "mean"})
            .reset_index()
        )
        df_agg["aa_substitutions"] = df_agg["aa_substitutions"].fillna("")
        data = multidms.Data(
            df_agg,
            alphabet=multidms.AAS_WITHSTOP_WITHGAP,
            reference=reference,
            assert_site_integrity=False,
            verbose=False,
            name=f"rep-{rep}",
        )
        datasets.append(data)
    logger.info(
        "Loaded spike datasets: %d replicates, %d total rows%s",
        len(datasets),
        len(func_score_df),
        f" (subsampled {subsample_frac})" if subsample_frac else "",
    )
    return datasets


def build_sim_datasets(csv_path, reference="h1"):
    """Load simulation training data, one multidms.Data per (library, func_score_type)."""
    import multidms

    func_score_df = pd.read_csv(csv_path).fillna({"aa_substitutions": ""})
    datasets = []
    for (library, measurement), fs_df in func_score_df.groupby(
        ["library", "func_score_type"]
    ):
        df = fs_df.rename(columns={"homolog": "condition"})[
            ["condition", "aa_substitutions", "func_score"]
        ].copy()
        df["aa_substitutions"] = df["aa_substitutions"].fillna("")
        data = multidms.Data(
            df,
            reference=reference,
            alphabet=multidms.AAS_WITHSTOP_WITHGAP,
            verbose=False,
            name=f"{library}_{measurement}",
        )
        datasets.append(data)
    logger.info(
        "Loaded sim datasets: %d groups, %d total rows",
        len(datasets),
        len(func_score_df),
    )
    return datasets


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------

def make_fit_params(datasets, fusionreg_values, fit_cfg):
    """Build fit_params dict matching build_fit_params() from _common.py."""
    return {
        "fusionreg": fusionreg_values,
        "dataset": datasets,
        "ge_type": [fit_cfg.get("ge_type", "Sigmoid")],
        "l2reg": [fit_cfg.get("l2reg", 1e-4)],
        "beta0_ridge": [fit_cfg.get("beta0_ridge", 0)],
        "maxiter": [fit_cfg.get("maxiter", 75)],
        "tol": [fit_cfg.get("tol", 1e-6)],
        "warmstart": [fit_cfg.get("warmstart", False)],
        "loss_kwargs": [fit_cfg.get("loss_kwargs", {"δ": 1.0})],
        "ge_kwargs": [fit_cfg.get("ge_kwargs", {
            "tol": 1e-5, "maxiter": 1000, "maxls": 40, "jit": True, "verbose": False,
        })],
        "cal_kwargs": [fit_cfg.get("cal_kwargs", {
            "tol": 1e-4, "maxiter": 1000, "maxls": 40, "jit": True, "verbose": False,
        })],
        "beta_clip_range": [tuple(fit_cfg.get("beta_clip_range", [-10, 10]))],
    }


DEFAULT_FIT_CFG_SPIKE = {
    "maxiter": 75,
    "tol": 1e-6,
    "ge_type": "Sigmoid",
    "l2reg": 1e-4,
    "beta0_ridge": 0,
    "warmstart": False,
    "loss_kwargs": {"δ": 1.0},
    "ge_kwargs": {"tol": 1e-5, "maxiter": 1000, "maxls": 40, "jit": True, "verbose": False},
    "cal_kwargs": {"tol": 1e-4, "maxiter": 1000, "maxls": 40, "jit": True, "verbose": False},
    "beta_clip_range": [-10, 10],
}

DEFAULT_FIT_CFG_SIM = {
    **DEFAULT_FIT_CFG_SPIKE,
    "maxiter": 100,
    "beta0_ridge": 1e-5,
}


def get_peak_rss_mb():
    """Get peak resident set size in MB."""
    rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Linux":
        return rss_raw / 1024  # KB on Linux
    return rss_raw / (1024 * 1024)  # bytes on macOS


def extract_losses(models_df):
    """Extract per-model total training loss from fit results DataFrame."""
    losses = []
    for _, row in models_df.iterrows():
        if row.get("model") is not None:
            try:
                losses.append(row.model.training_loss["total"])
            except Exception:
                pass
    return losses


# ---------------------------------------------------------------------------
# Experiment 1 & 5: Throughput benchmarks
# ---------------------------------------------------------------------------

def run_throughput_benchmark(
    datasets, gpu_ids, n_processes, fusionreg_values, fit_cfg, label, warmup=True,
):
    """Run fit_models with timing and memory metrics.

    Returns a dict with benchmark results.
    """
    import multidms.model_collection

    fit_params = make_fit_params(datasets, fusionreg_values, fit_cfg)
    n_total = len(fusionreg_values) * len(datasets)

    # Optional warm-up: single short fit to JIT-compile
    jit_time = None
    if warmup:
        warm_params = {
            **fit_params,
            "fusionreg": [fusionreg_values[0]],
            "dataset": [datasets[0]],
            "maxiter": [3],
        }
        t0 = time.perf_counter()
        try:
            multidms.model_collection.fit_models(
                warm_params, gpu_ids=gpu_ids, n_processes=n_processes,
                failures="error",
            )
        except Exception as e:
            logger.warning("Warmup failed: %s", e)
        jit_time = round(time.perf_counter() - t0, 1)

    # Full benchmark
    t0 = time.perf_counter()
    try:
        n_fit, n_failed, models = multidms.model_collection.fit_models(
            fit_params, gpu_ids=gpu_ids, n_processes=n_processes,
            failures="error",
        )
        wall_time = round(time.perf_counter() - t0, 1)
        losses = extract_losses(models)
        retried = False
    except multidms.model_collection.ModelCollectionFitError as e:
        wall_time_initial = round(time.perf_counter() - t0, 1)
        logger.warning("Initial fit failed (%s), retrying on single GPU/process", e)

        # Retry sequentially (mirrors robust_fit_models)
        single_gpu = gpu_ids[:1] if gpu_ids else None
        t0 = time.perf_counter()
        n_fit, n_failed, models = multidms.model_collection.fit_models(
            fit_params, gpu_ids=single_gpu, n_processes=1,
            failures="error",
        )
        wall_time = round(time.perf_counter() - t0, 1)
        losses = extract_losses(models)
        retried = True
        logger.info(
            "Retry succeeded: %d/%d fits in %.1fs (initial attempt: %.1fs)",
            n_fit, n_total, wall_time, wall_time_initial,
        )

    result = {
        "label": label,
        "gpu_ids": [int(g) for g in gpu_ids] if gpu_ids else None,
        "n_processes": n_processes,
        "n_total": n_total,
        "n_fit": n_fit,
        "n_failed": n_failed,
        "retried": retried,
        "jit_warmup_s": jit_time,
        "wall_time_s": wall_time,
        "time_per_fit_s": round(wall_time / max(n_fit, 1), 1),
        "peak_rss_mb": round(get_peak_rss_mb(), 1),
        "mean_loss": round(sum(losses) / len(losses), 4) if losses else None,
        "loss_std": round(pd.Series(losses).std(), 4) if len(losses) > 1 else None,
    }
    return result


def experiment_1(args):
    """Experiment 1: GPU vs CPU fitting throughput (spike full dataset)."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: GPU vs CPU fitting throughput")
    logger.info("=" * 60)

    datasets = build_spike_datasets(SPIKE_DATA)
    fusionreg = [0.0, 2.0, 4.0, 8.0]

    configs = [
        ("cpu-1", None, 1),
        ("cpu-4", None, 4),
        ("cpu-8", None, 8),
        ("gpu-1", [0], 1),
    ]
    # Only include gpu-4 if we have 4+ visible GPUs
    try:
        import jax
        n_gpus = len(jax.devices("gpu"))
        if n_gpus >= 4:
            configs.append(("gpu-4", [0, 1, 2, 3], 1))
        elif n_gpus >= 2:
            configs.append((f"gpu-{n_gpus}", list(range(n_gpus)), 1))
    except Exception:
        logger.info("No GPUs detected, skipping GPU configs")
        configs = [c for c in configs if c[1] is None]

    results = []
    for trial in range(args.repeats):
        for label, gpu_ids, n_proc in configs:
            run_label = f"spike-full/{label}" + (f"/trial-{trial}" if args.repeats > 1 else "")
            logger.info("--- %s ---", run_label)
            r = run_throughput_benchmark(
                datasets, gpu_ids, n_proc, fusionreg,
                DEFAULT_FIT_CFG_SPIKE, run_label,
            )
            results.append(r)
            logger.info(json.dumps(r, indent=2))

    return results


def experiment_5(args):
    """Experiment 5: Dataset size scaling (sim + subsampled spike)."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 5: Dataset size scaling")
    logger.info("=" * 60)

    reduced_configs = [
        ("cpu-1", None, 1),
        ("cpu-4", None, 4),
        ("gpu-1", [0], 1),
    ]

    # Check GPU availability
    try:
        import jax
        n_gpus = len(jax.devices("gpu"))
        if n_gpus == 0:
            reduced_configs = [c for c in reduced_configs if c[1] is None]
    except Exception:
        reduced_configs = [c for c in reduced_configs if c[1] is None]

    results = []
    fusionreg = [0.0, 2.0, 4.0, 8.0]

    # Spike subsampled 1%
    logger.info("--- Spike 1%% subsample ---")
    spike_sub = build_spike_datasets(SPIKE_DATA, subsample_frac=0.01)
    for label, gpu_ids, n_proc in reduced_configs:
        run_label = f"spike-1pct/{label}"
        logger.info("--- %s ---", run_label)
        r = run_throughput_benchmark(
            spike_sub, gpu_ids, n_proc, fusionreg,
            DEFAULT_FIT_CFG_SPIKE, run_label,
        )
        results.append(r)
        logger.info(json.dumps(r, indent=2))

    # Simulation
    logger.info("--- Simulation datasets ---")
    sim_ds = build_sim_datasets(SIM_DATA)
    for label, gpu_ids, n_proc in reduced_configs:
        run_label = f"sim/{label}"
        logger.info("--- %s ---", run_label)
        r = run_throughput_benchmark(
            sim_ds, gpu_ids, n_proc, fusionreg,
            DEFAULT_FIT_CFG_SIM, run_label,
        )
        results.append(r)
        logger.info(json.dumps(r, indent=2))

    return results


# ---------------------------------------------------------------------------
# Experiment 2: Multi-GPU failure characterization
# ---------------------------------------------------------------------------

def _patch_multidms_logging():
    """Monkey-patch _fit_models_gpu to log exceptions instead of swallowing them."""
    import multidms.model_collection as mc

    original_fit_models_gpu = mc._fit_models_gpu
    mc._original_fit_models_gpu = original_fit_models_gpu

    def _fit_models_gpu_logged(exploded_params, gpu_ids):
        """Patched version that logs exceptions."""
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import jax

        devices = jax.devices("gpu")
        n_gpus = len(gpu_ids)
        gpu_semaphores = {i: threading.Semaphore(1) for i in range(n_gpus)}
        fit_logger = logging.getLogger("multidms.gpu_fit")

        def fit_on_gpu(task):
            idx, gpu_idx, kwargs = task
            device = devices[gpu_ids[gpu_idx]]
            semaphore = gpu_semaphores[gpu_idx]

            with semaphore:
                with jax.default_device(device):
                    try:
                        result = mc.fit_one_model(**kwargs)
                    except Exception as e:
                        fit_logger.exception(
                            "Model %d FAILED on GPU %d (device=%s): %s",
                            idx, gpu_ids[gpu_idx], device, e,
                        )
                        result = None
            return idx, result

        tasks = [(i, i % n_gpus, kw) for i, kw in enumerate(exploded_params)]
        results = [None] * len(tasks)

        with ThreadPoolExecutor(max_workers=n_gpus) as executor:
            futures = {executor.submit(fit_on_gpu, t): t for t in tasks}
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return results

    mc._fit_models_gpu = _fit_models_gpu_logged
    logger.info("Patched _fit_models_gpu with exception logging")


def _unpatch_multidms_logging():
    """Restore original _fit_models_gpu."""
    import multidms.model_collection as mc
    if hasattr(mc, "_original_fit_models_gpu"):
        mc._fit_models_gpu = mc._original_fit_models_gpu
        del mc._original_fit_models_gpu
        logger.info("Restored original _fit_models_gpu")


def experiment_2(args):
    """Experiment 2: Multi-GPU failure characterization."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 2: Multi-GPU failure characterization")
    logger.info("=" * 60)

    import multidms.model_collection

    try:
        import jax
        n_gpus = len(jax.devices("gpu"))
    except Exception:
        logger.error("No GPUs available, skipping experiment 2")
        return []

    if n_gpus < 2:
        logger.error("Need >= 2 GPUs for experiment 2, found %d", n_gpus)
        return []

    gpu_ids = list(range(min(n_gpus, 4)))
    datasets = build_spike_datasets(SPIKE_DATA)
    fusionreg = [0.0, 2.0, 4.0, 8.0]
    fit_params = make_fit_params(datasets, fusionreg, DEFAULT_FIT_CFG_SPIKE)
    n_total = len(fusionreg) * len(datasets)

    _patch_multidms_logging()

    results = []
    repeats = args.repeats if args.repeats > 1 else 10  # default 10 for this experiment

    for trial in range(repeats):
        logger.info("--- Trial %d/%d (gpu-%d) ---", trial + 1, repeats, len(gpu_ids))

        t0 = time.perf_counter()
        try:
            n_fit, n_failed, models = multidms.model_collection.fit_models(
                fit_params, gpu_ids=gpu_ids, failures="tolerate",
            )
        except Exception as e:
            # stack_fit_models may crash on None — count failures manually
            logger.error("fit_models crashed: %s", e)
            n_fit, n_failed = 0, n_total
            models = None

        wall_time = round(time.perf_counter() - t0, 1)

        result = {
            "label": f"gpu-{len(gpu_ids)}/trial-{trial}",
            "gpu_ids": gpu_ids,
            "n_total": n_total,
            "n_fit": n_fit,
            "n_failed": n_failed,
            "failure_rate": round(n_failed / n_total, 3) if n_total > 0 else 0,
            "wall_time_s": wall_time,
        }
        results.append(result)
        logger.info(json.dumps(result, indent=2))

    _unpatch_multidms_logging()

    # Summary
    failure_rates = [r["failure_rate"] for r in results]
    logger.info(
        "Multi-GPU failure summary: mean=%.1f%%, min=%.1f%%, max=%.1f%% over %d trials",
        100 * sum(failure_rates) / len(failure_rates),
        100 * min(failure_rates),
        100 * max(failure_rates),
        len(results),
    )

    return results


# ---------------------------------------------------------------------------
# Experiment 3: JIT compilation overhead
# ---------------------------------------------------------------------------

def experiment_3(args):
    """Experiment 3: JIT compilation overhead (cold vs warm)."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 3: JIT compilation overhead")
    logger.info("=" * 60)

    import jax
    import multidms.model_collection

    # Determine device
    try:
        n_gpus = len(jax.devices("gpu"))
        use_gpu = n_gpus > 0
    except Exception:
        use_gpu = False

    gpu_ids = [0] if use_gpu else None
    device_label = "gpu" if use_gpu else "cpu"

    datasets = build_spike_datasets(SPIKE_DATA)
    single_params = make_fit_params(
        [datasets[0]], [4.0], DEFAULT_FIT_CFG_SPIKE,
    )

    results = []

    for trial in range(max(args.repeats, 1)):
        # Cold fit — clear JIT cache
        jax.clear_caches()

        t0 = time.perf_counter()
        multidms.model_collection.fit_models(
            single_params, gpu_ids=gpu_ids, failures="error",
        )
        cold_time = round(time.perf_counter() - t0, 1)

        # Warm fit — JIT cached in-process
        t0 = time.perf_counter()
        multidms.model_collection.fit_models(
            single_params, gpu_ids=gpu_ids, failures="error",
        )
        warm_time = round(time.perf_counter() - t0, 1)

        jit_overhead = round(cold_time - warm_time, 1)
        jit_pct = round(100 * jit_overhead / cold_time, 1) if cold_time > 0 else 0

        result = {
            "label": f"jit-{device_label}/trial-{trial}",
            "device": device_label,
            "cold_fit_s": cold_time,
            "warm_fit_s": warm_time,
            "jit_overhead_s": jit_overhead,
            "jit_overhead_pct": jit_pct,
        }
        results.append(result)
        logger.info(json.dumps(result, indent=2))

    # Test persistent compilation cache if env var is set
    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if cache_dir:
        logger.info("Testing persistent JAX compilation cache at %s", cache_dir)
        jax.clear_caches()
        t0 = time.perf_counter()
        multidms.model_collection.fit_models(
            single_params, gpu_ids=gpu_ids, failures="error",
        )
        cached_time = round(time.perf_counter() - t0, 1)
        results.append({
            "label": f"jit-{device_label}/persistent-cache",
            "device": device_label,
            "cold_fit_with_disk_cache_s": cached_time,
            "cache_dir": cache_dir,
        })
        logger.info("Persistent cache cold fit: %.1fs", cached_time)

    return results


# ---------------------------------------------------------------------------
# Experiment 4: GPU memory footprint
# ---------------------------------------------------------------------------

def experiment_4(args):
    """Experiment 4: GPU memory footprint per fit."""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 4: GPU memory footprint per fit")
    logger.info("=" * 60)

    try:
        import jax
        n_gpus = len(jax.devices("gpu"))
    except Exception:
        logger.error("No GPUs available, skipping experiment 4")
        return []

    if n_gpus == 0:
        logger.error("No GPUs detected, skipping experiment 4")
        return []

    import multidms.model_collection

    datasets = build_spike_datasets(SPIKE_DATA)
    single_params = make_fit_params(
        [datasets[0]], [4.0], DEFAULT_FIT_CFG_SPIKE,
    )

    results = []

    # Helper: parse nvidia-smi CSV output
    def parse_gpu_memory_trace(csv_path):
        """Parse nvidia-smi CSV and return per-GPU peak memory in MB."""
        try:
            df = pd.read_csv(csv_path, skipinitialspace=True)
            # Column names from nvidia-smi have leading spaces
            df.columns = [c.strip() for c in df.columns]

            per_gpu = {}
            if "index" in df.columns and "memory.used [MiB]" in df.columns:
                df["memory_mb"] = df["memory.used [MiB]"].str.replace(" MiB", "").astype(float)
                for gpu_idx, group in df.groupby("index"):
                    per_gpu[int(gpu_idx)] = {
                        "peak_mb": round(group["memory_mb"].max(), 1),
                        "mean_mb": round(group["memory_mb"].mean(), 1),
                        "min_mb": round(group["memory_mb"].min(), 1),
                    }
            return per_gpu
        except Exception as e:
            logger.warning("Failed to parse GPU trace: %s", e)
            return {}

    def parse_gpu_utilization_trace(csv_path):
        """Parse nvidia-smi CSV for GPU utilization %."""
        try:
            df = pd.read_csv(csv_path, skipinitialspace=True)
            df.columns = [c.strip() for c in df.columns]

            per_gpu = {}
            if "index" in df.columns and "utilization.gpu [%]" in df.columns:
                df["util_pct"] = df["utilization.gpu [%]"].str.replace(" %", "").astype(float)
                for gpu_idx, group in df.groupby("index"):
                    per_gpu[int(gpu_idx)] = {
                        "mean_util_pct": round(group["util_pct"].mean(), 1),
                        "max_util_pct": round(group["util_pct"].max(), 1),
                    }
            return per_gpu
        except Exception as e:
            logger.warning("Failed to parse utilization trace: %s", e)
            return {}

    # --- Baseline: JAX import overhead ---
    logger.info("Measuring JAX import GPU memory baseline...")
    trace_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    trace_file.close()
    monitor = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv", "-l", "1",
        ],
        stdout=open(trace_file.name, "w"),
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3)  # Let baseline stabilize
    monitor.terminate()
    monitor.wait()
    baseline_memory = parse_gpu_memory_trace(trace_file.name)
    os.unlink(trace_file.name)
    results.append({
        "label": "baseline/jax-idle",
        "per_gpu_memory": baseline_memory,
    })
    logger.info("Baseline GPU memory: %s", json.dumps(baseline_memory, indent=2))

    # --- Single fit on GPU 0 ---
    logger.info("Measuring single fit GPU memory (GPU 0)...")
    trace_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    trace_file.close()
    monitor = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv", "-l", "1",
        ],
        stdout=open(trace_file.name, "w"),
        stderr=subprocess.DEVNULL,
    )

    t0 = time.perf_counter()
    multidms.model_collection.fit_models(
        single_params, gpu_ids=[0], failures="error",
    )
    fit_time = round(time.perf_counter() - t0, 1)

    time.sleep(2)  # Let memory settle
    monitor.terminate()
    monitor.wait()

    single_fit_memory = parse_gpu_memory_trace(trace_file.name)
    single_fit_util = parse_gpu_utilization_trace(trace_file.name)
    os.unlink(trace_file.name)

    results.append({
        "label": "single-fit/gpu-0",
        "fit_time_s": fit_time,
        "per_gpu_memory": single_fit_memory,
        "per_gpu_utilization": single_fit_util,
    })
    logger.info(
        "Single fit: %.1fs, memory: %s",
        fit_time, json.dumps(single_fit_memory, indent=2),
    )

    # --- Multi-GPU fit (all available GPUs) ---
    if n_gpus >= 2:
        gpu_ids = list(range(min(n_gpus, 4)))
        multi_params = make_fit_params(
            datasets, [0.0, 2.0, 4.0, 8.0], DEFAULT_FIT_CFG_SPIKE,
        )

        logger.info("Measuring multi-GPU fit memory (GPUs %s)...", gpu_ids)
        trace_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        trace_file.close()
        monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv", "-l", "1",
            ],
            stdout=open(trace_file.name, "w"),
            stderr=subprocess.DEVNULL,
        )

        t0 = time.perf_counter()
        try:
            multidms.model_collection.fit_models(
                multi_params, gpu_ids=gpu_ids, failures="tolerate",
            )
        except Exception as e:
            logger.warning("Multi-GPU fit error (expected): %s", e)
        multi_time = round(time.perf_counter() - t0, 1)

        time.sleep(2)
        monitor.terminate()
        monitor.wait()

        multi_memory = parse_gpu_memory_trace(trace_file.name)
        multi_util = parse_gpu_utilization_trace(trace_file.name)
        os.unlink(trace_file.name)

        results.append({
            "label": f"multi-fit/gpu-{len(gpu_ids)}",
            "gpu_ids": gpu_ids,
            "fit_time_s": multi_time,
            "per_gpu_memory": multi_memory,
            "per_gpu_utilization": multi_util,
        })
        logger.info(
            "Multi-GPU fit: %.1fs, memory: %s",
            multi_time, json.dumps(multi_memory, indent=2),
        )

    # Key ratio analysis
    if single_fit_memory and 0 in single_fit_memory:
        peak_mb = single_fit_memory[0]["peak_mb"]
        total_mb = None
        try:
            # Get total GPU memory from nvidia-smi
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                text=True,
            )
            total_mb = float(out.strip().split("\n")[0])
        except Exception:
            total_mb = 46080  # L40S default
        if total_mb:
            fits_per_gpu = int(total_mb / peak_mb) if peak_mb > 0 else 0
            logger.info(
                "Key ratio: peak=%.0f MB / %.0f MB total = %.1f%% → %d fits could share 1 GPU",
                peak_mb, total_mb, 100 * peak_mb / total_mb, fits_per_gpu,
            )
            results.append({
                "label": "memory-ratio",
                "peak_single_fit_mb": peak_mb,
                "total_gpu_mb": total_mb,
                "pct_used": round(100 * peak_mb / total_mb, 1),
                "max_concurrent_fits_per_gpu": fits_per_gpu,
            })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def save_results(results, experiment_name):
    """Save results to JSON file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"experiment_{experiment_name}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved %d results to %s", len(results), path)
    return path


def print_summary_table(results, title=""):
    """Print a human-readable summary table."""
    if not results:
        return
    print(f"\n{'=' * 70}")
    if title:
        print(f"  {title}")
        print(f"{'=' * 70}")

    # Find common keys
    keys = ["label", "wall_time_s", "time_per_fit_s", "n_fit", "n_failed",
            "jit_warmup_s", "peak_rss_mb", "mean_loss"]
    present_keys = [k for k in keys if any(k in r for r in results)]

    # Header
    widths = {k: max(len(k), max(len(str(r.get(k, ""))) for r in results)) for k in present_keys}
    header = " | ".join(k.ljust(widths[k]) for k in present_keys)
    print(header)
    print("-" * len(header))

    # Rows
    for r in results:
        row = " | ".join(str(r.get(k, "")).ljust(widths[k]) for k in present_keys)
        print(row)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark multidms fit_models() across GPU/CPU configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiments", "-e",
        default="1,2,3,4,5",
        help="Comma-separated experiment numbers to run (default: 1,2,3,4,5)",
    )
    parser.add_argument(
        "--repeats", "-r",
        type=int, default=1,
        help="Number of repeat trials per config (default: 1, experiment 2 defaults to 10)",
    )
    parser.add_argument(
        "--maxiter", "-m",
        type=int, default=10,
        help="Max fitting iterations (default: 10 — sufficient for profiling, "
             "use 75+ only if you need converged results)",
    )
    args = parser.parse_args()

    experiments = [int(x.strip()) for x in args.experiments.split(",")]

    # Override default maxiter for all experiments
    DEFAULT_FIT_CFG_SPIKE["maxiter"] = args.maxiter
    DEFAULT_FIT_CFG_SIM["maxiter"] = args.maxiter

    logger.info(
        "Benchmark starting: experiments=%s, repeats=%d, maxiter=%d",
        experiments, args.repeats, args.maxiter,
    )
    logger.info("Platform: %s, Python: %s", platform.platform(), sys.version.split()[0])

    # Check data availability
    for path in [SPIKE_DATA, SIM_DATA]:
        if not os.path.exists(path):
            logger.error("Data file not found: %s", path)
            logger.error("Run the production pipeline first to generate intermediate data.")
            sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = {}

    if 1 in experiments:
        results = experiment_1(args)
        all_results["1_throughput"] = results
        save_results(results, "1_throughput")
        print_summary_table(results, "Experiment 1: GPU vs CPU Throughput (spike full)")

    if 5 in experiments:
        results = experiment_5(args)
        all_results["5_scaling"] = results
        save_results(results, "5_scaling")
        print_summary_table(results, "Experiment 5: Dataset Size Scaling")

    if 2 in experiments:
        results = experiment_2(args)
        all_results["2_failure_mode"] = results
        save_results(results, "2_failure_mode")
        print_summary_table(
            [{"label": r["label"], "n_fit": r["n_fit"], "n_failed": r["n_failed"],
              "failure_rate": r["failure_rate"], "wall_time_s": r["wall_time_s"]}
             for r in results],
            "Experiment 2: Multi-GPU Failure Characterization",
        )

    if 3 in experiments:
        results = experiment_3(args)
        all_results["3_jit_overhead"] = results
        save_results(results, "3_jit_overhead")

    if 4 in experiments:
        results = experiment_4(args)
        all_results["4_gpu_memory"] = results
        save_results(results, "4_gpu_memory")

    # Save combined results
    combined_path = os.path.join(RESULTS_DIR, "all_experiments.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("All results saved to %s", combined_path)

    logger.info("Benchmark complete.")


if __name__ == "__main__":
    main()
