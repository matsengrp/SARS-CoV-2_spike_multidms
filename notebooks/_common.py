"""Shared utilities for pipeline notebooks."""

import copy
import os
from functools import reduce

import pandas as pd
import yaml


def deep_merge(base, override):
    """Recursively merge override dict into base dict.

    Values in override take precedence. Nested dicts are merged recursively
    rather than replaced wholesale.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_run_name(run_name):
    """Resolve run_name from explicit value, env var, or symlink.

    Resolution chain:
    1. Explicit ``run_name`` parameter (if truthy)
    2. ``MULTIDMS_RUN`` environment variable
    3. ``runs/current`` symlink target basename
    4. None (no named run)
    """
    if run_name:
        return run_name
    env_val = os.environ.get("MULTIDMS_RUN", "")
    if env_val:
        return env_val
    symlink = "runs/current"
    if os.path.islink(symlink):
        return os.path.basename(os.readlink(symlink))
    return None


def load_config(config_path="config/config.yaml", profile=None, run_name=None):
    """Load config.yaml with optional profile overrides and named-run redirection.

    Designed as a papermill-friendly function: all parameters can be
    injected directly into a notebook parameters cell.

    Parameters
    ----------
    config_path : str
        Path to main config YAML (default: ``config/config.yaml``).
    profile : str or None
        Optional profile name (e.g. ``test``) which loads
        ``config/profile_{profile}.yaml`` and deep-merges it over the base.
    run_name : str or None
        Optional named run. When resolved (via explicit value, ``MULTIDMS_RUN``
        env var, or ``runs/current`` symlink), output directories are rewritten
        to ``runs/<name>/...``.

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if profile:
        profile_path = f"config/profile_{profile}.yaml"
        with open(profile_path) as f:
            overrides = yaml.safe_load(f)
        if overrides:
            config = deep_merge(config, overrides)

    resolved = _resolve_run_name(run_name)
    if resolved:
        base = f"runs/{resolved}"
        if "simulation" in config:
            config["simulation"]["output_dir"] = f"{base}/simulation"
        if "spike" in config:
            config["spike"]["output_dir"] = f"{base}/spike_analysis"

    return config


def setup_altair():
    """Enable the HTML renderer for Altair charts.

    Ensures interactive Altair charts survive headless papermill execution
    and nbconvert HTML export.
    """
    import altair as alt

    alt.renderers.enable("html")


def build_phenotype_fxn_dict(
    sigmoid_phenotype, ge_scale, wt_latent, is_reference=True
):
    """Build a phenotype function dict from a SigmoidPhenotypeSimulator.

    Parameters
    ----------
    sigmoid_phenotype : dms_variants.simulate.SigmoidPhenotypeSimulator
        The simulator object with ``latentPhenotype`` method.
    ge_scale : float
        Scale of the sigmoid global epistasis function.
    wt_latent : float
        Wildtype latent phenotype value (for the reference homolog).
    is_reference : bool
        If True, observed phenotype is ``g(latent)``. If False,
        observed phenotype is ``g(latent) - g(wt_latent_h2)`` to center
        the non-reference homolog at zero.

    Returns
    -------
    dict
        Maps ``"latentPhenotype"``, ``"observedPhenotype"``, and
        ``"observedEnrichment"`` to callable functions.
    """
    import numpy as np

    def g(z):
        ge_bias = -ge_scale / (1 + np.exp(-wt_latent))
        return ge_bias + (ge_scale / (1 + np.exp(-z)))

    fxn_dict = {"latentPhenotype": sigmoid_phenotype.latentPhenotype}

    if is_reference:
        fxn_dict["observedPhenotype"] = lambda x: g(
            float(fxn_dict["latentPhenotype"](x))
        )
    else:
        offset = g(float(sigmoid_phenotype.wt_latent))
        fxn_dict["observedPhenotype"] = lambda x: g(
            float(fxn_dict["latentPhenotype"](x))
        ) - offset

    fxn_dict["observedEnrichment"] = lambda x: 2 ** (
        fxn_dict["observedPhenotype"](x)
    )
    return fxn_dict


def reconstruct_simulators(sim_config, mut_effects_df, seed):
    """Reconstruct phenotype simulators and function dicts from config and saved mutation effects.

    Re-seeds the RNG and regenerates the same gene sequences and
    SigmoidPhenotypeSimulators that ``sim_01_data_simulation`` created,
    then rebuilds the phenotype function dicts for both homologs.

    Parameters
    ----------
    sim_config : dict
        The ``simulation`` section of the pipeline config.
    mut_effects_df : pandas.DataFrame
        The saved mutation effects CSV (``simulated_muteffects.csv``).
    seed : int
        Global random seed from the pipeline config.

    Returns
    -------
    tuple of (dict, dict)
        ``(phenotype_fxn_dict_h1, phenotype_fxn_dict_h2)`` where each dict
        maps ``"latentPhenotype"``, ``"observedPhenotype"``, and
        ``"observedEnrichment"`` to callable functions.
    """
    import random

    import Bio.Seq
    import dms_variants.simulate
    import numpy as np
    from dms_variants.constants import (
        AAS_NOSTOP,
        AAS_WITHSTOP,
        CODONS_NOSTOP,
    )

    wt_latent = sim_config["wt_latent"]
    sigmoid_phenotype_scale = sim_config["sigmoid_phenotype_scale"]

    random.seed(seed)
    np.random.seed(seed)

    geneseq_h1 = "".join(random.choices(CODONS_NOSTOP, k=sim_config["genelength"]))
    aaseq_h1 = str(Bio.Seq.Seq(geneseq_h1).translate())

    mut_pheno_args = {
        "geneseq": geneseq_h1,
        "wt_latent": wt_latent,
        "seed": seed,
        "stop_effect": sim_config["stop_effect"],
        "norm_weights": tuple(tuple(w) for w in sim_config["norm_weights"]),
    }
    SigmoidPhenotype_h1 = dms_variants.simulate.SigmoidPhenotypeSimulator(
        **mut_pheno_args
    )
    SigmoidPhenotype_h2 = dms_variants.simulate.SigmoidPhenotypeSimulator(
        **mut_pheno_args
    )

    for mutation in SigmoidPhenotype_h2.muteffects.keys():
        SigmoidPhenotype_h2.muteffects[mutation] = mut_effects_df.loc[
            mut_effects_df["mutation"] == mutation, "beta_h2"
        ].values[0]

    wt_latent_phenotype_shift = mut_effects_df.query("bundle_mut")["beta_h2"].sum()
    SigmoidPhenotype_h2.wt_latent = (
        SigmoidPhenotype_h1.wt_latent + wt_latent_phenotype_shift
    )

    non_identical_sites = sorted(
        random.sample(
            range(1, len(aaseq_h1) + 1), sim_config["n_non_identical_sites"]
        )
    )
    aaseq_h2 = ""
    for aa_n, aa in enumerate(aaseq_h1, 1):
        if aa_n in non_identical_sites:
            valid = [
                m
                for m in AAS_NOSTOP
                if m != aa
                and SigmoidPhenotype_h1.muteffects[f"{aa}{aa_n}{m}"]
                > sim_config["min_muteffect_in_bundle"]
                and SigmoidPhenotype_h1.muteffects[f"{aa}{aa_n}{m}"]
                < sim_config["max_muteffect_in_bundle"]
            ]
            aaseq_h2 += random.choice(valid)
        else:
            aaseq_h2 += aa

    for idx, row in mut_effects_df.query("bundle_mut").iterrows():
        for aa_mut in AAS_WITHSTOP:
            if aa_mut == row.mut_aa:
                continue
            non_ref_mutation = f"{row.mut_aa}{row.site}{aa_mut}"
            if aa_mut == "*":
                SigmoidPhenotype_h2.muteffects[non_ref_mutation] = sim_config[
                    "stop_effect"
                ]
            elif aa_mut == row.wt_aa:
                SigmoidPhenotype_h2.muteffects[non_ref_mutation] = -row.beta_h2
            else:
                ref_mut = f"{row.wt_aa}{row.site}{aa_mut}"
                ref_mut_effect = mut_effects_df.loc[
                    mut_effects_df["mutation"] == ref_mut, "beta_h2"
                ].values[0]
                SigmoidPhenotype_h2.muteffects[non_ref_mutation] = (
                    -row.beta_h2 + ref_mut_effect
                )

    phenotype_fxn_dict_h1 = build_phenotype_fxn_dict(
        SigmoidPhenotype_h1, sigmoid_phenotype_scale, wt_latent, is_reference=True
    )
    phenotype_fxn_dict_h2 = build_phenotype_fxn_dict(
        SigmoidPhenotype_h2, sigmoid_phenotype_scale, wt_latent, is_reference=False
    )
    return phenotype_fxn_dict_h1, phenotype_fxn_dict_h2


def all_muts_known(substitutions, known_muts):
    """Check if all mutations in a substitution string are in the known set.

    Parameters
    ----------
    substitutions : str
        Space-separated mutation strings (may be NaN or empty).
    known_muts : set
        Set of known mutation strings.

    Returns
    -------
    bool
    """
    if pd.isna(substitutions) or substitutions.strip() == "":
        return True
    return all(m in known_muts for m in substitutions.split())


def build_fit_params(fit_config, datasets):
    """Build a standard fitting parameter dict from a config section.

    Parameters
    ----------
    fit_config : dict
        The ``fitting`` subsection of the pipeline config
        (e.g., ``config["spike"]["fitting"]``).
    datasets : list
        List of ``multidms.Data`` objects to fit.

    Returns
    -------
    dict
        Ready to pass to ``multidms.model_collection.fit_models()``.
    """
    return {
        "maxiter": [fit_config["maxiter"]],
        "tol": [fit_config["tol"]],
        "fusionreg": fit_config["fusionreg_values"],
        "l2reg": [fit_config["l2reg"]],
        "beta0_ridge": [fit_config["beta0_ridge"]],
        "ge_type": [fit_config["ge_type"]],
        "ge_kwargs": [fit_config["ge_kwargs"]],
        "cal_kwargs": [fit_config["cal_kwargs"]],
        "loss_kwargs": [fit_config["loss_kwargs"]],
        "warmstart": [fit_config["warmstart"]],
        "beta0_init": [fit_config["beta0_init"]],
        "alpha_init": [fit_config["alpha_init"]],
        "beta_clip_range": [tuple(fit_config["beta_clip_range"])],
        "dataset": datasets,
    }


def combine_replicate_muts(
    fit_dict, predicted_func_scores=False, how="inner", **kwargs
):
    """Combine mutation DataFrames from replicate model fits.

    Takes a dictionary of fit objects (keyed by replicate name), extracts
    per-replicate mutation DataFrames, merges them, and computes replicate
    averages for each parameter column.

    Parameters
    ----------
    fit_dict : dict
        Maps replicate name (str) to a fitted model object with a
        ``get_mutations_df(**kwargs)`` method.
    predicted_func_scores : bool
        If False (default), columns containing ``predicted_func_score``
        are excluded from the output.
    how : str
        Merge strategy passed to ``pd.merge`` (default ``"inner"``).
    **kwargs
        Forwarded to each model's ``get_mutations_df()``.

    Returns
    -------
    pandas.DataFrame
        Merged DataFrame with per-replicate and ``avg_`` columns.
    """
    mutations_dfs = []
    for replicate, fit in fit_dict.items():
        fit_mut_df = fit.get_mutations_df(**kwargs).reset_index()
        # Ensure 'mutation' column exists for merging
        if "mutation" not in fit_mut_df.columns and fit_mut_df.index.name == "mutation":
            fit_mut_df = fit_mut_df.reset_index()
        fit_mut_df = fit_mut_df.drop(
            [c for c in fit_mut_df.columns if "times_seen" in c], axis=1
        )
        new_column_name_map = {
            c: f"{replicate}_{c}"
            for c in fit_mut_df.columns
            if c != "mutation"
        }
        fit_mut_df = fit_mut_df.rename(new_column_name_map, axis=1)
        mutations_dfs.append(fit_mut_df)

    mut_df = reduce(
        lambda left, right: pd.merge(
            left, right, on="mutation", how=how
        ),
        mutations_dfs,
    )

    # Collect the union of all non-meta columns across all models
    all_cols = set()
    for fit in fit_dict.values():
        ref_df = fit.get_mutations_df(**kwargs)
        all_cols.update(c for c in ref_df.columns if "times_seen" not in c)

    meta_cols = ["mutation", "wts", "sites", "muts"]
    param_cols = sorted(
        c for c in all_cols if c not in meta_cols and c != "mutation"
    )

    # Extract shared meta columns from first replicate
    first_rep = list(fit_dict.keys())[0]
    for mc_col in meta_cols:
        col_name = f"{first_rep}_{mc_col}"
        if col_name in mut_df.columns:
            mut_df[mc_col] = mut_df[col_name]
    # Drop all replicate versions of meta columns
    drop_meta = [
        f"{rep}_{mc_col}"
        for rep in fit_dict.keys()
        for mc_col in meta_cols
        if f"{rep}_{mc_col}" in mut_df.columns
    ]
    mut_df.drop(drop_meta, axis=1, inplace=True)

    # Compute replicate averages for parameter columns
    column_order = []
    for c in param_cols:
        if not predicted_func_scores and "predicted_func_score" in c:
            continue

        cols_to_combine = [
            f"{rep}_{c}"
            for rep in fit_dict.keys()
            if f"{rep}_{c}" in mut_df.columns
        ]
        if not cols_to_combine:
            continue

        mut_df[f"avg_{c}"] = mut_df[cols_to_combine].mean(axis=1)
        column_order += cols_to_combine + [f"avg_{c}"]

    return mut_df.loc[:, meta_cols + column_order]


def robust_fit_models(params, gpu_ids=None, n_processes=1, **kwargs):
    """Wrapper around multidms.model_collection.fit_models with GPU retry.

    Multi-GPU parallel fitting can fail transiently (e.g., during first
    JIT compilation). If the initial fit raises ModelCollectionFitError,
    retry all fits sequentially on a single GPU.

    Parameters are passed through to ``multidms.model_collection.fit_models``.
    """
    import multidms.model_collection

    try:
        return multidms.model_collection.fit_models(
            params, gpu_ids=gpu_ids, n_processes=n_processes, **kwargs
        )
    except multidms.model_collection.ModelCollectionFitError as e:
        single_gpu = gpu_ids[:1] if gpu_ids else None
        print(
            f"WARNING: {e} — retrying all fits sequentially"
            f" on GPU {single_gpu}"
        )
        return multidms.model_collection.fit_models(
            params, gpu_ids=single_gpu, n_processes=1, **kwargs
        )
