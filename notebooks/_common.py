"""Shared utilities for pipeline notebooks."""

import copy
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


def load_config(config_path="config/config.yaml", profile=None):
    """Load config.yaml with optional profile overrides.

    Designed as a papermill-friendly function: both parameters can be
    injected directly into a notebook parameters cell.

    Parameters
    ----------
    config_path : str
        Path to main config YAML (default: ``config/config.yaml``).
    profile : str or None
        Optional profile name (e.g. ``test``) which loads
        ``config/profile_{profile}.yaml`` and deep-merges it over the base.

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
        fit_mut_df = fit.get_mutations_df(**kwargs)
        fit_mut_df = fit_mut_df.drop(
            [c for c in fit_mut_df.columns if "times_seen" in c], axis=1
        )
        new_column_name_map = {
            c: f"{replicate}_{c}" for c in fit_mut_df.columns
        }
        fit_mut_df = fit_mut_df.rename(new_column_name_map, axis=1)
        mutations_dfs.append(fit_mut_df)

    mut_df = reduce(
        lambda left, right: pd.merge(
            left, right, left_index=True, right_index=True, how=how
        ),
        mutations_dfs,
    )

    # Collect the union of all non-meta columns across all models
    all_cols = set()
    for fit in fit_dict.values():
        ref_df = fit.get_mutations_df(**kwargs)
        all_cols.update(c for c in ref_df.columns if "times_seen" not in c)

    meta_cols = ["wts", "sites", "muts"]
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
