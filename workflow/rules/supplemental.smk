"""Snakemake rules for supplemental analysis notebooks."""


rule supplemental_sim_figures:
    input:
        muteffects=f"{SIM_OUT}/simulated_muteffects.csv",
        func_scores=f"{SIM_OUT}/simulated_func_scores.csv",
        model_vs_truth=f"{SIM_OUT}/model_vs_truth_beta_shift.csv",
        sparsity=f"{SIM_OUT}/fit_sparsity.csv",
        replicate_corr=f"{SIM_OUT}/library_replicate_correlation.csv",
        cv_loss=f"{SIM_OUT}/cross_validation_loss.csv",
    output:
        executed_notebook=f"{SUP_OUT}/supplemental_sim_figures.ipynb",
        html=f"{HTML_BASE}/supplemental/supplemental_sim_figures.html",
    params:
        notebook="notebooks/supplemental/supplemental_sim_figures.ipynb",
        pm_args=PAPERMILL_ARGS,
        jax_env=JAX_ENV,
    shell:
        """
        {params.jax_env} papermill {params.notebook} {output.executed_notebook} \
            {params.pm_args} && \
        jupyter nbconvert --to html {output.executed_notebook} \
            --output-dir $(dirname {output.html}) \
            --output $(basename {output.html})
        """


rule supplemental_structure:
    input:
        mutations_df=f"{SPIKE_OUT}/mutations_df.csv",
        alignment="data/clustalo-I20230702-193723-0021-19090519-p1m.clustal_num",
    output:
        nbr_score=f"{SPIKE_OUT}/nbr_score_df.csv",
        executed_notebook=f"{SUP_OUT}/supplemental_structure.ipynb",
        html=f"{HTML_BASE}/supplemental/supplemental_structure.html",
    params:
        notebook="notebooks/supplemental/supplemental_structure.ipynb",
        pm_args=PAPERMILL_ARGS,
        jax_env=JAX_ENV,
    shell:
        """
        {params.jax_env} papermill {params.notebook} {output.executed_notebook} \
            {params.pm_args} && \
        jupyter nbconvert --to html {output.executed_notebook} \
            --output-dir $(dirname {output.html}) \
            --output $(basename {output.html})
        """
