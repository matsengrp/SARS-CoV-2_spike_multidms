"""Snakemake rules for the simulation validation pipeline (4 consolidated notebooks)."""


rule sim_01_data_simulation:
    input:
        config="config/config.yaml",
    output:
        muteffects=f"{SIM_OUT}/simulated_muteffects.csv",
        func_scores=f"{SIM_OUT}/simulated_func_scores.csv",
        executed_notebook=f"{SIM_OUT}/sim_01_data_simulation.ipynb",
        html=f"{HTML_BASE}/simulation/sim_01_data_simulation.html",
    params:
        notebook="notebooks/simulation/sim_01_data_simulation.ipynb",
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


rule sim_02_model_fitting:
    input:
        func_scores=f"{SIM_OUT}/simulated_func_scores.csv",
    output:
        fit_collection=f"{SIM_OUT}/fit_collection.pkl",
        executed_notebook=f"{SIM_OUT}/sim_02_model_fitting.ipynb",
        html=f"{HTML_BASE}/simulation/sim_02_model_fitting.html",
    params:
        notebook="notebooks/simulation/sim_02_model_fitting.ipynb",
        pm_args=PAPERMILL_ARGS,
        jax_env=JAX_ENV,
    resources:
        gpu=GPU_FIT,
    shell:
        """
        {params.jax_env} papermill {params.notebook} {output.executed_notebook} \
            {params.pm_args} && \
        jupyter nbconvert --to html {output.executed_notebook} \
            --output-dir $(dirname {output.html}) \
            --output $(basename {output.html})
        """


rule sim_03_evaluation:
    input:
        fit_collection=f"{SIM_OUT}/fit_collection.pkl",
        muteffects=f"{SIM_OUT}/simulated_muteffects.csv",
    output:
        model_vs_truth=f"{SIM_OUT}/model_vs_truth_beta_shift.csv",
        sparsity=f"{SIM_OUT}/fit_sparsity.csv",
        replicate_corr=f"{SIM_OUT}/library_replicate_correlation.csv",
        phenotype=f"{SIM_OUT}/model_vs_truth_variant_phenotype.csv",
        cv_loss=f"{SIM_OUT}/cross_validation_loss.csv",
        executed_notebook=f"{SIM_OUT}/sim_03_evaluation.ipynb",
        html=f"{HTML_BASE}/simulation/sim_03_evaluation.html",
    params:
        notebook="notebooks/simulation/sim_03_evaluation.ipynb",
        pm_args=PAPERMILL_ARGS,
        jax_env=JAX_ENV,
    resources:
        gpu=GPU_FIT,
    shell:
        """
        {params.jax_env} papermill {params.notebook} {output.executed_notebook} \
            {params.pm_args} && \
        jupyter nbconvert --to html {output.executed_notebook} \
            --output-dir $(dirname {output.html}) \
            --output $(basename {output.html})
        """


rule sim_04_visualization:
    input:
        fit_collection=f"{SIM_OUT}/fit_collection.pkl",
        muteffects=f"{SIM_OUT}/simulated_muteffects.csv",
        model_vs_truth=f"{SIM_OUT}/model_vs_truth_beta_shift.csv",
        sparsity=f"{SIM_OUT}/fit_sparsity.csv",
        replicate_corr=f"{SIM_OUT}/library_replicate_correlation.csv",
        phenotype=f"{SIM_OUT}/model_vs_truth_variant_phenotype.csv",
        cv_loss=f"{SIM_OUT}/cross_validation_loss.csv",
    output:
        executed_notebook=f"{SIM_OUT}/sim_04_visualization.ipynb",
        html=f"{HTML_BASE}/simulation/sim_04_visualization.html",
    params:
        notebook="notebooks/simulation/sim_04_visualization.ipynb",
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
