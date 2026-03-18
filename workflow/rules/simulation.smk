"""Snakemake rules for the simulation validation pipeline (4 consolidated notebooks)."""


rule sim_01_data_simulation:
    input:
        config="config/config.yaml",
    output:
        muteffects="results/simulation/simulated_muteffects.csv",
        func_scores="results/simulation/simulated_func_scores.csv",
        executed_notebook="results/simulation/sim_01_data_simulation.ipynb",
        html="results/html/simulation/sim_01_data_simulation.html",
    params:
        notebook="notebooks/simulation/sim_01_data_simulation.ipynb",
        pm_args=PAPERMILL_ARGS,
    log:
        "logs/sim_01_data_simulation.log",
    shell:
        """
        bash workflow/scripts/run_notebook.sh \
            {params.notebook} {output.executed_notebook} {output.html} {log} \
            {params.pm_args}
        """


rule sim_02_model_fitting:
    input:
        config="config/config.yaml",
        func_scores="results/simulation/simulated_func_scores.csv",
    output:
        fit_collection="results/simulation/fit_collection.pkl",
        executed_notebook="results/simulation/sim_02_model_fitting.ipynb",
        html="results/html/simulation/sim_02_model_fitting.html",
    params:
        notebook="notebooks/simulation/sim_02_model_fitting.ipynb",
        pm_args=PAPERMILL_ARGS,
    log:
        "logs/sim_02_model_fitting.log",
    resources:
        gpu=1,
    shell:
        """
        bash workflow/scripts/run_notebook.sh \
            {params.notebook} {output.executed_notebook} {output.html} {log} \
            {params.pm_args}
        """


rule sim_03_evaluation:
    input:
        config="config/config.yaml",
        fit_collection="results/simulation/fit_collection.pkl",
        muteffects="results/simulation/simulated_muteffects.csv",
    output:
        model_vs_truth="results/simulation/model_vs_truth_beta_shift.csv",
        sparsity="results/simulation/fit_sparsity.csv",
        replicate_corr="results/simulation/library_replicate_correlation.csv",
        phenotype="results/simulation/model_vs_truth_variant_phenotype.csv",
        cv_loss="results/simulation/cross_validation_loss.csv",
        executed_notebook="results/simulation/sim_03_evaluation.ipynb",
        html="results/html/simulation/sim_03_evaluation.html",
    params:
        notebook="notebooks/simulation/sim_03_evaluation.ipynb",
        pm_args=PAPERMILL_ARGS,
    log:
        "logs/sim_03_evaluation.log",
    shell:
        """
        bash workflow/scripts/run_notebook.sh \
            {params.notebook} {output.executed_notebook} {output.html} {log} \
            {params.pm_args}
        """


rule sim_04_visualization:
    input:
        config="config/config.yaml",
        fit_collection="results/simulation/fit_collection.pkl",
        muteffects="results/simulation/simulated_muteffects.csv",
        model_vs_truth="results/simulation/model_vs_truth_beta_shift.csv",
        sparsity="results/simulation/fit_sparsity.csv",
        replicate_corr="results/simulation/library_replicate_correlation.csv",
        phenotype="results/simulation/model_vs_truth_variant_phenotype.csv",
        cv_loss="results/simulation/cross_validation_loss.csv",
    output:
        executed_notebook="results/simulation/sim_04_visualization.ipynb",
        html="results/html/simulation/sim_04_visualization.html",
    params:
        notebook="notebooks/simulation/sim_04_visualization.ipynb",
        pm_args=PAPERMILL_ARGS,
    log:
        "logs/sim_04_visualization.log",
    shell:
        """
        bash workflow/scripts/run_notebook.sh \
            {params.notebook} {output.executed_notebook} {output.html} {log} \
            {params.pm_args}
        """
