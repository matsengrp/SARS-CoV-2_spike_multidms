"""Snakemake rules for supplemental analysis notebooks."""


rule supplemental_sim_figures:
    input:
        muteffects="results/simulation/simulated_muteffects.csv",
        func_scores="results/simulation/simulated_func_scores.csv",
        model_vs_truth="results/simulation/model_vs_truth_beta_shift.csv",
        sparsity="results/simulation/fit_sparsity.csv",
        replicate_corr="results/simulation/library_replicate_correlation.csv",
        cv_loss="results/simulation/cross_validation_loss.csv",
    output:
        executed_notebook="results/supplemental/supplemental_sim_figures.ipynb",
        html="results/html/supplemental/supplemental_sim_figures.html",
    params:
        notebook="notebooks/supplemental/supplemental_sim_figures.ipynb",
        pm_args=PAPERMILL_ARGS,
    shell:
        """
        papermill {params.notebook} {output.executed_notebook} \
            {params.pm_args} && \
        jupyter nbconvert --to html {output.executed_notebook} \
            --output-dir $(dirname {output.html}) \
            --output $(basename {output.html})
        """


rule supplemental_structure:
    input:
        mutations_df="results/spike_analysis/mutations_df.csv",
        alignment="data/clustalo-I20230702-193723-0021-19090519-p1m.clustal_num",
    output:
        nbr_score="results/spike_analysis/nbr_score_df.csv",
        executed_notebook="results/supplemental/supplemental_structure.ipynb",
        html="results/html/supplemental/supplemental_structure.html",
    params:
        notebook="notebooks/supplemental/supplemental_structure.ipynb",
        pm_args=PAPERMILL_ARGS,
    shell:
        """
        papermill {params.notebook} {output.executed_notebook} \
            {params.pm_args} && \
        jupyter nbconvert --to html {output.executed_notebook} \
            --output-dir $(dirname {output.html}) \
            --output $(basename {output.html})
        """
