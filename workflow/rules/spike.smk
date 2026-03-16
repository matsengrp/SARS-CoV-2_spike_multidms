"""Snakemake rules for the spike analysis pipeline (12 notebooks)."""


rule spike_01_data_loading:
    input:
        delta="data/Delta/functional_selections.csv",
        ba1="data/Omicron_BA1/functional_selections.csv",
        ba2="data/Omicron_BA2/functional_selections.csv",
    output:
        func_scores="results/spike_analysis/training_functional_scores.csv",
        executed_notebook="results/spike_analysis/spike_01_data_loading.ipynb",
        html="results/html/spike/spike_01_data_loading.html",
    params:
        notebook="notebooks/spike/spike_01_data_loading.ipynb",
        pm_args=PAPERMILL_ARGS,
    shell:
        """
        papermill {params.notebook} {output.executed_notebook} \
            {params.pm_args} && \
        jupyter nbconvert --to html {output.executed_notebook} \
            --output-dir $(dirname {output.html}) \
            --output $(basename {output.html})
        """


rule spike_02_exploratory_stats:
    input:
        func_scores="results/spike_analysis/training_functional_scores.csv",
    output:
        replicate_corr="results/spike_analysis/replicate_functional_score_correlation_scatter.pdf",
        executed_notebook="results/spike_analysis/spike_02_exploratory_stats.ipynb",
        html="results/html/spike/spike_02_exploratory_stats.html",
    params:
        notebook="notebooks/spike/spike_02_exploratory_stats.ipynb",
        pm_args=PAPERMILL_ARGS,
    shell:
        """
        papermill {params.notebook} {output.executed_notebook} \
            {params.pm_args} && \
        jupyter nbconvert --to html {output.executed_notebook} \
            --output-dir $(dirname {output.html}) \
            --output $(basename {output.html})
        """


rule spike_03_fit_models:
    input:
        func_scores="results/spike_analysis/training_functional_scores.csv",
    output:
        models="results/spike_analysis/full_models.pkl",
        executed_notebook="results/spike_analysis/spike_03_fit_models.ipynb",
        html="results/html/spike/spike_03_fit_models.html",
    params:
        notebook="notebooks/spike/spike_03_fit_models.ipynb",
        pm_args=PAPERMILL_ARGS,
    resources:
        gpu=1,
    shell:
        """
        papermill {params.notebook} {output.executed_notebook} \
            {params.pm_args} && \
        jupyter nbconvert --to html {output.executed_notebook} \
            --output-dir $(dirname {output.html}) \
            --output $(basename {output.html})
        """


rule spike_04_model_evaluation:
    input:
        models="results/spike_analysis/full_models.pkl",
    output:
        mutations_df="results/spike_analysis/mutations_df.csv",
        executed_notebook="results/spike_analysis/spike_04_model_evaluation.ipynb",
        html="results/html/spike/spike_04_model_evaluation.html",
    params:
        notebook="notebooks/spike/spike_04_model_evaluation.ipynb",
        pm_args=PAPERMILL_ARGS,
    shell:
        """
        papermill {params.notebook} {output.executed_notebook} \
            {params.pm_args} && \
        jupyter nbconvert --to html {output.executed_notebook} \
            --output-dir $(dirname {output.html}) \
            --output $(basename {output.html})
        """


rule spike_05_cross_validation:
    input:
        func_scores="results/spike_analysis/training_functional_scores.csv",
        models="results/spike_analysis/full_models.pkl",
    output:
        cv_models="results/spike_analysis/cv_models.pkl",
        shrinkage="results/spike_analysis/shrinkage_analysis_trace_plots_beta.pdf",
        executed_notebook="results/spike_analysis/spike_05_cross_validation.ipynb",
        html="results/html/spike/spike_05_cross_validation.html",
    params:
        notebook="notebooks/spike/spike_05_cross_validation.ipynb",
        pm_args=PAPERMILL_ARGS,
    resources:
        gpu=1,
    shell:
        """
        papermill {params.notebook} {output.executed_notebook} \
            {params.pm_args} && \
        jupyter nbconvert --to html {output.executed_notebook} \
            --output-dir $(dirname {output.html}) \
            --output $(basename {output.html})
        """


rule spike_06_global_epistasis:
    input:
        models="results/spike_analysis/full_models.pkl",
    output:
        ge_fig="results/spike_analysis/global_epistasis_and_prediction_correlations.pdf",
        executed_notebook="results/spike_analysis/spike_06_global_epistasis.ipynb",
        html="results/html/spike/spike_06_global_epistasis.html",
    params:
        notebook="notebooks/spike/spike_06_global_epistasis.ipynb",
        pm_args=PAPERMILL_ARGS,
    shell:
        """
        papermill {params.notebook} {output.executed_notebook} \
            {params.pm_args} && \
        jupyter nbconvert --to html {output.executed_notebook} \
            --output-dir $(dirname {output.html}) \
            --output $(basename {output.html})
        """


rule spike_07_shifted_mutations:
    input:
        models="results/spike_analysis/full_models.pkl",
    output:
        interactive_chart="results/spike_analysis/interactive_shift_chart.html",
        heatmap="results/spike_analysis/shift_by_site_heatmap_zoom.pdf",
        executed_notebook="results/spike_analysis/spike_07_shifted_mutations.ipynb",
        html="results/html/spike/spike_07_shifted_mutations.html",
    params:
        notebook="notebooks/spike/spike_07_shifted_mutations.ipynb",
        pm_args=PAPERMILL_ARGS,
    shell:
        """
        papermill {params.notebook} {output.executed_notebook} \
            {params.pm_args} && \
        jupyter nbconvert --to html {output.executed_notebook} \
            --output-dir $(dirname {output.html}) \
            --output $(basename {output.html})
        """


rule spike_08_naive_comparison:
    input:
        func_scores="results/spike_analysis/training_functional_scores.csv",
        models="results/spike_analysis/full_models.pkl",
    output:
        naive_corr="results/spike_analysis/shift_distribution_correlation_naive.pdf",
        executed_notebook="results/spike_analysis/spike_08_naive_comparison.ipynb",
        html="results/html/spike/spike_08_naive_comparison.html",
    params:
        notebook="notebooks/spike/spike_08_naive_comparison.ipynb",
        pm_args=PAPERMILL_ARGS,
    resources:
        gpu=1,
    shell:
        """
        papermill {params.notebook} {output.executed_notebook} \
            {params.pm_args} && \
        jupyter nbconvert --to html {output.executed_notebook} \
            --output-dir $(dirname {output.html}) \
            --output $(basename {output.html})
        """


rule spike_09_linear_comparison:
    input:
        func_scores="results/spike_analysis/training_functional_scores.csv",
        models="results/spike_analysis/full_models.pkl",
    output:
        linear_shrinkage="results/spike_analysis/shrinkage_analysis_linear_models.pdf",
        executed_notebook="results/spike_analysis/spike_09_linear_comparison.ipynb",
        html="results/html/spike/spike_09_linear_comparison.html",
    params:
        notebook="notebooks/spike/spike_09_linear_comparison.ipynb",
        pm_args=PAPERMILL_ARGS,
    resources:
        gpu=1,
    shell:
        """
        papermill {params.notebook} {output.executed_notebook} \
            {params.pm_args} && \
        jupyter nbconvert --to html {output.executed_notebook} \
            --output-dir $(dirname {output.html}) \
            --output $(basename {output.html})
        """


rule spike_10_validation:
    input:
        models="results/spike_analysis/full_models.pkl",
        titers="data/viral_titers.csv",
        validation="data/spike_validation_data.csv",
    output:
        validation_fig="results/spike_analysis/validation_titer_fold_change.pdf",
        executed_notebook="results/spike_analysis/spike_10_validation.ipynb",
        html="results/html/spike/spike_10_validation.html",
    params:
        notebook="notebooks/spike/spike_10_validation.ipynb",
        pm_args=PAPERMILL_ARGS,
    shell:
        """
        papermill {params.notebook} {output.executed_notebook} \
            {params.pm_args} && \
        jupyter nbconvert --to html {output.executed_notebook} \
            --output-dir $(dirname {output.html}) \
            --output $(basename {output.html})
        """


rule spike_11_reference_sensitivity:
    input:
        func_scores="results/spike_analysis/training_functional_scores.csv",
    output:
        ref_comparison="results/spike_analysis/reference_model_comparison_params_scatter.pdf",
        executed_notebook="results/spike_analysis/spike_11_reference_sensitivity.ipynb",
        html="results/html/spike/spike_11_reference_sensitivity.html",
    params:
        notebook="notebooks/spike/spike_11_reference_sensitivity.ipynb",
        pm_args=PAPERMILL_ARGS,
    resources:
        gpu=1,
    shell:
        """
        papermill {params.notebook} {output.executed_notebook} \
            {params.pm_args} && \
        jupyter nbconvert --to html {output.executed_notebook} \
            --output-dir $(dirname {output.html}) \
            --output $(basename {output.html})
        """


rule spike_12_sparsity_correlation:
    input:
        models="results/spike_analysis/full_models.pkl",
        mutations_df="results/spike_analysis/mutations_df.csv",
    output:
        sparsity_line="results/spike_analysis/percent_shifts_under_x_lineplot.pdf",
        shift_corr="results/spike_analysis/shift_corr_Delta_BA2.pdf",
        executed_notebook="results/spike_analysis/spike_12_sparsity_correlation.ipynb",
        html="results/html/spike/spike_12_sparsity_correlation.html",
    params:
        notebook="notebooks/spike/spike_12_sparsity_correlation.ipynb",
        pm_args=PAPERMILL_ARGS,
    shell:
        """
        papermill {params.notebook} {output.executed_notebook} \
            {params.pm_args} && \
        jupyter nbconvert --to html {output.executed_notebook} \
            --output-dir $(dirname {output.html}) \
            --output $(basename {output.html})
        """
