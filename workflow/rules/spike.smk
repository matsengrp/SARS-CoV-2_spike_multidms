"""Snakemake rules for the spike analysis pipeline (12 notebooks)."""


rule spike_01_data_loading:
    input:
        delta="data/Delta/functional_selections.csv",
        ba1="data/Omicron_BA1/functional_selections.csv",
        ba2="data/Omicron_BA2/functional_selections.csv",
    output:
        func_scores=f"{SPIKE_OUT}/training_functional_scores.csv",
        executed_notebook=f"{SPIKE_OUT}/spike_01_data_loading.ipynb",
        html=f"{HTML_BASE}/spike/spike_01_data_loading.html",
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
        func_scores=f"{SPIKE_OUT}/training_functional_scores.csv",
    output:
        replicate_corr=f"{SPIKE_OUT}/replicate_functional_score_correlation_scatter.pdf",
        executed_notebook=f"{SPIKE_OUT}/spike_02_exploratory_stats.ipynb",
        html=f"{HTML_BASE}/spike/spike_02_exploratory_stats.html",
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
        func_scores=f"{SPIKE_OUT}/training_functional_scores.csv",
    output:
        models=f"{SPIKE_OUT}/full_models.pkl",
        executed_notebook=f"{SPIKE_OUT}/spike_03_fit_models.ipynb",
        html=f"{HTML_BASE}/spike/spike_03_fit_models.html",
    params:
        notebook="notebooks/spike/spike_03_fit_models.ipynb",
        pm_args=PAPERMILL_ARGS,
    resources:
        gpu=GPU_FIT,
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
        models=f"{SPIKE_OUT}/full_models.pkl",
    output:
        mutations_df=f"{SPIKE_OUT}/mutations_df.csv",
        executed_notebook=f"{SPIKE_OUT}/spike_04_model_evaluation.ipynb",
        html=f"{HTML_BASE}/spike/spike_04_model_evaluation.html",
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
        func_scores=f"{SPIKE_OUT}/training_functional_scores.csv",
        models=f"{SPIKE_OUT}/full_models.pkl",
    output:
        cv_models=f"{SPIKE_OUT}/cv_models.pkl",
        shrinkage=f"{SPIKE_OUT}/shrinkage_analysis_trace_plots_beta.pdf",
        executed_notebook=f"{SPIKE_OUT}/spike_05_cross_validation.ipynb",
        html=f"{HTML_BASE}/spike/spike_05_cross_validation.html",
    params:
        notebook="notebooks/spike/spike_05_cross_validation.ipynb",
        pm_args=PAPERMILL_ARGS,
    resources:
        gpu=GPU_FIT,
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
        models=f"{SPIKE_OUT}/full_models.pkl",
    output:
        ge_fig=f"{SPIKE_OUT}/global_epistasis_and_prediction_correlations.pdf",
        executed_notebook=f"{SPIKE_OUT}/spike_06_global_epistasis.ipynb",
        html=f"{HTML_BASE}/spike/spike_06_global_epistasis.html",
    params:
        notebook="notebooks/spike/spike_06_global_epistasis.ipynb",
        pm_args=PAPERMILL_ARGS,
    resources:
        gpu=GPU_LOAD,
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
        models=f"{SPIKE_OUT}/full_models.pkl",
    output:
        interactive_chart=f"{SPIKE_OUT}/interactive_shift_chart.html",
        heatmap=f"{SPIKE_OUT}/shift_by_site_heatmap_zoom.pdf",
        executed_notebook=f"{SPIKE_OUT}/spike_07_shifted_mutations.ipynb",
        html=f"{HTML_BASE}/spike/spike_07_shifted_mutations.html",
    params:
        notebook="notebooks/spike/spike_07_shifted_mutations.ipynb",
        pm_args=PAPERMILL_ARGS,
    resources:
        gpu=GPU_LOAD,
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
        func_scores=f"{SPIKE_OUT}/training_functional_scores.csv",
        models=f"{SPIKE_OUT}/full_models.pkl",
    output:
        naive_corr=f"{SPIKE_OUT}/shift_distribution_correlation_naive.pdf",
        executed_notebook=f"{SPIKE_OUT}/spike_08_naive_comparison.ipynb",
        html=f"{HTML_BASE}/spike/spike_08_naive_comparison.html",
    params:
        notebook="notebooks/spike/spike_08_naive_comparison.ipynb",
        pm_args=PAPERMILL_ARGS,
    resources:
        gpu=GPU_FIT,
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
        func_scores=f"{SPIKE_OUT}/training_functional_scores.csv",
        models=f"{SPIKE_OUT}/full_models.pkl",
    output:
        linear_shrinkage=f"{SPIKE_OUT}/shrinkage_analysis_linear_models.pdf",
        executed_notebook=f"{SPIKE_OUT}/spike_09_linear_comparison.ipynb",
        html=f"{HTML_BASE}/spike/spike_09_linear_comparison.html",
    params:
        notebook="notebooks/spike/spike_09_linear_comparison.ipynb",
        pm_args=PAPERMILL_ARGS,
    resources:
        gpu=GPU_FIT,
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
        models=f"{SPIKE_OUT}/full_models.pkl",
        titers="data/viral_titers.csv",
        validation="data/spike_validation_data.csv",
    output:
        validation_fig=f"{SPIKE_OUT}/validation_titer_fold_change.pdf",
        executed_notebook=f"{SPIKE_OUT}/spike_10_validation.ipynb",
        html=f"{HTML_BASE}/spike/spike_10_validation.html",
    params:
        notebook="notebooks/spike/spike_10_validation.ipynb",
        pm_args=PAPERMILL_ARGS,
    resources:
        gpu=GPU_LOAD,
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
        func_scores=f"{SPIKE_OUT}/training_functional_scores.csv",
    output:
        ref_comparison=f"{SPIKE_OUT}/reference_model_comparison_params_scatter.pdf",
        executed_notebook=f"{SPIKE_OUT}/spike_11_reference_sensitivity.ipynb",
        html=f"{HTML_BASE}/spike/spike_11_reference_sensitivity.html",
    params:
        notebook="notebooks/spike/spike_11_reference_sensitivity.ipynb",
        pm_args=PAPERMILL_ARGS,
    resources:
        gpu=GPU_FIT,
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
        models=f"{SPIKE_OUT}/full_models.pkl",
        mutations_df=f"{SPIKE_OUT}/mutations_df.csv",
    output:
        sparsity_line=f"{SPIKE_OUT}/percent_shifts_under_x_lineplot.pdf",
        shift_corr=f"{SPIKE_OUT}/shift_corr_Delta_BA2.pdf",
        executed_notebook=f"{SPIKE_OUT}/spike_12_sparsity_correlation.ipynb",
        html=f"{HTML_BASE}/spike/spike_12_sparsity_correlation.html",
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
