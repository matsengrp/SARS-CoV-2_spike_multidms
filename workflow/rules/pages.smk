"""Snakemake rules for generating the GitHub Pages index."""


rule generate_index:
    input:
        ALL_HTMLS,
    output:
        "results/html/index.html",
    params:
        script="workflow/scripts/generate_index.py",
    shell:
        """
        python {params.script}
        """
