"""Snakemake rules for generating the GitHub Pages index."""


rule generate_index:
    input:
        ALL_HTMLS,
    output:
        "results/html/index.html",
    log:
        "logs/generate_index.log",
    script:
        "../scripts/generate_index.py"
