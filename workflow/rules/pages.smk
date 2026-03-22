"""Snakemake rules for generating the GitHub Pages index."""


rule generate_index:
    input:
        ALL_HTMLS,
    output:
        f"{HTML_BASE}/index.html",
    params:
        script="workflow/scripts/generate_index.py",
        html_dir=HTML_BASE,
    log:
        "logs/generate_index.log",
    shell:
        """
        python {params.script} --html-dir {params.html_dir} 2>&1 | tee {log}
        """
