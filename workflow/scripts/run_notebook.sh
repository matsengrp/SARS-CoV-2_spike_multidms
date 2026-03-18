#!/usr/bin/env bash
# Shared wrapper for papermill notebook execution + HTML export.
# Usage: run_notebook.sh <notebook> <executed_notebook> <html_output> <log> [papermill_args...]
set -euo pipefail

notebook="$1"; shift
executed_notebook="$1"; shift
html_output="$1"; shift
log="$1"; shift

{
    papermill "$notebook" "$executed_notebook" "$@" &&
    jupyter nbconvert --to html "$executed_notebook" \
        --output-dir "$(dirname "$html_output")" \
        --output "$(basename "$html_output")"
} 2>&1 | tee "$log"
