# SARS-CoV-2 Spike multidms analysis

Analysis of SARS-CoV-2 spike homologs as seen in our manuscript 
_Jointly modeling deep mutational scans identifies shifted mutational effects among SARS-CoV-2 spike homologs_.
Please see the 
[web page](https://matsengrp.github.io/SARS-CoV-2_spike_multidms/)
to explore the analysis interactively.

This repository contains the code in the form of a jupyter notebook,
as well as the source code for generating the page.

To run the notebook:
1. clone the repository 
```
git clone https://github.com/matsengrp/SARS-CoV-2_spike_multidms.git
```
2. (recommended) create a new environment
```
mamba create --name multidms-spike python=3.11
mamba activate multidms-spike
```
2. install the requirements
```
pip install -r analysis_requirements.txt
```
3. run jupyter notebook
```
jupyter notebook
```

## Key files
1. [results/spike_analysis/mutations_df.csv](results/mutations_df.csv) contains the respective parameters and phenotype effects for all mutations analyzed in this study.
2. [results/spike_analysis/training_functional_scores.csv](results/training_functional_scores.csv) contains the curated model training set of barcoded variants and their respective functional scores, for each of the two replicate.

