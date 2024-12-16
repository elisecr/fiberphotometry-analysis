# fiberphotometry-analysis
Jupyter notebook and Python code for analysis of pre-processed gCAMP fiber photometry signals, as published in:
> Rawlinson, E. C., McCullough, M. H., Marighetto, A., Al Abed, A. S., & Dehorter, N. (2024). Maladaptation of Memory Systems Impairs Dorsal CA1-Dependent Temporal Binding and Declarative Memory in The Cntnap2 Knockout Mouse Model of Autism. bioRxiv, 2024.2010.2029.620866. doi:10.1101/2024.10.29.620866

## Overview
This repository contains:
* The Jupyter notebook Peak analysis.ipynb which counts the occurrence of peaks and their corresponding amplitude from pre-processed data.
* The Jupyter notebook Behaviour analysis.ipynb which counts the occurence of keydowns representing behaviour across an experiment.
* The Jupyter notebook TFC peri-event plots.ipynb which graphs fiber photometry traces around experimental paradigm timings in trace fear conditioning. 
* YAML file fiber-photometry_2024.yml for installing the conda environment.

## Installation and usage
To run the notebook:
1. Clone the repository.
1. Install the environment with `conda` using the included YAML file.
``` 
conda env create -f fiber-photometry_2024.yml
```
1. Run all cells in order.

## Data availability
The data for this project are available upon request.

## References
Please see the companion paper cited above for details of the methods and packages used in this code.

## Citing this code
If you use or adapt our code or methods in your research, please cite the companion paper linked above.

## Questions 
Please email elise.rawlinson@anu.edu.au if you have questions about the code.
