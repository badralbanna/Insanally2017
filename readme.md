# Readme for Insanally 2017



This repository contains the code used for analysis in Insanally, et al 2017. 

## Requirements

All packages and scripts used to analyze our data were
custom written in `python` (2.7.13) using `jupyter` (1.0.0) and `ipython` (5.3.0). Parallelization for multi-core processors was accomplished using `ipyparallel` (6.0.2). The RNN script is written in `matlab`

Additional packages required are:

- `numpy` (1.13.1)
- `scipy` (0.19.1)
- `matplotlib` (2.0.2)
- `h5py` (2.7.0)
- `scikit-learn` (0.19.0)
- `statsmodels` (0.8.0)

## Files and Directories

- **`Readme.md`:** This file
- **`data/`**: Directory containing two examples each from ACtx and FR2, one responsive and one non-responsive.
- **`animal_info.py`**: Python file containing a dictionary ANIMALS with relelvant infomation about each recording session needed to load the data. 
- **`bayseian_neural_decoding/`:** this python packge contains the analysis tools for:
  - the ISI-based baysian deocder
  - the Poisson-based ISI deocder
  - the first-spike latency-based decoder
- **`MI_beh_plots.py`**: Python module containing the plotting functions used to generate all figures.
- **`Defining non-responsiveness.ipynb`**: Jupyter notebook containing the scripts used to identify non-responsive cells.  
- **`Calculating cell firing statistics and receptive field.ipynb`**: Jupyter notebook containing the scripts used to calucalte all cell firing statistics shown in Extended Data Figure 7. 
- **`Decoding responses.ipynb`:** Script for decoding all recording sessions referenced in `animal_info.py`.
- **`RNN.m`**: Matlab script for the RNN analysis shown in Extended Data Figure 15

