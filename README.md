## Correlated Product of Experts for Sparse Gaussian Process Regression

Implements the "Correlated Product of Experts" (CPoE) regression algorithm from the paper "Correlated Product of Experts for Sparse Gaussian Process Regression" by Manuel Schuerch, Dario Azzimonti, Alessio Benavoli and Marco Zaffalon.


## Usage

We provide the source code for CPoE, comparisons and examples to demonstrate the usage.
The main code is in "source/CPoE.py" and the folder "experiments" contains several jupyter notebooks for running examples and experiments.
In order to run the algorithm, you need [GPy](https://github.com/SheffieldML/GPy) (tested up to version 1.9.6) since we use their implementation of the kernels and likelihoods.

## Overview of available notebooks

- [downloading and preprocessing of public data](https://github.com/manschuer/CPoE/blob/main/experiments/download_data.ipynb)

- [example 1D](https://github.com/manschuer/CPoE/blob/main/experiments/example_1D.ipynb)

- [example 2D](https://github.com/manschuer/CPoE/blob/main/experiments/example_2D.ipynb)

- [experiment synthetic data](https://github.com/manschuer/CPoE/blob/main/experiments/syntheticData.ipynb)

- [experiments with real world data and deterministic optimization](https://github.com/manschuer/CPoE/blob/main/experiments/realData1.ipynb)

- [experiments with real world data and stochastic optimization](https://github.com/manschuer/CPoE/blob/main/experiments/realData2.ipynb)

- [experiment time series](https://github.com/manschuer/CPoE/blob/main/experiments/timeSeries.ipynb)

- [comparison with deep mixture GPs](https://github.com/manschuer/CPoE/blob/main/experiments//comparisonDSM_py.ipynb)

- [comparison with non-GP methods](https://github.com/manschuer/CPoE/blob/main/experiments/comparisonNonGP.ipynb)


## Contributors

Schuerch, M. and Azzimonti, D. and Benavoli A. and Zaffalon M.

## Reference

```
@article{schurch2021correlated,
  title={Correlated Product of Experts for Sparse Gaussian Process Regression},
  author={Sch{\"u}rch, Manuel and Azzimonti, Dario and Benavoli, Alessio and Zaffalon, Marco},
  journal={arXiv preprint arXiv:2112.09519},
  year={2021}
}
```