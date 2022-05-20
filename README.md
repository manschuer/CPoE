## Correlated Product of Experts for Sparse Gaussian Process Regression

Implements the "Correlated Product of Experts" (CPoE) regression algorithm from the paper "Correlated Product of Experts for Sparse Gaussian Process Regression" by Manuel Schuerch, Dario Azzimonti, Alessio Benavoli and Marco Zaffalon.


## Usage

We provide the source code for CPoE, comparisons and examples to demonstrate the usage.
The main code is in source/CPoE.py and the jupyter notebooks experiments/example_comparison_1D.ipynb and experiments/example_comparison_1D.ipynb show the usage and comparisons to other methods.
In order to run the algorithm, you need [GPy](https://github.com/SheffieldML/GPy) (tested up to version 1.9.6) since we use their implementation of the kernels.
Currently, we are working on scalable implementation in Tensorflow/GPflow.


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