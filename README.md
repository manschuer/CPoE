## Stochastic Recursive Gaussian Process Regression

Implements the "Correlated Product of Experts" (CPoE) Regression algorithm from the paper "Correlated Product of Experts for Sparse Gaussian Process Regression" by Manuel Schuerch, Dario Azzimonti, Alessio Benavoli and Marco Zaffalon.


## Usage

We provide the code for CPoE and an example to demonstrate the usage.
The main code is in CPoE.py and the example in the jupyter notebook example.ipynb.
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
```# CPoE
