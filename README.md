# ConditionalGMM [![Build Status](https://travis-ci.com/tmcclintock/ConditionalGMM.svg?branch=master)](https://travis-ci.com/tmcclintock/ConditionalGMM)

This repository contains helpful functions for computing conditional distributions of multivariate normal distributions and Gaussian mixture models (GMMs). It is able to return not only the conditional means (expectation values) but also the conditional covariances and conditional component weights (probabilities).

Note that this package is NOT intended for use in training GMMs. There are much better tools out there for this purpose such as those in [`scikit-learn`](https://scikit-learn.org/stable/modules/mixture.html).

## Installation

Clone the repository

`git clone https://github.com/tmcclintock/ConditionalGMM`

install the requirements (only `numpy` and `scipy`) using either `conda`:

`conda install --file requirements.txt`

or `pip`:

`pip install -r requirements.txt`

Install this package

`python setup.py install`

Run the tests with `pytest`.

## Usage

Suppose you had some data described by some number of Gaussians:
![alt text](https://github.com/tmcclintock/ConditionalGMM/blob/master/notebooks/figures/scatter_contour_2comps.png "2 component data")

Once you have a GMM describing the data (the lines) you can create a conditional GMM object:
```python
import ConditionalGMM

cGMM = ConditionalGMM.CondGMM(weights, means, covs, fixed_indices)
```
where `fixed_indices` is an array of the dimensions that you will take conditionals on. In this example, we will look at `y` conditional on `x`, so we would have `fixed_indices = [0]`.

Given some observations of `x` with this package you can quickly compute conditional probability distributions:
```python
y = np.linspace(-12, 0, 200)
x_obs = np.array([-1, 4, 7])
for x in x_obs:
    y_cpdf = np.array([cGMM.conditional_pdf([yi], x) for yi in y])
```
![alt text](https://github.com/tmcclintock/ConditionalGMM/blob/master/notebooks/figures/cPDF_2comps.png "conditional PDF")

This package also lets you draw random values (RVs) from the conditional PDF. For instance, here are ten thousand draws from each of the conditional PDFs shown above:
```python
x_obs = np.array([-1, 4, 7])
bins = np.arange(-12, 0, 0.5)
N = 100000
for x in x_obs:
    y_rvs = cGMM.rvs(x, size=N)
```
![alt text](https://github.com/tmcclintock/ConditionalGMM/blob/master/notebooks/figures/hist_2comps.png "conditional RVs")
