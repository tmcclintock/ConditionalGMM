# ConditionalGMM [![Build Status](https://travis-ci.com/tmcclintock/ConditionalGMM.svg?branch=master)](https://travis-ci.com/tmcclintock/ConditionalGMM)

This repository contains helpful functions for computing conditional distributions of multivariate normal distributions and Gaussian mixture models. It is able to return not only the conditional means (expectation values) but also the conditional covariances and conditional component weights (probabilities).

Note that this package is NOT intended for use in training GMMs. There are much better tools out there for this purpose such as those in [`scikit-learn`](https://scikit-learn.org/stable/modules/mixture.html).

## Installation

Clone the repository

`git clone https://github.com/tmcclintock/ConditionalGMM`

install the requirements using either `conda`:

`conda install --file requirements.txt`

or with `pip`:

`pip install -r requirements.txt`

and then install this package

`python setup.py install`