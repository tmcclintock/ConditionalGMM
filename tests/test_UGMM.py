"""Tests of the univariate GMM
"""
from ConditionalGMM import UnivariateGMM
import numpy as np
import numpy.testing as npt
import pytest

def test_ugmm_basic():
    #Smoke tests
    weights = [0.5, 0.5]
    means = [0., 1.]
    vars = [1., 2.]
    ugmm = UnivariateGMM.UniGMM(weights, means, vars)

def test_statistical_moments():
    weights = [0.5, 0.5]
    means = [0., 0.]
    vars = [1., 2.]
    ugmm = UnivariateGMM.UniGMM(weights, means, vars)

    npt.assert_equal(ugmm.mean(), 0)
    npt.assert_equal(ugmm.median(), 0)

    means = [2., 2.]
    vars = [1., 2.]
    ugmm = UnivariateGMM.UniGMM(weights, means, vars)

    npt.assert_equal(ugmm.mean(), 2)
    npt.assert_equal(ugmm.median(), 2)
