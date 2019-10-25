"""Tests of the conditional Gaussian mixture model.
"""
import ConditionalGMM as cgmm
import numpy as np
import numpy.testing as npt
import pytest

def test_cGMM_basic():
    #Smoke tests
    #2d - 1 component
    weights = [1.]
    means = [[0.5, -0.2]]
    covs = [[[2.0, 0.3], [0.3, 0.5]]]
    fixed_inds = [1]
    cGMM = cgmm.CondGMM(weights, means, covs, fixed_inds)

    #2d - 2 component
    weights = [0.5, 0.5]
    means = [[0.5, -0.2], [0.2, -0.2]]
    covs = [[[2.0, 0.3], [0.3, 0.5]], [[2.0, 0.3], [0.3, 0.5]]]
    fixed_inds = [1]
    cGMM = cgmm.CondGMM(weights, means, covs, fixed_inds)

def test_cGMM_exceptions():
    weights = [1.]
    means = [[0.5, -0.2]]
    covs = [[[2.0, 0.3], [0.3, 0.5]]]
    fixed_inds = [1]
    with pytest.raises(AssertionError):
        cGMM = cgmm.CondGMM(weights, means, covs, 1)
