"""Tests of the conditional Gaussian mixture model.
"""
import ConditionalGMM as cgmm
import numpy as np
import numpy.testing as npt
import pytest

def test_cGMM_basic():
    #Smoke tests
    weights = [1]
    means = [[0.5, -0.2]]
    covs = [[2.0, 0.3], [0.3, 0.5]]
    cGMM = cgmm.CondGMM(weights, means, covs)
