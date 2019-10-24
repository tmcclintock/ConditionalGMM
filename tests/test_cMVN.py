"""Tests of the basic conditional multivariate normal
"""
import ConditionalGMM as cGMM
import numpy as np
import numpy.testing as npt
import pytest

def test_cMN_basic():
    #Smoke tests
    #2D
    means = [0.5, -0.2]
    cov = [[2.0, 0.3], [0.3, 0.5]]
    ind = [0]
    cMN = cGMM.MNorm.CondMNorm(means, cov, ind)

    #3D
    means = [0.5, -0.2, 1.0]
    cov = [[2.0, 0.3, 0.0], [0.3, 0.5, 0.0], [0.0, 0.0, 1.0]]
    inds = [1, 2]
    cMN = cGMM.MNorm.CondMNorm(means, cov, inds)
    
def test_cMN_exceptions():
    #3D
    means = [0.5, -0.2, 1.0]
    cov = [[2.0, 0.3, 0.0], [0.3, 0.5, 0.0], [0.0, 0.0, 1.0]]
    inds = [1, 2]

    with pytest.raises(AssertionError):
        cGMM.MNorm.CondMNorm(means, cov, [-1])
    with pytest.raises(AssertionError):
        cGMM.MNorm.CondMNorm(means, cov, [3])
    with pytest.raises(AssertionError):
        cGMM.MNorm.CondMNorm(123, cov, [1, 2])
    with pytest.raises(AssertionError):
        cGMM.MNorm.CondMNorm(means, 123, [1, 2])
    with pytest.raises(AssertionError):
        cGMM.MNorm.CondMNorm(means[:2], cov, [1, 2])
    with pytest.raises(AssertionError):
        cGMM.MNorm.CondMNorm(means, cov[:2], [1, 2])
    with pytest.raises(AssertionError):
        cGMM.MNorm.CondMNorm(means, cov, 2)
    with pytest.raises(AssertionError):
        cGMM.MNorm.CondMNorm(means, cov, 2)
    with pytest.raises(AssertionError):
        cGMM.MNorm.CondMNorm(means, cov, [0, 1, 2, 2])
    with pytest.raises(AssertionError):
        cGMM.MNorm.CondMNorm(means, cov, [2, 2])
