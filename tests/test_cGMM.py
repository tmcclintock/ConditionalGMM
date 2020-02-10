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

def test_conditionals_moments():
    weights = [0.5, 0.5]
    means = np.array([[0.5, -0.2], [0.2, -0.2]])
    covs = np.array([[[2.0, 0.3], [0.3, 0.5]], [[2.0, 0.3], [0.3, 0.5]]])
    fixed_inds = [1]
    cGMM = cgmm.CondGMM(weights, means, covs, fixed_inds)
    mu2 = cGMM.conditional_component_means()
    npt.assert_equal(np.squeeze(mu2), means[:, 0])

    #TODO
    #Cs = cGMM.conditional_component_covs()
    #npt.assert_equal(covs[1], Cs)
    

def test_cGMM_exceptions():
    weights = [1.]
    means = [[0.5, -0.2]]
    covs = [[[2.0, 0.3], [0.3, 0.5]]]
    fixed_inds = [1]
    with pytest.raises(AssertionError):
        cGMM = cgmm.CondGMM(weights, means, covs, 1)

def test_rvs():
    #1 component
    weights = [1.]
    means = [[0.5, -0.2, 1.0]]
    cov = [[[2.0, 0.3, 0.0], [0.3, 0.5, 0.0], [0.0, 0.0, 1.0]]]
    ind = [2]
    cGMM = cgmm.CondGMM(weights, means, cov, ind)
    N = 10000
    x1_realizations = cGMM.rvs([1], size=N, random_state=42)
    npt.assert_equal(x1_realizations.shape, [N, 2])

    #2 components
    weights = [0.2, 0.8]
    means = [[0.5, -0.2, 1.0], [1., -0.1, 1.0]]
    cov = [[[2.0, 0.3, 0.0], [0.3, 0.5, 0.0], [0.0, 0.0, 1.0]],
           [[2.0, 0.3, 0.0], [0.3, 0.5, 0.0], [0.0, 0.0, 1.0]]]
    ind = [2]
    cGMM = cgmm.CondGMM(weights, means, cov, ind)
    N = 10000
    x1_realizations, labels = cGMM.rvs([1], size=N, random_state=42,
                                       component_labels=True)
    npt.assert_equal(x1_realizations.shape, [N, 2])
    npt.assert_equal(len(labels), N)
    npt.assert_equal(len(labels[labels == 0]) > 0, True)
    npt.assert_equal(len(labels[labels == 1]) > 0, True)

    
if __name__ == "__main__":
    test_rvs()
