"""Tests of the univariate GMM"""

import numpy as np
import numpy.testing as npt
import scipy as sp

from ConditionalGMM import UnivariateGMM


def test_ugmm_basic():
    # Smoke tests
    weights = [0.5, 0.5]
    means = [0.0, 1.0]
    vars = [1.0, 2.0]
    ugmm = UnivariateGMM.UniGMM(weights, means, vars)
    ugmm.logcdf(0.5)  # smoke test for this function, for now


def test_uggm_pdf():
    weights = [0.5, 0.5]
    means = [0.0, 1.0]
    vars = [1.0, 2.0]
    ugmm = UnivariateGMM.UniGMM(weights, means, vars)
    for x in np.linspace(-2, 4):
        pdf = ugmm.pdf(x)
        truepdf = np.dot(
            np.array(weights),
            np.array([sp.stats.norm.pdf(x, mi, vi) for mi, vi in zip(means, vars)]),
        )
        npt.assert_equal(pdf, truepdf)

        logpdf = ugmm.logpdf(x)
        truelogpdf = np.log(truepdf)
        npt.assert_equal(logpdf, truelogpdf)


def test_statistical_moments():
    weights = [0.5, 0.5]
    means = [0.0, 0.0]
    vars = [1.0, 2.0]
    ugmm = UnivariateGMM.UniGMM(weights, means, vars)

    npt.assert_equal(ugmm.mean(), 0)
    npt.assert_equal(ugmm.median(), 0)

    means = [2.0, 2.0]
    vars = [1.0, 2.0]
    ugmm = UnivariateGMM.UniGMM(weights, means, vars)

    npt.assert_equal(ugmm.mean(), 2)
    npt.assert_equal(ugmm.median(), 2)
