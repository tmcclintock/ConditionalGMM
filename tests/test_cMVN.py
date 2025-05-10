"""Tests of the basic conditional multivariate normal"""

import numpy.testing as npt
import pytest
import scipy.stats as ss

from ConditionalGMM.MNorm import CondMNorm


def test_cMN_basic():
    # Smoke tests
    # 2D
    means = [0.5, -0.2]
    cov = [[2.0, 0.3], [0.3, 0.5]]
    ind = [0]
    cMN = CondMNorm(means, cov, ind)

    # 3D
    means = [0.5, -0.2, 1.0]
    cov = [[2.0, 0.3, 0.0], [0.3, 0.5, 0.0], [0.0, 0.0, 1.0]]
    inds = [1, 2]
    cMN = CondMNorm(means, cov, inds)


def test_cMN_exceptions():
    # 3D
    means = [0.5, -0.2, 1.0]
    cov = [[2.0, 0.3, 0.0], [0.3, 0.5, 0.0], [0.0, 0.0, 1.0]]
    inds = [1, 2]

    with pytest.raises(AssertionError):
        CondMNorm(means, cov, [-1])
    with pytest.raises(AssertionError):
        CondMNorm(means, cov, [3])
    with pytest.raises(AssertionError):
        CondMNorm(123, cov, [1, 2])
    with pytest.raises(AssertionError):
        CondMNorm(means, 123, [1, 2])
    with pytest.raises(AssertionError):
        CondMNorm(means[:2], cov, [1, 2])
    with pytest.raises(AssertionError):
        CondMNorm(means, cov[:2], [1, 2])
    with pytest.raises(AssertionError):
        CondMNorm(means, cov, 2)
    with pytest.raises(AssertionError):
        CondMNorm(means, cov, 2)
    with pytest.raises(AssertionError):
        CondMNorm(means, cov, [0, 1, 2, 2])
    with pytest.raises(AssertionError):
        CondMNorm(means, cov, [2, 2])


def test_conditional_mean_and_cov():
    # 2D
    means = [0.5, -0.2]
    cov = [[2.0, 0.0], [0.0, 0.5]]
    ind = [1]
    cMN = CondMNorm(means, cov, ind)
    Sigma11 = cMN.conditional_cov()
    npt.assert_array_equal(Sigma11, [2.0])
    mu1 = cMN.conditional_mean([1])
    npt.assert_equal(means[:1], mu1)

    # 3D
    means = [0.5, -0.2, 1.0]
    cov = [[2.0, 0.3, 0.0], [0.3, 0.5, 0.0], [0.0, 0.0, 1.0]]
    ind = [2]
    cMN = CondMNorm(means, cov, ind)
    Sigma11 = cMN.conditional_cov()
    npt.assert_array_equal(Sigma11, [[2.0, 0.3], [0.3, 0.5]])
    mu1 = cMN.conditional_mean([1])
    npt.assert_equal(means[:2], mu1)
    mu1 = cMN.conditional_mean()
    npt.assert_equal(means[:2], mu1)


def test_conditional_probs():
    means = [0.5, -0.2, 1.0]
    cov = [[2.0, 0.3, 0.0], [0.3, 0.5, 0.0], [0.0, 0.0, 1.0]]
    ind = [2]
    cMN = CondMNorm(means, cov, ind)
    mu1 = cMN.conditional_mean([1])
    Sigma11 = cMN.conditional_cov()

    means2 = [0.5, -0.2]
    cov2 = [[2.0, 0.3], [0.3, 0.5]]

    npt.assert_equal(
        ss.multivariate_normal.pdf(means2, mean=mu1, cov=Sigma11),
        ss.multivariate_normal.pdf(means2, mean=means2, cov=cov2),
    )
    npt.assert_equal(
        ss.multivariate_normal.logpdf(means2, mean=mu1, cov=Sigma11),
        ss.multivariate_normal.logpdf(means2, mean=means2, cov=cov2),
    )


def test_rvs():
    means = [0.5, -0.2, 1.0]
    cov = [[2.0, 0.3, 0.0], [0.3, 0.5, 0.0], [0.0, 0.0, 1.0]]
    ind = [2]
    cMN = CondMNorm(means, cov, ind)
    N = 10000
    x1_realizations = cMN.rvs([1], size=N, random_state=42)

    npt.assert_equal(x1_realizations.shape, [N, 2])


def test_jointpdfs():
    means = [0.5, -0.2, 1.0]
    cov = [[2.0, 0.3, 0.0], [0.3, 0.5, 0.0], [0.0, 0.0, 1.0]]
    ind = [2]
    cMN = CondMNorm(means, cov, ind)
    x = [0.2, 0.0, 0.0]
    x1 = x[:2]
    x2 = x[2:]
    pdf = cMN.joint_pdf(x1, x2)
    logpdf = cMN.joint_logpdf(x1, x2)

    npt.assert_equal(ss.multivariate_normal.pdf(x, means, cov), pdf)
    npt.assert_equal(ss.multivariate_normal.logpdf(x, means, cov), logpdf)

    logpdf = cMN.joint_logpdf(x1)
    npt.assert_equal(cMN.joint_logpdf(x1, means[2:]), logpdf)


def test_pdf():
    means = [0.5, -0.2, 1.0]
    cov = [[2.0, 0.3, 0.0], [0.3, 0.5, 0.0], [0.0, 0.0, 1.0]]
    ind = [2]
    cMN = CondMNorm(means, cov, ind)
    x = [0.2, 0.0, 0.0]
    x1 = x[:2]
    x2 = x[2:]
    mu1 = cMN.conditional_mean(x2)
    Sigma1 = cMN.conditional_cov()
    pdf = cMN.pdf(x1, x2)
    npt.assert_equal(pdf, ss.multivariate_normal.pdf(x1, mean=mu1, cov=Sigma1))
    logpdf = cMN.logpdf(x1, x2)
    npt.assert_equal(logpdf, ss.multivariate_normal.logpdf(x1, mean=mu1, cov=Sigma1))


def test_conditional_mean():
    means = [0.5, -0.2, 1.0]
    cov = [[2.0, 0.3, 0.1], [0.3, 0.5, 0.1], [0.1, 0.1, 1.0]]
    ind = [2]
    cMN = CondMNorm(means, cov, ind)
    x2 = [1.0]
    cmean = cMN.conditional_mean(x2)
    npt.assert_array_equal(means[:2], cmean)
