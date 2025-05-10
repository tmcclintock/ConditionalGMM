"""Helpful functions that can only be computed (easily) for univarate GMMs."""

import numpy as np
import scipy as sp


class UniGMM(object):
    """Conditional Gaussian mixture model of a single random variable.
    This class exists to provide helpful routines, such as computing CDFs
    and PPFs of the GMM.

    Args:
        weights (`numpy.ndarray`): N component weights that sum to 1
        means (`numpy.ndarray`): N mean values of the component Gaussians
        variances (`numpy.ndarray`): N list of variances of the components

    """

    def __init__(self, weights, means, variances):
        assert isinstance(weights, (list, np.ndarray))
        assert isinstance(means, (list, np.ndarray))
        assert isinstance(variances, (list, np.ndarray))
        weights = np.atleast_1d(weights)
        means = np.atleast_1d(means)
        variances = np.atleast_1d(variances)
        assert weights.ndim == 1
        assert means.ndim == 1
        assert variances.ndim == 1
        assert len(weights) == len(means)
        assert len(weights) == len(variances)
        np.testing.assert_almost_equal(weights.sum(), 1.0)

        self.weights = weights
        self.means = means
        self.variances = variances
        self.stddevs = np.sqrt(self.variances)

    def pdf(self, x):
        """Probability density function of `x`.

        Args:
            x (float or array-like): random variable

        Returns:
            PDF

        """
        x = np.atleast_1d(x)
        assert np.ndim(x) < 2
        # TODO vectorize
        pdfs = np.array(
            [sp.stats.norm.pdf(x, mi, vi) for mi, vi in zip(self.means, self.variances)]
        )
        return np.dot(self.weights, pdfs)

    def logpdf(self, x):
        """Probability density function of `x`.

        Args:
            x (float or array-like): random variable

        Returns:
            log PDF

        """
        return np.log(self.pdf(x))

    def cdf(self, x):
        """Cumulative probability density function of `x`.

        Args:
            x (float or array-like): random variable

        Returns:
            CDF

        """
        cdfs = np.array(
            [sp.stats.norm.cdf(x, mi, vi) for mi, vi in zip(self.means, self.variances)]
        )
        return np.dot(self.weights, cdfs)

    def logcdf(self, x):
        """Log cumulative probability density function of `x`.

        Args:
            x (float or array-like): random variable

        Returns:
            log CDF

        """
        return np.log(self.cdf(x))

    def ppf(self, q):
        """Percent point function of quantile `q` (inverse `cdf`)
        of the RV.

        Args:
            q (float or array-like): lower tail probability

        Returns:
            (float or array-like) quantile corresponding to `q`

        """
        f = lambda x: self.cdf(x) - q
        return sp.optimize.newton(func=f, x0=self.mean(), fprime=self.pdf)

    def mean(self):
        """Mean of the RV for the GMM.

        Args:
            None

        Returns:
            mean of the random variable

        """
        return np.sum(self.weights * self.means)

    def median(self):
        """Median of the RV for the GMM.

        Args:
            None

        Returns:
            median of the random variable

        """
        return self.ppf(0.5)
