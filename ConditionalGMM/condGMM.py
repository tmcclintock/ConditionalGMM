"""Conditional Gaussian mixture model.
"""
import numpy as np
import scipy as sp
from .MNorm import *

class CondGMM(object):
    """Conditional Gaussian mixture model. Built from a collection of
    conditional multivariate normal (CondMNorm) distributions weighted
    to be properly normalized.

    Args:
        weights (`numpy.ndarray`): N component weights that sum to 1
        means (`numpy.ndarray`): NxM mean vectors of the component Gaussians
        covs (`numpy.ndarray`): NxMxM list of covariance of the components
        fixed_indices (array-like): list of indices for the fixed variables;
            default is `None`
        fixed_components (array-like): list of indices for the fixed components;
            default is `None`

    """
    def __init__(self, weights, means, covs,
                 fixed_indices, fixed_components = None):
        assert isinstance(weights, (list, np.ndarray))
        assert isinstance(means, (list, np.ndarray))
        assert isinstance(covs, (list, np.ndarray))
        assert isinstance(fixed_indices, (list, np.ndarray))
        weights = np.asarray(weights)
        means = np.asarray(means)
        covs = np.asarray(covs)
        fixed_indices = np.asarray(fixed_indices, dtype=np.int)
        assert weights.ndim == 1
        assert means.ndim == 2
        assert covs.ndim == 3
        assert fixed_indices.ndim == 1
        assert len(weights) == len(means)
        assert len(weights) == len(covs)
        np.testing.assert_almost_equal(weights.sum(), 1.)
        
        self.weights = weights
        self.means = means
        self.covs = covs
        self.fixed_indices = fixed_indices
        self.fixed_components = fixed_components

        #Create conditional multivarate normal distributions
        cMVNs = [CondMNorm(means[i], covs[i], fixed_indices)
                 for i in range(len(means))]
        self.conditionalMVNs = cMVNs

    def conditional_component_means(self, x2 = None):
        """Compute the conditional mean (expectation value)
        of the free variables given the value of the fixed variables
        for each component in the mixture model.

        Args:
            x2 (float or array-like): values of the fixed variables;
                default is `None`, yielding the unconditional means

        Returns:
            conditional mean of the free variables (x1) for each component

        """
        return np.array([d.conditional_mean(x2) for d in self.conditionalMVNs])

    def conditional_component_covs(self):
        """The conditional covariance of the free variables for
        each component in the mixture model.
        
        Args:
            None

        Returns:
            conditional covariance of the free variables (x1) for each component

        """
        return np.array([d.conditional_cov() for d in self.conditionalMVNs])

    def conditional_weights(self, x2 = None):
        pass
        
    def conditional_mean(self, x2 = None):
        pass

    def pdf(self, x1, x2 = None):
        pass

    def logpdf(self, x1, x2 = None):
        pass

    def rvs(self, x2, size = 1, random_state = None, component_labels = False):
        """Draw random samples from the conditional GMM
        conditioned on `x2`.

        Args:
            x2 (array-like): observation of the fixed variable
            size (int): number of random samples; default is 1
            random_state (int): state that numpy uses for drawing samples
            component_labels (bool): if `True`, return the label for which
                component each RV was drawn from

        Returns:
            random variable distributed according to the conditional GMM

        """
        assert size >= 1
        pass
