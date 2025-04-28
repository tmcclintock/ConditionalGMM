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
        fixed_indices = np.asarray(fixed_indices, dtype=np.int64)
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

        self.x2_ndim = len(self.fixed_indices)
        self.x1_ndim = len(self.means[0]) - self.x2_ndim
        
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

    def unconditional_pdf_x2(self, x2 = None, component_probs = False):
        """The unconditional probability of the fixed variable (x2), f(x2)
        This is required in order to properly normalize the conditional
        probability of x1, f(x1 | x2).

        Args:
            x2 (float or array-like): values of the fixed variables;
                default is `None`, yielding the unconditional means
            component_probs (bool): if True then return the probabilities of
                x2 having been drawn from each component, otherwise
                return the sum of these probabilities

        Returns:
            unconditional probability of x2

        """
        w = self.weights
        dists = self.conditionalMVNs
        mus = np.array([d._mu_2() for d in dists])
        covs = np.array([d._Sigma_22() for d in dists])

        probs = w*np.array([sp.stats.multivariate_normal.pdf(x2, mean=mus[i], cov=covs[i])
                                       for i in range(len(w))])
        if component_probs:
            return probs
        else:
            return probs.sum()

    def unconditional_logpdf_x2(self, x2 = None):
        """The unconditional log-probability of the fixed variable (x2).

        Args:
            x2 (float or array-like): values of the fixed variables;
                default is `None`, yielding the unconditional means

        Returns:
            unconditional log-probability of x2, ln(f(x2))

        """
        return np.log(self.unconditional_pdf_x2(x2))

    def unconditional_x2_mean(self):
        """Unconditional mean (considering all components) of x2.

        Args:
            None

        Returns:
            unconditional mean of x2 (E[x2])

        """
        mu2s = np.array([d._mu_2() for d in self.conditionalMVNs])
        return self.weights * mu2s 
    
    def conditional_weights(self, x2 = None):
        """Conditional weights (pi_i|x2) of each component conditioned on
        the observation of x2.

        Args:
            x2 (float or array-like): values of the fixed variables;
                default is `None`, yielding the unconditional means

        Returns:
            conditional component weights (pi_i|x2)

        """
        probs = self.unconditional_pdf_x2(x2, True)
        return probs / probs.sum()
        
    def conditional_mean(self, x2 = None):
        """Conditional mean of x1 (E[x1|x2]).

        Args:
            x2 (float or array-like): values of the fixed variables;
                default is `None`, which uses the unconditional
                mean of x2 over all components

        Returns:
            conditional mean of x1 conditioned on x2

        """
        if x2 is None:
            x2 = self.unconditional_x2_mean()
        c_weights = self.conditional_weights(x2)
        mus = np.array([d.conditional_mean(x2) for d in self.conditionalMVNs])
        return np.sum(c_weights * mus, axis = 0)
        
    def conditional_median(self, x2 = None):
        pass

    def conditional_pdf(self, x1, x2 = None):
        """The conditional probability of x1.

        Args:
            x1 (array-like): free variable
            x2 (float or array-like): values of the fixed variables;
                default is `None`, which uses the unconditional mean of x2

        Returns:
            conditional probability of x1, f(x1|x2)

        """
        return np.exp(self.conditional_logpdf(x1, x2))

    def conditional_logpdf(self, x1, x2 = None):
        """The conditional log probability of x1.

        Args:
            x1 (array-like): free variable
            x2 (float or array-like): values of the fixed variables;
                default is `None`, which uses the unconditional mean of x2

        Returns:
            conditional log probability of x1, f(x1|x2)

        """
        f_x2 = self.unconditional_pdf_x2(x2)
        return self.joint_logpdf(x1, x2) - np.log(f_x2)

    def joint_logpdf(self, x1, x2 = None):
        """The joint log probability of (x1, x2)

        Args:
            x1 (array-like): free variable
            x2 (float or array-like): values of the fixed variables;
                default is `None`, which uses the unconditional mean of x2

        Returns:
            log joint probability of (x1, x2)

        """
        dists = self.conditionalMVNs
        joint_pdfs = np.array([d.joint_pdf(x1, x2) for d in dists])
        return np.log(np.sum(self.weights * joint_pdfs))

    def joint_pdf(self, x1, x2 = None):
        """The joint probability of (x1, x2)

        Args:
            x1 (array-like): free variable
            x2 (float or array-like): values of the fixed variables;
                default is `None`, which uses the unconditional mean of x2

        Returns:
            log joint probability of (x1, x2)

        """
        return np.exp(self.joint_logpdf(x1, x2))
    
    def rvs(self, x2, size = 1, random_state = None, component_labels = False):
        """Draw random samples from the conditional GMM
        conditioned on `x2`.

        Args:
            x2 (array-like): observation of the fixed variable;
                default is `None`, which uses the unconditional
                mean of x2 over all components
            size (int): number of random samples; default is 1
            random_state (int): state that numpy uses for drawing samples
            component_labels (bool): if `True`, also return the label for 
                which component each RV was drawn from

        Returns:
            random variable distributed according to the conditional GMM

        """
        assert size >= 1

        if x2 is None:
            x2 = self.unconditional_x2_mean()

        if random_state is not None:
            np.random.seed(random_state)
        
        #Output array
        rvs = np.zeros((size, self.x1_ndim))
        #rvs = np.squeeze(rvs)

        #Choose which components the data come from
        c_weights = self.conditional_weights(x2)
        inds = np.arange(len(c_weights))
        components = np.random.choice(inds, size = size, p = c_weights)
        #_, counts = np.unique(components, return_counts = True)

        #Get RVs from each component
        dists = self.conditionalMVNs
        for i in inds:
            n = len(components[components == i])
            if n == 0: #Skip if no draws
                continue
            rvs_i = np.atleast_2d(dists[i].rvs(x2 = x2, size = n))
            rvs[i == components] = rvs_i

        if component_labels:
            return rvs, components
        else:
            return rvs
