"""Conditional multivariate normal distribution.
"""
import numpy as np
import scipy as sp

class CondMNorm(object):
    """Conditional multivariate normal. Given a joint mean vector and
    joint covariance matrix, precompute everything that can be computed
    in order to solve conditional expressions.

    Args:
        joint_means (`numpy.ndarray`): joint mean of all random variables (RVs)
        joint_cov (`numpy.ndarray`): joint covariance matrix of all RVs
        fixed_indices (array-like): list of indices for the fixed variables

    """
    def __init__(self, joint_means, joint_cov, fixed_indices):
        #Do some error checking
        assert isinstance(joint_means, (list, np.ndarray))
        assert isinstance(joint_cov, (list, np.ndarray))
        joint_means = np.asarray(joint_means)
        joint_cov = np.asarray(joint_cov)
        assert len(joint_means) == len(joint_cov)
        assert len(joint_cov) == len(joint_cov[0])
        assert isinstance(fixed_indices, (list, np.ndarray))
        fixed_indices = np.asarray(fixed_indices, dtype=np.int)
        assert len(fixed_indices) < len(joint_means)
        assert all(fixed_indices > -1)
        assert np.max(fixed_indices) < len(joint_means)
        #No repetition
        assert len(np.unique(fixed_indices)) == len(fixed_indices)
        
        #Save the unconditional properties
        self.joint_means = joint_means
        self.joint_cov = joint_cov
        self.fixed_indices = fixed_indices

        #Save the submatrices
        inds = np.arange(len(joint_means))
        free_indices = np.delete(inds, fixed_indices)
        mu_1 = joint_means[free_indices]
        mu_2 = joint_means[fixed_indices]
        Sigma_11 = joint_cov[free_indices]
        Sigma_11 = Sigma_11[:, free_indices]
        Sigma_12 = joint_cov[free_indices]
        Sigma_12 = Sigma_12[:, fixed_indices]
        Sigma_22 = joint_cov[fixed_indices]
        Sigma_22 = Sigma_22[:, fixed_indices]

        #Compute Sigma_{12} dot Sigma_22
        Sigma12_dot_Sigma22I = np.linalg.solve(Sigma_22, Sigma_12.T).T
        #Compute the conditional covariance
        Sigma_c = Sigma_11 - np.dot(Sigma12_dot_Sigma22I, Sigma_12.T)
        
        #Save everything
        self.mus = {"mu_1": mu_1, "mu_2": mu_2}
        self.Sigmas = {"Sigma_11": Sigma_11, "Sigma_12": Sigma_12,
                       "Sigma_22": Sigma_22, "Sigma_c": Sigma_c,
                       "Sigma12_dot_Sigma22I": Sigma12_dot_Sigma22I}

    def conditional_mean(self, x2 = None):
        """Compute the conditional mean (expectation value)
        of the free variables given the value of the fixed variables.

        Args:
            x2 (float or array-like): values of the fixed variables;
                default is `None`, yielding the unconditional mean

        Returns:
            conditional mean of the free variables (x1)

        """
        
        if x2 is None:
            return self.mus["mu_1"]

        assert isinstance(x2, (list, np.ndarray))
        mu_2 = self.mus["mu_2"]
        Sigs = self.Sigmas["Sigma12_dot_Sigma22I"]
        assert len(x2) == len(mu_2)

        return self.mus["mu_1"] + np.dot(Sigs, (x2 - mu_2))

    def conditional_cov(self):
        """The conditional covariance of the free variables.
        
        Args:
            None

        Returns:
            conditional covariance of the free variables (x_1)
        """
        return np.squeeze(self.Sigmas["Sigma_c"])

    def pdf(x1, x2 = None):
        """Conditional probability distribution function of `x1` 
        conditional on `x2`.

        Args:
            x1 (array-like): free variable
            x2 (array-like): observation of the fixed variable; 
                default is None, equivalent to using `x2 = mu_2`
        
        Returns:
            conditional probability distribution function

        """
        
        assert isinstance(x1, (list, np.ndarray))
        x1 = np.asarray(x1)
        assert len(x1) == len(self.mus["mu_1"])
        
        mu_1 = self.conditional_mean(x2)
        Sigma_1 = self.conditional_cov()

        return sp.stats.multivariate_normal.pdf(x1, mean=mu_1, cov=Sigma_1)

    def logpdf(x1, x2 = None):
        """Natural log of the conditional probability distribution 
        function of `x1` conditional on `x2`.

        Args:
            x1 (array-like): free variable
            x2 (array-like): observation of the fixed variable; 
                default is None, equivalent to using `x2 = mu_2`
        
        Returns:
            log of the conditional probability distribution function

        """

        assert isinstance(x1, (list, np.ndarray))
        x1 = np.asarray(x1)
        assert len(x1) == len(self.mus["mu_1"])
        
        mu_1 = self.conditional_mean(x2)
        Sigma_1 = self.conditional_cov()

        return sp.stats.multivariate_normal.logpdf(x1, mean=mu_1, cov=Sigma_1)

    def rvs(x2 = None, size = 1, random_state = None):
        """Draw random samples from the conditional multivariate
        normal distribution conditioned on `x2`.

        Args:
            x2 (array-like): observation of the fixed variable; 
                default is None, equivalent to using `x2 = mu_2`
            size (int): number of random samples; default is 1
            random_state: state that numpy uses for drawing samples

        Returns:
            random variable distributed according to the conditional PDF

        """
        assert size >= 1

        mu_1 = self.conditional_mean(x2)
        Sigma_1 = self.conditional_cov()

        return sp.stats.multivariate_normal.rvs(mean=mu_1, cov=Sigma_1,
                                                size = size,
                                                random_state = random_state)
