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
        fixed_indices (array-like): list of indices for the fixed variabces
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
