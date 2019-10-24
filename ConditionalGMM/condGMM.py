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
                 fixed_indices = None, fixed_components = None):
        self.weights = weights
        self.means = means
        self.covs = covs
        self.fixed_indices = fixed_indices
        self.fixed_components = fixed_components

