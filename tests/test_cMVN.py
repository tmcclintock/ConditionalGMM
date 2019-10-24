"""Tests of the basic conditional multivariate normal
"""
import ConditionalGMM as cGMM
import numpy as np
import numpy.testing as npt

def test_cMN_basic():
    means = [0.5, -0.2]
    cov = [[2.0, 0.3], [0.3, 0.5]]
    ind = [0]
    #Smoke tests
    cMN = cGMM.MNorm.CondMNorm(means, cov, ind)
    
#def test_cMN_exceptions():
    
