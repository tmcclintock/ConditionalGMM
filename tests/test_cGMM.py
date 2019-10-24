"""Tests of the conditional Gaussian mixture model.
"""
import ConditionalGMM as cgmm
import numpy as np
import numpy.testing as npt
import pytest

def test_cGMM_basic():
    #Smoke tests
    cGMM = cgmm.CondGMM()
