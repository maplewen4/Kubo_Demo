"""
Unit and regression test for the kubo_demo_bowen package.
"""

# Import package, test suite, and other packages as needed
import kubo_demo_bowen as trial
import pytest
import sys
import scipy.fftpack as fourier_transform
import numpy as np

def test_kubo_demo_bowen_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "kubo_demo_bowen" in sys.modules

def test_ct():
    """Test if Ct calculation work"""
    sample_time = np.loadtxt('ref.txt', usecols=[0])
    reference_ct = np.loadtxt('ref.txt', usecols=[1])
    
    instance = trial.xixihaha.Kubo(delta=1 , tau=1)

    calculate_ct = instance.calculate_Ct(time=sample_time)[:,1]

    assert np.allclose(calculate_ct , reference_ct)
