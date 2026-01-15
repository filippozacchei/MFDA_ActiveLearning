import numpy as np
from gp_active_mcmc.utils import in_box

def test_in_box():
    low = np.array([0.0, 0.0])
    high = np.array([1.0, 2.0])

    assert in_box(np.array([0.0, 0.0]), low, high)
    assert in_box(np.array([1.0, 2.0]), low, high)
    assert not in_box(np.array([-1.0, 0.0]), low, high)
    assert not in_box(np.array([0.5, 3.0]), low, high)
