from __future__ import annotations

import numpy as np
import pytest

from gp_active_mcmc.utils.metrics import coverage, rmse


def test_rmse() -> None:
    y_true = np.array([0.0, 1.0, 2.0])
    y_hat = np.array([0.0, 2.0, 2.0])
    assert rmse(y_hat, y_true) == pytest.approx(np.sqrt(1.0 / 3.0))

    with pytest.raises(ValueError):
        _ = rmse([1.0], [1.0, 2.0])


def test_coverage() -> None:
    y_true = np.array([0.0, 1.0])
    y_hat = np.array([0.0, 1.0])
    y_std = np.array([0.1, 0.1])
    assert coverage(y_true, y_hat, y_std, z=0.0) == pytest.approx(1.0)

    with pytest.raises(ValueError):
        _ = coverage(y_true, y_hat, y_std, z=-1.0)
