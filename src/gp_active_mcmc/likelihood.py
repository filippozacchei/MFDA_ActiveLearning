import numpy as np
from tinyDA import AdaptiveGaussianLogLike


class GaussianLogLikeWithGP(AdaptiveGaussianLogLike):
    """
    Gaussian likelihood where GP predictive variance is added
    to observational noise covariance.
    """

    def __init__(self, data: np.ndarray, covariance: np.ndarray):
        super().__init__(data, covariance)

    def loglike(self, y_pred: np.ndarray):
        y_pred = np.atleast_1d(y_pred)

        self.total_cov = self.cov + self.cov_bias

        if hasattr(y_pred, "variance") and y_pred.variance is not None:
            variance = np.atleast_1d(y_pred.variance)
            self.total_cov += np.diag(variance)

        self.cov_inverse = np.linalg.inv(self.total_cov)

        return super().loglike(y_pred)
