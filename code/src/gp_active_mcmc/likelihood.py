import numpy as np
from tinyDA import AdaptiveGaussianLogLike


class GaussianLogLikeWithGP(AdaptiveGaussianLogLike):
    """
    Gaussian likelihood where GP variance is added to observational noise.
    """

    def __init__(self, data, covariance):
        super().__init__(data, covariance)

    def loglike(self, y_pred: np.ndarray):
        if hasattr(y_pred, "variance") and y_pred.variance is not None:
            self.total_cov = self.cov + self.cov_bias + np.diag(y_pred.variance)
        else:
            self.total_cov = self.cov + self.cov_bias

        self.cov_inverse = np.diag(1.0 / np.diag(self.total_cov))

        return super().loglike(y_pred)
