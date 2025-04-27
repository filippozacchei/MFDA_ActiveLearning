import numpy as np
from scipy.optimize import minimize

class RBFGaussianProcess:
    def __init__(self, scale=1, reg=0):
        self.scale = scale 
        self.reg = reg
        self.k_xx_inv = None

    def rbf_kernel_incr_inv(self, B, C, D):
        temp = np.linalg.inv(D - C @ self.k_xx_inv @ B)
        block1 = self.k_xx_inv + self.k_xx_inv @ B @ temp @ C @ self.k_xx_inv
        block2 = - self.k_xx_inv @ B @ temp
        block3 = - temp @ C @ self.k_xx_inv
        block4 = temp
        res1 = np.concatenate((block1, block2), axis=1)
        res2 = np.concatenate((block3, block4), axis=1)
        res = np.concatenate((res1, res2), axis=0)
        return res

    def rbf_kernel(self, a, b):
        a = np.atleast_2d(a)
        b = np.atleast_2d(b)
        dists = np.sum((a[:, None, :] - b[None, :, :])**2, axis=2)
        return np.exp(-self.scale * dists)
    
    def fit(self, x=np.array([]), y=np.array([])):
        x, y = np.atleast_2d(x), np.atleast_2d(y)
        if self.k_xx_inv is None:
            self.x = x
            self.y = y
            K_xx = self.rbf_kernel(x, x) + self.reg * np.eye(x.shape[0])
            self.k_xx_inv = np.linalg.inv(K_xx)
        else:
            B = self.rbf_kernel(self.x, x)
            C = B.T
            D = self.rbf_kernel(x, x) + self.reg * np.eye(x.shape[0])
            self.k_xx_inv = self.rbf_kernel_incr_inv(B, C, D)
            self.x = np.vstack((self.x, x))
            self.y = np.vstack((self.y, y))
        return self

    def predict(self, x_predict):
        k = self.rbf_kernel(x_predict, self.x)

        mu_hat = k @ self.k_xx_inv @ self.y
        sigma_hat = 1 - np.diag(k @ self.k_xx_inv @ k.T)

        return mu_hat.ravel(), sigma_hat.ravel()
    
class ARDRBFGaussianProcess:
    def __init__(self, reg=1e-2, init_scales=None):
        self.reg = reg
        self.scales = init_scales
        self.k_xx_inv = None
        self.x = None
        self.y = None

    def rbf_kernel(self, a, b, scales=None):
        a = np.atleast_2d(a)
        b = np.atleast_2d(b)
        if scales is None:
            scales = self.scales
        scales = np.atleast_1d(scales).reshape(1, 1, -1)
        diff = a[:, None, :] - b[None, :, :]  # (N, M, d)
        dists = np.sum(scales * diff**2, axis=2)
        return np.exp(-dists)

    def rbf_kernel_incr_inv(self, B, C, D):
        """
        Incremental update to the inverse of kernel matrix using blockwise inversion (Woodbury identity).
        """
        temp = np.linalg.inv(D - C @ self.k_xx_inv @ B)
        block1 = self.k_xx_inv + self.k_xx_inv @ B @ temp @ C @ self.k_xx_inv
        block2 = -self.k_xx_inv @ B @ temp
        block3 = -temp @ C @ self.k_xx_inv
        block4 = temp
        res1 = np.concatenate((block1, block2), axis=1)
        res2 = np.concatenate((block3, block4), axis=1)
        return np.concatenate((res1, res2), axis=0)

    def neg_log_marginal_likelihood(self, log_scales):
        scales = np.exp(log_scales)
        K = self.rbf_kernel(self.x, self.x, scales) + self.reg * np.eye(self.x.shape[0])
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
            nlml = (
                0.5 * self.y.T @ alpha +
                np.sum(np.log(np.diag(L))) +
                0.5 * self.x.shape[0] * np.log(2 * np.pi)
            )
            return nlml.ravel()[0]
        except np.linalg.LinAlgError:
            return 1e6  # Penalize bad configurations

    def fit(self, x, y, optimize_scales=True):
        x, y = np.atleast_2d(x), np.atleast_2d(y)

        if self.scales is None:
            self.scales = np.ones(x.shape[1])

        if self.k_xx_inv is None:
            # First fit — store and optimize if needed
            self.x = x
            self.y = y

            if optimize_scales:
                res = minimize(self.neg_log_marginal_likelihood,
                               x0=np.log(self.scales),
                               method='L-BFGS-B',
                               bounds=[(-5, 5)] * x.shape[1])
                self.scales = np.exp(res.x)

            K = self.rbf_kernel(self.x, self.x) + self.reg * np.eye(self.x.shape[0])
            self.k_xx_inv = np.linalg.inv(K)

        else:
            # Incremental fit — no optimization
            B = self.rbf_kernel(self.x, x)
            C = B.T
            D = self.rbf_kernel(x, x) + self.reg * np.eye(x.shape[0])

            self.k_xx_inv = self.rbf_kernel_incr_inv(B, C, D)
            self.x = np.vstack((self.x, x))
            self.y = np.vstack((self.y, y))

        return self

    def predict(self, x_predict):
        x_predict = np.atleast_2d(x_predict)
        k = self.rbf_kernel(x_predict, self.x)
        mu_hat = k @ self.k_xx_inv @ self.y
        sigma_hat = 1 - np.sum(k @ self.k_xx_inv * k, axis=1)
        return mu_hat.ravel(), sigma_hat.ravel()
    


class ARDMatern52GaussianProcess:
    def __init__(self, reg=1e-2, init_scales=None):
        self.reg = reg
        self.scales = init_scales
        self.k_xx_inv = None
        self.x = None
        self.y = None

    def matern52_kernel(self, a, b, scales=None):
        a = np.atleast_2d(a)
        b = np.atleast_2d(b)
        if scales is None:
            scales = self.scales
        scales = np.atleast_1d(scales).reshape(1, 1, -1)

        diff = a[:, None, :] - b[None, :, :]
        dists_sq = np.sum(scales * diff**2, axis=2)
        dists = np.sqrt(dists_sq + 1e-12)

        sqrt5_d = np.sqrt(5.0 * dists_sq)
        term = 1 + sqrt5_d + (5.0 / 3.0) * dists_sq
        return term * np.exp(-np.sqrt(5.0) * dists)

    def matern52_kernel_incr_inv(self, B, C, D):
        S = D - C @ self.k_xx_inv @ B
        temp = np.linalg.inv(S)

        block1 = self.k_xx_inv + self.k_xx_inv @ B @ temp @ C @ self.k_xx_inv
        block2 = -self.k_xx_inv @ B @ temp
        block3 = -temp @ C @ self.k_xx_inv
        block4 = temp

        top = np.concatenate((block1, block2), axis=1)
        bottom = np.concatenate((block3, block4), axis=1)
        return np.concatenate((top, bottom), axis=0)

    def neg_log_marginal_likelihood(self, log_scales):
        scales = np.exp(log_scales)
        K = self.matern52_kernel(self.x, self.x, scales) + self.reg * np.eye(self.x.shape[0])
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
            nlml = (
                0.5 * self.y.T @ alpha +
                np.sum(np.log(np.diag(L))) +
                0.5 * self.x.shape[0] * np.log(2 * np.pi)
            )
            return nlml.ravel()[0]
        except np.linalg.LinAlgError:
            return 1e6

    def fit(self, x, y, optimize_scales=True):
        x, y = np.atleast_2d(x), np.atleast_2d(y)

        if self.scales is None:
            self.scales = np.ones(x.shape[1])

        if self.k_xx_inv is None:
            # First training step
            self.x = x
            self.y = y

            if optimize_scales:
                res = minimize(self.neg_log_marginal_likelihood,
                               x0=np.log(self.scales),
                               method='L-BFGS-B',
                               bounds=[(-5, 5)] * x.shape[1])
                self.scales = np.exp(res.x)

            K = self.matern52_kernel(x, x, self.scales) + self.reg * np.eye(x.shape[0])
            self.k_xx_inv = np.linalg.inv(K)
        else:
            # Online update via Woodbury identity
            B = self.matern52_kernel(self.x, x, self.scales)
            C = B.T
            D = self.matern52_kernel(x, x, self.scales) + self.reg * np.eye(x.shape[0])
            self.k_xx_inv = self.matern52_kernel_incr_inv(B, C, D)

            print(np.linalg.cond(self.k_xx_inv))
            self.x = np.vstack((self.x, x))
            self.y = np.vstack((self.y, y))

        return self

    def predict(self, x_predict):
        x_predict = np.atleast_2d(x_predict)
        k = self.matern52_kernel(x_predict, self.x, self.scales)
        mu_hat = k @ self.k_xx_inv @ self.y
        sigma_hat = 1 - np.sum(k @ self.k_xx_inv * k, axis=1)
        sigma_hat = np.maximum(sigma_hat, 0.0)
        return mu_hat.ravel(), np.sqrt(sigma_hat)