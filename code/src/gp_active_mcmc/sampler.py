import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
from .priors import FlatPrior

# =========================
# Base Surrogate MCMC
# =========================

class SurrogateMCMC(ABC):
    """Base class for surrogate-based MCMC samplers with optional history logging."""

    def __init__(
        self,
        gp,
        fw_true,
        loglike_surrogate,
        prior=None,
        constraint_fn=None,
        proposal=None,
        log_theta_ref: np.ndarray | None = None,
    ):
        self.gp = gp
        self.fw_true = fw_true
        self.prior = prior or FlatPrior()
        self.constraint_fn = constraint_fn or (lambda th: True)
        self.proposal = proposal
        self.loglike_surrogate = loglike_surrogate

        self.chain = None
        self.accepted = None
        self.used_forward = None

        self.log_theta_ref = log_theta_ref
        self.gp_pred_ref = []

    def mh_accept(self, logpost_new: float, logpost_old: float) -> bool:
        if np.isneginf(logpost_new):
            return False
        delta = logpost_new - logpost_old
        return np.log(np.random.rand()) < delta

    @abstractmethod
    def step(self, theta_n: np.ndarray) -> tuple[np.ndarray, bool, bool]:
        pass

    def run(self, 
            theta0: np.ndarray, 
            n_total: int, 
            store_gp_ref: bool = True,
            n_gp_update: int | None = None, 
            n_gp_update_params: int | None = None
            ) -> dict:
        d = len(theta0)
        self.chain = np.zeros((n_total + 1, d))
        self.accepted = np.zeros(n_total, dtype=bool)
        self.used_forward = np.zeros(n_total, dtype=bool)
        self.chain[0] = theta0
        
        self.gp_active = True

        if store_gp_ref and self.log_theta_ref is not None:
            self.gp_pred_ref = []  


        for n in tqdm(range(n_total)):
            theta_n = self.chain[n]
            theta_next, is_acc, used_fw = self.step(theta_n)
            self.chain[n + 1] = theta_next
            self.accepted[n] = is_acc
            self.used_forward[n] = used_fw
            
            if n_gp_update is not None and n >= n_gp_update:
                self.gp_active = False
                
            if n_gp_update_params is not None and n == n_gp_update_params:
                self.gp.stop_optimize()

            if store_gp_ref and self.log_theta_ref is not None:
                y_pred_ref, y_var_ref = self.gp.predict(self.log_theta_ref)
                self.gp_pred_ref.append((y_pred_ref.copy(), y_var_ref.copy()))

        result = {
            "chain": self.chain,
            "accepted": self.accepted,
            "accept_rate": np.mean(self.accepted),
            "used_forward": self.used_forward,
        }

        if store_gp_ref and self.log_theta_ref is not None:
            result["gp_pred_ref"] = self.gp_pred_ref

        if hasattr(self, "gamma_var"):
            result["gamma_var_hist"] = getattr(self, "gamma_var_hist", None)

        return result

# =========================
# Active Learning MCMC
# =========================

class ALMCMC(SurrogateMCMC):
    """Standard ALMCMC: HF only if GP variance exceeds gamma_var."""

    def __init__(self, *args, gamma_var: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma_var = gamma_var

    def step(self, theta_n: np.ndarray) -> tuple[np.ndarray, bool, bool]:
        theta_star = self.proposal.propose(theta_n)
        used_fw = False

        if not self.constraint_fn(theta_star):
            return theta_n, False, False

        lp_star = self.prior.logpdf(theta_star)
        if np.isneginf(lp_star):
            return theta_n, False, False

        y_gp, var_gp = self.gp.predict(theta_star)
        ubar = float(np.mean(var_gp))

        if ubar < self.gamma_var or not self.gp_active:
            loglike_star = self.loglike_surrogate(theta_star)
        else:
            y_true = self.fw_true(theta_star)
            self.gp.update(theta_star, y_true)
            loglike_star = self.loglike_surrogate(theta_star)
            used_fw = True

        logpost_star = loglike_star + float(lp_star)
        logpost_old = self.loglike_surrogate(theta_n) + float(self.prior.logpdf(theta_n))
        is_acc = self.mh_accept(logpost_star, logpost_old)

        theta_next = theta_star if is_acc else theta_n
        self.proposal.update(theta_next, is_acc)
        return theta_next, is_acc, used_fw

# =========================
# Randomized ALMCMC
# =========================

class RALMCMC(ALMCMC):
    """ALMCMC but HF is called randomly even if GP variance is low."""

    def __init__(self, *args, loglike, subsample_rate: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.subsample_rate = subsample_rate
        self.last_hf_theta = None
        self.loglike = loglike
        self.hf_errors = []
        self.max_err_hist = 50

    def step(self, theta_n: np.ndarray) -> tuple[np.ndarray, bool, bool]:

        theta_star = self.proposal.propose(theta_n)
        if self.last_hf_theta is None:
            self.last_hf_theta = theta_n
        used_fw = False

        if not self.constraint_fn(theta_star):
            return theta_n, False, False

        lp_star = self.prior.logpdf(theta_star)
        if np.isneginf(lp_star):
            return theta_n, False, False

        y_gp, var_gp = self.gp.predict(theta_star)
        ubar = float(np.mean(var_gp))

        do_hf = (self.gp_active and (ubar >= self.gamma_var)) or (np.random.rand() < self.subsample_rate)

        if do_hf:
            y_true = self.fw_true(theta_star)
            used_fw = True

            loglike_F_star = self.loglike(theta_star)
            loglike_F_start = self.loglike(self.last_hf_theta)

            loglike_C_star = self.loglike_surrogate(theta_star)
            loglike_C_start = self.loglike_surrogate(self.last_hf_theta)

            logpost_star = loglike_F_star + loglike_C_star
            logpost_old = loglike_F_start + loglike_C_start
            
            if self.gp_active:
                self.gp.update(theta_star, y_true)

            err = (y_true - y_gp) / np.sqrt(var_gp + 1e-12)
            self.hf_errors.append(float(np.mean(err**2)))
            self.hf_errors = self.hf_errors[-self.max_err_hist:]
        else:
            loglike_star = self.loglike_surrogate(theta_star)
            loglike_n = self.loglike_surrogate(theta_n)
            logpost_star = loglike_star + float(lp_star)
            logpost_old = loglike_n + float(self.prior.logpdf(theta_n))

        is_acc = self.mh_accept(logpost_star, logpost_old)

        if do_hf:
            theta_next = theta_star if is_acc else self.last_hf_theta
            if is_acc:
                self.last_hf_theta = theta_star
        else:
            theta_next = theta_star if is_acc else theta_n

        self.proposal.update(theta_next, is_acc)
        return theta_next, is_acc, used_fw

# =========================
# Adaptive Randomized ALMCMC
# =========================

class ARALMCMC(RALMCMC):
    """
    Adaptive RALMCMC:
    - Adjusts subsample_rate over the chain based on GP consistency (MLL improvement).
    """
    def __init__(self, *args, target_mll_gain: float = 0.01, adapt_rate: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_mll_gain = target_mll_gain
        self.adapt_rate = adapt_rate
        self.total_steps = 0


    def step(self, theta_n: np.ndarray) -> tuple[np.ndarray, bool, bool]:
        self.total_steps += 1

        # Call RALMCMC step
        theta_next, is_acc, used_fw = super().step(theta_n)

        # Adapt subsample_rate every N steps
        if self.total_steps % 10 == 0 and len(self.hf_errors) > 5:
            npe = np.mean(self.hf_errors)

            # target npe ≈ 1
            delta = np.clip((npe - 1.0), -1.0, 1.0)
            self.subsample_rate *= np.exp(self.adapt_rate * delta)
            self.subsample_rate = np.clip(self.subsample_rate, 0.01, 1.0)

        return theta_next, is_acc, used_fw
