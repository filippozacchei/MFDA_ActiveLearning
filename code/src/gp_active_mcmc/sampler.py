import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
from .priors import FlatPrior

# =========================
# Base Surrogate MCMC
# =========================

# =========================
# SurrogateMCMC with history logging
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
        log_theta_ref: np.ndarray | None = None,  # track GP predictions at a reference theta
    ):
        self.gp = gp
        self.fw_true = fw_true
        self.prior = prior or FlatPrior()
        self.constraint_fn = constraint_fn or (lambda th: True)
        self.proposal = proposal
        self.loglike_surrogate = loglike_surrogate

        # Chain info
        self.chain = None
        self.accepted = None
        self.used_forward = None

        # Optional reference logging
        self.log_theta_ref = log_theta_ref
        self.gp_pred_ref = []

    def mh_accept(self, logpost_new: float, logpost_old: float) -> bool:
        """Metropolis-Hastings acceptance decision."""
        if np.isneginf(logpost_new):
            return False
        delta = logpost_new - logpost_old
        return np.log(np.random.rand()) < delta

    @abstractmethod
    def step(self, theta_n: np.ndarray) -> tuple[np.ndarray, bool, bool]:
        """One MCMC iteration: return (theta_next, accepted, used_forward)."""
        pass

    def run(
        self,
        theta0: np.ndarray,
        n_total: int,
        store_gp_ref: bool = True,
    ) -> dict:
        """Run the MCMC chain with optional logging of GP predictions at reference theta."""

        d = len(theta0)
        self.chain = np.zeros((n_total + 1, d))
        self.accepted = np.zeros(n_total, dtype=bool)
        self.used_forward = np.zeros(n_total, dtype=bool)
        self.chain[0] = theta0

        if store_gp_ref and self.log_theta_ref is not None:
            self.gp_pred_ref = []

        for n in tqdm(range(n_total)):
            theta_n = self.chain[n]
            theta_next, is_acc, used_fw = self.step(theta_n)

            self.chain[n + 1] = theta_next
            self.accepted[n] = is_acc
            self.used_forward[n] = used_fw

            if store_gp_ref and self.log_theta_ref is not None:
                y_pred_ref, y_var_ref = self.gp.predict(self.log_theta_ref)
                self.gp_pred_ref.append((y_pred_ref.copy(), y_var_ref.copy()))

        result = {
            "chain": self.chain,
            "accepted": self.accepted,
            "accept_rate": np.sum(self.accepted)/n,
            "used_forward": self.used_forward,
        }

        # Include reference GP predictions if tracked
        if store_gp_ref and self.log_theta_ref is not None:
            result["gp_pred_ref"] = self.gp_pred_ref

        # Include subclass-specific attributes (like gamma_var)
        if hasattr(self, "gamma_var"):
            result["gamma_var_hist"] = getattr(self, "gamma_var_hist", None)

        return result
    
# =========================
# Active Learning MCMC
# =========================

class ALMCMC(SurrogateMCMC):
    """Active Learning MCMC: calls true forward if GP variance exceeds threshold."""

    def __init__(self, *args, gamma_var: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma_var = gamma_var

    def step(self, theta_n: np.ndarray) -> tuple[np.ndarray, bool, bool]:
        """Perform one MCMC iteration with GP-guided active learning."""
        # --- propose ---
        theta_star = self.proposal.propose(theta_n)

        # --- constraint check ---
        if not self.constraint_fn(theta_star):
            return theta_n, False, False

        # --- prior check ---
        lp_star = self.prior.logpdf(theta_star)
        if np.isneginf(lp_star):
            return theta_n, False, False

        # --- GP surrogate ---
        y_gp, var_gp = self.gp.predict(theta_star)
        ubar = float(np.mean(var_gp))
        used_fw = False

        if ubar < self.gamma_var:
            loglike_star = self.loglike_surrogate(theta_star)
        else:
            # Call true forward and update GP
            y_true = self.fw_true(theta_star)
            self.gp.update(theta_star, y_true)
            loglike_star = self.loglike_surrogate(theta_star)
            used_fw = True

        # --- MH acceptance ---
        logpost_star = loglike_star + float(lp_star)
        logpost_old = self.loglike_surrogate(theta_n) + float(self.prior.logpdf(theta_n))
        is_acc = self.mh_accept(logpost_star, logpost_old)

        # --- update proposal if adaptive ---
        theta_next = theta_star if is_acc else theta_n
        self.proposal.update(theta_next, is_acc)

        return theta_next, is_acc, used_fw


class DAMCMC(SurrogateMCMC):
    """
    Delayed Acceptance MCMC:
    - Stage 1: GP surrogate guides preliminary acceptance
    - Stage 2: True forward only if Stage 1 accepted
    - Optional subsampling of true forward evaluations
    """

    def __init__(self, *args, gamma_var: float = 0.01, subsample_rate: float = 1.0, **kwargs):
        """
        Parameters
        ----------
        gamma_var : float
            Threshold on GP variance for using surrogate.
        subsample_rate : float
            Fraction of iterations where true forward is forced even if GP variance is low.
        """
        super().__init__(*args, **kwargs)
        self.gamma_var = gamma_var
        self.subsample_rate = subsample_rate
        self.iteration = 0

    def step(self, theta_n: np.ndarray) -> tuple[np.ndarray, bool, bool]:
        """One iteration with delayed acceptance."""
        self.iteration += 1
        used_fw = False

        # --- propose ---
        theta_star = self.proposal.propose(theta_n)

        # --- constraint check ---
        if not self.constraint_fn(theta_star):
            return theta_n, False, False

        # --- prior check ---
        lp_star = self.prior.logpdf(theta_star)
        if np.isneginf(lp_star):
            return theta_n, False, False

        # --- Stage 1: GP surrogate ---
        y_gp, var_gp = self.gp.predict(theta_star)
        ubar = float(np.mean(var_gp))
        stage1_accept = False

        # Decide if surrogate is confident or forced by subsample
        force_fw = np.random.rand() < self.subsample_rate
        if ubar < self.gamma_var and not force_fw:
            loglike_star_stage1 = loglike_theta_gp(theta_star, self.gp)
            loglike_n_stage1 = loglike_theta_gp(theta_n, self.gp)
            alpha1 = min(1.0, np.exp(loglike_star_stage1 - loglike_n_stage1))
            if np.random.rand() < alpha1:
                stage1_accept = True
        else:
            stage1_accept = True  # go to Stage 2

        # --- Stage 2: True forward if Stage 1 accepted ---
        if stage1_accept:
            y_true_star = self.fw_true(theta_star)
            y_true_n = self.fw_true(theta_n)
            self.gp.update(theta_star, y_true_star)
            used_fw = True

            loglike_star = loglike_theta(theta_star, self.fw_true)
            loglike_n = loglike_theta(theta_n, self.fw_true)
            alpha2 = min(1.0, np.exp(loglike_star - loglike_n))
            is_acc = np.random.rand() < alpha2
        else:
            is_acc = False

        # --- update proposal if adaptive ---
        theta_next = theta_star if is_acc else theta_n
        self.proposal.update(theta_next, is_acc)

        return theta_next, is_acc, used_fw

class AGAMCMC(DAMCMC):
    """
    Adaptive Gamma MCMC:
    - Delayed Acceptance MCMC with adaptive GP variance threshold.
    - Adjusts gamma_var online to control the fraction of true forward calls.
    """

    def __init__(self, *args, target_fw_frac: float = 0.3, adapt_rate: float = 0.05, **kwargs):
        """
        Parameters
        ----------
        target_fw_frac : float
            Desired fraction of iterations where true forward is evaluated.
        adapt_rate : float
            Adaptation rate for adjusting gamma_var (proportional control).
        """
        super().__init__(*args, **kwargs)
        self.target_fw_frac = target_fw_frac
        self.adapt_rate = adapt_rate
        self.true_fw_calls = 0
        self.total_steps = 0

    def step(self, theta_n: np.ndarray) -> tuple[np.ndarray, bool, bool]:
        """One iteration with adaptive gamma_var."""
        self.total_steps += 1

        # Call parent step
        theta_next, is_acc, used_fw = super().step(theta_n)

        # Track true forward calls
        if used_fw:
            self.true_fw_calls += 1

        # --- adapt gamma_var ---
        if self.total_steps % 10 == 0:  # adapt every 10 steps
            fw_frac = self.true_fw_calls / self.total_steps
            # Proportional control: increase gamma if too many FWs, decrease if too few
            self.gamma_var *= 1.0 + self.adapt_rate * (fw_frac - self.target_fw_frac)
            # Keep gamma_var in reasonable bounds
            self.gamma_var = np.clip(self.gamma_var, 1e-6, 1.0)

        return theta_next, is_acc, used_fw
