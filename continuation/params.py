"""The tunable CLF-DR-CBF hyperparameters and the quantities derived from them.

WHAT THE DR CONSTRAINT ACTUALLY ENFORCES (audit A2-A4)
    The frozen DRCCP block is   r_W * |u_bar| / eps  <=  t - sum(s_i) / (N eps)   (componentwise),
    u_bar = [1, alpha, u_x, u_y]. Over the per-axis box |u_x|, |u_y| <= max_v <= 1 the binding
    component of |u_bar| is max(1, alpha), so the left side is

        tau_eff = r_W * max(1, alpha) / eps.

    If eps * N <= 1 the CVaR supremum is attained at the worst sample and the whole constraint is
    exactly  min_i CBC_i >= tau_eff  ("collapsed" regime). r_W and eps then matter ONLY through
    tau_eff. If eps * N > 1 the constraint averages the tail ("non_collapsed") and is a different
    mathematical object; the main continuation chain never uses it.

    N here is the number of DR samples = the scan-buffer length k_scans (one sample per buffered
    scan, audit A1); the frozen controller's n_keep = 5 truncation is never binding for k_scans <= 5.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from dr_control.drccp_controller import K_V_REFERENCE

N_KEEP = 5          # frozen ClfCbfDrccpController default; not a continuation parameter


@dataclass(frozen=True)
class DRCBFParams:
    alpha: float = 0.4               # cbf_rate
    clf_rate: float = 1.0            # lambda_V
    wasserstein_r: float = 0.004     # r_W
    epsilon: float = 0.1
    k_v: float = K_V_REFERENCE       # 0.05
    k_scans: int = 5                 # N

    def __post_init__(self):
        if not (self.alpha > 0 and self.clf_rate > 0 and self.epsilon > 0 and self.k_v > 0):
            raise ValueError(f"non-positive parameter in {self}")
        if self.wasserstein_r < 0:
            raise ValueError("r_W must be >= 0")
        if int(self.k_scans) != self.k_scans or not 1 <= self.k_scans <= N_KEEP:
            raise ValueError(f"k_scans must be an integer in 1..{N_KEEP}")

    @property
    def n_samples(self):
        return min(int(self.k_scans), N_KEEP)

    @property
    def eps_n(self):
        return self.epsilon * self.n_samples

    @property
    def regime(self):
        return "collapsed" if self.eps_n <= 1.0 + 1e-12 else "non_collapsed"

    def tau_eff(self, max_v=1.0):
        if max_v > 1.0 + 1e-12:
            raise ValueError("tau_eff derivation assumes max_v <= 1")
        return self.wasserstein_r * max(1.0, self.alpha) / self.epsilon

    def effective_class(self):
        """(alpha, tau_eff, regime) plus the parameters that are NOT absorbed into tau_eff.

        In the collapsed regime two configurations with equal class are the same controller up
        to solver tolerance. In the non-collapsed regime r_W and eps are not interchangeable, so
        they stay in the key.
        """
        base = (round(self.alpha, 10), round(self.tau_eff(), 10), self.regime,
                round(self.clf_rate, 10), round(self.k_v, 10), int(self.k_scans))
        if self.regime == "non_collapsed":
            base += (round(self.wasserstein_r, 10), round(self.epsilon, 10))
        return base

    def as_dict(self):
        d = asdict(self)
        d.update(tau_eff=self.tau_eff(), regime=self.regime, eps_n=self.eps_n)
        return d

    def tag(self):
        return (f"a{self.alpha:g}_rW{self.wasserstein_r:g}_e{self.epsilon:g}"
                f"_lV{self.clf_rate:g}_kv{self.k_v:g}_N{self.k_scans}")

    @classmethod
    def from_dict(cls, d):
        keys = ("alpha", "clf_rate", "wasserstein_r", "epsilon", "k_v", "k_scans")
        return cls(**{k: d[k] for k in keys})


#: The verified reference configuration of the frozen controller (audit section 1.1).
REFERENCE = DRCBFParams()


# ----------------------------------------------------------------------- search spaces
H1_ALPHA = (0.2, 0.4, 0.8, 1.2, 2.0)
H1_RW = (0.0, 0.001, 0.002, 0.004, 0.008, 0.012)
H1_EPS_COLLAPSED = (0.05, 0.10, 0.20)
H1_EPS_NON_COLLAPSED = (0.30,)
H2_CLF_RATE = (0.25, 0.5, 1.0, 2.0, 4.0)
H2_K_V = (0.025, 0.05, 0.10)
C1_K_SCANS = (1, 3, 5)


def h1_grid():
    out = []
    for a in H1_ALPHA:
        for r in H1_RW:
            for e in H1_EPS_COLLAPSED + H1_EPS_NON_COLLAPSED:
                out.append(DRCBFParams(alpha=a, wasserstein_r=r, epsilon=e))
    return out
