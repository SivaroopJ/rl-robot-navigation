"""TunableDRCBFPolicy: the frozen DRCBFPolicy with every DR-CBF/CLF hyperparameter exposed.

`dr_control/policy.py` is FROZEN and is not edited. This subclass changes exactly two
construction details and nothing else:
    * the controller is built with (alpha, clf_rate, r_W, eps, k_v) from DRCBFParams instead of
      the constructor defaults, and
    * reset() rebuilds the barrier source with k_scans = params.k_scans (the frozen reset()
      hard-codes 5) using the SAME classes and arguments the frozen reset() uses.
predict() is inherited unchanged, so observation slicing (nothing past index 28), the planner,
the carrot follower and the u = 0 infeasibility fallback are the frozen ones. At REFERENCE the
policy is required to be bit-identical to DRCBFPolicy (tests + the A12 equivalence gate).

There is NO T1 velocity cap here: the tuned arms F-B/F-C/F-D use the original barrier source
(decision D2). The historical C0-B arm is built from Phase7Policy directly, see `build_arm`.
"""
from __future__ import annotations

from dr_control.capped_velocity import V_CAP_DERIVED
from dr_control.drccp_controller import ClfCbfDrccpController
from dr_control.estimated_cbf import EstimatedLidarBarrierSource
from dr_control.policy import DRCBFPolicy
from dr_control.policy_phase7 import Phase7Policy
from dr_control.recovery import RecoveryLadder
from dr_control.velocity_tracker import LidarVelocityTracker

from continuation.params import REFERENCE, DRCBFParams
from continuation.random_recovery import RandomRecovery, RandomSpec


class TunableDRCBFPolicy(DRCBFPolicy):
    def __init__(self, *, params: DRCBFParams = REFERENCE, **kwargs):
        if "alpha" in kwargs:
            raise TypeError("pass alpha through params, not as a keyword")
        super().__init__(alpha=params.alpha, **kwargs)
        self.params = params
        # Same class as the frozen constructor; only the hyperparameters differ.
        self.ctrl = ClfCbfDrccpController(max_v=self.max_speed, cbf_rate=params.alpha,
                                          clf_rate=params.clf_rate,
                                          wasserstein_r=params.wasserstein_r,
                                          epsilon=params.epsilon, k_v=params.k_v)

    def reset(self, obs, ego_pose):
        out = super().reset(obs, ego_pose)            # frozen: plans, resets ctrl + counters
        tracker = LidarVelocityTracker(r_nominal=self.r_nominal, dt=self.dt,
                                       n_rays=self.n_rays, lidar_range=self.lidar_range)
        self.src = EstimatedLidarBarrierSource(r_robot=self.agent_radius,
                                               k_scans=int(self.params.k_scans),
                                               n_rays=self.n_rays,
                                               lidar_range=self.lidar_range, tracker=tracker)
        return out


# --------------------------------------------------------------------------- arm registry
#: kind -> description. "original" is the frozen DRCBFPolicy itself (C0-A);
#: "c0b" is the exact historical Phase7Policy(T1, 0.96) + RecoveryLadder("ladder").
KINDS = ("original", "c0b", "tuned", "ladder", "random")


def policy_kwargs(env):
    return dict(world_size=env.WORLD_SIZE, agent_radius=env.AGENT_RADIUS,
                static_obstacles=env.static_obstacles, max_speed=env.MAX_SPEED, dt=env.dt,
                lidar_range=env.LIDAR_RANGE, n_rays=env.N_LIDAR_RAYS, use_planner=True)


def build_arm(kind, env, *, params=REFERENCE):
    """Construct the policy for one episode. The recovery wrapper is attached separately by
    `attach_recovery`, AFTER pol.reset(), which is the order exp7_4 / exp8 use."""
    if kind not in KINDS:
        raise ValueError(kind)
    if kind == "original":
        if params != REFERENCE:
            raise ValueError("C0-A is the frozen controller; it has no tunable parameters")
        return DRCBFPolicy(**policy_kwargs(env))
    if kind == "c0b":
        if params != REFERENCE:
            raise ValueError("C0-B is the frozen historical controller; it is never tuned")
        return Phase7Policy(arm="T1_projection_cap", v_cap=V_CAP_DERIVED, **policy_kwargs(env))
    return TunableDRCBFPolicy(params=params, **policy_kwargs(env))


def attach_recovery(kind, pol, *, params=REFERENCE, random_spec=None, rng=None):
    if kind in ("original", "tuned"):
        return None
    if kind == "c0b":
        return RecoveryLadder(pol.ctrl, mode="ladder")          # historical defaults, verbatim
    if kind == "ladder":
        if params.regime != "collapsed":
            raise ValueError("LADDER's rung logic assumes the collapsed CVaR regime")
        # tau passed explicitly: the ladder's default r_W/eps is wrong for alpha > 1 (audit A4)
        return RecoveryLadder(pol.ctrl, mode="ladder", tau=params.tau_eff())
    if kind == "random":
        if params.regime != "collapsed":
            raise ValueError("random recovery scores min_i CBC_i, the collapsed-regime constraint")
        if random_spec is None or rng is None:
            raise ValueError("random arm needs a RandomSpec and a controller RNG")
        return RandomRecovery(pol.ctrl, spec=random_spec, rng=rng, tau=params.tau_eff())
    raise AssertionError(kind)
