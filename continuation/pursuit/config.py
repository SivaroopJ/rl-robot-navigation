"""Pursuit experiment configuration. Every field the plan requires to be explicit is here.

All values are the ones resolved with the user on 2026-09-12 (audit section 7) and are logged
with every run.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

#: Outcome priority on a contested step, fixed for every condition (audit section 7).
OUTCOME_PRIORITY = ("protagonist_collision", "capture", "goal", "timeout")
CONDITIONS = ("P-Random", "P-Tuned")
INFO_MODELS = ("known_state", "lidar_estimate")   # Track A adds the perception model
UPDATE_MODES = ("simultaneous",)


@dataclass(frozen=True)
class PursuitConfig:
    condition: str = "P-Random"          # P-Random (primary) | P-Tuned (ablation)
    capture_delta: float = 0.05          # d_capture = r_p + r_h + delta = 0.65 m
    hunter_max_speed: float = 1.0        # primary; 1.25 is the separate speed sweep
    hunter_radius: float = 0.3
    hunter_start_min_dist: float = 3.0   # hunter spawns >= this from the protagonist start
    info_model: str = "known_state"
    update_mode: str = "simultaneous"
    motion: str = "randomized"           # stochastic obstacles (SmoothStochasticMotion)
    max_steps: int = 500

    def __post_init__(self):
        if self.condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}")
        if self.info_model not in INFO_MODELS:
            raise ValueError(f"info_model must be one of {INFO_MODELS}")
        if self.update_mode not in UPDATE_MODES:
            raise ValueError(f"update_mode must be one of {UPDATE_MODES}")
        if self.capture_delta < 0:
            raise ValueError("capture tolerance must be >= 0")
        if not 0 < self.hunter_max_speed <= 1.5:
            raise ValueError("hunter speed bound outside the documented range")

    @property
    def hunter_visible_to_lidar(self):
        """Track A only: the hunter becomes a LiDAR-visible disc so it can be perceived.

        In the known-state baseline it stays invisible, so the oracle row is its only channel and
        the audited results are unaffected.
        """
        return self.info_model == "lidar_estimate"

    def capture_distance(self, agent_radius):
        return float(agent_radius) + float(self.hunter_radius) + float(self.capture_delta)

    def contact_distance(self, agent_radius):
        return float(agent_radius) + float(self.hunter_radius)

    def as_dict(self):
        d = asdict(self)
        d["capture_distance@r0.3"] = self.capture_distance(0.3)
        d["contact_distance@r0.3"] = self.contact_distance(0.3)
        d["outcome_priority"] = list(OUTCOME_PRIORITY)
        return d


# --------------------------------------------------------------------------- seed blocks
#: Fresh and disjoint from every navigation block (those occupy 10 000 000 - 10 999 999) and from
#: every Week 5 / 5.5 block. Env seeds only; controller RNG is a separate stream
#: (continuation.seeds.controller_rng), so the two can never be confused.
PURSUIT_BLOCKS = {
    "P1_sanity": (11_000_000, 20),
    "P2_static": (11_100_000, 20),
    "P3_smoke": (11_200_000, 10),
    "P3_main": (11_300_000, 200),
}


def pursuit_seeds(name, n=None):
    base, size = PURSUIT_BLOCKS[name]
    n = size if n is None else int(n)
    if n > size:
        raise ValueError(f"block {name} holds {size} episodes, requested {n}")
    return [base + i for i in range(n)]
