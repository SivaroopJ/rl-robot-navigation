"""HD seed blocks (HD_DESIGN.md section 5). Disjoint from every earlier reservation.

HD_FINAL is sealed: `seed_block("HD_FINAL")` raises unless `allow_final=True`, and
tests/test_highdim.py asserts that only experiments/highdim/hd0_baselines.py (floor + ceiling)
and hd4_final.py (PPO arms) pass that flag.

The per-episode Random RNG is `continuation.seeds.controller_rng(episode_seed)`, reused unchanged.
"""
from __future__ import annotations

from continuation import seeds as CS
from continuation.pursuit.config import PURSUIT_BLOCKS

BLOCKS = {
    "HD_DEV": (14_000_000, 100),
    "HD_FINAL": (14_100_000, 200),
}
TRAIN_BASE = 14_500_000
#: Implementation cap, not a pre-registered rule: 100 runs x 1000 envs keeps every training seed
#: inside [14 500 000, 14 600 000), clear of the dev and final blocks.
TRAIN_RUNS, TRAIN_ENVS = 100, 1000

#: Earlier reservations the HD blocks must avoid, beyond continuation.seeds.
OTHER_RESERVED = [(11_000_000, 12_000_000),     # pursuit (continuation.pursuit.config)
                  (12_000_000, 13_000_000),     # Track B dev + eval
                  (13_000_000, 14_000_000)]     # Track A dev + eval


def seed_block(name, n=None, *, allow_final=False):
    if name not in BLOCKS:
        raise KeyError(name)
    if name == "HD_FINAL" and not allow_final:
        raise PermissionError("HD_FINAL is sealed; only hd0_baselines.py and hd4_final.py open it")
    base, size = BLOCKS[name]
    n = size if n is None else int(n)
    if n > size:
        raise ValueError(f"block {name} holds {size} episodes, requested {n}")
    return [base + i for i in range(n)]


def train_env_seed(run, env_index):
    """Initial seed of training env `env_index` in PPO run `run`."""
    if not (0 <= run < TRAIN_RUNS and 0 <= env_index < TRAIN_ENVS):
        raise ValueError(f"run {run} / env {env_index} outside the HD_TRAIN range")
    return TRAIN_BASE + 1000 * int(run) + int(env_index)


def _spans():
    spans = [(b, b + s) for b, s in BLOCKS.values()]
    spans.append((TRAIN_BASE, TRAIN_BASE + 1000 * TRAIN_RUNS))
    return spans


def check_disjoint():
    reserved = [(b, b + CS.BLOCK_STRIDE) for b, _ in CS.BLOCKS.values()]
    reserved += list(CS.FORBIDDEN_RANGES) + OTHER_RESERVED
    reserved += [(b, b + s) for b, s in PURSUIT_BLOCKS.values()]
    spans = sorted(_spans())
    for (a0, a1), (b0, _) in zip(spans, spans[1:]):
        assert a1 <= b0, "HD blocks overlap each other"
    for lo, hi in spans:
        for r0, r1 in reserved:
            assert hi <= r0 or lo >= r1, f"HD span [{lo}, {hi}) hits reservation [{r0}, {r1})"


check_disjoint()
