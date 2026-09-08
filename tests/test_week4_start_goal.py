"""Week 4 start/goal gate: items 10-11 of the checklist.

These exist because the sampler's whole purpose is a DISTRIBUTIONAL property. Nothing in
a single episode can show that regions are covered or that distances vary, so the only
honest test is a large sample with explicit floors. The floors are set where the measured
behaviour comfortably sits, not at the measurement itself, so ordinary sampling noise does
not produce a red suite.
"""
import json

import numpy as np
import pytest

from robot_env.robot_nav_env import RobotNavEnv
from robot_env.start_goal import StartGoalSampler

N_SAMPLES = 6000


@pytest.fixture(scope="module")
def sampler():
    cfg = json.load(open("config.json"))
    env_cfg, sg = cfg["environment"], cfg["start_goal"]
    return StartGoalSampler(
        env_cfg["world_size"],
        max(env_cfg["agent_radius"] * 2, 0.5),
        env_cfg["static_obstacles"],
        region_grid=sg["region_grid"],
        region_free_threshold=sg["region_free_threshold"],
        distance_bins=sg["distance_bins"],
        min_separation=sg["min_separation"],
        max_attempts=sg["max_attempts"])


@pytest.fixture(scope="module")
def draws(sampler):
    rng = np.random.default_rng(0)
    starts, goals, distances = [], [], []
    for _ in range(N_SAMPLES):
        s, g = sampler.sample(rng)
        starts.append(sampler.region_index_of(s))
        goals.append(sampler.region_index_of(g))
        distances.append(float(np.linalg.norm(g - s)))
    return dict(starts=np.array(starts), goals=np.array(goals),
                distances=np.array(distances), sampler=sampler)


# ------------------------------------------------------------------- 10. region coverage
def test_every_usable_region_is_used_as_start_and_as_goal(draws):
    """No region of the map may be starved.

    The floor is half the uniform share. Exact uniformity is not achievable and not
    desirable: corner regions genuinely have more distant partners than central ones, so
    forcing the marginals flat would have to distort the distance strata instead. Half the
    share is the point at which 'the policy never trains here' stops being a concern.
    """
    sampler = draws["sampler"]
    expectation = N_SAMPLES / len(sampler.usable_regions)
    for label in ("starts", "goals"):
        counts = np.bincount(draws[label], minlength=sampler.region_grid ** 2)
        for region in sampler.usable_regions:
            assert counts[region] >= 0.5 * expectation, (
                f"{label}: region {region} drawn {counts[region]} times against an "
                f"expectation of {expectation:.0f}")


def test_unusable_regions_are_never_drawn(draws):
    """A region excluded for being mostly wall must not receive any pair."""
    sampler = draws["sampler"]
    unusable = set(range(sampler.region_grid ** 2)) - set(sampler.usable_regions)
    for region in unusable:
        assert (draws["starts"] == region).sum() == 0
        assert (draws["goals"] == region).sum() == 0


# ------------------------------------------------------------------ 11. distance variety
def test_distance_bands_are_evenly_populated(draws):
    """Each band within +-20% of an equal share.

    Equal mass per band is what makes the start-goal distance an experimental variable
    rather than a nuisance one: results can be reported per band, so 'does SR help on the
    long crossings?' is answerable instead of averaged into the middle.
    """
    sampler = draws["sampler"]
    share = 1.0 / len(sampler.distance_bins)
    for low, high in sampler.distance_bins:
        got = float(((draws["distances"] >= low) & (draws["distances"] <= high)).mean())
        assert abs(got - share) <= 0.2 * share, (
            f"band ({low}, {high}) holds {got:.1%} of pairs, expected ~{share:.1%}")


def test_pairs_are_far_apart(draws):
    """The floor the old uniform scheme lacked.

    On `main` both points were drawn uniformly with a 2.0 minimum, giving mean separation
    5.21 and 31.7% of pairs closer than 4.0 -- which is what produced 4-6 step episodes.
    """
    cfg = json.load(open("config.json"))
    assert draws["distances"].min() >= cfg["start_goal"]["min_separation"] - 1e-6
    assert draws["distances"].mean() > 6.0


def test_fallback_rate_is_negligible(draws):
    """Rejection sampling must almost never give up on the requested band.

    A high rate would mean the realised distance distribution is not the configured one,
    silently reshaping the task. Counted rather than hidden precisely so that a future
    layout or bin change cannot quietly break the stratification.
    """
    assert draws["sampler"].fallback_rate < 0.01


# ------------------------------------------------------------------------ validity
def test_sampled_points_are_always_free(draws, sampler):
    """Every pair must satisfy the ORIGINAL validity predicate.

    The sampler changes which legal points are likely, never which points are legal.
    """
    rng = np.random.default_rng(99)
    for _ in range(1500):
        s, g = sampler.sample(rng)
        assert sampler._is_free(s) and sampler._is_free(g)
        for cx, cy, hw, hh in sampler.static_obstacles:
            for p in (s, g):
                assert not (abs(p[0] - cx) < hw and abs(p[1] - cy) < hh), \
                    "a point was placed inside a static rectangle"


def test_env_reproduces_pairs_from_a_seed(week4_config):
    """Same seed -> same pair, which the paired PPO/PPO+SR evaluation depends on."""
    def pair(seed):
        env = RobotNavEnv(config_path=week4_config, n_dynamic_obstacles=6)
        env.reset(seed=seed)
        return env.agent_position.copy(), env.target_position.copy()

    for seed in (0, 5, 123):
        a, b = pair(seed), pair(seed)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
    assert not np.array_equal(pair(0)[0], pair(1)[0])


def test_uniform_mode_restores_the_original_distribution(legacy_config):
    """The legacy sampling mode must still produce the old, shorter pairs.

    Without this, 'reproduce the original baseline' would silently mean 'reproduce it on a
    different task distribution'.
    """
    env = RobotNavEnv(config_path=legacy_config, n_dynamic_obstacles=3)
    distances = []
    for seed in range(800):
        env.reset(seed=seed)
        distances.append(env.initial_distance)
    assert 4.5 < np.mean(distances) < 6.0, "legacy mode should reproduce the ~5.2 mean"
    assert np.min(distances) < 4.0, "legacy mode had no 4.0 floor"
