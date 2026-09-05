"""Shared fixtures for the Week 4 tests.

`legacy_config` and `week4_config` write temporary config files rather than mutating the
project's `config.json`, so a failing test can never leave the repo in a state where the
next training run silently uses the wrong environment.
"""
import json
import collections

import pytest

CONFIG = "config.json"


def _variant(tmp_path, name, **overrides):
    cfg = json.load(open(CONFIG), object_pairs_hook=collections.OrderedDict)
    for dotted, value in overrides.items():
        section, _, key = dotted.partition(".")
        if key:
            cfg[section][key] = value
        else:
            cfg[section] = value
    path = tmp_path / name
    path.write_text(json.dumps(cfg, indent=2))
    return str(path)


# Week 5 added `observation.obstacle_mode` and defaulted it to "paired" in the project's
# config.json. Every Week 4 fixture below pins it back to "legacy" so those tests keep
# describing the environment their docstrings claim -- in particular the bit-identity
# test, which would otherwise be comparing a 52-dim observation against main's 34-dim one.
LEGACY_OBS = {"observation.obstacle_mode": "legacy"}


@pytest.fixture
def legacy_config(tmp_path):
    """The environment exactly as it was on `main`: no dt, uniform pairs, no randomization."""
    return _variant(tmp_path, "legacy.json", dt=1.0,
                    **{"start_goal.sampling": "uniform",
                       "dynamic_obstacles.randomize": False,
                       **LEGACY_OBS})


@pytest.fixture
def week4_config(tmp_path):
    """Week 4's environment: dt, stratified pairs, smoothly stochastic obstacles."""
    return _variant(tmp_path, "week4.json",
                    **{"dynamic_obstacles.randomize": True, **LEGACY_OBS})


@pytest.fixture
def week4_fixed_config(tmp_path):
    """Week 4 timing and pairs, but the ORIGINAL obstacle motion (Experiment 1)."""
    return _variant(tmp_path, "week4_fixed.json",
                    **{"dynamic_obstacles.randomize": False, **LEGACY_OBS})


@pytest.fixture
def week5_config(tmp_path):
    """Week 5's environment: Week 4 plus the paired obstacle observation."""
    return _variant(tmp_path, "week5.json",
                    **{"dynamic_obstacles.randomize": True,
                       "observation.obstacle_mode": "paired"})
