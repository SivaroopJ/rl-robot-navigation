"""Week5-Phase7 / Stage 0 / trace-corpus integrity gate.

Stage 1's whole equivalence argument is "replay the frozen controller's recorded inputs and
compare". That is worth nothing unless the recording is faithful and self-contained. These
tests establish that it is, BEFORE any accelerated controller is written:

    1. the corpus is present, complete, and matches its own checksums
    2. every recorded step REPLAYS BIT-IDENTICALLY from its recorded inputs alone
    3. the ragged (CSR-style) packing is internally consistent
    4. sample provenance is what the frozen buffer semantics imply (row index == scan age)
    5. every solved step satisfies the reduced DR constraint min_i CBC_i >= tau
    6. the enriched infeasible corpus contains exactly the infeasible steps
    7. ground truth lives only in explicitly labelled truth_* channels

Skipped with a clear message if the corpus has not been built yet.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.exp7_0_trace_corpus import OUT_DIR, sha256, verify_replayable

pytestmark = pytest.mark.skipif(
    not (OUT_DIR / "INDEX.json").exists(),
    reason="trace corpus not built; run experiments.exp7_0_trace_corpus")

TAU = 0.004 / 0.1
CONTROL_PATH_KEYS = {
    "ep_index", "step", "p", "gamma", "u_prev", "u", "u_nom", "xi_flat", "xi_ptr",
    "xi_kept_flat", "xi_kept_ptr", "sample_age", "sample_track_id", "status_code",
    "delta", "h_crit", "box_overshoot", "total_time", "solver_time", "canon_time",
    "p1", "p3", "V", "min_cbc_kept", "min_cbc_all",
}
TRUTH_KEYS = {"truth_clearance", "truth_obs_pos", "truth_obs_vel"}


@pytest.fixture(scope="module")
def index():
    return json.loads((OUT_DIR / "INDEX.json").read_text())


@pytest.fixture(scope="module")
def dev_part():
    meta = json.loads((OUT_DIR / "meta_dev_fixed.json").read_text())
    return meta, np.load(OUT_DIR / meta["npz"])


# ------------------------------------------------------------------ 1: presence + checksums
def test_every_recorded_part_matches_its_checksum(index):
    assert index["parts"], "empty corpus index"
    for meta in index["parts"]:
        npz = OUT_DIR / meta["npz"]
        assert npz.exists(), npz
        assert sha256(npz) == meta["npz_sha256"], f"{meta['npz']} changed after recording"


def test_labels_identify_the_stage(index):
    assert index["label"].startswith("Week5-Phase7 / Stage 0")
    for meta in index["parts"]:
        assert meta["label"].startswith("Week5-Phase7 / Stage 0")


def test_controller_parameters_recorded_are_the_frozen_ones(index):
    for meta in index["parts"]:
        p = meta["controller_params"]
        assert (p["cbf_rate"], p["wasserstein_r"], p["epsilon"]) == (0.4, 0.004, 0.1)
        assert (p["max_v"], p["n_keep"], p["solver"]) == (1.0, 5, "SCS")
        assert p["tau"] == pytest.approx(TAU)


# ------------------------------------------------------------------ 2: replayability
def test_recorded_steps_replay_bit_identically(dev_part):
    """THE gate item. If this fails the corpus cannot support Stage 1's equivalence test."""
    meta, d = dev_part
    n_checked, mismatches = verify_replayable(
        d, meta["controller_params"], 120, np.random.default_rng(7))
    assert n_checked >= 100
    assert not mismatches, f"{len(mismatches)} steps did not replay: {mismatches[:5]}"


def test_recorded_replay_verification_was_run_for_every_part(index):
    for meta in index["parts"]:
        r = meta["replay_verification"]
        assert r["checked"] > 0 and r["bit_identical"], (meta["label"], r)


# ------------------------------------------------------------------ 3: packing consistency
def test_ragged_offsets_are_consistent(dev_part):
    _, d = dev_part
    n = len(d["u"])
    for ptr, flat in (("xi_ptr", "xi_flat"), ("xi_kept_ptr", "xi_kept_flat")):
        p = d[ptr]
        assert len(p) == n + 1 and p[0] == 0
        assert np.all(np.diff(p) > 0), "a step recorded zero samples"
        assert p[-1] == len(d[flat])
    assert d["xi_flat"].shape[1] == 4 and d["xi_kept_flat"].shape[1] == 4
    assert len(d["sample_age"]) == len(d["xi_flat"]) == len(d["sample_track_id"])
    for key in ("p", "gamma", "u", "u_prev", "u_nom"):
        assert d[key].shape == (n, 2)


def test_kept_samples_never_exceed_n_keep_and_are_a_subset(dev_part):
    meta, d = dev_part
    n_keep = meta["controller_params"]["n_keep"]
    for i in range(0, len(d["u"]), 37):
        a = d["xi_flat"][d["xi_ptr"][i]:d["xi_ptr"][i + 1]]
        k = d["xi_kept_flat"][d["xi_kept_ptr"][i]:d["xi_kept_ptr"][i + 1]]
        assert len(k) == min(len(a), n_keep)
        for row in k:                       # every kept row came from the full sample set
            assert np.any(np.all(np.isclose(a, row), axis=1)), row


# ------------------------------------------------------------------ 4: sample provenance
def test_sample_age_is_the_buffer_index(dev_part):
    """LidarBarrierSource.samples iterates the buffer newest-first and emits one row per
    scan, so row index IS scan age. Stage 2's staleness attribution depends on this."""
    _, d = dev_part
    for i in range(0, len(d["u"]), 29):
        a0, a1 = d["xi_ptr"][i], d["xi_ptr"][i + 1]
        np.testing.assert_array_equal(d["sample_age"][a0:a1], np.arange(a1 - a0))


def test_geometry_recompute_never_disagreed(index):
    """The recorder re-derives the nearest-point argmin to recover the owning track id. That
    duplication is validated by requiring the recomputed h to equal the frozen h exactly."""
    for meta in index["parts"]:
        assert meta["geometry_recompute_mismatch"] == 0, meta["label"]


def test_track_ids_are_either_valid_or_the_no_track_sentinel(dev_part):
    _, d = dev_part
    assert np.all(d["sample_track_id"] >= -1)
    assert (d["sample_track_id"] >= 0).any(), "no sample was ever bound to a track"


# ------------------------------------------------------------------ 5: the tau floor
def test_every_solved_step_meets_the_reduced_dr_constraint(index):
    """min_i CBC_i >= tau on the kept samples, for every step the solver called optimal.

    This is the reduced form of the DR constraint (see test_frozen_phase1_6). Finding it to
    hold across the whole corpus is what licenses Stage 1's fast path.

    "SOLVED" means status == "optimal" EXACTLY, as EpisodeDiagnostics defines it -- not the
    complement of "infeasible". Stage 0 originally got this wrong: two `optimal_inaccurate`
    steps (which the frozen controller answers with u = 0) pooled in with the solved ones and
    reported a floor of -0.371 instead of 0.03990. The npz data was always correct; only the
    summary field was. Recorded here so the distinction stays visible.

    TOLERANCE: measured worst deviation across the whole corpus is tau - 1.05e-4, one-sided
    and 0.26 % of tau, consistent with the 0.03989-0.03996 that Phases 2-6 recorded with the
    same first-order solver. 5e-4 is a factor of ~5 above the observed worst.
    """
    for meta in index["parts"]:
        d = np.load(OUT_DIR / meta["npz"])
        table = {int(k): v for k, v in meta["status_table"].items()}
        optimal = np.array([table[int(c)] == "optimal" for c in d["status_code"]])
        if not optimal.any():
            continue
        m = float(np.nanmin(d["min_cbc_kept"][optimal]))
        assert m >= TAU - 5e-4, (meta["label"], m)
        assert m >= 0.99 * TAU, (meta["label"], m)      # the floor is tau, not merely positive


def test_non_optimal_steps_are_reported_separately_from_infeasible(index):
    """`optimal_inaccurate` and solver errors are non-optimal but not infeasible. They must
    be counted in their own bucket, never folded into either neighbour."""
    for meta in index["parts"]:
        assert (meta["n_solved_steps"] + meta["n_infeasible_steps"]
                + meta["n_non_optimal_non_infeasible"]) == meta["steps"], meta["label"]
        assert meta["min_cbc_solved_minus_tau"] > -5e-4, meta["label"]


def test_failed_steps_command_zero(index):
    """The frozen fallback. Direction 1 replaces it only in NEW code."""
    for meta in index["parts"]:
        d = np.load(OUT_DIR / meta["npz"])
        table = {int(k): v for k, v in meta["status_table"].items()}
        failed = np.array([table[int(c)] != "optimal" for c in d["status_code"]])
        if failed.any():
            np.testing.assert_array_equal(d["u"][failed], np.zeros((failed.sum(), 2)))


def test_actions_respect_the_action_box(index):
    for meta in index["parts"]:
        d = np.load(OUT_DIR / meta["npz"])
        assert np.max(np.abs(d["u"])) <= meta["controller_params"]["max_v"] + 1e-12


# ------------------------------------------------------------------ 6: enriched corpus
def test_enriched_infeasible_corpus_holds_exactly_the_infeasible_steps(index):
    path = OUT_DIR / "meta_infeasible_enriched.json"
    if not path.exists():
        pytest.skip("no infeasible steps recorded")
    meta = json.loads(path.read_text())
    d = np.load(OUT_DIR / meta["npz"], allow_pickle=False)
    expected = sum(m["n_infeasible_steps"] for m in index["parts"])
    assert meta["steps"] == expected == len(d["u"])
    np.testing.assert_array_equal(d["u"], np.zeros_like(d["u"]))     # all took the fallback
    assert len(d["xi_ptr"]) == len(d["u"]) + 1 and d["xi_ptr"][-1] == len(d["xi_flat"])


def test_enriched_corpus_is_large_enough_to_test_status_agreement(index):
    """Stage 1's L1 gate needs infeasible steps to test against."""
    path = OUT_DIR / "meta_infeasible_enriched.json"
    if not path.exists():
        pytest.skip("no infeasible steps recorded")
    assert json.loads(path.read_text())["steps"] >= 200


# ------------------------------------------------------------------ 7: information ledger
def test_ground_truth_is_confined_to_labelled_channels(dev_part):
    """Ground truth is recorded for later metrics, and must be impossible to mistake for a
    control-path input. Every array is either a known control-path key or truth_*-prefixed."""
    _, d = dev_part
    keys = set(d.files)
    unknown = keys - CONTROL_PATH_KEYS - TRUTH_KEYS
    assert not unknown, f"unclassified arrays in the corpus: {unknown}"
    assert TRUTH_KEYS <= keys
    assert all(k.startswith("truth_") for k in keys - CONTROL_PATH_KEYS)


def test_control_path_arrays_carry_no_obstacle_state(dev_part):
    """xi is 4 columns wide: [dh_dt, h, grad_h_x, grad_h_y]. There is nowhere for an obstacle
    position or velocity to hide in the recorded control-path inputs."""
    _, d = dev_part
    assert d["xi_flat"].shape[1] == 4
    assert d["p"].shape[1] == 2 and d["gamma"].shape[1] == 2
