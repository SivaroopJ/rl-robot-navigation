"""Week5-Phase7 / Stage 0 / frozen-file manifest.

WHAT THIS IS FOR
    Phase 7 adds new modules beside the Phase 1-6 system and must never edit it. "Must never"
    is worth exactly as much as the check that enforces it, so every frozen file is hashed
    once here and re-hashed by tests/test_frozen_phase1_6.py on every test run. Drift in a
    frozen controller, in robot_env/, in evaluation/, in a Phase 1-6 experiment or test, or in
    any Phase 1-6 result file is then a hard test failure rather than something noticed later.

    This replaces the ad-hoc checksum step that Phases 5 and 6 ran by hand.

WHAT IS COVERED, AND WHY EACH GROUP
    robot_env/**                 the environment. tests/test_week4_env.py already asserts the
                                 legacy path is bit-identical; this covers the source too.
    evaluation/**                ShortestPathOracle is the SPL denominator of every prior
                                 result, and evaluate_week4 defines EVAL_SEED_BASE.
    dr_control/*.py              the frozen Phase 1-6 controller and perception stack.
    optional_navigation/*.py     the frozen Phase 5 planner.
    experiments/exp0..6*.py      the frozen experiments, so a result stays re-derivable.
    tests/test_dr_control*.py    the frozen gates themselves.
    tests/test_week4_env.py      the environment gate.
    results/phase1..6/**         the frozen RESULTS. Requirement: no result file may be
                                 modified. Hashing them is how that is enforced.
    config.json                  the environment configuration every phase was run under.

WHAT IS DELIBERATELY NOT COVERED
    requirements.txt  -- Phase 7 may legitimately add a dependency, and a manifest entry that
                         is expected to change teaches people to ignore manifest failures.
    *.log             -- console transcripts of the phase gates. They are gitignored by repo
                         convention and therefore absent from a fresh clone, so hashing them
                         would make the manifest fail for a reason that has nothing to do with
                         drift. The manifest covers VERSION-CONTROLLED artefacts only; the
                         JSON result files, which carry the same numbers, are covered.
    Phase 7's own files -- they are under active development by definition. Modules that
                         live in dr_control/ are listed EXPLICITLY in PHASE7_PATHS; Phase 7
                         experiments and tests carry "exp7_"/"phase7" in the basename.
    __pycache__, .venv, models/, and anything git ignores.

USAGE
    python -m experiments.exp7_0_frozen_manifest --write     # generate (once, at Stage 0)
    python -m experiments.exp7_0_frozen_manifest --verify    # check; exit 1 on drift
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "MD_files" / "week5" / "phase7" / "FROZEN_MANIFEST.sha256"
LABEL = "Week5-Phase7 / Stage 0 / frozen-file manifest"

#: (glob, description). Order only affects the report; the manifest itself is sorted.
FROZEN_GLOBS = [
    ("robot_env/**/*.py", "environment"),
    ("evaluation/**/*.py", "evaluation and SPL oracle"),
    ("dr_control/*.py", "frozen Phase 1-6 controller stack"),
    ("optional_navigation/*.py", "frozen Phase 5 planner"),
    ("experiments/exp0_analytic.py", "Phase 1 experiment"),
    ("experiments/exp0_5_infeasibility.py", "Phase 1.5 experiment"),
    ("experiments/exp1_lidar_static.py", "Phase 2 experiment"),
    ("experiments/exp3_dynamic_oracle.py", "Phase 3 experiment"),
    ("experiments/exp4_dynamic_estimated.py", "Phase 4 experiment"),
    ("experiments/exp5_astar_waypoint.py", "Phase 5 experiment"),
    ("experiments/exp6_ppo_comparison.py", "Phase 6 experiment"),
    ("tests/test_dr_control*.py", "frozen Phase 1-6 gates"),
    ("tests/test_week4_env.py", "environment gate"),
    ("results/phase1/**/*", "Phase 1 and 1.5 results"),
    ("results/phase2/**/*", "Phase 2 results"),
    ("results/phase3/**/*", "Phase 3 results"),
    ("results/phase4/**/*", "Phase 4 results"),
    ("results/phase5/**/*", "Phase 5 results"),
    ("results/phase6/**/*", "Phase 6 results"),
    ("config.json", "environment configuration"),
]

EXCLUDE_PARTS = {"__pycache__"}
EXCLUDE_SUFFIXES = {".log", ".pyc"}
#: Phase 7 lives in the same directories as the frozen code and is under active development,
#: so its files are excluded -- but EXPLICITLY, by the path list the plan's file layout (§2.1)
#: declares, not by a name heuristic. An explicit list is the safer construction: a marker rule
#: such as "any file with phase7 in the name" would silently exempt a frozen file that someone
#: renamed, which is exactly the drift this manifest exists to catch.
#:
#: Adding a path here is a deliberate act that shows up in review. It was needed once already:
#: the gate's first live run flagged dr_control/fast_drccp.py, because the Stage 0 marker
#: heuristic disagreed with the file names the approved plan specifies.
PHASE7_PATHS = {
    "dr_control/fast_drccp.py",        # Stage 1, Direction 2
    "dr_control/recovery.py",          # Stage 4, Direction 1
    "dr_control/predictive_cbf.py",    # Stage 3, Direction 3a
    "dr_control/tracking2.py",         # Stage 5, Direction 3b/c
    "dr_control/uncertainty.py",       # Stage 6, Direction 3d
    "dr_control/policy_phase7.py",     # composed policy
}
#: Kept as a convenience for the experiment and test files, which DO carry the marker.
EXCLUDE_MARKERS = ("phase7", "exp7_")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_files():
    """Every frozen path, relative to the repo root, sorted and de-duplicated."""
    out = set()
    for pattern, _ in FROZEN_GLOBS:
        for path in REPO.glob(pattern):
            if not path.is_file():
                continue
            if EXCLUDE_PARTS & set(path.parts) or path.suffix in EXCLUDE_SUFFIXES:
                continue
            rel = path.relative_to(REPO).as_posix()
            if rel in PHASE7_PATHS or any(m in path.name for m in EXCLUDE_MARKERS):
                continue
            out.add(path.relative_to(REPO).as_posix())
    return sorted(out)


def compute():
    return {rel: _sha256(REPO / rel) for rel in frozen_files()}


def write():
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    digests = compute()
    lines = [
        f"# {LABEL}",
        "# sha256 of every Phase 1-6 file that Phase 7 must not modify.",
        "# Verified by tests/test_frozen_phase1_6.py on every test run.",
        "# Regenerate ONLY with an explicit, reviewed reason -- a diff here is the point.",
        f"# files: {len(digests)}",
        "",
    ]
    lines += [f"{d}  {rel}" for rel, d in digests.items()]
    MANIFEST.write_text("\n".join(lines) + "\n")
    return digests


def read_manifest():
    """{relpath: sha256} from the manifest file."""
    if not MANIFEST.exists():
        raise FileNotFoundError(f"missing frozen manifest: {MANIFEST}")
    out = {}
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, rel = line.split("  ", 1)
        out[rel] = digest
    return out


def verify():
    """(ok, modified, missing, untracked). Pure; raises nothing on drift."""
    expected = read_manifest()
    present = set(frozen_files())
    modified, missing = [], []
    for rel, digest in expected.items():
        path = REPO / rel
        if not path.exists():
            missing.append(rel)
        elif _sha256(path) != digest:
            modified.append(rel)
    untracked = sorted(present - set(expected))
    ok = not (modified or missing or untracked)
    return ok, sorted(modified), sorted(missing), untracked


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    print(f"### {LABEL}\n")
    if args.write:
        digests = write()
        print(f"hashed {len(digests)} frozen files -> {MANIFEST.relative_to(REPO)}")
        for pattern, desc in FROZEN_GLOBS:
            n = sum(1 for rel in digests if Path(rel).match(pattern)
                    or rel.startswith(pattern.split("*")[0]))
            print(f"   {desc:38s} {pattern}")
        return 0

    ok, modified, missing, untracked = verify()
    print(f"manifest: {len(read_manifest())} files")
    for name, items in (("MODIFIED", modified), ("MISSING", missing),
                        ("UNTRACKED (new frozen-area file)", untracked)):
        if items:
            print(f"\n{name}:")
            for rel in items:
                print(f"   {rel}")
    print("\nPASS -- every frozen file matches" if ok else "\nFAIL -- frozen area has drifted")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
