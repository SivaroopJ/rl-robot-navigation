"""SHA-256 snapshot of everything RS Experiment 1 treats as FROZEN (RS_DESIGN.md section 3).

Candidates may change planning, parameters and perception only by runtime composition around the
frozen stack; the CBF/DR formulation and the dynamics never change. No file below may change: the
canonical environment (dynamics, collision, LiDAR, observation, reward, the M0 motion models), the
A* planner and carrot follower, the CLF-DR-CBF controller and its LiDAR pipeline, Random recovery
and its frozen parameters, the Continuation modules the baseline arm uses, the HD modules that
build the frozen astar_random arm and run its episodes, and the exp6 helpers they reuse.

`--write` creates the manifest ONCE (refuses to overwrite); tests/test_robustsuite.py verifies it
on every run. Same pattern as experiments/highdim/protect_manifest.py.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "MD_files/robustsuite/PROTECTED_MANIFEST.sha256"
GLOBS = ["optional_navigation/*.py", "evaluation/shortest_path.py",
         "evaluation/evaluate_week4.py", "dr_control/*.py", "continuation/*.py",
         "robot_env/*.py", "config.json", "experiments/exp6_ppo_comparison.py",
         "highdim/policy.py", "highdim/harness.py", "highdim/blocks.py", "highdim/subgoal.py",
         "highdim/gates.py", "experiments/week6_continuation/common.py",
         # package __init__ files on the import path, and the reservations
         # robustsuite.seeds proves its blocks disjoint from
         "highdim/__init__.py", "highdim/seeds.py", "evaluation/__init__.py",
         "experiments/__init__.py", "experiments/week6_continuation/__init__.py",
         "continuation/pursuit/config.py",
         "results/week6_continuation/H2/tuned_frozen.json",
         "results/week6_continuation/H2/tuned_frozen.sha256",
         "results/week6_continuation/R3/random_frozen.json",
         "results/week6_continuation/R3/random_frozen.sha256"]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute(root=REPO):
    out = {}
    for g in GLOBS:
        for p in Path(root).glob(g):
            if p.is_file() and "__pycache__" not in p.parts:
                out[str(p.relative_to(root))] = _sha(p)
    return dict(sorted(out.items()))


def read():
    rows = {}
    for line in MANIFEST.read_text().splitlines():
        if line and not line.startswith("#"):
            h, rel = line.split("  ", 1)
            rows[rel] = h
    return rows


def verify(root=REPO):
    """(modified, missing) relative paths, each sorted. Both empty means untouched."""
    root = Path(root)
    modified, missing = [], []
    for rel, h in read().items():
        p = root / rel
        if not p.is_file():
            missing.append(rel)
        elif _sha(p) != h:
            modified.append(rel)
    return sorted(modified), sorted(missing)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if a.write:
        if MANIFEST.exists():
            raise SystemExit(f"{MANIFEST} exists; never regenerated")
        rows = compute()
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text("# RS Experiment 1 frozen-file manifest (sha256)\n"
                            + "".join(f"{h}  {k}\n" for k, h in rows.items()))
        print(f"wrote {len(rows)} entries")
    else:
        m, x = verify()
        print("modified:", m, "missing:", x)
