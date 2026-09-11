"""SHA-256 snapshot of the Week 5 / Week 5.5 artefacts the continuation must never touch (audit A11).

The Week 5.5 files are untracked in git and outside the Phase-7 frozen manifest, so without this
snapshot "untouched" could not be verified. `--write` creates the manifest ONCE (refuses to
overwrite); tests/test_week6_continuation.py verifies it on every run.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "MD_files/week6_continuation/PROTECTED_MANIFEST.sha256"
GLOBS = ["generalization/*.py", "experiments/exp9_*.py", "tests/test_generalization_maps.py",
         "results/week6_generalization/**/*", "MD_files/week6/**/*",
         "results/week5_phase7/**/*", "MD_files/week5/**/*",
         "experiments/exp7_*.py", "experiments/exp8_final_comparison.py",
         "dr_control/capped_velocity.py", "dr_control/policy_phase7.py",
         "dr_control/recovery.py", "dr_control/tracking2.py", "dr_control/fast_drccp.py"]


def compute():
    out = {}
    for g in GLOBS:
        for p in REPO.glob(g):
            if p.is_file() and "__pycache__" not in p.parts:
                out[str(p.relative_to(REPO))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return dict(sorted(out.items()))


def read():
    rows = {}
    for line in MANIFEST.read_text().splitlines():
        if line and not line.startswith("#"):
            h, rel = line.split("  ", 1)
            rows[rel] = h
    return rows


def verify():
    want, have = read(), compute()
    modified = [k for k in want if k in have and have[k] != want[k]]
    missing = [k for k in want if k not in have]
    return modified, missing


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if a.write:
        if MANIFEST.exists():
            raise SystemExit(f"{MANIFEST} exists; never regenerated")
        rows = compute()
        MANIFEST.write_text("# Week6-Continuation protected-file manifest (sha256)\n"
                            + "".join(f"{h}  {k}\n" for k, h in rows.items()))
        print(f"wrote {len(rows)} entries")
    else:
        m, x = verify()
        print("modified:", m, "missing:", x)
