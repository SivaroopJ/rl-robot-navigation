"""HD Experiment 1: train one PPO seed (highdim.train; HD_DESIGN.md section 9).

    python -m experiments.highdim.hd_train --run 0 --seed 0 --steps 1000000 --out models/highdim/pilot_s0
    python -m experiments.highdim.hd_train --smoke --out models/highdim/smoke   # ~20k steps

A real run refuses uncommitted code changes, so every checkpoint's recorded commit is the code
that produced it. --smoke is a plumbing check only: checkpoint and evaluation every 16 384
steps on 2 HD_DEV seeds, and its output is never a result.
"""
from __future__ import annotations

import argparse
import json

from highdim import train as HT
from highdim.seeds import SMOKE_RUN


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--hold", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=250_000)
    ap.add_argument("--eval-seeds", type=int, default=50)
    ap.add_argument("--eval-workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    if a.smoke:
        kw = dict(run=SMOKE_RUN, seed=0, total_timesteps=20_000, every=16_384, eval_seeds=2)
    else:
        if None in (a.run, a.seed, a.steps):
            raise SystemExit("--run, --seed and --steps are required (or --smoke)")
        if a.run == SMOKE_RUN:
            raise SystemExit(f"run {SMOKE_RUN} is reserved for smoke runs")
        changes = HT.code_changes()
        if changes:
            raise SystemExit(f"refusing to train: uncommitted code changes\n{changes}")
        kw = dict(run=a.run, seed=a.seed, total_timesteps=a.steps, every=a.every,
                  eval_seeds=a.eval_seeds)
    final = HT.train(out_dir=a.out, hold=a.hold, eval_workers=a.eval_workers, **kw)
    print(json.dumps({k: final[k] for k in ("steps", "solver", "hold", "wall_s", "eval_s",
                                            "steps_per_s")}, indent=1))
    return final


if __name__ == "__main__":
    main()
