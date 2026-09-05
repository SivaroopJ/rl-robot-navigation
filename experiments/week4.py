"""Week 4 orchestration: the four training arms, the paired evaluation, and zero-shot.

THE THREE EXPERIMENTS, AND THE ONE SUB-EXPERIMENT
-------------------------------------------------
  Experiment 1  PPO vs PPO+SR on the ORIGINAL constant-velocity obstacles.
  Experiment 2  PPO TRAINED ON randomized obstacles.
  Experiment 3  PPO+SR TRAINED ON randomized obstacles.
  Zero-shot     The Experiment 1 models, evaluated on randomized obstacles, NOT retrained.

The distinction between Experiment 2/3 and the zero-shot analysis is the one most easily
lost: Experiments 2 and 3 train under stochasticity, and answer "does SR help when the
training environment itself is stochastic?". The zero-shot analysis trains under
determinism and only tests under stochasticity, answering the different question "does SR
transfer better?". They are reported separately and never merged into one table.

WHY EVERY ARM IS EVALUATED BY THE SAME FUNCTION
-----------------------------------------------
`evaluate_model` takes `randomize` (which environment to test on) independently of how the
model was trained, so the zero-shot analysis is not a separate code path that could drift
from the main one -- it is the same evaluation with a different argument.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os

import numpy as np

from evaluation.evaluate_week4 import evaluate_model, paired_comparison
from training.train_flat import ARMS, train_arm

RESULTS_ROOT = "results/week4"
DEFAULT_SEEDS = (0, 1, 2)


def model_path_for(arm, seed):
    return os.path.join("models", "week4", f"{arm}_seed{seed}", "final_model.zip")


def eval_dir_for(arm):
    spec = ARMS[arm]
    return os.path.join(RESULTS_ROOT, spec["experiment"], spec["label"])


# --------------------------------------------------------------------------- training
def run_training(arms, seeds, timesteps, epsilon, **kwargs):
    for arm, seed in itertools.product(arms, seeds):
        print(f"\n=== training {arm} seed {seed} ===", flush=True)
        train_arm(arm, seed, total_timesteps=timesteps, epsilon=epsilon,
                  results_root=RESULTS_ROOT, **kwargs)


# ------------------------------------------------------------------------ evaluation
def run_evaluation(arms, seeds, *, n_eval_episodes=200, n_obstacles=6,
                   obstacle_speed=1.0, zero_shot=False):
    """Evaluate each arm/seed on its OWN training environment, or zero-shot on randomized."""
    records = {}
    for arm, seed in itertools.product(arms, seeds):
        spec = ARMS[arm]
        path = model_path_for(arm, seed)
        if not os.path.exists(path):
            print(f"  skipping {arm} seed {seed}: no model at {path}")
            continue

        # Zero-shot means: take a model trained on FIXED obstacles and test it on
        # randomized ones. It is only meaningful for the Experiment 1 arms.
        if zero_shot:
            if spec["randomize"]:
                continue
            evaluate_on_randomized = True
            out_dir = os.path.join(RESULTS_ROOT, "zero_shot",
                                   f"{spec['label']}_fixed_to_random")
        else:
            evaluate_on_randomized = spec["randomize"]
            out_dir = eval_dir_for(arm)

        out_path = os.path.join(out_dir, f"eval_seed{seed}.json")
        print(f"  evaluating {arm} seed {seed} on "
              f"{'randomized' if evaluate_on_randomized else 'fixed'} obstacles", flush=True)
        records[(arm, seed)] = evaluate_model(
            path, sr=spec["sr"], randomize=evaluate_on_randomized,
            n_dynamic_obstacles=n_obstacles, obstacle_speed=obstacle_speed,
            n_eval_episodes=n_eval_episodes, out_path=out_path,
            label=f"{arm}_seed{seed}" + ("_zeroshot" if zero_shot else ""))
    return records


# --------------------------------------------------------------------------- analysis
def aggregate(paths):
    """Mean and standard deviation across seeds for one arm.

    Reported as mean +- std over SEEDS, not over episodes: the episode-level spread
    describes scenario difficulty, which is identical across arms by construction and so
    says nothing about the policy. Seed spread is what a claim about an algorithm needs.
    """
    records = [json.load(open(p)) for p in sorted(paths)]
    if not records:
        return None
    keys = ["success_rate", "collision_rate", "timeout_rate", "avg_episode_length",
            "avg_path_length", "mean_return", "avg_final_distance", "mean_spl"]
    out = {"n_seeds": len(records), "seeds": [r["label"] for r in records]}
    for key in keys:
        values = [r[key] for r in records if r.get(key) is not None]
        out[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))} if values else None
    for key in ("wall_collisions", "static_collisions", "dynamic_collisions"):
        out[key] = {"mean": float(np.mean([r[key] for r in records])),
                    "std": float(np.std([r[key] for r in records]))}
    return out


def build_result_table(out_path=os.path.join(RESULTS_ROOT, "result_table.json")):
    """The six-row summary table, with the zero-shot rows explicitly labelled."""
    rows = []
    for arm in ("ppo_fixed", "ppo_sr_fixed", "ppo_random", "ppo_sr_random"):
        spec = ARMS[arm]
        stats = aggregate(glob.glob(os.path.join(eval_dir_for(arm), "eval_seed*.json")))
        if stats:
            rows.append({"model": spec["label"], "trained_on": "randomized" if spec["randomize"] else "fixed",
                         "evaluated_on": "randomized" if spec["randomize"] else "fixed",
                         "zero_shot": False, **stats})
    for label in ("ppo", "ppo_sr"):
        stats = aggregate(glob.glob(os.path.join(
            RESULTS_ROOT, "zero_shot", f"{label}_fixed_to_random", "eval_seed*.json")))
        if stats:
            rows.append({"model": f"{label}_fixed", "trained_on": "fixed",
                         "evaluated_on": "randomized", "zero_shot": True, **stats})

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(rows, handle, indent=2)
    return rows


def build_paired_comparisons(seeds=DEFAULT_SEEDS,
                             out_path=os.path.join(RESULTS_ROOT, "paired_comparisons.json")):
    """PPO vs PPO+SR, matched episode by episode, per seed.

    The primary Week 4 comparison is Experiment 2 vs Experiment 3; the Experiment 1 and
    zero-shot pairs are reported alongside but answer different questions.
    """
    comparisons = {}
    pairings = {
        "experiment1_fixed": ("ppo_fixed", "ppo_sr_fixed", eval_dir_for),
        "primary_randomized": ("ppo_random", "ppo_sr_random", eval_dir_for),
    }
    for name, (arm_a, arm_b, resolver) in pairings.items():
        per_seed = []
        for seed in seeds:
            path_a = os.path.join(resolver(arm_a), f"eval_seed{seed}.json")
            path_b = os.path.join(resolver(arm_b), f"eval_seed{seed}.json")
            if os.path.exists(path_a) and os.path.exists(path_b):
                per_seed.append({"seed": seed, **paired_comparison(
                    json.load(open(path_a)), json.load(open(path_b)))})
        if per_seed:
            comparisons[name] = per_seed

    zs = []
    for seed in seeds:
        a = os.path.join(RESULTS_ROOT, "zero_shot", "ppo_fixed_to_random", f"eval_seed{seed}.json")
        b = os.path.join(RESULTS_ROOT, "zero_shot", "ppo_sr_fixed_to_random", f"eval_seed{seed}.json")
        if os.path.exists(a) and os.path.exists(b):
            zs.append({"seed": seed, **paired_comparison(json.load(open(a)), json.load(open(b)))})
    if zs:
        comparisons["zero_shot"] = zs

    with open(out_path, "w") as handle:
        json.dump(comparisons, handle, indent=2)
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "evaluate", "zero-shot", "table"])
    parser.add_argument("--arms", nargs="+", default=sorted(ARMS), choices=sorted(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--n-obstacles", type=int, default=6)
    parser.add_argument("--obstacle-speed", type=float, default=1.0)
    args = parser.parse_args()

    if args.command == "train":
        run_training(args.arms, args.seeds, args.timesteps, args.epsilon,
                     n_dynamic_obstacles=args.n_obstacles, obstacle_speed=args.obstacle_speed)
    elif args.command == "evaluate":
        run_evaluation(args.arms, args.seeds, n_eval_episodes=args.episodes,
                       n_obstacles=args.n_obstacles, obstacle_speed=args.obstacle_speed)
    elif args.command == "zero-shot":
        run_evaluation(["ppo_fixed", "ppo_sr_fixed"], args.seeds,
                       n_eval_episodes=args.episodes, n_obstacles=args.n_obstacles,
                       obstacle_speed=args.obstacle_speed, zero_shot=True)
    elif args.command == "table":
        print(json.dumps(build_result_table(), indent=2))
        build_paired_comparisons(args.seeds)


if __name__ == "__main__":
    main()
