"""Render a Week 4 results root as a readable summary.

Written so the overnight pipeline leaves something legible behind without anyone having to
reconstruct it from training logs. Reports mean +- std across SEEDS, never across episodes:
evaluation episodes are identical across arms by construction, so their spread measures
scenario difficulty and says nothing about the policy.

Usage: python week4_summary.py results/week4_speed06_075
"""
import glob
import json
import os
import sys

import numpy as np

ARMS = [("PPO            trained fixed  -> eval fixed",  "experiment1/ppo"),
        ("PPO + SR       trained fixed  -> eval fixed",  "experiment1/ppo_sr"),
        ("PPO            trained random -> eval random", "experiment2/ppo_randomized"),
        ("PPO + SR       trained random -> eval random", "experiment3/ppo_sr_randomized"),
        ("PPO    (ZERO-SHOT) fixed -> random",           "zero_shot/ppo_fixed_to_random"),
        ("PPO+SR (ZERO-SHOT) fixed -> random",           "zero_shot/ppo_sr_fixed_to_random")]


def load(root, sub):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(root, sub, "eval_seed*.json")))]


def ms(values):
    v = [x for x in values if x is not None]
    return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan"))


def main(root):
    print("=" * 100)
    print("WEEK 4 RESULTS  --  %s" % root)
    print("=" * 100)

    cfgs = sorted(glob.glob(os.path.join(root, "**", "run_config.json"), recursive=True))
    if cfgs:
        c = json.load(open(cfgs[0]))
        print("\nCONFIGURATION")
        print("  timesteps/run %-12s obstacles %-4s speed_range %s"
              % (c["total_timesteps"], c["n_dynamic_obstacles"], c.get("obstacle_speed_range")))
        print("  dt %-6s start-goal bands %s" % (c["dt"], c["start_goal"]["distance_bins"]))
        print("  git %s on %s" % (c["git_commit"][:10], c["git_branch"]))
    for name in ("randomized", "fixed"):
        p = os.path.join(root, "calibration", "epsilon_%s.json" % name)
        if os.path.exists(p):
            k = json.load(open(p))
            print("  epsilon (%s) %.4f   [rho p50 under an untrained policy, %d pairs]"
                  % (name, k["recommended_epsilon"], k["n_pairs"]))

    print("\nMAIN TABLE  (200 eval episodes/seed, deterministic, mean +- std ACROSS SEEDS)\n")
    print("  %-46s %-15s %-15s %-13s %s" % ("arm", "success", "collision", "SPL", "seeds"))
    print("  " + "-" * 96)
    for label, sub in ARMS:
        rs = load(root, sub)
        if not rs:
            print("  %-46s %s" % (label, "(no results)"))
            continue
        s = ms([r["success_rate"] for r in rs])
        c = ms([r["collision_rate"] for r in rs])
        p = ms([r["mean_spl"] for r in rs])
        print("  %-46s %.3f +- %.3f   %.3f +- %.3f   %.3f+-%.3f   %d"
              % (label, s[0], s[1], c[0], c[1], p[0], p[1], len(rs)))

    print("\nPRIMARY COMPARISON  (RQ3: PPO vs PPO+SR, both TRAINED on randomized obstacles)")
    a, b = load(root, "experiment2/ppo_randomized"), load(root, "experiment3/ppo_sr_randomized")
    if a and b:
        sa, sb = ms([r["success_rate"] for r in a]), ms([r["success_rate"] for r in b])
        ca, cb = ms([r["collision_rate"] for r in a]), ms([r["collision_rate"] for r in b])
        print("    success   PPO %.3f +- %.3f   vs   PPO+SR %.3f +- %.3f   delta %+.3f"
              % (sa[0], sa[1], sb[0], sb[1], sb[0] - sa[0]))
        print("    collision PPO %.3f +- %.3f   vs   PPO+SR %.3f +- %.3f   delta %+.3f  <- primary metric"
              % (ca[0], ca[1], cb[0], cb[1], cb[0] - ca[0]))
        pooled = np.hypot(sa[1], sb[1])
        if pooled > 0 and abs(sb[0] - sa[0]) < pooled:
            print("    NOTE: the success difference is smaller than the seed spread. At three seeds")
            print("          this does not support a claim in either direction.")
    else:
        print("    (incomplete)")

    pc = os.path.join(root, "paired_comparisons.json")
    if os.path.exists(pc):
        print("\nPAIRED (McNemar) COUNTS -- episodes matched by seed, so scenario difficulty cancels")
        for key, seeds in json.load(open(pc)).items():
            for s in seeds:
                print("    %-22s seed %s: only-PPO %3d | only-SR %3d | both %3d | neither %3d"
                      % (key, s["seed"], s["only_a_success"], s["only_b_success"],
                         s["both_success"], s["neither_success"]))

    print("\nCOLLISION BREAKDOWN  (the research question is about DYNAMIC obstacles)\n")
    print("  %-46s %8s %8s %8s" % ("arm", "dynamic", "static", "wall"))
    for label, sub in ARMS:
        rs = load(root, sub)
        if not rs:
            continue
        d = sum(r["dynamic_collisions"] for r in rs)
        st = sum(r["static_collisions"] for r in rs)
        w = sum(r["wall_collisions"] for r in rs)
        tot = max(d + st + w, 1)
        print("  %-46s %8s %8d %8d" % (label, "%d (%.0f%%)" % (d, 100 * d / tot), st, w))

    print("\nSUCCESS BY START-GOAL DISTANCE BAND\n")
    print("  %-46s %s" % ("arm", "".join("%12s" % b for b in
                                         ["4.0-5.5", "5.5-7.0", "7.0-8.5", "8.5-10.5"])))
    for label, sub in ARMS:
        rs = load(root, sub)
        if not rs:
            continue
        cells = []
        for i in range(4):
            v = [r["by_distance_band"][str(i)]["success_rate"]
                 for r in rs if str(i) in r["by_distance_band"]]
            cells.append("%12s" % ("%.3f" % np.mean(v) if v else "-"))
        print("  %-46s %s" % (label, "".join(cells)))

    print("\nSR HEALTH  (a null result is only about SR if the acceptance rule was actually active)")
    for arm in ("ppo_sr_fixed", "ppo_sr_random"):
        for f in sorted(glob.glob(os.path.join(root, "launch", "%s_seed*.log" % arm))):
            txt = open(f).read()
            def last(k):
                import re
                m = re.findall(r"\| *%s *\| *([-0-9.e+]+)" % k, txt)
                return float(m[-1]) if m else float("nan")
            print("    %-26s accept_rate %.2f  rho_p50 %.2f  eps %.2f  approx_kl %.4f  std %.3f"
                  % (os.path.basename(f)[:-4], last("accept_rate_closer"), last("rho_p50"),
                     last("epsilon"), last("approx_kl"), last("std")))
    print("\n  Healthy: accept_rate away from 0 and 1, approx_kl ~0.02, std well above 0.1.")
    print("  accept_rate near 1 means SR degenerated into PPO and any null is about epsilon.")
    print("=" * 100)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/week4")
