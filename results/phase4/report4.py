"""Full Phase 4 breakdown: where the projected error lives, and why estimated != oracle."""
import json, sys
import numpy as np

def cellrow(c):
    if not c or not c.get("n"): return f"{c.get('cell','?'):26s} n=0"
    return (f"{c['cell']:26s} n={c['n']:>7d}  mean={c['mean']:+.4f}  P(e>0)={c['p_optimistic']:.3f}"
            f"  p50={c['p50']:+.4f} p90={c['p90']:+.4f} p95={c['p95']:+.4f} p99={c['p99']:+.4f}"
            f"  max={c['max']:+.4f}")

for path in sys.argv[1:]:
    d = json.load(open(path))
    print(f"\n{'='*100}\n### {path}   motion={d['config']['motion']}  episodes={d['config']['episodes']}")
    for arm in d["arms"]:
        s = d["arms"][arm]["summary"]
        err = s.get("error", {})
        if arm == "oracle":     # e == 0 by construction; nothing to break down
            print(f"\n--- {arm}: e == 0 by construction (n={err.get('overall',{}).get('n')})")
            continue
        print(f"\n--- {arm} : projected error e = dh_dt_est - dh_dt_true")
        print("  " + cellrow(err.get("overall", {})))
        print("  " + cellrow(err.get("headline_optimistic", {})) + "   <-- HEADLINE")
        for grp in ("by_clearance", "by_binding", "by_critical", "by_ttc", "by_since_bounce"):
            print(f"  [{grp}]")
            for c in err.get(grp, []):
                print("    " + cellrow(c))
        cf = s.get("confirmation", {})
        if cf:
            print("  [confirmation latency]")
            for k, v in cf.items():
                print(f"    {k:48s} {v}")
    if "paired" in d:
        print("\n  paired:")
        for k, v in d["paired"].items():
            sig = "SIGNIFICANT" if v["p"] < 0.05 else "NOT RESOLVED"
            print(f"    {k}: diff={v['diff']:+.4f} CI95=[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] "
                  f"McNemar p={v['p']:.5f}  -> {sig}")
