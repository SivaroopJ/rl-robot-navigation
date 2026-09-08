"""Harvest reproduced evaluations and emit the REPRODUCTION_RESULTS.md comparison tables.

Run from the repository root AFTER main.py completes and after reproduced JSONs have been
copied into reproduction_results/evaluations/. Reads nothing from the live results/ tree.
"""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # reproduction_results/
REF  = json.load(open(os.path.join(ROOT, "configs", "reported_reference.json")))
REPRO_DIR = os.path.join(ROOT, "evaluations")

LABELS = {
    "e1": ("E1 — obstacle density (speed = 1.0)", {
        "obs3_spd1.0": "N=3", "obs6_spd1.0": "N=6 (training condition)", "obs10_spd1.0": "N=10"}),
    "e2": ("E2 — obstacle speed (N = 6)", {
        "obs6_spd0.5": "speed=0.5", "obs6_spd1.0": "speed=1.0 (training condition)", "obs6_spd1.5": "speed=1.5"}),
    "e3": ("E3 — generalization to unseen configurations", {
        "obs5_spd0.8": "N=5, speed=0.8", "obs7_spd1": "N=7, speed=1.0", "obs8_spd1.2": "N=8, speed=1.2",
        "obs9_spd1.3": "N=9, speed=1.3", "obs10_spd1.5": "N=10, speed=1.5", "obs12_spd2": "N=12, speed=2.0",
        "obs15_spd0.3": "N=15, speed=0.3"}),
    "e4": ("E4 — reward-shaping ablation (N=6, speed=1.0)", {
        "obs6_spd1.0_shaping": "With shaping", "obs6_spd1.0_no_shaping": "Without shaping"}),
}

def load_repro(exp, key):
    p = os.path.join(REPRO_DIR, exp, f"eval_{key}.json")
    return json.load(open(p)) if os.path.exists(p) else None

def pp(x): return "—" if x is None else f"{x*100:.1f}%"
def dpp(a, b): return "—" if (a is None or b is None) else f"{(a-b)*100:+.1f}pp"

out, rows_all = [], []
for exp, (title, names) in LABELS.items():
    out.append(f"### {title}\n")
    out.append("| Configuration | Reported (JSON) | Reported (RESULTS.md) | Reproduced | Diff vs JSON | Diff vs prose | Repro collisions | Repro timeouts |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for key, label in names.items():
        ref = REF[exp][key]; r = load_repro(exp, key)
        rs = r["success_rate"] if r else None
        cr = (r["n_collision"]/r["n_episodes"]) if r else None
        tr = (r["n_truncated"]/r["n_episodes"]) if r else None
        out.append(f"| {label} | {pp(ref['json_success'])} | {pp(ref['prose_success'])} | **{pp(rs)}** | "
                   f"{dpp(rs, ref['json_success'])} | {dpp(rs, ref['prose_success'])} | {pp(cr)} | {pp(tr)} |")
        if rs is not None:
            rows_all.append((rs - ref["json_success"], rs - ref["prose_success"]))
    out.append("")

if rows_all:
    import statistics as st
    dj = [a for a, _ in rows_all]; dp_ = [b for _, b in rows_all]
    out.append("### Aggregate\n")
    out.append("| Metric | vs committed JSONs | vs RESULTS.md prose |")
    out.append("|---|---:|---:|")
    out.append(f"| Configurations compared | {len(dj)} | {len(dp_)} |")
    out.append(f"| Mean difference | {st.mean(dj)*100:+.1f}pp | {st.mean(dp_)*100:+.1f}pp |")
    out.append(f"| Median difference | {st.median(dj)*100:+.1f}pp | {st.median(dp_)*100:+.1f}pp |")
    out.append(f"| Min / Max | {min(dj)*100:+.1f} / {max(dj)*100:+.1f}pp | {min(dp_)*100:+.1f} / {max(dp_)*100:+.1f}pp |")
    out.append(f"| Mean |difference| | {st.mean(abs(x) for x in dj)*100:.1f}pp | {st.mean(abs(x) for x in dp_)*100:.1f}pp |")
    out.append("")

print("\n".join(out))
