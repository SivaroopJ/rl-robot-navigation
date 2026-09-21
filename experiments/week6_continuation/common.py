"""Shared plumbing for the continuation stage scripts: run, persist, verify, print."""
from __future__ import annotations

import hashlib
import json
import time
import warnings
from pathlib import Path

from continuation.harness import RESULTS, manifest_rows, run_block, write_new
from continuation.stats import summaries

RULES = Path("MD_files/week6_continuation/SELECTION_RULES.md")
RULES_HASH = RESULTS / "selection_rules.sha256"

warnings.filterwarnings("ignore", message="Solution may be inaccurate")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_rules():
    want = RULES_HASH.read_text().split()[0]
    have = sha256(RULES)
    if want != have:
        raise SystemExit(f"SELECTION_RULES.md changed after it was hashed ({have} != {want})")
    return have


def verify_gate():
    g = json.loads((RESULTS / "gate" / "gate.json").read_text())
    if not g.get("GATE_PASS"):
        raise SystemExit("reference-equivalence gate has not passed; stop")


def run_stage(stage, arms, seeds, *, workers=8, tag=None, progress=200):
    verify_gate()
    verify_rules()
    out_dir = RESULTS / stage
    name = tag or stage
    for f in (f"{name}_records.json", f"{name}_summary.json", f"{name}_manifest.jsonl"):
        if (out_dir / f).exists():
            raise SystemExit(f"{out_dir / f} exists; never overwritten")
    t0 = time.time()
    print(f"[{stage}] {len(arms)} arms x {len(seeds)} seeds x 2 conditions = "
          f"{len(arms) * len(seeds) * 2} episodes, {workers} workers", flush=True)
    ck = out_dir / f"{name}_partial.jsonl"
    res = run_block(arms, seeds, workers=workers, progress=progress, checkpoint=ck)
    wall = time.time() - t0
    summ = {a.name: summaries(res[a.name], a.name) for a in arms}
    meta = {"stage": stage, "tag": name, "seeds": [seeds[0], seeds[-1]], "n_seeds": len(seeds),
            "conditions": ["fixed", "randomized"], "arms": [a.describe() for a in arms],
            "wall_s_this_invocation": wall, "rules_sha256": sha256(RULES)}
    write_new(out_dir / f"{name}_records.json", {"meta": meta, "records": res})
    write_new(out_dir / f"{name}_summary.json", {"meta": meta, "summary": summ})
    write_new(out_dir / f"{name}_manifest.jsonl", manifest_rows(res, arms, stage))
    ck.unlink()                       # the full records file now supersedes the checkpoint
    print(f"[{stage}] done in {wall:.0f}s", flush=True)
    return res, summ, meta


KEYS = [("success_rate", "succ"), ("collision_rate", "coll"), ("dynamic_collision_rate", "dyn"),
        ("static_collision_rate", "stat"), ("wall_collision_rate", "wall"),
        ("timeout_rate", "tout"), ("min_clearance_mean", "clr"), ("mean_spl", "spl"),
        ("infeasible_step_rate", "inf/st"), ("infeasible_events_per_episode", "ev/ep"),
        ("P_collision_given_infeasibility", "P(c|inf)"), ("recovery_success_rate", "recOK"),
        ("recovery_fraction", "recfrac"), ("step_time_ms_mean_ep", "ms"),
        ("step_time_ms_p95_ep", "p95ms")]


def table(summ, names=None, conds=("fixed", "randomized", "pooled"), keys=KEYS):
    names = names or list(summ)
    lines = []
    head = f"| arm | cond | " + " | ".join(k[1] for k in keys) + " |"
    lines += [head, "|" + "---|" * (len(keys) + 2)]
    for n in names:
        for c in conds:
            s = summ[n][c]
            cells = []
            for k, _ in keys:
                v = s.get(k)
                cells.append("—" if v is None or v != v else
                             (f"{v:.3f}" if isinstance(v, float) else str(v)))
            lines.append(f"| {n} | {c} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_json(p):
    return json.loads(Path(p).read_text())


# --------------------------------------------------------------------------- frozen configs
def _load_frozen(stem):
    p = RESULTS / stem
    want = (p.with_suffix(".sha256")).read_text().split()[0]
    if sha256(p) != want:
        raise SystemExit(f"{p} changed after it was frozen")
    return load_json(p)


def load_tuned():
    from continuation.params import DRCBFParams
    f = _load_frozen("H2/tuned_frozen.json")
    return DRCBFParams.from_dict(f["params"]), f


def load_random():
    from continuation.random_recovery import RandomSpec
    f = _load_frozen("R3/random_frozen.json")
    s = f["spec"]
    return RandomSpec(K=s["K"], H=s["H"], speeds=tuple(s["speeds"]), accept_m=s["accept_m"],
                      name="Random-CLF-DR-CBF"), f


def print_paired(label, cmp, keys=("success", "collision", "dynamic_collision", "timeout")):
    for c in ("fixed", "randomized", "pooled"):
        for k in keys:
            v = cmp[c][k]
            print(f"  {label} {c:10s} {k:18s} diff {v['diff']:+.3f} "
                  f"CI [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] n10={v['only_a']} "
                  f"n01={v['only_b']} p={v['p']:.4f}")
