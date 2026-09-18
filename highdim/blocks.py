"""Blocks of paired HD episodes: parallel, checkpointed, resumable.

Mirrors continuation.harness.run_block (fingerprinted .jsonl checkpoint, torn-line tolerant
resume, fork workers recycled every `maxtasksperchild` episodes), for one HD arm per call.

LIGHT RECORDS AND TRACES. The checkpoint holds light records. The per-step `trace` and
`trajectory` (derived bulk) go to a sidecar:
  * while the block runs: `<traces>.partial.jsonl`, plain appended JSON lines (torn-line
    tolerant). The trace row is written BEFORE the checkpoint row, and an episode counts as done
    only when BOTH exist, so a kill between the two re-runs the episode;
  * when the block completes: rewritten as `traces` (.jsonl.gz), deduplicated, sorted by
    (condition, seed), with a fixed gzip header (mtime 0), so its sha256 depends only on the
    episodes, not on worker completion order or resume history. The partial file is removed.
Episodes are deterministic given the seed, so any trace can be rebuilt by re-running it.

Seeds are passed in, never block names: only the entry points open HD_FINAL (highdim.seeds).

`check=(solver, shadow)` runs the fast-solver suitability check's episodes
(highdim.harness.run_check_episode) instead of evaluation episodes; it is part of the fingerprint.
`model=<checkpoint path>` (with `hold`) drives the PPO arms; each worker loads the checkpoint
once, and the fingerprint carries the checkpoint's sha256 and the hold length.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import time
from multiprocessing import get_context
from pathlib import Path

from continuation.harness import _js
from evaluation.shortest_path import ShortestPathOracle
from experiments.exp6_ppo_comparison import make_env
from highdim.harness import run_check_episode, run_episode

HEAVY = ("trace", "trajectory")
_W = {}          # per-process cache: condition -> (env, oracle, env default MAX_STEPS)
_MODELS = {}     # per-process cache: checkpoint path -> loaded PPO model


def _load(path):
    if path not in _MODELS:
        from stable_baselines3 import PPO
        _MODELS[path] = PPO.load(path, device="cpu")
    return _MODELS[path]


def _worker(job):
    kind, cond, seed, max_steps, check, *drive = job
    model, hold = drive[0] if drive and drive[0] is not None else (None, 1)
    if cond not in _W:
        env = make_env(cond == "randomized")
        _W[cond] = (env, ShortestPathOracle(env.WORLD_SIZE, env.AGENT_RADIUS,
                                            env.static_obstacles), env.MAX_STEPS)
    env, oracle, default_steps = _W[cond]
    env.MAX_STEPS = default_steps if max_steps is None else int(max_steps)
    if check is None:
        if model is not None:
            return cond, seed, run_episode(kind, cond, seed, env=env, oracle=oracle,
                                           model=_load(model), hold=hold)
        return cond, seed, run_episode(kind, cond, seed, env=env, oracle=oracle)
    solver, shadow = check
    return cond, seed, run_check_episode(kind, cond, seed, env=env, oracle=oracle,
                                         solver=solver, shadow=shadow)


def fingerprint(kind, seeds, conditions, max_steps, tag, check=None, model=None, hold=1):
    d = {"kind": kind, "seeds": list(seeds), "conditions": list(conditions),
         "max_steps": max_steps, "tag": tag}
    if check is not None:                  # evaluation fingerprints are unchanged by `check`
        d["check"] = list(check)
    if model is not None:                  # ... and by `model` when there is none
        d["model_sha256"] = hashlib.sha256(Path(model).read_bytes()).hexdigest()
        d["hold"] = int(hold)
    blob = json.dumps(d, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _read_jsonl(path):
    rows = []
    if path.exists():
        for ln in path.read_text().splitlines():
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                continue                          # a torn line from a kill
    return rows


def _partial(traces):
    return traces.with_name(traces.name.replace(".jsonl.gz", "") + ".partial.jsonl")


def _load_traces(traces):
    """{(cond, seed): row} from the finished gz (if any) and the partial file; last wins."""
    got = {}
    if traces.exists():
        with gzip.open(traces, "rt") as f:
            for ln in f:
                r = json.loads(ln)
                got[(r["cond"], r["seed"])] = r
    for r in _read_jsonl(_partial(traces)):
        got[(r["cond"], r["seed"])] = r
    return got


def _finalise_traces(traces, rows):
    """Deterministic gz: sorted rows, sorted keys, gzip mtime 0."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        for r in rows:
            gz.write((json.dumps(r, sort_keys=True, default=_js) + "\n").encode())
    tmp = traces.with_name(traces.name + ".tmp")
    tmp.write_bytes(buf.getvalue())
    tmp.replace(traces)
    _partial(traces).unlink(missing_ok=True)


def run_block(kind, seeds, *, conditions=("fixed", "randomized"), workers=8, checkpoint,
              traces, tag="", progress=None, max_steps=None, maxtasksperchild=40, check=None,
              model=None, hold=1):
    """{condition: [light records sorted by seed]} for one arm.

    A rerun with the IDENTICAL fingerprint (kind, seeds, conditions, max_steps, tag, plus
    `check`, and the checkpoint sha256 and hold when `model` is given) resumes,
    skipping finished episodes; any other definition refuses the checkpoint with SystemExit.
    `tag` is the code identity (the entry point passes the git commit), so a resume after a code
    change is refused rather than mixing two versions in one block.
    """
    checkpoint, traces = Path(checkpoint), Path(traces)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if model is not None and check is not None:
        raise ValueError("the fast-solver check runs no PPO model")
    fp = fingerprint(kind, seeds, conditions, max_steps, tag, check, model, hold)
    drive = None if model is None else (str(model), int(hold))
    rows = _read_jsonl(checkpoint)
    if rows:
        if rows[0].get("fingerprint") != fp:
            raise SystemExit(f"{checkpoint} belongs to a different block definition")
    else:                                         # new, or empty after a kill at creation
        checkpoint.write_text(json.dumps({"fingerprint": fp, "kind": kind}) + "\n")
    trace_rows = _load_traces(traces)
    out = {c: {} for c in conditions}
    for row in rows[1:]:
        if (row["cond"], row["seed"]) in trace_rows:      # done = checkpoint AND trace
            out[row["cond"]][row["seed"]] = row["rec"]
    if rows[1:]:
        print(f"  resuming: {sum(len(v) for v in out.values())} episodes already in "
              f"{checkpoint}", flush=True)

    jobs = [(kind, c, s, max_steps, check, drive) for c in conditions for s in seeds if s not in out[c]]
    t0 = time.time()
    pool = None
    if workers <= 1:
        it = map(_worker, jobs)
    else:
        pool = get_context("fork").Pool(workers, maxtasksperchild=maxtasksperchild)
        it = pool.imap_unordered(_worker, jobs, chunksize=1)
    try:
        with open(checkpoint, "a") as ck, open(_partial(traces), "a") as tr:
            for i, (cond, seed, rec) in enumerate(it, 1):
                heavy = {k: rec.pop(k) for k in HEAVY}
                row = {"cond": cond, "seed": seed, **heavy}
                tr.write(json.dumps(row, default=_js) + "\n")
                tr.flush()
                ck.write(json.dumps({"cond": cond, "seed": seed, "rec": rec},
                                    default=_js) + "\n")
                ck.flush()
                out[cond][seed] = rec
                trace_rows[(cond, seed)] = row
                if progress and (i % progress == 0 or i == len(jobs)):
                    print(f"  {kind}: {i}/{len(jobs)} episodes, {time.time() - t0:.0f}s",
                          flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    _finalise_traces(traces, [json.loads(json.dumps(trace_rows[(c, s)], default=_js))
                              for c in conditions for s in seeds])
    return {c: [out[c][s] for s in seeds] for c in conditions}


def check_paired(res_a, res_b):
    """Same conditions, same seeds in the same order, same start and goal. RuntimeError if not."""
    if set(res_a) != set(res_b):
        raise RuntimeError("the two blocks cover different conditions")
    for c in res_a:
        if len(res_a[c]) != len(res_b[c]):
            raise RuntimeError(f"{c}: different episode counts")
        for a, b in zip(res_a[c], res_b[c]):
            if not (a["seed"] == b["seed"] and a["start"] == b["start"]
                    and a["goal"] == b["goal"]):
                raise RuntimeError(f"pairing broken at {c}/{a['seed']}")
