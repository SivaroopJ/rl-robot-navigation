"""Blocks of RS episodes for one (arm, cell): parallel, checkpointed, resumable.

A NEW module: highdim/blocks.py is frozen (RS_DESIGN.md section 3). This is highdim.blocks.run_block
with the cell added, and it reuses that module's checkpoint and trace-sidecar helpers unchanged:
the fingerprinted .jsonl checkpoint with torn-line tolerant resume, the trace sidecar written
BEFORE the checkpoint row (an episode is done only when both exist), and the deterministic gz
(sorted rows, gzip mtime 0) written when the block completes. Sidecar rows are keyed by
("cond", seed), where "cond" is the motion condition.

The fingerprint adds the cell and the generator version to highdim's (arm, seeds, motions,
max_steps, tag), so a checkpoint is refused by any other cell or any other generator. A
candidate's configuration joins it too (phase 4); the baseline's fingerprint is unchanged.

The checkpoint, trace and gz helpers are highdim.blocks' private functions, imported rather than
copied: the manifest hash-freezes them, so they cannot drift under this module.

Seeds are passed in, never block names: only rs6_final.py opens RS_FINAL (robustsuite.seeds).
"""
from __future__ import annotations

import hashlib
import json
import time
from multiprocessing import get_context
from pathlib import Path

from continuation.harness import _js
from evaluation.shortest_path import ShortestPathOracle
from highdim.blocks import _finalise_traces, _load_traces, _partial, _read_jsonl
from robustsuite import scenario as SC
from robustsuite import seeds as RS
from robustsuite.harness import check_cell, check_config, run_episode
from robustsuite.scenario_env import RSScenarioEnv

#: Kept out of the checkpoint rows and written to the trace sidecar. The spec is regenerable
#: from (cell, seed) and is the bulk of a record.
HEAVY = ("trace", "trajectory", "spec")
_W = {}          # per-process cache: motion -> (env, env default MAX_STEPS)
_M0_ORACLE = []  # per-process cache: the anchor's SPL oracle (M0's map never changes)


def _m0_oracle(env):
    if not _M0_ORACLE:
        _M0_ORACLE.append(ShortestPathOracle(env.WORLD_SIZE, env.AGENT_RADIUS, env.m0_static))
    return _M0_ORACLE[0]


def _worker(job):
    arm, cell, motion, seed, max_steps, config = job
    if motion not in _W:
        env = RSScenarioEnv(motion)
        _W[motion] = (env, env.MAX_STEPS)
    env, default_steps = _W[motion]
    env.MAX_STEPS = default_steps if max_steps is None else int(max_steps)
    oracle = _m0_oracle(env) if cell == RS.ANCHOR else None
    return motion, seed, run_episode(arm, cell, motion, seed, env=env, oracle=oracle,
                                     config=config)


def fingerprint(arm, cell, seeds, motions, max_steps, tag, config=None):
    d = {"arm": arm, "cell": list(cell), "seeds": list(seeds), "motions": list(motions),
         "max_steps": max_steps, "tag": tag, "generator_version": SC.GENERATOR_VERSION}
    if config is not None:                  # the baseline's fingerprint is unchanged by phase 4
        d["config"] = check_config(arm, config).name
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


def run_block(arm, cell, seeds, *, motions=RS.MOTIONS, workers=8, checkpoint, traces, tag="",
              progress=None, max_steps=None, maxtasksperchild=40, config=None):
    """{motion: [light records sorted by seed]} for one arm on one cell.

    A rerun with the IDENTICAL fingerprint resumes, skipping finished episodes; any other
    definition refuses the checkpoint with SystemExit. `tag` is the code identity (the entry
    point passes the git commit), so a resume after a code change is refused.
    """
    check_cell(cell)
    checkpoint, traces = Path(checkpoint), Path(traces)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    check_config(arm, config)
    fp = fingerprint(arm, cell, seeds, motions, max_steps, tag, config)
    rows = _read_jsonl(checkpoint)
    if rows:
        if rows[0].get("fingerprint") != fp:
            raise SystemExit(f"{checkpoint} belongs to a different block definition")
    else:                                         # new, or empty after a kill at creation
        checkpoint.write_text(json.dumps({"fingerprint": fp, "arm": arm, "cell": list(cell)})
                              + "\n")
    trace_rows = _load_traces(traces)
    out = {m: {} for m in motions}
    for row in rows[1:]:
        if (row["cond"], row["seed"]) in trace_rows:      # done = checkpoint AND trace
            out[row["cond"]][row["seed"]] = row["rec"]
    if rows[1:]:
        print(f"  resuming: {sum(len(v) for v in out.values())} episodes already in "
              f"{checkpoint}", flush=True)

    jobs = [(arm, cell, m, s, max_steps, config)
            for m in motions for s in seeds if s not in out[m]]
    t0 = time.time()
    pool = None
    if workers <= 1:
        it = map(_worker, jobs)
    else:
        pool = get_context("fork").Pool(workers, maxtasksperchild=maxtasksperchild)
        it = pool.imap_unordered(_worker, jobs, chunksize=1)
    try:
        with open(checkpoint, "a") as ck, open(_partial(traces), "a") as tr:
            for i, (motion, seed, rec) in enumerate(it, 1):
                heavy = {k: rec.pop(k) for k in HEAVY}
                row = {"cond": motion, "seed": seed, **heavy}
                tr.write(json.dumps(row, default=_js) + "\n")
                tr.flush()
                ck.write(json.dumps({"cond": motion, "seed": seed, "rec": rec},
                                    default=_js) + "\n")
                ck.flush()
                out[motion][seed] = rec
                trace_rows[(motion, seed)] = row
                if progress and (i % progress == 0 or i == len(jobs)):
                    print(f"  {arm} {cell.family}/{cell.obstacles}: {i}/{len(jobs)} episodes, "
                          f"{time.time() - t0:.0f}s", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    _finalise_traces(traces, [json.loads(json.dumps(trace_rows[(m, s)], default=_js))
                              for m in motions for s in seeds])
    return {m: [out[m][s] for s in seeds] for m in motions}


def load_block(arm, cell, seeds, *, checkpoint, traces, tag, motions=RS.MOTIONS,
               max_steps=None, config=None):
    """A finished block read back without running or rewriting anything.

    Returns ({motion: [light records sorted by seed]}, {(motion, seed): trace row}).
    SystemExit if the stored fingerprint is not this definition's or an episode is missing."""
    checkpoint, traces = Path(checkpoint), Path(traces)
    rows = _read_jsonl(checkpoint)
    fp = fingerprint(arm, cell, seeds, motions, max_steps, tag, config)
    if not rows or rows[0].get("fingerprint") != fp:
        raise SystemExit(f"{checkpoint} is not the block {arm!r} {tuple(cell)} with tag {tag!r}")
    trace_rows = _load_traces(traces)
    got = {(r["cond"], r["seed"]): r["rec"] for r in rows[1:]}
    missing = [(m, s) for m in motions for s in seeds
               if (m, s) not in got or (m, s) not in trace_rows]
    if missing:
        raise SystemExit(f"{checkpoint}: {len(missing)} episodes missing, e.g. {missing[:3]}")
    return ({m: [got[(m, s)] for s in seeds] for m in motions},
            {(m, s): trace_rows[(m, s)] for m in motions for s in seeds})
