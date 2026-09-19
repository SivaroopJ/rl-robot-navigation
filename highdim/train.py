"""PPO trainer for HD Experiment 1 (HD_DESIGN.md section 9).

    PPO          SB3, the config.json training hyperparameters (training.train_flat's
                 shared_ppo_kwargs: lr 3e-4 linear, n_steps 2048, batch 256, 10 epochs,
                 gamma 0.99, lambda 0.95, entropy 0.01, target KL 0.02), MLP [256, 256], CPU,
                 no observation normalisation
    environment  8 HDSubgoalEnv (highdim.wrapper): env index 0-3 fixed obstacle motion, 4-7
                 randomized (HD_DESIGN section 13); each first reset on its HD_TRAIN seed
                 train_env_seed(run, index). PPO(seed=s) re-seeds a vec env with s + index, so
                 build_model re-seeds it with the HD_TRAIN seeds after construction
    solver       the one the fast-solver check selected (results/highdim/HD1/analysis.json);
                 evaluation is always frozen SCS (highdim.harness)
    checkpoints  every `every` control steps (250k) plus the final model, each with a metadata
                 JSON: commit, solver, L, hold, frozen-parameter sha256, seed, run, step counts.
                 A checkpoint is saved at the first rollout boundary at or after its mark, so it
                 always holds an updated policy; `ckpt_<mark>` records the mark and the exact
                 PPO step count (e.g. mark 250 000 -> 262 144 with 8 x 2048-step rollouts)
    hold k       SB3 counts PPO decisions; one decision spans k control steps. Budget and marks
                 are given in control steps and divided by k; the exact control-step count
                 (from the per-step diagnostics) is recorded too
    evaluation   at every checkpoint: the checkpoint, deterministic, on the first `eval_seeds`
                 (50) HD_DEV seeds x 2 conditions, through highdim.blocks; HD_FINAL is never
                 opened here
    health       section 9.1 sanity inputs: `train_log.jsonl`, one row per policy update (SB3's
                 train/* metrics, epochs run, early stop, finiteness of losses and parameters),
                 and a `health` summary in final.json (non-finite rollout buffers, diagnostic
                 records missing a key or with a non-finite action, and steps whose info
                 carries none). `early_stop` means the target-KL stop fired: the update made
                 fewer optimizer steps than n_epochs x minibatches
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from experiments.week6_continuation.common import RESULTS as FROZEN_DIR
from highdim import blocks as HB
from highdim.ppo import NET_ARCH
from highdim.seeds import seed_block, train_env_seed
from highdim.subgoal import L
from highdim.wrapper import DIAGNOSTIC_KEYS, HDSubgoalEnv
from robot_env.robot_nav_env import load_config
from training.train_flat import pin_torch_threads, shared_ppo_kwargs

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "config.json"
SOLVER_DECISION = REPO / "results/highdim/HD1/analysis.json"
FROZEN_PARAM_FILES = ("H2/tuned_frozen.json", "R3/random_frozen.json")
N_ENVS = 8
BUFFER_FIELDS = ("actions", "values", "log_probs", "rewards", "returns", "advantages")
TRAIN_KEYS = ("approx_kl", "clip_fraction", "explained_variance", "loss", "value_loss",
              "policy_gradient_loss", "entropy_loss", "std")
FINITE_DIAGNOSTICS = ("gamma", "u_exec", "a_raw", "a_disc")


def training_solver():
    """The training solver the fast-solver check selected (HD_DESIGN.md section 8)."""
    d = json.loads(Path(SOLVER_DECISION).read_text())["decision"]
    if d["solver"] not in ("fast", "frozen") or (d["solver"] == "fast") != bool(d["pass"]):
        raise RuntimeError(f"inconsistent fast-solver decision: {d}")
    return d["solver"]


def env_specs(run, n_envs=N_ENVS):
    """[(condition, first-reset seed)]: the first half fixed, the second half randomized."""
    return [("fixed" if i < n_envs // 2 else "randomized", train_env_seed(run, i))
            for i in range(n_envs)]


def make_vec_env(run, *, solver, hold=1, n_envs=N_ENVS, vec="subproc"):
    fns = [(lambda c=c, s=s: HDSubgoalEnv(c, seed=s, solver=solver, hold=hold))
           for c, s in env_specs(run, n_envs)]
    venv = SubprocVecEnv(fns, start_method="fork") if vec == "subproc" else DummyVecEnv(fns)
    return VecMonitor(venv, info_keywords=("is_success",))


def build_model(venv, *, seed, run=None, overrides=None):
    """PPO on `venv`. With `run`, the envs' first resets are put back on their HD_TRAIN seeds
    (PPO's own seeding has just set them to seed + index)."""
    training = load_config(str(CONFIG))["training"]
    kw = {**shared_ppo_kwargs(training), **(overrides or {})}
    model = PPO("MlpPolicy", venv, seed=int(seed), device="cpu", verbose=0,
                policy_kwargs={"net_arch": list(NET_ARCH)}, **kw)
    if run is not None:
        seeds = [s for _, s in env_specs(run, venv.num_envs)]
        if seeds != [seeds[0] + i for i in range(len(seeds))]:
            raise RuntimeError("HD_TRAIN seeds of one run must be contiguous")
        venv.seed(seeds[0])                 # VecEnv.seed(s) -> env i resets on s + i
    return model


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout.strip()


def code_changes():
    """Uncommitted changes (and untracked files) under the code paths."""
    return _git("status", "--porcelain", "--", "highdim", "experiments", "continuation",
                "dr_control", "robot_env", "training", "config.json")


def metadata(*, run, seed, solver, hold, steps, n_envs, **extra):
    specs = env_specs(run, n_envs)
    return {"commit": _git("rev-parse", "HEAD"), "code_changes": code_changes(),
            "solver": solver, "L": L, "hold": int(hold), "seed": int(seed), "run": int(run),
            "steps": int(steps),
            "frozen_params_sha256": {f: _sha(FROZEN_DIR / f) for f in FROZEN_PARAM_FILES},
            "config_sha256": _sha(CONFIG),
            "env_conditions": [c for c, _ in specs], "env_seeds": [s for _, s in specs],
            "sb3": stable_baselines3.__version__, "torch": torch.__version__, **extra}


def evaluate(ckpt, out_dir, steps, *, eval_seeds, workers, hold, max_steps=None):
    """Deterministic dev evaluation of one checkpoint: eval_seeds HD_DEV seeds x 2 conditions."""
    seeds = seed_block("HD_DEV", eval_seeds)
    stem = Path(out_dir) / "evals" / f"ckpt_{steps}"
    res = HB.run_block("ppo_random", seeds, workers=workers, max_steps=max_steps,
                       checkpoint=f"{stem}.jsonl", traces=f"{stem}.traces.jsonl.gz",
                       model=str(ckpt), hold=hold)
    recs = [r for c in res for r in res[c]]
    rate = {k: float(np.mean([r[k] for r in recs])) for k in ("success", "collision", "timeout")}
    return {"steps": int(steps), "seeds": seeds, "n": len(recs), **rate,
            **{f"success_{c}": float(np.mean([r["success"] for r in res[c]])) for c in res}}


def buffer_finite(buf):
    """True if every stored rollout quantity PPO learns from is finite."""
    return all(np.isfinite(np.asarray(getattr(buf, k), float)).all() for k in BUFFER_FIELDS)


def diagnostic_complete(d):
    """True if a per-step diagnostic record has every key and finite reference and actions."""
    return (all(k in d for k in DIAGNOSTIC_KEYS)
            and all(np.isfinite(np.asarray(d[k], float)).all() for k in FINITE_DIAGNOSTICS))


class TrainingHealth(BaseCallback):
    """Section 9.1 sanity inputs; observes only, never changes training.

    SB3 records train/* inside train(), which runs after the rollout's log dump, so an update's
    metrics are read at the next rollout start (and once more by `collect` after learn())."""

    def __init__(self, log_path):
        super().__init__()
        self.log_path = Path(log_path)
        self.rollouts = self.nonfinite_rollouts = 0
        self.updates = self.nonfinite_updates = 0
        self.diag_steps = self.diag_incomplete = 0
        self._seen = 0                                 # model._n_updates already logged
        self._grad_steps = 0                           # optimizer steps since the last row

    def _on_training_start(self):
        opt = self.model.policy.optimizer
        inner = opt.step

        def counted(*a, **k):
            self._grad_steps += 1
            return inner(*a, **k)
        opt.step = counted
        m = self.model
        self.full_update = m.n_epochs * -(-m.n_steps * m.n_envs // m.batch_size)

    def _on_step(self):
        for info in self.locals["infos"]:
            diags = info.get("diagnostics")
            if not diags:
                self.diag_incomplete += 1              # a step that logged nothing
                continue
            for d in diags:
                self.diag_steps += 1
                self.diag_incomplete += not diagnostic_complete(d)
        return True

    def _on_rollout_end(self):
        self.rollouts += 1
        self.nonfinite_rollouts += not buffer_finite(self.model.rollout_buffer)

    def _on_rollout_start(self):
        self.collect()

    def collect(self):
        epochs = self.model._n_updates - self._seen
        if epochs == 0:
            return
        self._seen = self.model._n_updates
        v = self.model.logger.name_to_value
        row = {"update": self.updates + 1, "ppo_steps": int(self.model.num_timesteps),
               "control_steps": self.diag_steps,
               **{k: float(v.get(f"train/{k}", float("nan"))) for k in TRAIN_KEYS},
               "epochs": int(epochs), "grad_steps": self._grad_steps,
               "early_stop": self._grad_steps < self.full_update}
        self._grad_steps = 0
        params = all(bool(torch.isfinite(q).all()) for q in self.model.policy.parameters())
        row["finite"] = params and all(np.isfinite(row[k]) for k in TRAIN_KEYS)
        self.updates += 1
        self.nonfinite_updates += not row["finite"]
        with open(self.log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def summary(self):
        return {"updates": self.updates, "nonfinite_updates": self.nonfinite_updates,
                "rollouts": self.rollouts, "nonfinite_rollouts": self.nonfinite_rollouts,
                "diag_steps": self.diag_steps, "diag_incomplete": self.diag_incomplete}


class Checkpoints(BaseCallback):
    """Every `every` control steps: save the model and its metadata, then evaluate it on HD_DEV.

    Checked at rollout boundaries (and once more after learn() by `flush`), where the policy has
    just been updated, never in the middle of a rollout.
    """

    def __init__(self, out_dir, every, hold, meta, eval_kw):
        super().__init__()
        self.out_dir, self.meta, self.eval_kw = Path(out_dir), meta, eval_kw
        self.hold = int(hold)
        self.every = int(every)                        # control steps
        self.next = self.every
        self.control_steps = 0
        self.eval_time = 0.0

    def _on_step(self):
        self.control_steps += sum(len(i.get("diagnostics", ())) for i in self.locals["infos"])
        return True

    def _on_rollout_start(self):
        self.flush()

    def flush(self):
        while self.num_timesteps * self.hold >= self.next:
            mark = self.next
            self.next += self.every
            ckpt = self.out_dir / f"ckpt_{mark}.zip"
            self.model.save(ckpt)
            (self.out_dir / f"ckpt_{mark}.json").write_text(json.dumps(
                self.meta(steps=self.num_timesteps, mark=mark,
                          control_steps=self.control_steps), indent=1))
            t0 = time.perf_counter()
            row = evaluate(ckpt, self.out_dir, mark, **self.eval_kw)
            self.eval_time += time.perf_counter() - t0
            row.update(ppo_steps=self.num_timesteps, control_steps=self.control_steps)
            with open(self.out_dir / "evals.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")


def train(*, run, seed, total_timesteps, out_dir, hold=1, n_envs=N_ENVS, every=250_000,
          eval_seeds=50, eval_workers=8, eval_max_steps=None, overrides=None, vec="subproc"):
    """Train one PPO seed for `total_timesteps` control steps. Returns the final metadata."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} is not empty; training runs are never overwritten")
    out_dir.mkdir(parents=True, exist_ok=True)
    pin_torch_threads(1)
    solver = training_solver()
    venv = make_vec_env(run, solver=solver, hold=hold, n_envs=n_envs, vec=vec)
    model = build_model(venv, seed=seed, run=run, overrides=overrides)

    def meta(**kw):
        return metadata(run=run, seed=seed, solver=solver, hold=hold, n_envs=n_envs,
                        overrides=overrides or {}, **kw)
    if every % hold:
        raise ValueError(f"checkpoint interval {every} is not a multiple of hold {hold}")
    cb = Checkpoints(out_dir, every, hold, meta, dict(
        eval_seeds=eval_seeds, workers=eval_workers, hold=hold, max_steps=eval_max_steps))
    health = TrainingHealth(out_dir / "train_log.jsonl")
    t0 = time.perf_counter()
    try:
        model.learn(total_timesteps=-(-int(total_timesteps) // hold),
                    callback=CallbackList([health, cb]))
    finally:
        venv.close()
    health.collect()                            # the last update
    cb.flush()                                  # a mark reached by the last update
    wall = time.perf_counter() - t0
    train_s = wall - cb.eval_time
    final = meta(steps=model.num_timesteps, control_steps=cb.control_steps, wall_s=wall,
                 eval_s=cb.eval_time,
                 steps_per_s=cb.control_steps / train_s if train_s > 0 else float("nan"),
                 health=health.summary())
    model.save(out_dir / "final.zip")
    (out_dir / "final.json").write_text(json.dumps(final, indent=1))
    return final
