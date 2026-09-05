# Zero-shot sub-experiment — do deterministic-trained policies transfer to stochastic motion?

**Status: implementation complete, smoke test complete, full training NOT run.**

## The question, and what it is not

Take the two models trained in Experiment 1 on the **original** constant-velocity obstacle
motion, and evaluate them — **without any retraining** — on the smoothly randomized
obstacles of Experiments 2 and 3.

```
PPO      trained on fixed  ->  evaluated on randomized
PPO + SR trained on fixed  ->  evaluated on randomized
```

This is Week 4's RQ4, and it is **supplementary**. It must not be confused with the primary
experiment, and the two are kept in separate result files and separate table rows for that
reason:

| | trained on | tested on | answers |
|---|---|---|---|
| **Question A** (primary, Exp 2 vs 3) | randomized | randomized | does SR help when the training environment is itself stochastic? |
| **Question B** (this document) | fixed | randomized | does SR *transfer* better to unseen dynamics? |

A model can do well on one and badly on the other. Merging them would let a transfer result
masquerade as a training result, so `results/week4/result_table.json` labels the two
zero-shot rows with `zero_shot: true` and the report never presents them unlabelled.

## Method

The zero-shot evaluation is not a separate code path. `evaluate_model` takes *which
environment to test on* independently of how the model was trained, so this analysis is the
same function called with `randomize=True` on an Experiment 1 checkpoint. That removes the
risk of the two evaluation paths drifting apart and producing an artefactual difference.

* Same 200 evaluation episodes, same seed block from `1_000_000`, as Experiments 2 and 3.
* Same deterministic policy setting, same metrics.
* Three seeds per arm; results reported as mean ± std across seeds.

Because the seed block is shared, the zero-shot models can be compared **pairwise** against
each other and against the randomized-trained models on identical scenarios.

## What is computed

For each arm, performance is reported in both environments and the degradation between them:

```
degradation_success  = success_rate(fixed)   - success_rate(randomized)
collision_increase   = collision_rate(random) - collision_rate(fixed)
path_length_change   = avg_path_length(random) - avg_path_length(fixed)
```

The transfer question is then whether SR's degradation is *smaller* than PPO's — not
whether SR's absolute randomized score is higher. Those can disagree: an arm that was better
on fixed obstacles can still transfer worse. Both are reported.

Because the research question is about dynamic obstacles specifically, the collision
breakdown matters more here than anywhere else. `collision_type` separates wall, static and
dynamic contacts, so an increase in failures under randomization can be attributed to the
moving obstacles rather than assumed to come from them. If a policy degrades mainly through
*static* collisions, that is a different and less interesting story — it would mean the
randomized obstacles perturbed the trajectory into a wall, not that the policy failed to
anticipate moving obstacles.

## Interpreting a null

The honest prior is that both arms degrade, because the randomized obstacles are genuinely
out of distribution for a fixed-trained policy. A difference between the arms' degradation
is only meaningful if it is larger than the seed spread, which is why three seeds are the
minimum and why mean ± std is reported rather than a point estimate.

**Do not assume SR is more robust.** SR's mechanism — escaping local optima during training
via anti-goal repulsion — has no obvious reason to confer robustness to a *test-time*
dynamics shift. If it does, that is a finding worth stating carefully; if it does not, the
null is equally worth reporting.

## Commands

```bash
# uses the Experiment 1 checkpoints; nothing is retrained
python -m experiments.week4 zero-shot --seeds 0 1 2 --episodes 200
python -m experiments.week4 table
```

Outputs:

```
results/week4/zero_shot/ppo_fixed_to_random/eval_seed{0,1,2}.json
results/week4/zero_shot/ppo_sr_fixed_to_random/eval_seed{0,1,2}.json
results/week4/paired_comparisons.json      -> key "zero_shot"
results/week4/result_table.json            -> the two rows with zero_shot: true
```
