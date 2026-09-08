# Week5-Phase7 / Stage 3 — M1 velocity intervention

**Dated 2026-09-07.** Pre-registered in `PHASE7_PLAN.md` §11.7 (`5284025`) and clarified in
§11.7.8 (`6566d76`), both before implementation. V0 frozen; α, τ, ε, the objective, sample
selection and the **`u = 0` fallback unchanged**; the intervention is strictly upstream, in ξ
construction. No recovery, IMM, CT or uncertainty method was implemented.

Chain preserved: `cc692ab` → `df42437` → `5415bc7` → `e2cda71` → `c2019ff` → `5284025` →
`6566d76` → this. Suite 247 passed; frozen manifest verifies.

---

## Gate G-S3: **efficacy PASSES for T1 and T2; the safety veto is NOT triggered.**

Final tier, 200 paired episodes per condition, Phase 6 protocol and seeds.

### Mechanism-specific efficacy (the pre-registered criterion)

| condition | arm | **singleton / M1** | **multi / M2** | total | clamp binds |
|---|---|---|---|---|---|
| fixed | C0 | **0.0148** | 0.0216 | 0.0365 | 0.000 |
| fixed | **T1** | **0.0000** | 0.0232 | 0.0232 | 0.094 |
| fixed | T2 | **0.0000** | 0.0204 | 0.0204 | 0.130 |
| fixed | D-oracle *(diag)* | 0.0000 | 0.0125 | 0.0125 | — |
| randomized | C0 | **0.0114** | 0.0271 | 0.0384 | 0.000 |
| randomized | **T1** | **0.0000** | 0.0283 | 0.0283 | 0.070 |
| randomized | T2 | **0.0000** | 0.0234 | 0.0234 | 0.110 |
| randomized | D-oracle *(diag)* | 0.0000 | 0.0202 | 0.0202 | — |

Absolute counts: singleton **405 → 0** (fixed) and **298 → 0** (randomized) under T1; multi
497 → 561 and 596 → 672.

> **Pre-registered prediction:** *"T1 reduces the singleton/M1 infeasibility rate relative to C0,
> but residual multi-constraint/M2 infeasibility remains."*
>
> **CONFIRMED.** The singleton class is eliminated exactly — 0.0000 in every treatment arm and
> in both conditions — and M2 remains, essentially undiminished.

**M2 slightly increased under T1** (0.0216 → 0.0232 fixed, 0.0271 → 0.0283 randomized). This is
a genuine closed-loop effect absent from Stage 2b's open-loop substitution: clamping changes the
commanded action, so the trajectory and the states visited differ, and new multi-constraint
conflicts appear at states C0 never reached. It is reported, not explained away.

Total infeasibility falls **−36.4 %** (T1 fixed) and **−26.3 %** (T1 randomized); T2 gives
−44.1 % and −39.1 %. **As pre-registered, these are smaller than Stage 2b's A4 (−52.1 %)**,
because `v_cap = 0.96` is looser than the privileged 0.675/0.75 that A4 used. The prediction
that A4 was an upper bound is confirmed.

### Safety — the veto is not triggered, and that is *not* proof of safety

| condition | arm | collision | McNemar p | CI95 | success | min clearance | worst clearance |
|---|---|---|---|---|---|---|---|
| fixed | C0 | 0.140 | — | — | 0.855 | 0.245 | −0.0555 |
| fixed | T1 | 0.135 | 1.000 | [−0.035, +0.025] | 0.860 | 0.252 | −0.0510 |
| fixed | T2 | 0.130 | 0.774 | [−0.045, +0.025] | 0.865 | 0.256 | −0.0530 |
| randomized | C0 | 0.275 | — | — | 0.725 | 0.182 | −0.0593 |
| randomized | T1 | 0.280 | 1.000 | [−0.035, +0.050] | 0.720 | 0.183 | −0.0629 |
| randomized | T2 | 0.290 | 0.701 | [−0.035, +0.070] | 0.710 | 0.177 | −0.0712 |

**No arm shows a significant collision increase, so the veto does not fire. At n = 200 the
paired collision test resolves only ~0.08 absolute, so this result FAILS TO DETECT HARM; it does
NOT establish safety.** Two point estimates are numerically adverse and must be stated: T1
randomized collision +0.005 and T2 randomized +0.015, with T2's worst-case clearance moving from
−0.0593 to −0.0712 and its mean clearance from 0.182 to 0.177.

All collisions in every arm are **dynamic**; static and wall collisions remain 0.

### Higher-n safety endpoints

`P(e > 0 | TTC < 1 s)`: fixed 0.405 → **0.397** (T1), 0.413 (T2); randomized 0.485 → **0.498**
(T1), 0.488 (T2). Small, mixed-sign movements — worse for T1 under randomized motion, better
under fixed.

**These numbers are NOT comparable to Phase 4's 0.142.** Phase 4 measured the projected error on
the *binding-constraint* subset in the standalone harness; this measures all rows in
`RobotNavEnv`. The C0 arm here (0.405 / 0.485) is the correct reference, and the comparison is
strictly within this table.

Collision attribution is essentially unmoved: after-infeasible 24 → 22 (fixed), 37 → 39
(randomized) under T1. **Eliminating the singleton class did not reduce fallback-associated
collisions**, which weakens — though does not refute — the Phase 6 conjecture that the `u = 0`
fallback is the dominant collision pathway.

---

## The diagnostic arm is the most informative result

D-oracle, **ground truth, information-ledger-breaking, never a headline arm**:

| condition | collision | p | CI95 | success | SPL |
|---|---|---|---|---|---|
| fixed | 0.140 → 0.130 | 0.845 | [−0.060, +0.040] | 0.855 → 0.870 | 0.761 → 0.809 |
| **randomized** | **0.275 → 0.195** | **0.0070** | **[−0.135, −0.025]** | 0.725 → **0.800** | 0.626 → 0.705 |

**Perfect velocity yields a large, statistically significant safety gain under randomized motion
(−0.08 absolute collision, −29 % relative) that neither T1 nor T2 captures.** Dynamic collisions
fall 55 → 39. The headroom in velocity estimation is therefore real and substantial — but
**capping alone does not access it**. Capping removes the *pessimistic* tail that causes
infeasibility; the collision benefit lives in the *accuracy* of the estimate, which a cap cannot
supply.

D-oracle still leaves 0.0125 / 0.0202 infeasibility, all multi-constraint — confirming M2 is
irreducible by any velocity intervention and sizing it for Stage 4.

---

## Verdict and routing

**Efficacy: PASS** for both T1 and T2, on the mechanism-specific criterion. **Safety veto: not
triggered**, with the power caveat stated above.

The pre-registered routing rule is *"accept for Stage 4 only if efficacy holds and no safety
endpoint worsens."* Efficacy holds. Whether the small adverse movements constitute "worsening"
is a judgement I am not making unilaterally:

* **T1** — nothing worsens in the fixed condition (collision, clearance and optimistic error all
  improve slightly); under randomized motion collision +0.005, clearance +0.001, and
  `P(e>0|TTC<1s)` +0.013. All well inside noise at this n.
* **T2** — clamps more often (0.110–0.130 vs 0.070–0.094) and buys more infeasibility reduction,
  but three randomized endpoints move adversely (collision +0.015, mean clearance −0.005,
  worst clearance −0.012). **I recommend T1 over T2** on that basis.

**My recommendation:** accept **T1** as the M1 intervention and proceed to Stage 4 sizing M2 at
the measured residual (0.0232 fixed / 0.0283 randomized, entirely multi-constraint). Record that
the intervention's benefit is confined to infeasibility and did **not** produce a measurable
navigation-outcome gain, and that the D-oracle result motivates — but does not authorise —
estimator-accuracy work at Stage 5+.

## Limitations

* n = 200 cannot resolve collision differences below ~0.08; every collision conclusion here is
  "no harm detected", never "safe".
* M2's increase under T1 is a closed-loop trajectory effect and is not separable, with this
  design, from a genuine worsening of multi-constraint conflict.
* `v_cap = 0.96` was derived and not swept; the pre-registered sensitivity arms (1.0, and
  `V_PHYS` as a diagnostic) were not run, so no claim is made about the cap's optimality.
* D-oracle uses ground truth and bounds the mechanism; it is not an achievable target.

**Stage 4 has not been started. No recovery, IMM, CT or uncertainty method exists in the tree.**
