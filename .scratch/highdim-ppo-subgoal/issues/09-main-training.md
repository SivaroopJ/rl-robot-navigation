# 09: Main training: 3 seeds × 3M steps, plus the extension rule

**What to build:** The researcher gets three final PPO-subgoal checkpoints trained under the pre-registered budget, with the extension decision taken on dev data by the fixed rule.

**Blocked by:** 08 (Pilot verdict = viable)

**Status:** ready-for-agent

- [ ] Seeds 0, 1 and 2 each train for 3M steps with 8 environments, the hold length from 08 and the chosen solver
- [ ] Dev learning curves (50 HD_DEV seeds × 2 conditions every 250k steps) are saved for each seed
- [ ] Extension rule: if ≥ 2 of 3 seeds gain ≥ 2 pp mean dev success between the 2.0–2.5M and 2.5–3.0M windows, all 3 seeds extend to 5M; the decision is recorded along with its numbers
- [ ] Each seed's final checkpoint and metadata are saved; best-on-dev checkpoints are not used for evaluation
- [ ] HD_FINAL is not touched
