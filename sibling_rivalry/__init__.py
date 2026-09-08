"""PPO + Sibling Rivalry (Trott et al., NeurIPS 2019) for RobotNavEnv.

Ported from the validated implementation in `~/projects/ppo-nav/ppo_sr/`, which was
written against the paper and ablated over epsilon and the anti-goal. The selection rule
and the terminal reward are transcriptions of that code; what is new here is the
integration with Stable-Baselines3 (ppo-nav's PPO is custom) and the episodic rollout
collection SR requires.
"""
from sibling_rivalry.select import Episode, PairStats, relabel_and_select
from sibling_rivalry.reward import sr_terminal_bonus, apply_sr_reward

__all__ = ["Episode", "PairStats", "relabel_and_select",
           "sr_terminal_bonus", "apply_sr_reward"]
