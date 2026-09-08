# results/week4

Only artifacts that are **measured** live here.

`start_goal_coverage/` and `trajectory_examples/` characterise the *environment* and were
produced by sampling it directly — they do not depend on any trained policy, and the numbers
in them are real.

Everything else (`experiment1/`, `experiment2/`, `experiment3/`, `zero_shot/`,
`terminal_heatmaps/`, `collision_heatmaps/`, `plots/`, `result_table.json`,
`paired_comparisons.json`) appears only once the training runs in `MD_files/week4/` have
actually been executed. The pipeline that writes them has been smoke-tested end to end, but
those smoke outputs were **deleted rather than kept**: they came from 4k-step models
evaluated on 12 episodes, and a file under `results/` that looks like a result but is not
one is worse than a missing file.
