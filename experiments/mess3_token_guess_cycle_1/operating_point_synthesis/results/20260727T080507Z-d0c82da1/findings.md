# Operating-point validation

Matched PPO and IQN arms, identical recipe, architecture, budget, seeds, and probe. Only the process differs.

| point | stay | alpha | floor | band | arm | belief R² | clears floor | share of accuracy range |
|---|---:|---:|---:|---:|---|---:|:--:|---:|
| shipped | 0.9 | 0.85 | 0.964 | 0.036 | ppo | 0.8461 ± 0.0065 | no | -0% |
| shipped | 0.9 | 0.85 | 0.964 | 0.036 | iqn | 0.9756 ± 0.0032 | yes | 47% |
| cantor_sharp | 0.95 | 0.85 | 0.944 | 0.056 | ppo | 0.9138 ± 0.0356 | no | 41% |
| cantor_sharp | 0.95 | 0.85 | 0.944 | 0.056 | iqn | 0.9793 ± 0.0015 | yes | 90% |
| cantor | 0.95 | 0.75 | 0.915 | 0.085 | ppo | 0.8990 ± 0.0188 | no | 63% |
| cantor | 0.95 | 0.75 | 0.915 | 0.085 | iqn | 0.9737 ± 0.0012 | yes | 92% |
| proposed | 0.96 | 0.55 | 0.832 | 0.168 | ppo | 0.8281 ± 0.0548 | no | 52% |
| proposed | 0.96 | 0.55 | 0.832 | 0.168 | iqn | 0.9646 ± 0.0056 | yes | 91% |

- `shipped`: IQN beats PPO by 0.1294 belief R², which is 361% of the usable band, and by 47 points of the accuracy range.
- `cantor_sharp`: IQN beats PPO by 0.0655 belief R², which is 117% of the usable band, and by 50 points of the accuracy range.
- `cantor`: IQN beats PPO by 0.0746 belief R², which is 88% of the usable band, and by 28 points of the accuracy range.
- `proposed`: IQN beats PPO by 0.1365 belief R², which is 81% of the usable band, and by 39 points of the accuracy range.

A gap wider than the band is not a larger effect, it is an off-scale reading: the arms differ by more than the range in which a difference is interpretable. Seed spread is not comparable across points either, because an arm pinned at a non-learning fixed point has almost no variance while an arm that sometimes learns has a lot.
