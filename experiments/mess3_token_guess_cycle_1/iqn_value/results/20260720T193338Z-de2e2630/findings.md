# IQN distributional-value comparison

| condition | held-out R² | greedy token accuracy |
|---|---:|---:|
| max_entropy | 0.8558 | 0.6733 |
| predictive_loss | 0.9319 | 0.6734 |
| reward_only | 0.8552 | 0.6733 |
| iqn_value | 0.9760 | 0.6787 |

The IQN condition changes only the value critic and its loss. Its mean quantile value supplies PPO's scalar GAE baseline, while sampled quantiles regress against on-policy lambda-return samples.
