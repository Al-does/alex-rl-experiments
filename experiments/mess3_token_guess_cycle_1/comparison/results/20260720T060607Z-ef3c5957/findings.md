# MESS3 token-guess belief comparison

All conditions use the same transformer, PPO recipe, seed, environment, and binary next-token reward.

| condition | held-out R² | greedy token accuracy | predictive λ | entropy α |
|---|---:|---:|---:|---:|
| reward_only | 0.8552 | 0.6733 | 0.000 | 0.000 |
| predictive_loss | 0.9319 | 0.6734 | 0.100 | 0.000 |
| max_entropy | 0.8558 | 0.6733 | 0.000 | 0.050 |

The affine probe is fit on one rollout seed and evaluated on a disjoint seed. Belief labels are never used during reward-only or max-entropy training.
