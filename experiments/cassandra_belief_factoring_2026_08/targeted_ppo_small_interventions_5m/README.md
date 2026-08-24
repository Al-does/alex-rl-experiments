# Targeted dim-64 PPO interventions

These four one-at-a-time variants hold the targeted Cassandra environment,
seed (`42`), five-million-step budget, dim-64 four-layer transformer, entropy
coefficient (`0.03`, matching the successful dim-64 control), discount (`0.990`), and disabled KL loss fixed.

| Leaf | Only baseline change |
|---|---|
| `vf_clip_100` | value clipping `10 → 100` |
| `lambda_098` | GAE lambda `0.95 → 0.98` |
| `bptt_64` | BPTT and transformer context `256 → 64` |
| `previous_reward` | append the immediately preceding scalar reward to the policy observation |

Run each leaf with `--seed 42`. The full recipes stop after 5,000,000 sampled
environment steps; `--smoke` stops after 4,096.
