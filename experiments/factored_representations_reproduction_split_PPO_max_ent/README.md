# Split-network PPO with reward-stream entropy

This experiment imports the matched cycle-2 split-network PPO recipe and changes
only its entropy treatment:

```text
reward_for_GAE = correctness_reward + 0.5 H(pi_behavior(.|history))
PPO entropy coefficient = 0.0
```

The behavior-policy entropy is detached and added before GAE. The critic
therefore fits entropy-augmented value targets while PPO retains the original
zero actor-entropy coefficient. Because discounting is zero, transitions are
action-independent, actor and critic parameters are disjoint, and the entropy
bonus is detached, this is a reward-stream ablation rather than a direct
maximum-entropy actor gradient.

At a uniform policy, the added reward is approximately `1.10` for the
two-factor 9-action environment and `1.65` for the three-factor 27-action
environment. Each factor cell trains for `10_000_000` environment steps.

Run both factor counts with:

```bash
uv run rl-harness \
  experiments.factored_representations_reproduction_split_PPO_max_ent.experiment \
  --smoke --hardware-profile cpu
```
