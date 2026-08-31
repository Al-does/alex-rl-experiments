# Split-network PPO cycle 2

This study preserves the environment, objectives, transformer, PPO
hyperparameters, checkpoint schedule, and probe battery from
`factored_representations_reproduction_PPO_2026_08`, while removing all
parameter sharing between the actor and critic.

Both networks receive the same observation history but own independent
four-layer transformers. The policy and value heads are also independent.
Longitudinal probes read only the actor transformer's final-block residual
after its MLP and before its final LayerNorm. In the auxiliary condition, the
next-token head is likewise attached only to the actor representation.

The maximum-entropy condition optimizes

```text
reward_for_GAE = r_correct + 0.5 H(pi_behavior(.|history))
PPO entropy coefficient = 0.0
```

Detached behavior-policy entropy is added to rewards before GAE, giving the
critic an entropy-augmented value target. PPO's differentiable entropy
coefficient remains zero, matching the original shared- and split-network PPO
runs. This isolates reward-stream entropy, but it should not be interpreted as
a direct actor maximum-entropy gradient: actions do not affect transitions,
discounting is zero, the entropy bonus is detached, and the actor and critic
share no parameters. At a uniform policy the added reward is approximately
`1.10` for 9 actions and `1.65` for 27 actions.

Each entry point runs the two- and three-factor environments:

```bash
uv run rl-harness \
  experiments.factored_representations_reproduction_split_PPO_cycle_2_2026_08.ppo.experiment \
  --smoke --hardware-profile cpu

uv run rl-harness \
  experiments.factored_representations_reproduction_split_PPO_cycle_2_2026_08.ppo_aux_ce.experiment \
  --smoke --hardware-profile cpu

uv run rl-harness \
  experiments.factored_representations_reproduction_split_PPO_cycle_2_2026_08.ppo_max_entropy.experiment \
  --smoke --hardware-profile cpu
```
