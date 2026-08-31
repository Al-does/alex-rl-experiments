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
E[r_correct + alpha H(pi(.|history))], alpha = 0.05.
```

Detached behavior-policy entropy is added to rewards before GAE, giving the
critic a soft value target. PPO's differentiable entropy term uses the same
coefficient so the actor receives the local entropy gradient. The latter is
essential here: actions do not affect transitions, discounting is zero, and
the actor and critic share no parameters, so reward augmentation alone would
not maximize actor entropy in expectation. The temperature matches the prior
single-factor MESS3 maximum-entropy comparison and is fixed in reward units
across the 9- and 27-action environments.

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
