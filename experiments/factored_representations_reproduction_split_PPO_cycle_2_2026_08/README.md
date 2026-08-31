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

Each entry point runs the two- and three-factor environments:

```bash
uv run rl-harness \
  experiments.factored_representations_reproduction_split_PPO_cycle_2_2026_08.ppo.experiment \
  --smoke --hardware-profile cpu

uv run rl-harness \
  experiments.factored_representations_reproduction_split_PPO_cycle_2_2026_08.ppo_aux_ce.experiment \
  --smoke --hardware-profile cpu
```
