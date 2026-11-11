# belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11

Requested by the human user: fresh persistent-nuisance v2 environment, smoke only.
Three variant_{1,2,3}_transformer leaves; all architecture and probes match PR #213.
Training uses PR #213's 2B-step target, constant entropy 0.0 and temperature 1.5
in both rollout and PPO likelihoods. Smoke retains the parent's 4,096 steps.
No full training is requested.

The shared environment.py in ../belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11 records explicit T0/E,
uniform initial distribution and reward state 2. Effect size 2.5 multiplies the
unchanged variant directions in the log-odds tilt of destination state 2.

Everything except the environment and the explicitly requested training settings
is inherited. Smoke outputs are ignored and are not scientific findings.

```bash
uv run --group jax rl-harness experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.variant_1_transformer.experiment --smoke --hardware-profile cpu --no-upload-artifacts
```
