# Emission asymmetry with PPO and an entropy bonus

Follow-up to Reviewer 1 W3 in [deep-beliefs](https://github.com/Al-does/deep-beliefs/blob/main/experiments/emission_asymmetry_eps02.md).
The [original recipe](https://github.com/Al-does/alex-rl-experiments/commit/4118df4625104100ae469abf6ad863b30592a2c4)
tracked the approximate quotient even where the exact reward marginal gave
better reference control. This variant tests whether a different training
recipe improves task reward and accessibility of the finer correction.

The `asym_{000,025,050,075,100}_transformer` leaves use the original process
at κ = 0, .025, .05, .075 and .10. The zero-perturbation leaf is a matched
anchor. Current token and previous action are the only inputs; the reward is
pre-transition occupancy of state 3. The emission perturbations, transitions,
tilts, resets, 64×4 banded RoPE transformer (32 positions per layer), sampling
temperature 1.5, no epsilon-greedy, 2B-step target (2,012,217,344 actual), and
checkpoint/probe schedule are unchanged. Use fresh seeds 42–44 to match the
original comparison. Step-0 actor initialization is unchanged for each seed.

The existing eps02 JAX PPO backend adds a shared-trunk critic, terminal GAE
(γ .99, λ .95), and four epochs of 64 whole-episode minibatches per rollout.
The actor loss subtracts an entropy bonus on the temperature-scaled
policy: coefficient 0.03 at step 0, linearly annealed to zero by 1.8B steps,
then held at zero. This mirrors the first-90% schedule of the cycle-7 ladder;
the bonus is not part of the environment reward or the critic targets.

## Verification and analysis

```bash
study=belief_quotient_emission_asymmetry_eps02_jax_temp15_ppo_2026_11
uv run --group jax rl-harness "experiments.$study.asym_100_transformer.experiment" \
  --seed 42 --smoke --hardware-profile cpu --no-upload-artifacts
uv run --group jax pytest -q tests/test_emission_asymmetry_eps02.py \
  tests/test_emission_asymmetry_eps02_ppo.py tests/test_history_jax_persistent_nuisance_eps02_ppo.py
```

Smoke is two CPU rollouts (4,096 steps), with step-0 and subsequent checkpoint
probes. For any future full run, use a CUDA profile and retain the checkpoints
(for remote runs, use `--upload-artifacts` and verify durability). Use the original
study's offline analysis on the matched step-0 and final checkpoints:

```bash
uv run --group jax python -m experiments.belief_quotient_emission_asymmetry_eps02_jax_temp15_2026_11.analysis \
  --run-dir RUN_RESULTS --checkpoint-dir CHECKPOINTS --output NEW_REPORT.json
```

Compare greedy reward and action fractions, including time spent in the
constant-Tilt-X plateau, and changes from initialization in affine accessibility
of the quotient, exact reward marginal, their difference, and state identity.
The held-out greedy offline evaluation excludes the entropy bonus. Use the same
reference controllers and probe streams as the original study. Improvement
over REINFORCE would show an effect of the combined recipe, not isolate the
bonus: that would require matched zero-bonus PPO at the same κ and seeds.
Smoke outputs are not research results.
