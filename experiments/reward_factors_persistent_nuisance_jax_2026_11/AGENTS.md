# Persistent-nuisance reward factors (pure JAX)

Eight leaves: {reward_both,reward_factor_1}_{transformer,memoryless,gru128,gru92}.
Source reward_factors_history_jax_2026_11 at PR #195 merge
`0ca34b25ea14efe9e5668c79d258b90fb2cd7200`; development origin/main
`2fb56dce3d63b4e01927100958a431babf5a79ab`.
See ../belief_quotient_persistent_nuisance_jax_2026_11/AGENTS.md for explicit T0/E,
tilt semantics, dependencies, analytical references and scientific limitations.

Each independent factor uses the new T0/E and variant-3 tilts, with the
source ACTION_PAIRS order; joint state/token use kron order 3*f1+f2.
Both conditions have identical transitions, emissions, initial distribution and
reward components. reward_both sums pre-transition S3 occupancy of both factors;
reward_factor_1 scores only factor 1. Nothing privileged enters actor inputs.
Resolved recipe metadata records the explicit factor matrices and directions.

Training/model configs are imported directly from the PR #195 factor shared.py:
200M budget, constant entropy .015, same optimizer and batching as the ladder.
Transformer window 10 (lookback 40); probes use pre-final-LayerNorm residuals
and report joint, factor_1, factor_2 exact-filter targets. Default temperature
remains 1.0. Initialization and later-checkpoint probing are unchanged.

```bash
uv sync --group jax  # or --group jax-cuda
uv run --group jax python -m experiments.reward_factors_persistent_nuisance_jax_2026_11.design
uv run --group jax rl-harness experiments.reward_factors_persistent_nuisance_jax_2026_11.reward_both_transformer.experiment --smoke --hardware-profile cpu --no-upload-artifacts
```
The design command reports the ladder's per-factor v2 references, not an
optimality claim for this variant-3 joint task. Probe R² is accessibility,
not causal use. Smoke scores are not research findings.

## Smoke verification

All 8 CPU smoke leaves passed: two training iterations and finite joint,
factor_1 and factor_2 probe r_squared at steps 0, 2,048 and 4,096.
Outputs stay in ignored .smoke/.
