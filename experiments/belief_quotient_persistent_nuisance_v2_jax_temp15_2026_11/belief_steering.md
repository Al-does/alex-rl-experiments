# Offline v2 steering runner

No training or infrastructure provisioning occurs in these entrypoints. From the
repository root with its JAX environment active (Vast `--run` convention):

```sh
python -m experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.belief_steering --study belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11 --variant 1
```

Choose either study (insert `_eps05` before `_2026_11`) and variant 1/2/3.
Forward B2 credentials; the runner downloads only the final parameter NPZ and
canonical durability manifest, verifies SHA256 even for cached files, and restores
without optimizer state. CPU and GPU use the same JAX code. Reports refuse
overwriting. `--smoke` reduces episode lengths/budgets, pulse times/horizon, and
norm-calibration iterations; smoke reports remain under ignored `artifacts/`.
Scientific runs also use the harness B2 uploader to persist maps, checkpoint and
report copies and emit a terminal manifest under
`results/belief_steering_v2_seed42_run/`, satisfying Vast's default required
durability/self-destruct contract. Upload failures propagate; they do not report
successful durability. The CLI itself never commits, pushes or destroys boxes.

Scientific protocol: 12k retained fit/test rows, seeds 42/43; affine decoder and
encoder at blocks 1–4 and final LayerNorm; exclusion 8 plus independently fitted
128-exclusion supplementary probes. Raw decoded beliefs are not clipped.
Tune at block 4, seed 1042, 16 complete 1024-step episodes; smallest λ at or below
the tuning random baseline, otherwise lowest occupancy (ties choose smallest λ).
Confirm seed 2042, 32 complete episodes; full ten-point grid at block 4 and final
LayerNorm, selected λ at all five sites, and norm-matched decoder-orthogonal
controls at both headline sites. Shadow caches are independently unsteered but
receive the same executed trajectory inputs. All episode steps count for behavior.

Additional δ point/mirror conditions at both headline sites use a scalar effective
λ calibrated on tuning episodes to match rho_flip λ=1 mean shift norm. Bracket
up to λ=1024, then 12 bisections. Report bracketing and actual tuning/confirmation
norms; closed-loop norms need not be monotone or match perfectly on new episodes.
No confirmation metric is used to choose the scalar. The nominal λ grid stays
unchanged. Orthogonal directions are fixed seeded unit vectors in the decoder
nullspace, with per-decision norm matched to the corresponding belief shift.

Pulse: independent intact seed-3042 episodes, branches from pre-decision carry at
256/512/768; blocks 1/2/3; mirror/rho_flip λ=1 once; same intact tokens and
previous actions, no branch actions fed back; lags 0..128 inclusive. Readout uses
block-4 decoder. Full exact filters start from the exact decision-time posterior;
variant-2 quotient prediction starts from its identical sufficient rho, with A/B
coarsened. Quotient delta has no interpretation. Signed/absolute rho/delta, action
disagreement, TV, decay multiplier and R² are retained. A fitted lag-1+ tail rate
does not quantify preservation of the lag-0 write.

Generate the separate numerical reference set locally:

```sh
python -m experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.steering_prediction --output experiments/belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11/results/belief_steering_v2_predictions.json
```

This computes variants 1/2/3 × gamma .97/.99/.995 × grids 60/100, all four
targets and λ values, with exact filtering and zero-probability-safe Bellman
updates. Predictions are shared across epsilon conditions because their physical
processes are identical. The runner's reference-action agreement uses .995/100.
Variant 2 uses the exact scalar reward-state quotient grid and rho-only policy
lookup: full-simplex coordinate rounding must not introduce spurious δ effects.
Bellman tolerance is 1e-9, cap 10k iterations. Bellman convergence is not grid
convergence or an optimality certificate; compare both grids and all discounts.
For norm-matched extrapolations λ>1, reference lookup projects negative installed
coordinates to zero and renormalizes; such agreement is supplementary.

Aggregate six completed JSON files explicitly:

```sh
python -m experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11.steering_aggregate --output experiments/belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11/results/belief_steering_v2_six_agents.json PATH1 PATH2 PATH3 PATH4 PATH5 PATH6
```

Matched episode differences quantify evaluation uncertainty only: there is one
training seed per arm. Nuisance-only is not a fifth independent belief target:
the S1/S2 distinction is δ itself and is reward-relevant in variant 3.
