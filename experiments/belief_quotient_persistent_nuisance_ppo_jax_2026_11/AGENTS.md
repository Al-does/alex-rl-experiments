# Persistent-nuisance history ladder, standard PPO (pure JAX)

Twelve leaves: variant_{1,2,3}_{transformer,memoryless,gru128,gru92}.
Source: `belief_quotient_persistent_nuisance_jax_2026_11` (PR #208), which
trains with critic-free Monte Carlo policy gradient (REINFORCE in the PPO shell).
This study uses the same process, encoders, batch geometry, entropy bonus,
budget, checkpoints and probes, but trains with standard PPO with a learned
critic (`ppo.py`). The source and historical studies are unchanged.

## What changes vs PR #208

| | PR #208 | this study |
|---|---|---|
| critic | none | linear value head on the shared post-final-LayerNorm embedding |
| advantages | standardized MC reward-to-go | GAE, batch-standardized |
| episode end | treated as terminal | truncation, bootstrapped from V(next obs) |
| gamma / lambda | .99 / 1 | .99 / .95 |
| epochs x minibatches | 1 x 64 | 4 x 64 (256 Adam steps per rollout) |
| Adam | lr 4.2e-4, eps 1e-8 | lr 3e-4, eps 1e-5 |
| grad clipping | none | global norm 0.5 |
| value loss | - | 0.5 x MSE, unclipped |
| PPO clip | .2 | .2 |
| entropy | .015 constant | .015 constant |
| KL penalty | none | off (`use_kl_loss=False`); when on, RLlib adaptive `kl_coeff` 0.2, `kl_target` .01 |

The critic regresses (1 - gamma)-scaled returns (per-step reward units) so its
targets are O(1) on the shared torso; standardized advantages are invariant to
the scale. The actor is initialized exactly as in PR #208 for the same seed; the
value head gets its own key. Rollout sampling consumes the same RNG stream as
the source trainer. Probes run the actor only, at the same checkpoints and on
the same representation (embedding), so results are comparable to PR #208.

Unchanged: 200M-step target (11 rollouts of 19,456 envs x 1,024 steps =
219,152,384 steps), temperature 1.0, seed default 42, transformer 64x4 banded
RoPE window 32, memoryless residual MLP 64x6, GRU 128x2 / 92x4 with 32-step
truncated BPTT from rollout hidden states. Checkpoints are `.npz` parameter
files and include the value head.

## Commands

```bash
uv run --group jax pytest -q tests/test_history_jax_persistent_nuisance_ppo.py
uv run --group jax rl-harness experiments.belief_quotient_persistent_nuisance_ppo_jax_2026_11.variant_2_gru128.experiment --smoke --hardware-profile cpu --no-upload-artifacts
```

Use `--group jax-cuda` and omit `--smoke` for full GPU runs. Smoke uses 2 envs,
2 minibatches, 4 epochs, two iterations (4,096 steps), with step 0, 2,048 and
4,096 probes; `.smoke` outputs are not scientific results.
