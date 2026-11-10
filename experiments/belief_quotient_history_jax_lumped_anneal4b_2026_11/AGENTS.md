# belief quotient history jax — lumped baseline, 4B steps, per-leaf entropy anneal

Companion of `belief_quotient_history_jax_lumped_2026_11` (same environment, models, critic-free PPO
recipe, probes). Differences: fresh runs of 4,000,000,000 env steps (201 rollouts)
and a per-leaf piecewise-linear entropy coefficient: 0.015 until the leaf's
anneal start, linear to 0 over the next 79,691,776 steps (4 rollouts), 0 after.
Windows are in `shared.ANNEAL_WINDOWS`, derived from the seed-42 0->6B greedy
reward curves of the constant-entropy runs (see `shared.py` docstring).
Smoke: `uv run rl-harness experiments.belief_quotient_history_jax_lumped_anneal4b_2026_11.<leaf>.experiment --smoke --hardware-profile cpu --no-upload-artifacts`.
