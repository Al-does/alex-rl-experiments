# Independent token-0/1 flip diagnostic

This Cycle-5 Variant-2 diagnostic distinguishes the projected full belief
`s_t = P(state=2 | exact token history)` from the separately evolved coarse
belief `c_t = P(B={state 2} | not-2/2 history)`.

See
[the three-state to two-state belief mapping](../../mess3_reward_state_action_symmetry_cycle_6/three_to_two_belief_mapping.md)
for the simplex geometry, filtering distinction, and intervention rationale.

It fits frozen affine decoders for both targets on factual greedy rollouts.
For each held-out history, it then independently exchanges every token 0/1
identity with probability 0.5 while preserving:

- every token-2 position;
- the factual previous-action sequence;
- the frozen model, policy heads, and probe weights.

The coarse target is exactly unchanged by construction, while the full
three-state filter is recomputed for every randomized history. A coarse-filter
representation should retain coarse-probe accuracy and stable policy outputs.
A fine-filter representation should instead make the frozen `s_t` decoder
track the counterfactual change in the exact full-filter target.

The diagnostic also reruns the frozen policy in closed loop with independently
flipped policy-visible tokens and reports the reward change.

Run against a direct RLlib Algorithm checkpoint or a source bundle containing
`final_checkpoint/`:

```bash
uv run rl-harness \
  experiments.mess3_reward_state_action_symmetry_cycle_5.independent_flip_diagnostic.experiment \
  --resume-from /path/to/checkpoint-or-bundle \
  --seed 42
```

The compact result is `independent_flip_diagnostic.json` under the run's
results directory.

To recover and analyze the final Cycle-5 Variant-2 checkpoints for seeds
42--44:

```bash
uv run python -m \
  experiments.mess3_reward_state_action_symmetry_cycle_5.independent_flip_diagnostic.seed_queue
```

Use `--checkpoint-name checkpoint_000001` to analyze a specific saved
checkpoint instead of the final checkpoint.
