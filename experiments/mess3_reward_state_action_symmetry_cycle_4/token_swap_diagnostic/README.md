# State-0/state-1 token-swap diagnostic

This analysis asks whether a cycle 4 or cycle 5 variant-2 checkpoint responds
to the exact state/token symmetry that its task does not need to break.

The diagnostic fits one affine belief decoder on factual greedy rollouts. On an
independent rollout it reconstructs each transformer history and replays it
twice through the frozen model:

1. with the factual token and previous-action features;
2. with token channels 0 and 1 exchanged at every history position while the
   factual action sequence remains fixed.

Because the initial distribution, emissions, and every variant-2
action-conditioned transition are equivariant under the same exchange, the
exact counterfactual target is `[b1, b0, b2]`. If the decoded representation is
equivariant, the intervention exchanges decoded `b0`/`b1`, preserves decoded
`b2`, and leaves held-out MSE approximately unchanged.

This is a controlled representational intervention. It does not let changed
policy actions feed back into the environment, so it does not by itself
establish closed-loop policy use.

Run it against a direct RLlib Algorithm checkpoint or a source bundle
containing `final_checkpoint/`:

```bash
uv run rl-harness \
  experiments.mess3_reward_state_action_symmetry_cycle_4.token_swap_diagnostic.experiment \
  --resume-from /path/to/checkpoint-or-bundle
```

Use the corresponding cycle-5 module for cycle-5 checkpoints. The compact
result is `token_swap_diagnostic.json` under the run's results directory.
