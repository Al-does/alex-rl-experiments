# Discrete-SAC factored-representation reproduction, cycle 2

This cycle keeps PR #63's task, architecture, probes, 50-million-step budget,
learning rates, discount, target update, and prioritized replay capacity. Its
preregistered hyperparameter changes are:

- replay batch: `65,536`, processed as eight sequential `8,192` minibatches;
- replay warm-up: `1,500` → `10,000` environment steps;
- replay intensity: explicit `1.0` trained transition per sampled transition;
- prioritized replay beta: fixed at `0.6`;
- target entropy: `0.3` or `0.6` of the categorical maximum `log(|A|)`;
- auxiliary CE coefficient: `0.1` or `0.3`.

The entropy target is recorded both as its dimensionless fraction and as the
resolved positive entropy value. This avoids RLlib's action-space-dependent
`"auto"` resolution from becoming an unrecorded experimental variable.

The batch size is the conservative result of an RTX 4090 capacity sweep. It
reserved 81.8% of GPU memory for the largest three-factor auxiliary cell;
`77,952` fit at 97.8%, while `78,080` OOMed. Eight minibatches amortize replay
and learner coordination while bounding activation memory.

## Arms

Reward-only:

- `sac_entropy_0p3`
- `sac_entropy_0p6`

SAC plus next-token CE:

- `sac_aux_0p1_entropy_0p3`
- `sac_aux_0p1_entropy_0p6`
- `sac_aux_0p3_entropy_0p3`
- `sac_aux_0p3_entropy_0p6`

Each arm runs the matched two-factor and three-factor cells sequentially. For
example:

```bash
uv run rl-harness \
  experiments.factored_representations_reproduction_SAC_cycle_2_2026_08.sac_entropy_0p3.experiment \
  --smoke --hardware-profile cpu
```

The model, environment adapter, learner extension, and analysis battery are
imported directly from the PR #63 package so cycle 2 does not fork those
scientific definitions.
