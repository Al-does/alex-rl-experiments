# Ladder causal-decoding pilot

The shared [findings and protocol](../reward_factors_causal_decoding_2026_11/findings.md)
and [tables](../reward_factors_causal_decoding_2026_11/tables.md) cover both
Reward-State and Full-State seed-42 checkpoints. Ladder-specific compact
reports are under this folder's `results/`; no existing studies were changed.

The main result is negative: DAS replay fidelity .972 did not yield a selective
Reward-State steering improvement. At the historical selection rule its
rho-flip occupancy was .1665 (Eq. (4): .0869), and its mirror-control action
change was .1655 (Eq. (4): .0117). A band decoder improved held-out belief R²
from .849 to .900, but its default gradient write selected zero strength.
Two episodes per split make these feasibility measurements, not reliable
effect estimates or a replacement for the previous three-seed study.

```sh
uv run --group jax python -m experiments.belief_quotient_causal_decoding_2026_11.run \
  --test ladder --seed 42 --method das --pilot --workers 1
```

Use `--test ladder_full` for Full-State; omit `--pilot` for full episode
budgets. Outputs refuse overwriting; the command above reproduces the
configuration but requires a checkout without an existing matching report.
