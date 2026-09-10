---
name: belief-geometry
triggers: [user, model]
description: Analyze sequential-agent belief geometry with the shared harness's held-out probes, predictive controls, and interpretation workflow.
---

Use the shared harness's generic belief-geometry workflow. Locate the installed
`analysis` package in the project Python environment and read its `README.md`.
Resolve the companion harness checkout from that package when source is needed;
do not assume an absolute worktree path.

Verify run provenance and target timing, select independent fit/test trajectories,
and check full-context warmup before fitting. Use `evaluate_belief_geometry`,
appropriate predictive/history controls, initialization, and task-derived contrasts.
Keep task performance separate from decoding; label reference-controller estimates
versus bounds and disclose sampling, variance, and selection limitations.

Keep task adapters and compact outputs in this repository. Delegate with exact
refs, adapter, checkpoint-selection rule, group/sampling protocol, seeds, controls,
and disjoint output paths. Return verified metrics, figures, provenance, and
caveats. Do not launch training, paid compute, or publication unless requested.
