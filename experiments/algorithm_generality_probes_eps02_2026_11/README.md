# Algorithm generality: final-checkpoint eps02 ladder probes

Frozen PPO and automatic-alpha SAC actors, seeds 42/44/46/49/52, all three
Belief-Quotient Ladder conditions. No training or paid compute.

```bash
.venv/bin/python -m pytest -q experiments/algorithm_generality_probes_eps02_2026_11/test_analysis.py
.venv/bin/python -m experiments.algorithm_generality_probes_eps02_2026_11.analyze --smoke
.venv/bin/python -m experiments.algorithm_generality_probes_eps02_2026_11.analyze --workers 8
.venv/bin/python -m experiments.algorithm_generality_probes_eps02_2026_11.report
```

The smoke uses one PPO Reward-State Quotient agent with 960/1,920 scored rows;
the full run uses the paper's 60,000/80,000. Checkpoints and logs belong in
ignored `artifacts/`; compact agent scores and comparisons are in `results/`.
Existing completed agent JSONs are resumed; use an isolated clean output
directory when changing the protocol rather than treating them as fresh runs.

## Protocol

- Final checkpoint step 2,012,217,344 and actual saved step-0 initialization.
  SHA-256 and size are checked against each canonical B2 durability manifest.
- Independent greedy streams `split(fold_in(PRNGKey(training_seed), 2))`,
  complete 1,024-step episodes, first 64 steps excluded, post-final-LayerNorm
  embeddings, affine ridge 1e-6, highest JAX matmul precision.
- Full belief, reward-state coordinate rho, and state-1/state-2 contrast delta
  are fit separately. Target-normalized error is 1-R².
- Reward-State Quotient: exact quotient c[B] from A/B-merged observations,
  independently checked against the merged-token full filter. Full-State
  Quotient: merged-token state-3 posterior only, **not an exact quotient**.
- For raw global errors above one, apply the paper's selective support
  constraint (simplex projection or scalar clipping). Preserve both scores and
  record the replacement. This retrospective test-error screening biases
  selected errors down; it is not a different fit or a general clipping rule.
- Within-action-class b/rho/delta fits use class-centered held-out variance.
  Replay final-policy complete histories through step-0 parameters, keeping
  targets, rows and final greedy action classes fixed. Compare full embeddings,
  the 62D null space of logit differences, and three logits. Missing fit classes
  yield explicitly excluded rows, never an invented fallback score. One-class
  Trivial Quotient results are global scores, not within-region evidence.
- Skip own-policy initialization rollouts, shuffled/random histories, label
  permutation, A/B behavior mixing and training curves. Accessibility is not
  evidence of causal use or a uniquely identified Bayesian implementation.

PPO adds only `value.w`/`value.b` to the common actor tree. The adapter requires
exactly those extra keys with expected shapes, then validates every actor key,
shape, dtype and finite value. SAC auto-alpha `step_*.npz` is actor-only;
`critic_step_*.npz` is not a probe input. Discovery intentionally excludes
fixed-alpha SAC and all non-transformer/continuation studies.

## Reused code and immutable provenance

The collector, models and replay implementation are imported from the existing
`experiments/belief_quotient_history_jax_2026_11` modules. Global regression
uses the harness `analysis.probes.linear` implementation.

`paper_*.py` contain mechanically extracted, **unmodified function bodies**
from sources not available as importable modules on experiment main:

| File | Original source | Revision |
|---|---|---|
| `paper_probe.py` | deep-beliefs `experiments/persistent_nuisance_v2/reprobe.py`: `lumped_model`, `coarse_filter`, `runtime`, `flatten`, `fit_score` | `177ba8d9081933b83a3834b7a1f92ac31bb3f02b` |
| `paper_constraints.py` | deep-beliefs `experiments/persistent_nuisance_v2/constrain_degenerate.py`: `constrain_predictions`, `constraint_method` | same |
| `paper_coarse.py` | deep-beliefs `experiments/persistent_nuisance_eps02/compare_coarse.py`: `merged_filter` | same |
| `paper_within.py` | alex-rl-experiments `experiments/within_class_belief_probes_2026_11/metrics.py`: `affine`, `score`, `refit`, `null_basis` | `c351a35032d6530daeaf42264ec32c64707517e9` |

The thin `within` wrapper omits the expensive perturbation controls from the
original `temp15_analyze.py::analyze_encoder`, preserving its full/head-blind/
logit classwise fits. `results/reinforce_reference.json` is a compact extraction
of existing final scores from the paper's `ladder_reprobes.json` and
`coarse_comparison.json`, plus that revision's
`results/ladder_eps02_within_class.json`. Source paths/revisions are embedded.
It contains all 15 REINFORCE seeds; tables also select the same five seeds.
No REINFORCE agents are rerun.

`results/findings.md` contains the interpretation and coverage caveats;
`results/tables.md` includes all condition means and same-five-seed rows.
The optional `diagnose_trivial` module reproduces the SAC Trivial Quotient
seed-44 rare-history association diagnostic without replacing primary scores.

Per-agent JSONs record training/analysis/harness revisions, source-file hashes,
checkpoint hashes and B2 keys, CPU affinity and elapsed time. The analysis HEAD
is the clean base before the analysis commit; source hashes identify the exact
uncommitted adapter contents executed. CPU/GPU greedy ties can change visited
histories relative to archived training-time probes.
