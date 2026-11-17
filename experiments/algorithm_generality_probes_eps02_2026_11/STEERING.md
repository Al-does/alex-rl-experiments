# Frozen-agent steering: eps02 PPO and auto-alpha SAC

This is Step 2 on the same branch as the ladder probes. No actor training,
paid compute, XOR, factor-2 suppression or uploads are involved. Forty final
actors are used: the thirty ladder actors from Step 1, plus the ten Joint
Reward actors from the corresponding PPO/auto-alpha SAC reward-factor studies.
The paper's moving-reward experiment does **not** include Marginal Reward.

## Run

From the repository root with the existing sibling editable harness:

```sh
.venv/bin/python -m experiments.algorithm_generality_probes_eps02_2026_11.vendor_steering
.venv/bin/python -m pytest -q experiments/algorithm_generality_probes_eps02_2026_11/test_steering.py experiments/algorithm_generality_probes_eps02_2026_11/test_analysis.py
.venv/bin/python -m experiments.algorithm_generality_probes_eps02_2026_11.steering_run --smoke --workers 2
.venv/bin/python -m experiments.algorithm_generality_probes_eps02_2026_11.steering_run --workers 8
.venv/bin/python -m experiments.algorithm_generality_probes_eps02_2026_11.steering_report
.venv/bin/python -m experiments.algorithm_generality_probes_eps02_2026_11.steering_validate
```

`--analysis ladder` and `--analysis moving` select one protocol. Smoke runs
use one PPO agent for each protocol. Completed output paths refuse overwrites;
use a separate output location for a genuinely new campaign rather than
treating old results as newly computed. All checkpoints are the actual final
step 2,012,217,344 actors, seeds 42/44/46/49/52. Restoration reuses Step 1's
strict PPO value-head removal and SHA-256/byte-size durability-manifest checks.

## Pinned protocols and minimum vendoring

The source drivers and dependencies are absent from this branch's base, so
`vendor_steering.py` mechanically extracts only needed functions/classes from:

- Ladder: PR #239, `45517b13a67d746048a002f9db52ea55a654b219`.
- Moving reward: PR #240, `3504bfb72455a411b725ef368a18ff892760555f`.

`steering_sources.json` records original paths, revisions and whole-source
hashes. All numerical helper bodies are unchanged. Tests compare their ASTs
with the pinned originals and compare the complete numerical portion of the
ladder driver's body byte-for-byte. Changes are restricted to import
relocation, final-actor discovery/loading, local output/provenance, and
selecting `REWARDS = ('moving',)`. The moving-reward driver calls the reused
strict loader through an already-verified actor closure. Generic trainer,
environment and model implementations remain imports, not copied code.

Regeneration, reference extraction and source-parity tests require these git
objects locally; fetch the two pinned revisions from origin if absent. The
vendored numerical modules themselves do not require the original branches.
After the completed run, a trailing space in the generated ladder report
header and terminal blank lines in generated modules were removed. Executed
hashes are retained verbatim; the validator accepts either current bytes or
the exact reconstruction with that one header space / one terminal newline.
No numerical statement changed.

### Ladder

- Five sites: post-block residuals 1–4 and final LayerNorm.
- Affine activation→belief and belief→activation maps; 12,000 fit/test rows,
  seeds 42/43, exclusion 8 with supplemental exclusion-128 probes,
  SVD cutoff 1e-6. Raw decoded beliefs are not clipped.
- Delta point, delta mirror, rho flip and rho point, with the original
  ten-strength grid. Tune at block 4 using seed 1042 / 16 episodes; confirm
  using seed 2042 / 32 episodes of 1,024 steps, counting every step.
- Original selection rule: smallest strength reaching the random-action
  occupancy, otherwise the grid strength with lowest tuning occupancy.
- Full-strength curves at block 4/final norm, selected strengths across all
  five sites, decoder-orthogonal controls and norm-matched delta calibration.
- Single-write mirror/rho-flip pulses after blocks 1–3 at steps 256/512/768,
  128 subsequent lags, evaluation seed 3042. Exact full/quotient-filter
  comparisons and finite-cache cutoffs are retained.
- Original numerical belief-grid policy, grid 100 / discount .995, for
  installed-belief action agreement. The paper's 256-episode closed-loop
  exact-filter reference is reused with immutable source provenance.

### Moving reward

- Joint Reward only; factor 1 always rewards state 3. Factor 2 alternates
  reward between state 3 and states 1/2 every 256 steps in a 1,024-step episode.
- Fit the original factored and joint targets at all five sites from 60,000
  rows, warmup 64, fit seed 42. Factored is the source's `marginal` target
  identifier, **not** a Marginal Reward agent.
- Both myopic and discounted (gamma .99) action maps, the original 13-strength
  grid, per-target/map tuning at block 4 on 64 episodes (seed 1042), and
  confirmation on 256 episodes (seed 2042). Smallest strength maximizing
  tuning total reward is reused unchanged across sites.
- Intact/random/exact-belief-map references and norm-matched
  decoder-orthogonal direction controls at block 4. Factor 1 is not steered.
  Component/phase rewards and both factors' action-change/agreement scores
  are retained. The external controller, not the frozen agent, knows phase.

## Results and interpretation boundaries

Compact reports and comparisons live in `results/steering/`; raw reports,
map arrays, checkpoints and logs stay under ignored `artifacts/`. Every
actor report records training/analysis/harness revisions, source hashes,
checkpoint hash/B2 keys, CPU affinity, versions and wall time.

`steering_report.py` extracts the paper's existing REINFORCE results (same
five seeds and all fifteen), preserving exact source-file hashes. It does not
rerun those agents. `tables.md` includes full-strength ladder effects,
tuning-selected site/null comparisons, moving-reward/site tables and per-seed
results, norm-matched delta and pulse comparisons. `summary.json` retains all per-seed/site/control/phase metrics;
ladder agent reports retain compact pulse comparisons.

The completed CPU campaign contains 30 full ladder reports and 10 full
moving-reward reports. Both reduced-budget smokes completed first. Ladder
agents took about 6–8 minutes each and moving-reward agents 11–14 minutes
each under eight-way CPU execution; those are observed, not planning estimates.

All means are descriptive across independently trained agents, not pooled
episode uncertainty. Null controls use tuning-selected strengths, whereas
the headline ladder table uses full strength, exactly as in the paper; do
not conflate them. Matched direction does not guarantee matching activation
norms under a different closed-loop trajectory. Large out-of-distribution
writes and collateral changes to the unsteered factor limit causal claims.
Numerical belief-grid and QMDP references are not certified POMDP optima.
