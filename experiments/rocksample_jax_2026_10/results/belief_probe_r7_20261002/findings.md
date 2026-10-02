# RockSample r7: repeated seeds, different learned rocks

**Probing and independent rollouts agree on every lower-return final policy.**
Five of the eight agents neglect one rock: seed 0 neglects rock 5 `(0,3)` in
both runs; seed 1 neglects rock 0 `(1,0)` in both; seed 3 neglects rock 0 in
run 1 but learns it in run 2. Both seed-2 policies and run-2 seed 3 sample every
initially-Good rock with at least 97% probability.

The missed coordinate has negative held-out on-policy marginal R² and a decoded
check-update response much smaller than the exact Bayesian update. This is not
just a failure to encounter information: the missed rock is checked in 83–86%
of independent episodes. The evidence supports behavioral neglect and poor
on-policy affine tracking, not complete absence of a nonlinear representation.

## Runs, measurements, and provenance

- Run 1: `20261002T182024Z-ad0c3a34`.
- Run 2: `20261002T191712Z-7f1dc1ef`.
- Arm: `d128_kl0.1`, runtime seed 42, four nominal seeds, RockSample `[5,7]`.
- Actual archived updates: **1, 2, 3, 4, 5, 6, 9, 14, 20, 29**, at
  1,048,576 through 30,408,704 environment steps. No intermediate values are
  inferred. Step zero is reconstructed from the sweep's four-key split and
  PPO's parameter-key split; there is no stored initialization checkpoint.
- **88 checkpoint/initialization measurements**, each on 512 held-out complete
  episodes sampled from the saved policy; 256 fit/256 test episodes, grouped
  training-only affine-probe CV. Reset rows are retained;
  terminal observations and post-exit rows are excluded. Probe activations
  precede the action associated with each decision-time Bayesian target.
- Every final policy receives **4,096 additional independent sampled-policy
  rollout episodes** using a separate stream, excluded from decoder fitting,
  checkpoint choice, and probe-test scoring. These held-out means differ from
  the training-run evaluation means because the evaluation episodes differ.
- Targets use actions, emitted observations, observed position and rock layout;
  rewards and hidden rock qualities are excluded. Hidden qualities are used
  only for calibration and conditional rollout diagnostics. Marginal M has
  seven coordinates; joint J has 128 configurations. Predictions are unprojected.
- All 80 checkpoints passed archived-manifest size/SHA-256 and environment-step
  checks. Matched-seed reconstructed initialization hashes agree across runs.
  Adapter/driver source hashes also agree across all 88 measurements.
- Training archive: https://github.com/Al-does/alex-rl-experiments/pull/175.
  Analysis and compact data: https://github.com/Al-does/alex-rl-experiments/pull/177.
  JSON includes source revisions, recipes, summary hashes, parameter hashes,
  validation errors, CV choices, target variances and coverage. The generated
  `report/manifest.json` fingerprints these inputs and all generated outputs.

## Final agent comparison

Return ± SEM is from the independent 4,096-episode confirmation. Probe R²
uses held-out episodes from each policy's separate 512-episode collection.

| Agent | Return ± SEM | M R² | J R² | Neglected rock |
|---|---:|---:|---:|---|
| run 1, seed 0 | 39.319 ± 0.197 | 0.767 | 0.392 | 5 `(0,3)` |
| run 1, seed 1 | 39.221 ± 0.198 | 0.773 | 0.388 | 0 `(1,0)` |
| run 1, seed 2 | 43.972 ± 0.214 | 0.907 | 0.594 | none |
| run 1, seed 3 | 39.126 ± 0.198 | 0.813 | 0.428 | 0 `(1,0)` |
| run 2, seed 0 | 39.072 ± 0.198 | 0.752 | 0.392 | 5 `(0,3)` |
| run 2, seed 1 | 39.006 ± 0.199 | 0.749 | 0.395 | 0 `(1,0)` |
| run 2, seed 2 | 43.875 ± 0.214 | 0.906 | 0.629 | none |
| run 2, seed 3 | 44.023 ± 0.214 | 0.922 | 0.617 | none |

All five lower-return agents have M R² 0.749–0.813 and J R² 0.388–0.428;
the three higher-return agents have M R² 0.906–0.922 and J R² 0.594–0.629.
These eight observations show a descriptive association, not a causal test.

## Which coordinate is missed?

The per-rock score below uses all held-out policy timesteps. Each missed
coordinate has nonzero target variance; these are defined scores, not
constant-target artifacts.

| Agent | Rock | On-policy R² | Informative check-change R² | Mean abs. Bayesian → decoded check change | Sampled / initially-Good episodes | Prior at end |
|---|---:|---:|---:|---:|---:|---:|
| run 1, seed 0 | 5 | −0.048 | 0.038 | 0.187 → 0.039 | 0 / 2,003 | 13.96% |
| run 1, seed 1 | 0 | −0.117 | −0.013 | 0.218 → 0.041 | 0 / 2,059 | 16.87% |
| run 1, seed 3 | 0 | −0.052 | 0.027 | 0.216 → 0.038 | 0 / 2,059 | 17.53% |
| run 2, seed 0 | 5 | −0.096 | 0.017 | 0.202 → 0.032 | 0 / 2,003 | 17.29% |
| run 2, seed 1 | 0 | −0.089 | 0.048 | 0.208 → 0.039 | 1 / 2,059 | 14.87% |

Change R² is scored on informative checks; the absolute-change means include
all eligible held-out checks. The nearly flat decoded coordinate changes by
only 16–21% of the mean absolute Bayesian change. Four policies never sampled
the missed rock in these 4,096 episodes. Run-2 seed 1 sampled rock 0 in one
initially-Good episode (0.049%), so its neglect is near-total rather than literal
zero. Missing-rock conditional sampling is the behavioral diagnostic; the
report's <5% flag is descriptive, not a representational threshold.

## When do the trajectories separate?

Seed 3 gives the clearest within-seed comparison. At 9.44M steps (update 9),
both runs have return ≈31 and poor policy-history rock-0 decoding. Between
updates 9 and 14 (9.44–14.68M steps), run 2 develops rock-0 accessibility and
sampling while run 1 does not:

| Agent/update | Probe-collection return | Rock-0 R², all held-out policy timesteps | Rock-0 sampled episode fraction |
|---|---:|---:|---:|
| run 1, seed 3, update 9 | 31.367 | −0.072 | 1.17% |
| run 2, seed 3, update 9 | 30.820 | 0.047 | 9.96% |
| run 1, seed 3, update 14 | 38.945 | −0.055 | 0.39% |
| run 2, seed 3, update 14 | 43.535 | 0.634 | 49.02% |
| run 1, seed 3, update 29 | 39.629 | −0.052 | 0% |
| run 2, seed 3, update 29 | 44.355 | 0.916 | 50.98% |

The sampled-episode fraction includes initially-Bad episodes, unlike the
conditional-Good column in the final diagnosis. About half the rocks start Good,
so ≈50% overall sampling corresponds to sampling almost every Good rock.
On the independent paired rollout stream, run-2 seed 3 exceeds run-1 seed 3 by
**4.897 ± 0.101 SEM** return.

Both seed-0 runs show early rock-5 suppression: its policy-history R² is
already negative at update 2 (2.10M steps), and sampling falls from 14.45% at
initialization to 2.5–2.7% at update 2 and essentially zero later. Thus the final
gap is preceded by a coordinate-specific failure rather than a late evaluation
artifact. The archive brackets the changes; it does not locate the exact
training update where they occurred.

Parameters differ by the first saved update for all four matched seeds, while
their reconstructed initialization parameters agree. **Nominal seed does not
uniquely determine the outcome**, as seed 3 directly demonstrates. Seeds 0, 1,
and 2 happen to retain their respective basins across these two new runs.
The observations are compatible with numerical/path dependence but do not
isolate a GPU, implementation, RNG, or scheduling cause.

## Initialization and interpretation

- The literal step-zero policies have on-policy M R² 0.340–0.415. The initialized
  networks refit on each final policy's identical histories score 0.462–0.535,
  versus trained M R² 0.749–0.922. These paired initialization controls are
  distinct from the step-zero points on the own-policy curves.
- M and J are different targets: a high seven-coordinate fit does not imply
  equally good affine access to every joint posterior configuration.
- The exact next-sensor-probability control scores J R² 0.393–0.454. The three
  higher-return decoders exceed it by 0.175–0.223 R²; four of five lower-return
  decoders do not exceed it (run-1 seed 3 exceeds it by just 0.022). Current
  observable controls score 0.092–0.184. These descriptive comparisons qualify
  any claim of geometry beyond immediately predictive information.
- Bootstrap bands are conditional on fixed fitted probes and resample complete
  held-out episodes. They do not estimate retraining/path variability. The full
  J null/nuisance/predictive battery is run only at final checkpoints;
  intermediate J scores use the same grouped affine fit without those expensive
  auxiliary controls.
- No causal intervention or decoder-transfer claim is made. No new training or
  paid compute was launched.

## Outputs and verification

`report/summary.csv` contains 88 checkpoint rows; `report/per_rock.csv` contains
616 per-rock rows. Figures include M/J learning
curves, reward versus 1−R², per-rock trajectories, final per-rock scores,
independent rollout diagnostics and first-held-out-episode coordinate traces
(selection fixed, not cherry-picked). See `report/report.md` for the generated
measurement contract and `BELIEF_PROBES.md` for reproduction commands.

Focused adapter/report tests: **10 passed**. Exact Bayesian filter tests:
**5 passed**. Changed-file Ruff and hash verification passed. The experiment
fast suite reports **879 passed, 30 failed, 1 skipped, 1 deselected**; all 30
failures match the already verified clean-base failures documented at
`rl-harness/docs/issues/open/2026-10-02-experiment-fast-suite-recipe-mismatches.md`.
The old affected recipes are unchanged between that base and current main.
