# Four-seed RockSample belief probing

## Finding: seed 1 neglects rock 0 at (1, 0)

On independently reset final-policy episodes, seed 1 checks rock 0 in 83.4%
of episodes but never samples it (0/512). The other seeds sample it in about
half the episodes and attain per-rock marginal R² of 0.913–0.936. Seed 1's
rock-0 R² is −0.125; on informed/relevant rows it remains −0.073, so the
effect is not explained by sampled rocks deterministically remaining Bad.
Its initialization encoder on the same final-policy histories actually has
rock-0 R² = 0.289; training made this coordinate less linearly accessible.

The coordinate responds weakly even when its own policy supplies evidence.
Across 481 informative held-out rock-0 checks, the average absolute Bayesian
update is 0.225 while the decoded update is 0.041 (about 18% as large).
Check-difference R² is −0.020. For seeds 0/2/3 those differences have R²
0.839/0.837/0.820, with decoded change comparable to the true update.
The deterministically selected first test episode shows the same pattern:
seed 1's rock-0 decoded coordinate stays near 0.5 after evidence drives the
target toward zero. See `report/coordinates_on_policy.png`.

Fresh returns are 44.08/39.30/44.00/44.39 for seeds 0/1/2/3. A deficit of
roughly five reward units is consistent with neglecting one rock that is
Good with prior probability one half and pays +10 when collected.

## Limit: this is not proof of a completely absent rock representation

When a common script supplies remote and distance-zero checks and visits all
rocks, seed 1's rock-0 marginal R² rises to 0.700 (0.583 on informed/relevant
rows), and check-difference R² rises to 0.788. The other seeds achieve
0.950–0.978 unrestricted rock-0 R² there. These are separately fitted probes
on the forced-history distribution, not transfer of an on-policy probe.

Thus the supported conclusion is **behavioral neglect plus poor on-policy
linear belief tracking for rock 0**, with partial accessibility under forced
histories. The evidence does not establish that rock 0 is never encoded,
that no nonlinear decoder could recover it, or that a decoded coordinate
causally drives actions. Rock 5 also has weak forced-history fits in some
successful agents, reinforcing that distribution-specific probe failures
alone are not evidence of task failure.

## Initialization-to-final measurements

All entries below are held-out 1−R². Initialization and final encoders replay
identical final-policy histories; the initialization parameter key is exactly
reconstructed from the archived sweep and PPO splits, not retrained.

| Seed | M init → final | J init → final |
|---|---:|---:|
| 0 | 0.5138 → 0.0937 | 0.7945 → 0.3732 |
| 1 | 0.5539 → 0.2195 | 0.8865 → 0.6112 |
| 2 | 0.4923 → 0.0926 | 0.7712 → 0.3603 |
| 3 | 0.4967 → 0.0728 | 0.8001 → 0.3370 |

The 95% episode-bootstrap intervals for final M are [0.0814, 0.1049],
[0.2032, 0.2367], [0.0783, 0.1068] and [0.0648, 0.0846], respectively.
Only initialization and the 30,408,704-step final state were measured.
Connecting lines are visual guides; no intermediate checkpoint is invented.

M is the factored seven-coordinate posterior; J is its nonlinear 128-state
product. The joint linear probe is therefore a distinct, harder diagnostic.
All final joint fits beat initialization, observable controls, and the tested
permutation/Gaussian nulls. The next-sensor marginal baseline is nearly
perfect by construction: it is an affine function of each exact marginal
given rover position. It is a predictive control, not independent evidence
for an internal Bayesian belief model.

## Reproduction and validation

- Recipe and commands: `../../BELIEF_PROBES.md`; generated method tables:
  `report/report.md`; complete metric/provenance inputs: `seed0.json`–`seed3.json`.
- Four archived final checkpoints verified against B2 manifest SHA-256 and
  byte counts; each restored state reports 30,408,704 environment steps.
- Each seed/distribution uses 512 complete reset-to-exit/cap episodes, with
  256 fit and 256 test episodes. Fit-only episode CV selects SVD cutoffs.
  Bootstrap resamples test episodes with probes held fixed.
- Exact targets use actions, symbols, observed positions and known sensor
  likelihoods only. Reward/truth invariance and independent direct-joint
  replay tests passed. Target normalization and marginal/joint consistency
  errors across this evaluation are below 3×10⁻¹⁵.
- 38 RockSample library tests, six adapter tests, a seed-0 real-checkpoint
  smoke, changed-file Ruff, and the library wheel build passed. Rendering
  consumes compact results without reopening checkpoints.
- Full fast suites are not green: library 946 passed / 2 failed; experiments
  875 passed / 30 failed / 1 skipped / 1 deselected. The failing tests were
  reproduced on their respective `origin/main` checkouts, with both original
  experiment and original harness source imports for the experiment comparison.
  They concern existing infrastructure safeguards and other study recipes.
- Analysis source SHA-256 is identical across all four reports and the
  checked-in adapter. The `experiment_revision` is the base revision before
  this new analysis was committed; source hashes identify the executed code.

Scientific interpretation is descriptive and specific to these four final
agents. Per-rock scores and confidence intervals do not estimate population
variation across training seeds.
