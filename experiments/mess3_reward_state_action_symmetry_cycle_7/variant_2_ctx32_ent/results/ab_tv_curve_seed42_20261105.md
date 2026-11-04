# A/B sensitivity across every saved checkpoint

## Main finding

The late-training decline is real, but sensitivity does not decrease
monotonically throughout training. Mean total variation falls early, rises
again from about 2.72M to 5.87M steps, then falls to the final checkpoint.
The same pattern appears on greedy and sampled on-policy trajectories.

| Saved state | Training steps | Greedy-trajectory mean TV | Sampled-trajectory mean TV |
| --- | ---: | ---: | ---: |
| Initialization | 0 | 0.0243318 | 0.0257799 |
| Checkpoint 6 | 2,721,424 | 0.00986395 | 0.0102466 |
| Checkpoint 14, mid-training peak | 5,867,152 | 0.0179236 | 0.0181435 |
| Checkpoint 23 | 9,406,096 | 0.00202775 | 0.00206938 |
| Checkpoint 24, penultimate | 9,799,312 | 0.00139074 | 0.00140306 |
| Checkpoint 25, final | 10,192,528 | 0.00131206 | 0.00130697 |

Final TV is 94.6% below initialization on greedy trajectories and 94.9%
below initialization on sampled trajectories. The sampled-trajectory global
maximum is 0.0267122 at checkpoint 1 (755,344 steps); the mid-training peak
in the table is not the global maximum.

The curves establish a training-time pattern in policy sensitivity, not its
mechanism. Each checkpoint has its own visitation distribution, and there
is only one training seed. A/B-sensitive probabilities alone do not identify
reward-state inference.

## Curves and data

All files share the stem `ab_tv_curve_seed42_20261105`:

- `.pdf`: full curve on a linear y-axis.
- `_log.pdf`: the same points and intervals on a logarithmic y-axis.
- `.csv`: all 54 checkpoint/action-mode estimates, intervals, and diagnostics.
- `.json`: complete measurements and provenance.
- `.plot_manifest.json`: plot-source/input/output hashes and physical sizing.

The PDFs are vector graphics, 5.5 by 3.4 inches, with embedded TrueType fonts.
No smoothing or invented intermediate checkpoint measurements are used.
Shading is pointwise 95% episode-bootstrap uncertainty, not variation across
training seeds or simultaneous curve confidence bands.

## Protocol

- Reward-State Quotient, training seed 42, the original cycle-6-prefixed run.
- Initialization plus every durable trained checkpoint, 0 through 25:
  27 actual saved states, all module files hash-verified against the canonical
  B2 manifest. Actual step counts come from the hash-verified `progress.csv`;
  initialization is the saved zero-step state.
- For each saved state and each action mode, evaluate 64 complete 1,024-decision
  episodes under that checkpoint's own original policy. Greedy mode chooses
  the highest-probability action; sampled mode uses the unmodified softmax.
- A separate policy cache receives independently randomized fair A/B inputs
  on that checkpoint's trajectory. C and the preceding executed original
  action are preserved. The shadow never controls an environment.
- Use the established evaluation seed 20261105 and separate environment,
  symbol-randomization, and action-uniform streams keyed by episode. These
  seeds are common across checkpoints. Sampled actions use inverse-CDF
  sampling with common uniforms, not forced common actions.
- Preserve full histories from reset, then exclude 128 decisions, covering the
  encoder receptive field. Each estimate uses 57,344 scored decisions;
  the full curve covers 3,096,576 scored decisions across its 54 evaluations.
- Use 2,000 episode-bootstrap resamples per estimate, with decision-weighted
  means and bootstrap seed 20261104.
- Checkpoints 23-25 reuse the previously verified, protocol-identical raw
  evaluations. Their summaries were recomputed and matched the old estimates.
  The other 24 saved states were evaluated locally on CPU. No training,
  paid GPU resources, or manuscript modifications were used.

## Greedy-choice diagnostics

The final checkpoint's greedy invariance does not hold everywhere earlier in
training. For example, at checkpoint 8 (3,507,856 steps), original and shadow
greedy choices differ on 6,400/57,344 decisions on greedy trajectories
(11.16%). Initialization and several other early checkpoints also show
disagreements. From checkpoint 10 (4,294,288 steps) through the final
checkpoint, no scored greedy-choice disagreements occur in either action mode.
These are shadow comparisons, not measurements of a randomized-input policy
controlling a separate environment.

## Verification and provenance

- Real-checkpoint smoke covers initialization and all 26 trained checkpoints
  under both modes; smoke estimates are not mixed into the final curve.
- The relevant tests report 31 passed, with the previously documented
  duplicate-run campaign-discovery test excluded.
- A source-provenance check initially identified the PyTorch inference
  decorator rather than the collector source. Source lookup now unwraps the
  decorator and has a regression test. The 24 completed points were preserved,
  both raw-data hashes checked, and all summaries recomputed exactly.
  Each adopted point retains its original collection specification, original
  point hash, and explicit correction record; raw data were not altered.
- The final report records the corrected collector hash, actual module hashes,
  raw-data hashes, source revisions, and runtime versions. Full checkpoints,
  raw trajectories, the original driver snapshot, and recovery audit remain
  under ignored `artifacts/ab_tv_curve_seed42/`.

## Reproduce

From the experiment repository, with B2 credentials configured for preparation
and the preceding three-checkpoint raw artifacts available for reuse:

```bash
.venv/bin/python -m experiments.mess3_reward_state_action_symmetry_cycle_7.ab_tv_curve prepare \
  --scratch experiments/mess3_reward_state_action_symmetry_cycle_7/variant_2_ctx32_ent/artifacts/ab_tv_curve_reproduction

.venv/bin/python -m experiments.mess3_reward_state_action_symmetry_cycle_7.ab_tv_curve evaluate \
  --index experiments/mess3_reward_state_action_symmetry_cycle_7/variant_2_ctx32_ent/artifacts/ab_tv_curve_reproduction/checkpoint_index.json \
  --scratch experiments/mess3_reward_state_action_symmetry_cycle_7/variant_2_ctx32_ent/artifacts/ab_tv_curve_reproduction/full \
  --output experiments/mess3_reward_state_action_symmetry_cycle_7/variant_2_ctx32_ent/artifacts/ab_tv_curve_reproduction/report.json \
  --workers 3

.venv/bin/python -m experiments.mess3_reward_state_action_symmetry_cycle_7.plot_ab_tv_curve \
  --report experiments/mess3_reward_state_action_symmetry_cycle_7/variant_2_ctx32_ent/artifacts/ab_tv_curve_reproduction/report.json \
  --output-prefix experiments/mess3_reward_state_action_symmetry_cycle_7/variant_2_ctx32_ent/artifacts/ab_tv_curve_reproduction/curve
```

Evaluation resumes only complete points whose protocol and raw hashes match.
Output reports and figures refuse overwriting.
