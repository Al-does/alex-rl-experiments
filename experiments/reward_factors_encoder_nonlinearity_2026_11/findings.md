# Eq. (4): nonlinear encoders and encoder-input source

Branch: `devin/1795210916-causal-decoding`. Frozen final checkpoints; no agent retraining.
The ladder companion is `belief_quotient_encoder_nonlinearity_2026_11`.

## Historical source audit

The historical belief→activation encoder uses **exact Bayes-filter beliefs in
all four tests**, not probe predictions. The new causal runner's generic inverse
was not used as evidence for this answer: these are the mappings used by its
actual `eq4`/`eq4_joint` branches.

| Test | Code path and encoder fitting source |
|---|---|
| Ladder (+ Full-State) | `algorithm_generality_probes_eps02_2026_11/steer_core.py:120–127`: `records['belief']` is flattened with the recorded activations; `affine(b[:, :2], h[:, site])` fits the encoder. |
| Moving | `reward_factors_replay_edit_2026_11/transfer.py:55–68`: `base.map_targets(rows['belief'])` supplies the inputs to `base.affine_fit(b, h)`. |
| XOR eps02 | Same `transfer.prepare` path, with eps02 checkpoints/processes. |
| XOR v2 | Same `transfer.prepare` path, with v2 checkpoints/processes. |

All paths above are relative to `experiments/`. For reward factors,
`algorithm_generality_probes_eps02_2026_11/steer_moving_base.py:121–123`
constructs the first two probabilities of each marginal (four coordinates), or
the first eight probabilities of the joint posterior. Its `collect` function
records the exact decision-time filter at lines 83, 92–93. The original paper
implementation also explicitly fits `embedding` from these exact `fit_targets`
in `steer_moving.py:191–201`. Ladder's `steer_core.rollout` initializes and
updates the exact action/token filter at lines 93–105. No rewards enter it.

The archived controls are dispatched in
`reward_factors_causal_decoding_2026_11/run.py:207–213,276–284` through
`steer_core.fit_maps` for ladder and `transfer.prepare` for moving/both XOR
cohorts. Cohort dispatch is at lines 58–61.

**Write-time current belief is always the probe's raw, unclipped decode**, for
every encoder arm. Ladder: `steer_core.py:31–38`; moving/XOR:
`steer_moving.py:95–98`. The historical target-construction functions are reused
unchanged (including any normalization internal to target construction). The
encoder's training source changes, never the write-time current input.

## Design and strict controls

- 2×2: affine or MLP, trained on exact or probe-decoded belief coordinates.
- Both historical factored/marginal and joint representations are run on
  moving and both XOR agents. Ladder/Full-State have one two-coordinate decoder;
  their entries repeated in joint rows are the same results, not extra runs.
- Affine uses the original float64 least squares, including the original
  cutoff (`1e-6` ladder, `1e-10` reward factors). Its displacement remains
  `(target-current) @ W`, so the historical arithmetic is preserved exactly.
- MLP: Linear → tanh → Linear, **128 hidden units for 64 output activations**.
  Train-only standardization of inputs/outputs; AdamW, learning rate `1e-3`,
  weight decay `1e-4`, batch 1,024, at most 200 epochs, validation patience 15,
  minimum improvement `1e-5`. Restore the best validation weights. No tuning
  against confirmation episodes. Initialization/shuffle seed is 42 for all
  agents and arms. The write is `lambda*(enc(target)-enc(current))`.
- No decoder, target, hook, action map, action rule, environment stream, horizon,
  metric, or checkpoint changes. Site 3 is block-4 latest, pre-final-norm.
- Full fit: ladder uses the historical first 12,000 post-warmup rows from 24
  intact episodes; reward factors use the historical 60,000 rows/fit seed.
  Validation: 12 intact episodes with stream seed 43. Test R²: 12 separate
  intact episodes with stream seed 44. Fit seed is 42. Warmup stays 8 ladder,
  64 reward-factor steps. Encoder R² predicts activations, **not beliefs**;
  each column is evaluated with its own specified input source.
- Selection: 16 episodes, stream 1042; confirmation: 32 × 1,024 decisions,
  stream 2042. Ladder grid `(0,.25,.4,.45,.5,.55,.6,.75,.9,1)`: first entry
  meeting the paired random-action floor, otherwise minimum occupancy. Mirror
  stays at lambda one. Reward-factor grid
  `(0,.25,.5,.75,.8,.85,.9,.95,1,1.25,1.5,2,3)`: maximize selection reward,
  first grid entry breaks ties. These are the unchanged historical rules.
- Prior fresh Eq. (4) campaign controls reused archived selected lambdas rather
  than selecting anew at 16 episodes. Therefore report **both**: (1) strict
  encoder-only effects, all arms at the archived lambda; (2) all four arms
  reselected at 16 episodes, including affine/exact. The latter can differ from
  the old control solely because of lambda selection. Both get paired 32-episode
  confirmation. Report generation checks the strict affine/exact control
  against every fresh Eq. (4) metric exactly; any discrepancy requires a
  separate exact reproduction by the unchanged historical runner on CPU.

## Pilot and execution

Seed 42 locally first: eight episodes per split for ladder/moving, two for
Full-State and both XOR tests; full 1,024-step episodes. All 32 arm reports
completed, using 644.7 allocated core-seconds for arm work; maximum worker time
111.6 seconds. Independent pilot summary/tables are in `results/pilot_*`.
Pilot MLP R² exceeded affine R² in all eight test/representation comparisons,
but closed-loop gains were mixed. Most MLPs reached the epoch cap; early-stopping
support is not a claim that all optimizations converged.

This measured pilot made the full run comfortably feasible inside two local
CPU-hours. Full evaluation therefore uses four two-core workers on the local
eight-core CPU. The 96-report campaign took **12.75 minutes** of wall time,
approximately **1.491 allocated core-hours**, excluding the later control
rechecks. **No paid compute rented.**

```bash
uv run --group jax python -m experiments.reward_factors_encoder_nonlinearity_2026_11.run --pilot --seed 42 --workers 4
uv run --group jax python -m experiments.reward_factors_encoder_nonlinearity_2026_11.run --seed 42 43 44 --workers 4
uv run --group jax python -m experiments.reward_factors_encoder_nonlinearity_2026_11.report
```

Each test/seed invocation shares collected fit/validation/test data across all
arms. Outputs resume per arm. Only compact reports, tables and manifests are
committed; no checkpoints, activation arrays, optimizer states or raw logs.

## Confirmation results

See `results/campaign_tables.md`: held-out encoder R², both complete performance
tables, and numerical paired differences with separate seed-42/43/44 columns.
`campaign_summary.json` retains mean ± sample SD, each seed's values, and raw
paired signs. `campaign_manifest.json` hashes every report and fresh control.

**Optimization limit:** 42/48 MLP fits reached the 200-epoch cap. Best validation
weights are restored, but convergence is not established. Conclusions below
are about this architecture and training budget, not optimal nonlinear encoders.

1. **Activation reconstruction improves substantially.** For example, moving
   factored R² is `.6485±.0095` affine/exact versus `.8576±.0132` MLP/exact;
   joint decoded-source R² is `.7937±.0202` versus `.9163±.0154`. Moving and XOR
   eps02 have identical full-budget encoder R² because they share the same
   frozen agents, intact processes, fit data and held-out data; only their
   intervention reward schedules/targets differ.
2. **No universal steering gain.** At the identical archived lambda, both MLPs
   worsen ladder occupancy for all three seeds despite better encoder R²:
   `.0949±.0178` historical versus `.1252±.0289` MLP/exact and `.1217±.0141`
   MLP/decoded. Mirror action changes also increase for all three seeds. Both
   MLPs worsen Full-State occupancy for all three seeds.
3. **A factored moving lead, not a joint replacement.** Fixed-lambda MLP/exact
   moving reward gains are `+.034058±.036495` (`+++`); factor-1 action changes
   fall by `.023448±.009986` (`−−−`). MLP/decoded reward gains are
   `+.021291±.010688` (`+++`), changes fall by `.040700±.009921` (`−−−`). But
   joint moving reward falls by `.031728±.010306` (`−−−`) for MLP/exact and
   `.041402±.029537` (`−−−`) for MLP/decoded. Factored MLP reward remains below
   the historical joint control on average.
4. **Changing only the affine fitting source has a ladder trade-off.** Decoded
   source lowers occupancy by `.007243±.004344` (`−−−`) while increasing mirror
   action changes by `.002991±.001909` (`+++`). Moving effects change sign
   across agents; it is not a general improvement.
5. **XOR is not a robust v2 improvement.** Fixed-lambda factored eps02 gains are
   `.014191±.012327` for MLP/exact and `.019572±.018860` for MLP/decoded, both
   `++0`. Seed 44 uses archived lambda zero, so equality there is expected and
   cannot demonstrate an ineffective nonzero write. v2 reward signs are mixed
   for factored MLPs, and joint MLP/decoded is worse for all three seeds.
6. **Lambda selection matters.** The separately reselected table includes an
   independently reselected affine/exact arm, not the archived-lambda control
   relabeled as a fresh selection. For example, factored moving rewards become
   `.9598±.0402` affine/exact, `1.0139±.0212` MLP/exact, `1.0011±.0179`
   MLP/decoded. Joint MLP/exact instead falls to `.9649±.0584`. Do not attribute
   changes between the fixed/reselected tables to encoder architecture alone.

### Exact-control audit and platform discrepancy

Nineteen of 24 historical affine/exact test/seed/representation controls match
every previously published metric exactly. Five differ: moving seed-43
factored; eps02 seed-43 factored and seed-44 joint; v2 seed-43/44 factored.
The v2 differences affect action-change fractions only. Each discrepant row
was rerun using the **unchanged** `reward_factors_causal_decoding_2026_11.run.worker`
on local CPU, preserving its legacy fitting, dispatch, lambda and evaluation.
Every local metric then matches the new affine/exact baseline exactly. The
largest published-to-local reward difference is `.001709` (moving seed 43).
This verifies that these discrepancies are not introduced by the new encoder
runner; the precise numerical/platform mechanism was not isolated. No tolerance
was used to waive a mismatch. Compact `cpu_recheck_*` files and the manifest
record the full discrepancies. Six additional unchanged-runner moving
intact/map CPU checks reproduce the published rewards exactly, confirming the
gap denominators. Paired effects use the local historical control, not a blend
of the remote result and a local candidate.

## Interpretation guardrails

These are descriptive across-agent comparisons for only three agents, not
population confidence intervals or independent episode uncertainty estimates.
Higher belief→activation R² need not improve steering: it is measured on intact
histories, whereas target/current inputs during interventions may be outside
that distribution, especially for exact-trained encoders fed probe predictions.
Decoded-source encoders reconstruct activations from a projection of those same
activations; their higher R² is not evidence of more accurate state beliefs.

This is a nonlinear **encoder** experiment, not the previous nonlinear decoder
plus probe-gradient write. A nonlinear encoder can write outside the linear
probe's readable subspace. Thus its behavioral effects do not overturn the
earlier observation that solving a correlational decoder's target error need
not change policy. Whole-activation contextual donors remain retrieval/lookup
controller references, not successful belief writes. No such donors or direct
policy-gradient targets are used here.
