# Nonergodic Belief Explorer · Alpha095

Compares **initialization versus final** for **alpha095 variant 2 and variant 3**
from PR 122. It restores native RLModules and verifies each downloaded file
against the run's B2 durability manifest. No training or Ray cluster is started.

## Generate

Use this experiment checkout with the editable sibling `rl-harness` including
`analysis.simplex` and its packaged viewer assets. The library owns generic
metrics, serialization and rendering; this experiment owns its controlled target,
checkpoint restoration, histories, activation extraction and scientific protocol.
B2 access uses the existing `B2_*`
environment variables; never put credentials in this directory.

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 uv run --extra visualization \
  python -m experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.simplex_data \
  --output experiments/nonergodic_mess3_reward_state_action_symmetry_cycle_5/artifacts/nonergodic-belief-explorer
```

The output directory must not exist. Default: 160 independent fit episodes and
160 test episodes per variant, 128 positions each (20,480 rows per split).
`--episodes 4` is a pipeline smoke, not a scientific result. `--download-only`
restores and loads the four checkpoints without running probes.

Choose any available sites with `--sites post_final_norm` or
`--sites layer_3 layer_4`. Only those representations are retained and fitted.
The default includes every encoder block and final normalization, deriving the
block count from the restored model. `--primary-site` must name a selected site;
by default use `post_final_norm` if selected, otherwise the last requested site.
Fix this choice before evaluation.

Open the generated `index.html` directly, or serve its directory with
`python -m http.server`. Plotly and data are bundled locally; the viewer requires
no CDN, server API, login, or Python after generation. Keep its HTML, CSS, scripts,
and three JSON reports (`report_0.json` through `report_2.json`) together.
The `raw/` NPZ files are optional for the viewer.
Do not commit generated point clouds, raw trajectories, or downloaded checkpoints.

## Geometry and controls

Each top panel plots one three-coordinate block of the exact six-state belief.
Its coordinate sum is the component posterior. In sequence mode the triangle's
vertices are `(w,0,0)`, `(0,w,0)`, `(0,0,w)`. The two lower panels show raw affine
activation predictions in those same coordinates, with initialization and final
overlaid or shown separately. Predicted planes use the raw predicted component
mass. Common axis limits include negative coordinates, excess mass, and plane
vertices; no projection, clipping, or renormalization is performed.

Points and planes blend state-vertex colors (red, green, blue) by local state
composition. Brightness interpolates from dark gray at weight zero to the state
color at weight one, using square-root weight. For raw overshoots, positive
coordinates determine hue and weight is bounded to [0,1] **for color only**.
Checkpoint colors remain on outlines; initialization uses diamonds,
final uses circles. Serialized coordinates retain float64 precision; captions
and hover values use four decimal places.
As the slider advances, past positions remain as small dots and only the current
position has a large dot. There are no sequence connector lines.

## Refresh existing geometry and attach scores

`simplex_scores.py` reads a previously exported alpha095 bundle and preserves
its coordinates and geometry metrics. It attaches archived checkpoint NTP
R²/MSE and empirical reward-state occupancy. Existing NTP probes use each
checkpoint's own greedy policy and 20,000 held-out steps; they are not the same
histories as the matched final-policy belief probes. The table labels these
archived scores and exposes their definitions.

The compact `results/explorer_predictive_scores.json` adds only the missing
activation-to-log-NTP probes at the last residual block (`layer_4`), for actual
initialization and final checkpoints of variants 2 and 3. They use the original
160-fit/160-test complete histories, natural log with a `1e-12` probability floor,
and training-only whole-episode SVD-cutoff CV. Replay is checked exactly against
saved held-out observations, beliefs, tokens, actions and component assignments.
No belief probes or task-success estimates are refitted.

Refresh using that saved supplement, with no model loading or fitting:

```bash
uv run --extra visualization \
  python -m experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.simplex_scores \
  --source PREVIOUS_EXPORT \
  --output NEW_EXPORT \
  --supplement experiments/nonergodic_mess3_reward_state_action_symmetry_cycle_5/results/explorer_predictive_scores.json
```

The saved supplement is bound to the original per-variant report hashes. For a
different source export, choose a new supplement path and explicitly add
`--fit-missing-log-ntp` only if new predictive probes are authorized. Geometry is
never regenerated by this refresh command. Keep the source raw arrays for replay.

No valid alpha095 Bayes-max occupancy is saved, so that score is omitted. The
cycle-5 reward audit uses different component parameters and evaluates a feasible
observer rather than a Bayes-optimal controller; it cannot supply this maximum.
The shared viewer accepts a future saved maximum/reference with its definition,
without requiring any layer probes to be rerun.

The sequence slider covers reset/BOS (`t=0`) through the terminal observation
(`t=127`): **128 contexts, 127 executed actions**. At `t>0`, the displayed
preceding action is `actions[t-1]`; the action selected after observing the
current token is `actions[t]`. At the terminal position it is `-1`.

The target is the **decision-time arrival belief**, adapted from the blog's
passive HMM to the experiment's delayed, action-controlled transducer. The
independent filter conditions on the visible token using the previous pending
edge kernel, then applies the executed transition. It is verified against public
environment diagnostics on every row. Rewards and latent states do not enter
the filter or policy inputs. True component labels only stratify example selection.

Within each variant, a greedy final-policy rollout supplies the same complete
histories to both checkpoints. Initialization is therefore an off-policy
representation control. Different variants have different histories/visitation;
cross-variant raw MSE is not an equal-distribution comparison.

Primary representation: **post-final RMSNorm**, fixed before evaluation.
Four pre-final-normalization residual streams are additional, separately fitted
comparisons. Full-episode causal forward passes are checked against prefix
forward passes and the stateful policy API.

The shared `evaluate_belief_geometry` battery provides:

- Independent whole-episode fit/test splits and training-only SVD-cutoff CV.
- Train-mean, current-token/action, exact pending-token and log-pending-token
  baselines (log floor `1e-12`), plus actual saved initialization.
- Three held-out label-permutation and covariance-matched Gaussian nulls.
- Six-state and informative-contrast metrics and 200 episode-bootstrap
  resamples for paired comparisons, holding each fitted probe fixed.

The viewer displays a seeded 6,144-row held-out subsample, identical across
checkpoints, plus the first four test episodes from each true component
(up to eight total, without selecting on decoding quality). Metrics use the
**full** held-out set. Report files retain target variance, raw MSE, normalized
MSE, R², coverage, overshoots, provenance, and control comparisons.

This analysis measures linear accessibility, not causal use or unique
identification of an internal Bayesian computation. One trained seed per
variant does not measure variation across independently trained models.

## Verification

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 uv run pytest -q \
  tests/test_nonergodic_simplex.py \
  tests/test_nonergodic_mess3_reward_state_action_symmetry_cycle_5.py
uv run pytest -q -m "not slow"
```

Viewer tests moved with the assets to the harness:
`node --test ../rl-harness/tests/test_nonergodic_belief_explorer.cjs`.
See the installed `analysis/nonergodic_belief_explorer/README.md` for the shared API and
[passive PR 119 adoption](../../nonergodic_mess3_token_guess_cycle_1/simplex_adoption.md)
for a concrete adaptation and reusable prompt.
