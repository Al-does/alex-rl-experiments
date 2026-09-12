# Alpha095 checkpoint simplex explorer

Compares **initialization versus final** for **alpha095 variant 2 and variant 3**
from PR 122. It restores native RLModules and verifies each downloaded file
against the run's B2 durability manifest. No training or Ray cluster is started.

## Generate

Use this experiment checkout with the editable sibling `rl-harness`. The
experiment's existing tests require harness commit
`bafed8e` (`devin/1789107129-record-rllib-reseed-issue`), which provides
`FreshEpisodeSingleAgentEnvRunner`. B2 access uses the existing `B2_*`
environment variables; never put credentials in this directory.

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 uv run --extra visualization \
  python -m experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.simplex_data \
  --output experiments/nonergodic_mess3_reward_state_action_symmetry_cycle_5/artifacts/simplex-viewer
```

The output directory must not exist. Default: 160 independent fit episodes and
160 test episodes per variant, 128 positions each (20,480 rows per split).
`--episodes 4` is a pipeline smoke, not a scientific result. `--download-only`
restores and loads the four checkpoints without running probes.

Open the generated `index.html` directly, or serve its directory with
`python -m http.server`. Plotly and data are bundled locally; the viewer requires
no CDN, server API, login, or Python after generation. Keep its HTML, CSS, scripts,
and three JSON reports together. The `raw/` NPZ files are optional for the viewer.
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
Checkpoint colors remain on outlines/trails; initialization uses diamonds,
final uses circles. Serialized coordinates retain float64 precision; captions
and hover values use four decimal places.

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
node --test tests/test_simplex_viewer.cjs
```
