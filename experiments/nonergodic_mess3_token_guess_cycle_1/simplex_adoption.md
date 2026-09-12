# Use the shared simplex viewer for PR #119

This guide targets [PR #119](https://github.com/Al-does/alex-rl-experiments/pull/119),
the passive `nonergodic_mess3_token_guess_cycle_1` study. It is an adoption recipe,
not a new set of trained-model findings. The alpha095 action-symmetry study is
the first migrated consumer; its checkpoint and rollout code can serve as a
structural example, but its target adapter cannot be reused here.

## Shared API and installation

Use the sibling RL-harness checkout containing `analysis.simplex`. Its packaged
`analysis/simplex_viewer/README.md` documents the complete contract.
From this experiment repo, run:

```bash
uv sync --group dev --extra visualization
uv run python -c "from analysis.simplex import build_simplex_run, write_simplex_viewer"
```

The visualization extra now requests `rl-harness[visualization]`. There is no
experiment-owned JavaScript to copy or modify. Metrics/export/rendering are
generic; checkpoint restoration, environment interaction, target timing and
representation extraction remain here.

## 1. Restore the real checkpoints

Start with `ppo/results/20260911T003313Z-d1b8abe3/run_manifest.json`, which records
a completed run and B2 durability-manifest location. `condition_summary.json`
records an `initial_checkpoint`, log-spaced training checkpoints and final
`checkpoint_000000`. Follow the canonical remote manifest to locate and verify
the native module files; archived absolute `/root/work/...` paths are provenance,
not paths expected on a new machine.

Use `analysis.checkpoints.load_module_only` to restore each actual native
module on CPU. Begin with initialization versus final on identical histories.
Do not reconstruct a random "initialization" from a seed. The later
`20260911T072936Z-cefb4090` manifest currently says `running` and has no completed
checkpoint inventory in this checkout; do not treat it as another completed
run without separately verifying durable completion.

## 2. Keep PR #119's passive filtering and delayed-token timing

Actions guess a token for reward. `NextTokenGuessTask.resolve_action` uses the
same model transition matrix for every guess; guesses do not change the
environment. Observations contain only the delayed token. **Do not add executed
actions, action-conditioned matrices, or rewards to the Bayesian update.**

The existing `analysis._target_adapter` returns a filtered **source belief**
`weighted_belief`. It obtains it by solving
`source @ model.transition_matrix = info["belief_current"]`.
The public diagnostic is an arrival belief, one transition later than the
source used to predict the pending token. The correct next-token control is
`source @ model.emission_matrix`, not `arrival @ model.emission_matrix`.
The existing test
`test_probe_next_token_target_uses_source_belief_not_an_extra_transition`
checks this distinction.

For complete freshly collected histories, a forward filter avoids inverting
the transition matrix. Start from the model prior at BOS; for each newly visible
delayed token, apply the passive edge-emitting operator:

```python
import numpy as np

from analysis.probes import predictive_belief_sequence
from experiments.nonergodic_mess3_token_guess_cycle_1.process import nonergodic_mess3_model

model = nonergodic_mess3_model()
source_beliefs = predictive_belief_sequence(
    model.initial_distribution,
    model.edge_transition_matrices[visible_tokens[1:]],  # skip BOS; 127 revealed tokens
)
pending_token_probabilities = source_beliefs @ model.emission_matrix
np.testing.assert_allclose(
    source_beliefs @ model.transition_matrix,
    arrival_diagnostics,  # public belief_current at all 128 positions
    atol=1e-10,
)
```

`test_simplex_passive_adoption.py` executes this complete-history filter,
compares it with the existing target adapter and public diagnostics, and checks
that different guesses leave histories and targets identical.

## 3. Collect matched complete episodes and extract activations

Use independent seeded trajectories for fitting and held-out evaluation. Keep
each complete trajectory within its split. PR #119 has 128 contexts: reset/BOS
plus 127 delayed-token observations; retain the terminal observation for the
slider even though no next action exists there.

Reset recurrent/transformer state at each episode. Obtain activations through
the experiment model's public encoding API, replaying the **same** observed
histories through initialization and final checkpoints. Default to the
post-final-normalization representation already used by the experiment; label
pre-normalization or per-layer sites as separate robustness comparisons.
Do not choose a representation using held-out MSE.

The existing `_collect` is useful for the target/extraction conventions, but its
concatenated rollout report is not already a complete-episode viewer payload.
Preserve explicit episode/position IDs during new collection; do not blindly
reshape previously flattened samples or reconstruct point clouds from scalar
summary metrics. Report whether all positions or a warmup-masked subset are
scored. The sequence viewer can include BOS through position 127 as a new
complete-history diagnostic, distinct from archived post-warmup metrics.

## 4. Fit and export with the shared code

Call `analysis.belief_geometry.evaluate_belief_geometry` on 2D activation/target
rows with whole-episode `train_groups` and `test_groups`, actual
`initialization_features`, predictive/log-predictive and observation controls,
and the study's component-posterior contrasts. Choose the SVD cutoff using
training groups only. Retain the battery's null and episode-bootstrap reports.

Reshape **aligned returned predictions** to the held-out target array's
`(episodes, 128, 6)` shape, then call:

```python
from pathlib import Path

import numpy as np

from analysis.simplex import build_simplex_run, write_simplex_viewer

run = build_simplex_run(
    name="PR 119 · token guess",
    description="Passive source-belief target; matched initialization/final histories",
    targets=test_source_beliefs,
    predictions={
        "Initialization": {
            site: battery.baseline_predictions[f"initialization/{site}"].reshape(test_source_beliefs.shape)
            for site in test_features
        },
        "Final": {
            site: values.reshape(test_source_beliefs.shape)
            for site, values in battery.predictions.items()
        },
    },
    components={"Component A": [0, 1, 2], "Component B": [3, 4, 5]},
    state_labels=model.state_labels,
    primary_site=primary_site,
    tokens=token_labels,  # (episodes, 128) strings, including "BOS"
    cloud_rows=heldout_cloud_rows,
    example_episodes=np.arange(min(8, len(test_source_beliefs))),
)
write_simplex_viewer(
    Path("experiments/nonergodic_mess3_token_guess_cycle_1/ppo/artifacts/simplex"),
    [run],
    title="PR 119 · nonergodic belief geometry",
    description=protocol_description,
    reports={"Probe battery": battery.report, "Provenance": provenance},
)
```

This handoff assumes real measured arrays from the preceding steps; it is not a
standalone script. `heldout_cloud_rows` indexes C-order flattened held-out rows.
Cloud subsampling changes the display, not metrics, which use every supplied
position. Preserve float64 targets and raw predictions, including negative and
out-of-simplex values. Pass no action notes for this passive study.

Record both repository commits, source and checkpoint hashes, run IDs, seeds,
fit/test budgets, exact target timing, representation sites, controls, policy
sampling and any warmup. The exporter bundles reports, Plotly and data for
offline use and refuses to overwrite its files. Keep the full output tree
under ignored `artifacts/`; publish compact findings separately.

## Example prompt

Copy this after adopting the shared-library and experiment migration PRs:

> Update the experiment from https://github.com/Al-does/alex-rl-experiments/pull/119
> to generate and display its belief geometry with RL-harness's shared
> `analysis.simplex` API. Follow this experiment's `simplex_adoption.md` and the
> harness's packaged simplex-viewer guide.
>
> Use the completed PPO run `20260911T003313Z-d1b8abe3`. Restore and verify its
> actual initialization and final native module checkpoints through the recorded
> B2 durability manifest and `load_module_only`; do not train new models.
> Collect independent complete fit/test episodes and replay identical held-out
> token histories through both checkpoints. Retain all 128 positions, including
> BOS and the terminal observation.
>
> This is a passive nonergodic process: guesses affect reward, not transitions.
> Preserve PR #119's filtered-source-belief/pending-token timing. Use only newly
> visible delayed tokens in the passive Bayesian update. Check
> `source @ transition == belief_current` and compute the predictive control
> from `source @ emission`; do not add action-conditioned kernels or rewards.
>
> Fit with `evaluate_belief_geometry`, keeping whole episodes grouped, training-only
> regularization, initialization, predictive/observation controls, nulls and
> episode-bootstrap comparisons. Use post-final normalization as primary.
> Export with `build_simplex_run` and `write_simplex_viewer`, with components
> [0,1,2] and [3,4,5], no action fields, raw affine coordinates, and full precision.
> Provide held-out point clouds, initialization/final controls and complete example
> histories with a timestep slider in the shared four-panel viewer.
>
> Run the relevant tests, validate against the existing passive target adapter,
> record provenance and sampling, and open a PR containing the adapter and compact
> findings. Give me a live preview and a downloadable offline viewer; keep
> checkpoints and large generated data out of Git.
