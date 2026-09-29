# Reward-State Quotient seed 42: A/B observation intervention

The final module from the original cycle-6-prefixed run was restored using
RLlib's native RLModule checkpoint and its three files were checked against the
canonical B2 durability manifest. The archived run ID and exact module hash
are in `ab_token_intervention_seed42.json`.

For each of 64 independently seeded, 1,024-decision episodes, the same trained
greedy policy ran in two separately seeded copies of the environment. One
copy received ordinary observations. In the other, each A or B observation
was independently replaced with a fair A/B draw before the policy's recurrent
update. C and the previous executed action were preserved. A third, unexecuted
policy state received the same random draws on the baseline trajectory,
allowing action comparisons without downstream environment divergence.
The reward fraction is reward-state occupancy, including the reset decision.
The uncertainty interval resamples the 64 *paired episodes*, conditioning on
this policy and this intervention distribution.

| Measure | Baseline | Randomized A/B |
| --- | ---: | ---: |
| Reward-state occupancy | 0.331223 | 0.331223 |
| Greedy action fractions (actions 0 / 1 / 2) | 0.331055 / 0.668945 / 0 | 0.331055 / 0.668945 / 0 |
| Paired occupancy difference (95% episode bootstrap interval) | | 0.000000 [0.000000, 0.000000] |

The resampling changed 49.80% of A/B tokens. Across all 65,536 decisions,
neither the closed-loop actions nor shadow-trajectory greedy actions differed.
The shadow action distributions were slightly different (mean total variation
0.001417), so the result establishes greedy-action invariance in these
rollouts rather than exact invariance of the policy probabilities. It does
not establish the same result for sampling from the trained policy.

## Reproduce locally

With the experiment repository and editable `rl-harness` installed via `uv`,
and the repository's B2 read credentials available, restore the module:

```bash
uv run python - <<'PY'
from pathlib import Path
from experiments.mess3_reward_state_action_symmetry_cycle_7.control_campaign import (
    WorkItem, _download_module, _select_records,
)

root = Path("experiments/mess3_reward_state_action_symmetry_cycle_7/variant_2_ctx32_ent")
run = root / "results/mess3_reward_state_action_symmetry_cycle_6-variant_2_ctx32_ent-seed42-10m"
curve = run / "checkpoint_probe_curve.json"
stage, record = _select_records(curve)[-1]
item = WorkItem(
    "variant_2_ctx32_ent", 2, 42, stage, 6,
    int(record["training_iteration"]), int(record["agent_steps"]),
    run.name, record["checkpoint_name"], curve, run / "run_manifest.json",
    root / "artifacts/ab_intervention/seed42_final",
    root / "artifacts/ab_intervention/download.json",
)
_download_module(item)
PY

uv run python -m experiments.mess3_reward_state_action_symmetry_cycle_7.ab_token_intervention \
  --checkpoint experiments/mess3_reward_state_action_symmetry_cycle_7/variant_2_ctx32_ent/artifacts/ab_intervention/seed42_final \
  --run-manifest experiments/mess3_reward_state_action_symmetry_cycle_7/variant_2_ctx32_ent/results/mess3_reward_state_action_symmetry_cycle_6-variant_2_ctx32_ent-seed42-10m/run_manifest.json \
  --output experiments/mess3_reward_state_action_symmetry_cycle_7/variant_2_ctx32_ent/artifacts/ab_intervention/reproduced.json
```

The evaluation refuses to overwrite an existing output file. The checkpoint
and scratch reports remain under ignored `artifacts/`.
