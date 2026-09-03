# Recovering Cycle 6 agents from B2

Compact results on `main` identify the scientific run; B2 stores the full
RLlib checkpoint directories. For variant 2, start with:

```text
experiments/mess3_reward_state_action_symmetry_cycle_6/battery/results/variant_2_agent_registry.json
```

The registry records, for each latest completed seed:

- final reward occupancy and held-out belief-probe metrics;
- the committed `run_manifest.json` and `tune_summary.json`;
- the canonical B2 durability-manifest key;
- the exact final RLlib checkpoint object prefix;
- the file count and byte total verified against B2.

At publication time, seed 42 is the strongest default for variant 2: all three
seeds select greedy action 1, while seed 42 has both the highest measured
reward-state occupancy and the lowest held-out belief MSE. Use multiple seeds
when the downstream test is intended to estimate variability rather than to
exercise one representative high-performing policy.

## Download

Cloud agents receive B2 credentials through `B2_BUCKET`, `B2_ENDPOINT`,
`B2_APPLICATION_KEY_ID`, and `B2_APPLICATION_KEY`. Read the desired
`b2_checkpoint_prefix` from the registry, then download the *whole* checkpoint
directory:

```bash
export AWS_ACCESS_KEY_ID="$B2_APPLICATION_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$B2_APPLICATION_KEY"

PREFIX="$(python - <<'PY'
import json
from pathlib import Path

registry = Path(
    "experiments/mess3_reward_state_action_symmetry_cycle_6/"
    "battery/results/variant_2_agent_registry.json"
)
agents = json.loads(registry.read_text())["agents"]
print(next(agent["b2_checkpoint_prefix"] for agent in agents if agent["seed"] == 42))
PY
)"

uv run --with awscli aws s3 cp --recursive \
  --endpoint-url "$B2_ENDPOINT" \
  "s3://$B2_BUCKET/$PREFIX" \
  artifacts/cycle6_variant2_seed42/checkpoint/
```

The destination should contain `rllib_checkpoint.json`,
`algorithm_state.pkl`, and the `learner_group/` and `env_runner/` component
trees. Restore the directory through RLlib's public checkpoint API, for example
`Algorithm.from_checkpoint(...)`; do not reconstruct an algorithm from only
one private component file.

## Verify before use

The run's committed `run_manifest.json` contains
`remote_artifacts.canonical_manifest_key`. It should match
`b2_manifest_key` in the registry. The canonical manifest in B2 lists SHA-256
and size for every checkpoint object. A recovery tool should verify those
values after download, especially when the agent will be used for a new
scientific result.

If a later rerun is added, regenerate or update the registry by selecting the
latest completed `run_manifest.json` for each seed, then resolve the final
variant-2 checkpoint from that run's `variant_2/tune_summary.json`. Do not pick
a run by lexicographic directory name alone.
