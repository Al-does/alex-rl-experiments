"""Run all action-symmetry variants sequentially."""

from dataclasses import replace

from harness.artifacts import RunArtifacts
from harness.context import RunContext

from experiments.mess3_reward_state_action_symmetry_cycle_1.shared import (
    run_condition,
)


def run(context: RunContext):
    summaries = {}
    for variant in (1, 2, 3):
        name = f"variant_{variant}"
        summaries[name] = run_condition(
            replace(
                context,
                results_dir=context.results_dir / name,
                artifacts_dir=context.artifacts_dir / name,
                resume_from=None,
            ),
            variant,
        )
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    summary = {
        "seed": context.seed,
        "smoke": context.smoke,
        "algorithm": "PPO",
        "variants": summaries,
    }
    outputs.write_json("battery_summary.json", summary)
    return summary
