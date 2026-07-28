"""Run all action-symmetry variants sequentially."""

from dataclasses import replace

from harness.artifacts import RunArtifacts
from harness.context import RunContext

from experiments.mess3_reward_state_action_symmetry_cycle_2.analysis import (
    build_battery_mse_report,
    plot_battery_mse_curves,
)
from experiments.mess3_reward_state_action_symmetry_cycle_2.shared import (
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
    mse_report = build_battery_mse_report(summaries)
    mse_report["figures"] = plot_battery_mse_curves(
        mse_report,
        results_dir=context.results_dir,
    )
    outputs.write_json("battery_mse_metrics.json", mse_report)
    summary = {
        "seed": context.seed,
        "smoke": context.smoke,
        "algorithm": "PPO",
        "variants": summaries,
        "mse_report": "battery_mse_metrics.json",
        "mse_figures": mse_report["figures"],
    }
    outputs.write_json("battery_summary.json", summary)
    return summary
