import json

from experiments.mess3_reward_state_action_symmetry_cycle_7.analysis import (
    probe_checkpoint,
)


def analyze_checkpoint(
    context,
    *,
    checkpoint,
    condition,
    checkpoint_label,
    agent_steps,
    training_iteration,
):
    result = probe_checkpoint(
        context,
        checkpoint=checkpoint,
        condition=condition,
        agent_steps=agent_steps,
    )
    report = {
        "condition": condition,
        "checkpoint_label": checkpoint_label,
        "agent_steps": agent_steps,
        "training_iteration": training_iteration,
        "probe": result.metrics,
        "policy": {
            "mean_reward": result.metrics["reward_state_2_fraction_greedy"],
            "greedy_action_fractions": result.metrics["greedy_action_fractions"],
        },
    }
    (context.results_dir / "checkpoint_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report
