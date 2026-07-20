"""Reproduce and visualize the first retained IQN checkpoint."""

from __future__ import annotations

from pathlib import Path

from ray import tune

from experiments.mess3_token_guess_cycle_1.analysis import probe_checkpoint
from experiments.mess3_token_guess_cycle_1.iqn_value.experiment import (
    build_config,
)
from experiments.mess3_token_guess_cycle_1.iqn_value_20m.experiment import (
    _steps,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.runners import run_tune


TARGET_ENV_STEPS = 800_000
SMOKE_ENV_STEPS = 4_096
ORIGINAL_CHECKPOINT_STEPS = 827_560
ORIGINAL_CHECKPOINT_R_SQUARED = 0.9901978873112481


def run(context: RunContext):
    if context.seed is None:
        raise ValueError("the IQN reproduction requires a resolved seed")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    target_steps = SMOKE_ENV_STEPS if context.smoke else TARGET_ENV_STEPS
    outputs.write_json(
        "resolved_recipe.json",
        {
            "condition": "iqn_first_checkpoint_reproduction",
            "target_env_steps": target_steps,
            "seed": context.seed,
            "original_checkpoint_steps": ORIGINAL_CHECKPOINT_STEPS,
            "original_checkpoint_r_squared": ORIGINAL_CHECKPOINT_R_SQUARED,
            "qualification": (
                "fresh rerun because the original checkpoint was not uploaded"
            ),
        },
    )
    result_grid = run_tune(
        build_config(context),
        context,
        stop={"env_runners/num_env_steps_sampled_lifetime": target_steps},
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(
                num_to_keep=1,
                checkpoint_at_end=True,
            ),
        },
    )
    results = list(result_grid)
    if len(results) != 1:
        raise RuntimeError(
            f"IQN reproduction expected one trial, got {len(results)}"
        )
    result = results[0]
    if result.error is not None:
        raise RuntimeError("IQN reproduction training failed") from result.error
    if result.checkpoint is None:
        raise RuntimeError("IQN reproduction produced no checkpoint")

    probe = probe_checkpoint(
        context,
        checkpoint=Path(result.checkpoint.path),
        condition="iqn_first_checkpoint_reproduction",
    )
    sampled_steps = _steps(result.metrics or {})
    if sampled_steps is None:
        raise RuntimeError("IQN reproduction result omitted sampled steps")
    summary = {
        "seed": context.seed,
        "smoke": context.smoke,
        "sampled_agent_steps": sampled_steps,
        "probe": probe.metrics,
        "original_checkpoint": {
            "sampled_agent_steps": ORIGINAL_CHECKPOINT_STEPS,
            "r_squared": ORIGINAL_CHECKPOINT_R_SQUARED,
        },
        "is_exact_historical_checkpoint": False,
        "figure": str(context.results_dir / "belief_simplex.png"),
    }
    outputs.write_json("reproduction_summary.json", summary)
    (context.results_dir / "findings.md").write_text(
        "\n".join(
            [
                "# First IQN checkpoint reproduction",
                "",
                f"- Reproduction steps: {sampled_steps:,}",
                f"- Reproduction held-out R²: "
                f"{probe.metrics['r_squared']:.4f}",
                f"- Historical held-out R²: "
                f"{ORIGINAL_CHECKPOINT_R_SQUARED:.4f}",
                "",
                "This is a fresh seed-42 rerun, not the destroyed historical "
                "checkpoint.",
                "",
            ]
        )
    )
    return summary
