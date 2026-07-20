"""Run IQN token guessing for 3M steps with gamma equal to one."""

from __future__ import annotations

from pathlib import Path

from ray import tune

from experiments.mess3_token_guess_cycle_1.analysis import probe_checkpoint
from experiments.mess3_token_guess_cycle_1.iqn_value.experiment import (
    build_config as build_standard_iqn_config,
)
from experiments.mess3_token_guess_cycle_1.iqn_value_20m.experiment import (
    _metric,
    _steps,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.runners import run_tune


TOTAL_ENV_STEPS = 3_000_000
SMOKE_ENV_STEPS = 4_096
GAMMA = 1.0


def build_config(context: RunContext):
    """Build the controlled IQN recipe with only gamma changed."""

    return build_standard_iqn_config(context).training(gamma=GAMMA)


def run(context: RunContext):
    if context.seed is None:
        raise ValueError("the gamma-one IQN run requires a resolved seed")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    outputs.write_json(
        "resolved_recipe.json",
        {
            "condition": "iqn_gamma_1_3m",
            "gamma": GAMMA,
            "total_env_steps": target_steps,
            "qualification": (
                "MESS3 episodes truncate and bootstrap, so gamma=1 defines "
                "an unbounded continuing-return value target"
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
            f"gamma-one IQN expected one trial, got {len(results)}"
        )
    result = results[0]
    if result.error is not None:
        raise RuntimeError("gamma-one IQN training failed") from result.error
    if result.checkpoint is None:
        raise RuntimeError("gamma-one IQN produced no final checkpoint")

    probe = probe_checkpoint(
        context,
        checkpoint=Path(result.checkpoint.path),
        condition="iqn_gamma_1_3m",
    )
    metrics = result.metrics or {}
    sampled_steps = _steps(metrics)
    if sampled_steps is None:
        raise RuntimeError("gamma-one IQN result omitted sampled steps")
    summary = {
        "seed": context.seed,
        "smoke": context.smoke,
        "gamma": GAMMA,
        "sampled_agent_steps": sampled_steps,
        "probe": probe.metrics,
        "training": {
            "episode_return_mean": _metric(
                metrics,
                "env_runners/episode_return_mean",
            ),
            "episode_len_mean": _metric(
                metrics,
                "env_runners/episode_len_mean",
            ),
            "iqn_loss": _metric(
                metrics,
                "learners/default_policy/iqn_value/loss",
            ),
            "mean_quantile_spread": _metric(
                metrics,
                "learners/default_policy/iqn_value/mean_quantile_spread",
            ),
            "value_explained_variance": _metric(
                metrics,
                "learners/default_policy/vf_explained_var",
            ),
        },
        "figure": str(context.results_dir / "belief_simplex.png"),
    }
    outputs.write_json("gamma_1_summary.json", summary)
    (context.results_dir / "findings.md").write_text(
        "\n".join(
            [
                "# IQN gamma=1.0 at 3M steps",
                "",
                f"- Sampled agent steps: {sampled_steps:,}",
                f"- Greedy token accuracy: "
                f"{100.0 * probe.metrics['token_accuracy_greedy']:.2f}%",
                f"- Held-out belief R²: {probe.metrics['r_squared']:.4f}",
                f"- IQN loss: {summary['training']['iqn_loss']}",
                f"- Mean quantile spread: "
                f"{summary['training']['mean_quantile_spread']}",
                "",
                "Gamma one is formally unbounded for this continuing, "
                "positive-reward process because truncations bootstrap.",
                "",
            ]
        )
    )
    return summary
