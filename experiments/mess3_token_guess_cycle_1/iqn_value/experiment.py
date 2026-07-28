"""PPO token guessing with an IQN distributional value critic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.plots import simplex_scatter
from envs.hmm import HMMEnv
from experiments.mess3_token_guess_cycle_1.analysis import (
    ProbeResult,
    probe_checkpoint,
)
from experiments.mess3_token_guess_cycle_1.comparison.experiment import (
    BASE_MODEL_CONFIG,
    ENV_CONFIG,
    SMOKE_ENV_STEPS,
    TOTAL_ENV_STEPS,
    _apply_runtime_resources,
)
from experiments.mess3_token_guess_cycle_1.iqn_value.iqn import (
    HUBER_KAPPA_KEY,
    LOSS_COEFFICIENT_KEY,
    NAMESPACE,
    IQNPPOTorchLearner,
    IQNTransformerModel,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.runners import run_tune


IQN_CONFIG = {
    "train_quantiles": 32,
    "value_quantiles": 64,
    "n_cosines": 64,
}
IQN_LOSS_COEFFICIENT = 0.5
IQN_HUBER_KAPPA = 1.0
MODEL_CONFIG = {
    **BASE_MODEL_CONFIG,
    NAMESPACE: IQN_CONFIG,
}
LEARNER_CONFIG = {
    LOSS_COEFFICIENT_KEY: IQN_LOSS_COEFFICIENT,
    HUBER_KAPPA_KEY: IQN_HUBER_KAPPA,
}


def build_config(context: RunContext) -> PPOConfig:
    """Build the controlled PPO configuration with only the critic changed."""

    profile = context.hardware or PROFILES["cpu"]
    config = (
        PPOConfig()
        .environment(HMMEnv, env_config=ENV_CONFIG)
        .framework(
            "torch",
            torch_compile_learner=(
                not context.smoke and profile.learner_device == "cuda"
            ),
            torch_compile_learner_what_to_compile="forward_train",
            torch_compile_learner_dynamo_backend="inductor",
            torch_compile_learner_dynamo_mode="reduce-overhead",
            torch_compile_worker=False,
        )
        .learners(
            learner_class=IQNPPOTorchLearner,
            learner_config_dict=LEARNER_CONFIG,
        )
        .training(
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            # The standard scalar critic loss is disabled and replaced by IQN.
            vf_loss_coeff=0.0,
            entropy_coeff=0.0,
            train_batch_size_per_learner=(
                2_048 if context.smoke else 32_768
            ),
            minibatch_size=256 if context.smoke else 4_096,
            num_epochs=6,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=IQNTransformerModel,
                model_config=MODEL_CONFIG,
            )
        )
        .debugging(seed=context.seed)
    )
    return _apply_runtime_resources(config, context)


def _prior_summary() -> tuple[Path, dict[str, Any]]:
    results_root = (
        Path(__file__).parents[1]
        / "comparison"
        / "results"
    )
    candidates = sorted(results_root.glob("*/comparison_summary.json"))
    if not candidates:
        raise FileNotFoundError(
            "IQN comparison requires a completed three-arm comparison"
        )
    path = candidates[-1]
    return path, json.loads(path.read_text())


def _comparison_metrics(
    prior: dict[str, Any],
    iqn: ProbeResult,
) -> dict[str, dict[str, float]]:
    metrics = {
        name: {
            "r_squared": float(values["r_squared"]),
            "token_accuracy_greedy": float(
                values["token_accuracy_greedy"]
            ),
        }
        for name, values in prior["conditions"].items()
    }
    metrics["iqn_value"] = {
        "r_squared": float(iqn.metrics["r_squared"]),
        "token_accuracy_greedy": float(
            iqn.metrics["token_accuracy_greedy"]
        ),
    }
    return metrics


def _plot_comparison(
    probe: ProbeResult,
    metrics: dict[str, dict[str, float]],
    *,
    path: Path,
) -> None:
    figure = plt.figure(figsize=(10.0, 9.2))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.5, 1.0))
    exact_axis = figure.add_subplot(grid[0, 0])
    decoded_axis = figure.add_subplot(grid[0, 1])
    comparison_axis = figure.add_subplot(grid[1, :])
    colors = np.clip(probe.target_display, 0.0, 1.0)
    simplex_scatter(
        exact_axis,
        probe.target_display,
        colors=colors,
        s=0.5,
        alpha=0.32,
        title="IQN: exact Bayesian belief",
        labels=("s0", "s1", "s2"),
    )
    simplex_scatter(
        decoded_axis,
        probe.decoded_display,
        colors=colors,
        s=0.5,
        alpha=0.32,
        title=(
            "IQN: rank-2 affine decode\n"
            f"held-out R²={probe.metrics['r_squared']:.4f}"
        ),
        labels=("s0", "s1", "s2"),
    )

    names = list(metrics)
    positions = np.arange(len(names))
    width = 0.36
    comparison_axis.bar(
        positions - width / 2,
        [metrics[name]["r_squared"] for name in names],
        width,
        label="belief-probe R²",
    )
    comparison_axis.bar(
        positions + width / 2,
        [metrics[name]["token_accuracy_greedy"] for name in names],
        width,
        label="greedy token accuracy",
    )
    comparison_axis.set_xticks(
        positions,
        [name.replace("_", " ") for name in names],
    )
    comparison_axis.set_ylim(0.0, 1.0)
    comparison_axis.set_ylabel("held-out score")
    comparison_axis.set_title("Controlled four-arm comparison")
    comparison_axis.legend()
    comparison_axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _findings(metrics: dict[str, dict[str, float]]) -> str:
    lines = [
        "# IQN distributional-value comparison",
        "",
        "| condition | held-out R² | greedy token accuracy |",
        "|---|---:|---:|",
    ]
    for name, values in metrics.items():
        lines.append(
            f"| {name} | {values['r_squared']:.4f} | "
            f"{values['token_accuracy_greedy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The IQN condition changes only the value critic and its loss. "
            "Its mean quantile value supplies PPO's scalar GAE baseline, while "
            "sampled quantiles regress against on-policy lambda-return samples.",
            "",
        ]
    )
    return "\n".join(lines)


def run(context: RunContext):
    if context.seed is None:
        raise ValueError("the IQN comparison requires a resolved seed")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    prior_path, prior = _prior_summary()
    outputs.write_json(
        "resolved_recipe.json",
        {
            "condition": "iqn_value",
            "comparison_source": str(prior_path),
            "environment": ENV_CONFIG,
            "model": MODEL_CONFIG,
            "algorithm": "PPO",
            "distributional_value_target": (
                "sampled on-policy GAE lambda-return"
            ),
            "iqn_loss": "quantile Huber",
            "iqn_loss_coefficient": IQN_LOSS_COEFFICIENT,
            "iqn_huber_kappa": IQN_HUBER_KAPPA,
            "total_env_steps": (
                SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
            ),
        },
    )
    result_grid = run_tune(
        build_config(context),
        context,
        stop={
            "env_runners/num_env_steps_sampled_lifetime": (
                SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
            ),
        },
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(
                num_to_keep=1,
                checkpoint_at_end=True,
            ),
        },
    )
    results = list(result_grid)
    if len(results) != 1:
        raise RuntimeError(f"IQN expected one trial, got {len(results)}")
    result = results[0]
    if result.error is not None:
        raise RuntimeError("IQN training failed") from result.error
    if result.checkpoint is None:
        raise RuntimeError("IQN training produced no final checkpoint")

    probe = probe_checkpoint(
        context,
        checkpoint=Path(result.checkpoint.path),
        condition="iqn_value",
    )
    metrics = _comparison_metrics(prior, probe)
    figure_path = context.results_dir / "iqn_belief_and_comparison.png"
    _plot_comparison(probe, metrics, path=figure_path)
    summary = {
        "seed": context.seed,
        "smoke": context.smoke,
        "comparison_source": str(prior_path),
        "conditions": metrics,
        "iqn_probe": probe.metrics,
        "comparison_figure": str(figure_path),
    }
    outputs.write_json("iqn_comparison_summary.json", summary)
    (context.results_dir / "findings.md").write_text(_findings(metrics))
    return summary
