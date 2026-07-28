"""Reproduce the early token-guess IQN result with RL Harness IQN."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from envs.hmm import HMMEnv
from experiments.mess3_token_guess_cycle_1.analysis import probe_checkpoint
from experiments.mess3_token_guess_cycle_1.comparison.experiment import (
    BASE_MODEL_CONFIG,
    ENV_CONFIG,
    _apply_runtime_resources,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.runners import run_tune
from learners import (
    HUBER_KAPPA_KEY,
    LOSS_COEFFICIENT_KEY,
    IQNPPOTorchLearner,
)
from learners.models import IQNValueMixin, TransformerModel
from learners.models.iqn_value import NAMESPACE as IQN_NAMESPACE


class GenericIQNTransformerModel(IQNValueMixin, TransformerModel):
    """Historical transformer recipe using the promoted IQN value option."""


TARGET_ENV_STEPS = 800_000
SMOKE_ENV_STEPS = 4_096
HISTORICAL_SAMPLED_STEPS = 827_560
HISTORICAL_R_SQUARED = 0.9906668133729878
HISTORICAL_GREEDY_ACCURACY = 0.6784
R_SQUARED_TOLERANCE = 0.02
ACCURACY_TOLERANCE = 0.02
IQN_CONFIG = {
    "train_quantiles": 32,
    "value_quantiles": 64,
    "n_cosines": 64,
}
LEARNER_CONFIG = {
    LOSS_COEFFICIENT_KEY: 0.5,
    HUBER_KAPPA_KEY: 1.0,
}
MODEL_CONFIG = {
    **BASE_MODEL_CONFIG,
    IQN_NAMESPACE: IQN_CONFIG,
}


def build_config(context: RunContext) -> PPOConfig:
    """Build the historical recipe through generic library components."""

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
                module_class=GenericIQNTransformerModel,
                model_config=MODEL_CONFIG,
            )
        )
        .debugging(seed=context.seed)
    )
    return _apply_runtime_resources(config, context)


def _metric(metrics: Mapping[str, Any], path: str) -> float | None:
    direct = metrics.get(path)
    if isinstance(direct, Real):
        return float(direct)
    value: Any = metrics
    for part in path.split("/"):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return float(value) if isinstance(value, Real) else None


def _sampled_steps(metrics: Mapping[str, Any]) -> int | None:
    for path in (
        "env_runners/num_env_steps_sampled_lifetime",
        "num_env_steps_sampled_lifetime",
    ):
        value = _metric(metrics, path)
        if value is not None:
            return int(value)
    return None


def run(context: RunContext) -> dict[str, Any]:
    if context.seed is None:
        raise ValueError("the generic IQN reproduction requires a resolved seed")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    target_steps = SMOKE_ENV_STEPS if context.smoke else TARGET_ENV_STEPS
    outputs.write_json(
        "resolved_recipe.json",
        {
            "condition": "iqn_generic_reproduction",
            "implementation": "rl_harness_ppo_iqn",
            "target_env_steps": target_steps,
            "seed": context.seed,
            "model": MODEL_CONFIG,
            "learner": LEARNER_CONFIG,
            "historical_reference": {
                "sampled_agent_steps": HISTORICAL_SAMPLED_STEPS,
                "r_squared": HISTORICAL_R_SQUARED,
                "token_accuracy_greedy": HISTORICAL_GREEDY_ACCURACY,
            },
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
            f"generic IQN reproduction expected one trial, got {len(results)}"
        )
    result = results[0]
    if result.error is not None:
        raise RuntimeError("generic IQN reproduction training failed") from result.error
    if result.checkpoint is None:
        raise RuntimeError("generic IQN reproduction produced no checkpoint")

    probe = probe_checkpoint(
        context,
        checkpoint=Path(result.checkpoint.path),
        condition="iqn_generic_reproduction",
    )
    sampled_steps = _sampled_steps(result.metrics or {})
    if sampled_steps is None:
        raise RuntimeError("generic IQN result omitted sampled steps")
    r_squared = float(probe.metrics["r_squared"])
    accuracy = float(probe.metrics["token_accuracy_greedy"])
    comparison = {
        "r_squared_delta": r_squared - HISTORICAL_R_SQUARED,
        "token_accuracy_greedy_delta": (
            accuracy - HISTORICAL_GREEDY_ACCURACY
        ),
        "within_reference_tolerance": (
            abs(r_squared - HISTORICAL_R_SQUARED) <= R_SQUARED_TOLERANCE
            and abs(accuracy - HISTORICAL_GREEDY_ACCURACY)
            <= ACCURACY_TOLERANCE
        ),
    }
    summary = {
        "seed": context.seed,
        "smoke": context.smoke,
        "sampled_agent_steps": sampled_steps,
        "probe": probe.metrics,
        "historical_reference": {
            "sampled_agent_steps": HISTORICAL_SAMPLED_STEPS,
            "r_squared": HISTORICAL_R_SQUARED,
            "token_accuracy_greedy": HISTORICAL_GREEDY_ACCURACY,
        },
        "comparison": comparison,
        "figure": str(context.results_dir / "belief_simplex.png"),
    }
    outputs.write_json("reproduction_summary.json", summary)
    (context.results_dir / "findings.md").write_text(
        "\n".join(
            [
                "# Generic PPO IQN reproduction",
                "",
                f"- Sampled agent steps: {sampled_steps:,}",
                f"- Held-out belief R²: {r_squared:.4f}",
                f"- Historical belief R²: {HISTORICAL_R_SQUARED:.4f}",
                f"- Greedy token accuracy: {accuracy:.4f}",
                (
                    "- Historical greedy token accuracy: "
                    f"{HISTORICAL_GREEDY_ACCURACY:.4f}"
                ),
                (
                    "- Within reference tolerance: "
                    f"{str(comparison['within_reference_tolerance']).lower()}"
                ),
                "",
            ]
        )
    )
    return summary
