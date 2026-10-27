from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from functools import partial

from envs.hmm import HMMEnv
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.runners import run_tune
from ray import tune

from experiments.factored_representations_reproduction_PPO_2026_08.shared import (
    _save_initial_checkpoint,
)
from experiments.pusher_b import rl as pusher_b_ppo
from experiments.pusher_b.process import EPISODE_LENGTH
from experiments.pusher_b.rl import PPOSettings
from experiments.storage.training_curves import write_training_curves

from .metrics import GuessOrPassMetricsCallback
from .process import ppo_environment_config
from .transfer import (
    FrozenEncoderPPOTorchLearner,
    initialize_algorithm_transfer,
    transfer_overrides,
    verify_frozen_update,
)

TOTAL_ENV_STEPS = 10_000_000
SMOKE_ENV_STEPS = pusher_b_ppo.SMOKE_ENV_STEPS
LEARNING_RATE = 5e-5
ENTROPY_COEFF_SCHEDULE = ((0, 0.01), (6_000_000, 0.01), (8_000_000, 0.0))

SETTINGS = PPOSettings(
    total_env_steps=TOTAL_ENV_STEPS,
    lr=LEARNING_RATE,
    entropy_coeff=ENTROPY_COEFF_SCHEDULE,
    next_token_aux=False,
)


def _metric(metrics: Mapping[str, object], path: str) -> object | None:
    value: object = metrics
    for part in path.split("/"):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _init_algorithm(
    *,
    algorithm,
    checkpoint_path: str,
    warm_start_path: str | None = None,
    transfer_metadata: dict[str, object] | None = None,
    recipe_path: str,
    **kwargs,
) -> None:
    if warm_start_path is not None:
        if transfer_metadata is None:
            raise ValueError("warm start requires transfer metadata")
        initialize_algorithm_transfer(
            algorithm,
            module_checkpoint=warm_start_path,
            algorithm_name="ppo",
            trunk_mode=str(transfer_metadata["trunk_mode"]),
            metadata=transfer_metadata,
            recipe_path=recipe_path,
        )
    _save_initial_checkpoint(
        algorithm=algorithm,
        checkpoint_path=checkpoint_path,
        **kwargs,
    )


def build_config(context: RunContext, *, cost: float):
    metadata = transfer_overrides()
    config = pusher_b_ppo.build_config(
        context,
        preset="b10",
        settings=SETTINGS,
    )
    config.environment(
        HMMEnv,
        env_config=ppo_environment_config(cost),
    )
    config.callbacks(
        callbacks_class=GuessOrPassMetricsCallback,
        on_algorithm_init=partial(
            _init_algorithm,
            checkpoint_path=str(context.artifacts_dir / "initial_checkpoint"),
            warm_start_path=(
                str(context.resume_from)
                if context.resume_from is not None
                else None
            ),
            transfer_metadata=metadata,
            recipe_path=str(context.artifacts_dir / "resolved_recipe.json"),
        ),
    )
    if metadata is not None and metadata.get("trunk_mode") == "frozen":
        config.learners(learner_class=FrozenEncoderPPOTorchLearner)
        callbacks = config.callbacks_on_train_result
        if not isinstance(callbacks, list):
            callbacks = [callbacks] if callbacks is not None else []
        config.callbacks(
            on_train_result=[
                *callbacks,
                partial(
                    verify_frozen_update,
                    recipe_path=str(
                        context.artifacts_dir / "resolved_recipe.json"
                    ),
                ),
            ]
        )
    return config


def resolved_recipe(
    context: RunContext,
    *,
    cost: float,
    transfer: dict[str, object] | None = None,
) -> dict[str, object]:
    recipe = pusher_b_ppo.resolved_recipe(
        context,
        preset="b10",
        settings=SETTINGS,
    )
    recipe.update(
        {
            "study": "pusher_b10_checkpoint_transfer_2026_10.guess_or_pass_rllib",
            "condition": "ppo_guess_or_pass",
            "pretraining_objective": (
                transfer.get("pretraining_objective", "ppo")
                if transfer is not None
                else "ppo"
            ),
            "target_finetuning_objective": "ppo without next-token CE",
            "next_token_aux_coefficient": 0.0,
            "cost": cost,
            "environment": ppo_environment_config(cost),
            "action_semantics": (
                "0/1 guess the pending token; 2 passes"
            ),
            "reward": (
                "1 for a correct guess, -cost for a wrong guess, "
                "0 for pass"
            ),
            "episode_length": EPISODE_LENGTH,
            "total_env_steps": (
                SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
            ),
            "learning_rate": LEARNING_RATE,
            "entropy_coeff": [list(row) for row in ENTROPY_COEFF_SCHEDULE],
        }
    )
    if transfer is not None:
        recipe["transfer"] = transfer
    return recipe


def run_ppo(context: RunContext, *, cost: float) -> dict[str, object]:
    if context.seed is None:
        raise ValueError("guess-or-pass PPO requires a resolved seed")
    transfer = transfer_overrides()
    if (context.resume_from is None) != (transfer is None):
        raise ValueError("warm-start checkpoint and transfer metadata must be paired")
    artifacts = RunArtifacts.from_context(context)
    artifacts.prepare()
    recipe = resolved_recipe(context, cost=cost, transfer=transfer)
    if transfer is not None and recipe.get("next_token_aux_coefficient") not in (
        None,
        0,
        0.0,
    ):
        raise ValueError("guess-or-pass transfer target must not use next-token CE")
    artifacts.write_json(
        "resolved_recipe.json",
        recipe,
    )
    tune_context = (
        replace(context, resume_from=None)
        if context.resume_from is not None
        else context
    )
    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    result_grid = run_tune(
        build_config(context, cost=cost),
        tune_context,
        stop={
            "env_runners/num_env_steps_sampled_lifetime": target_steps,
        },
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(
                num_to_keep=1,
                checkpoint_at_end=True,
            )
        },
    )
    results = list(result_grid)
    if len(results) != 1 or results[0].error is not None:
        raise RuntimeError("guess-or-pass PPO training failed")
    write_training_curves(context)
    metrics = results[0].metrics
    summary = {
        "study": "pusher_b10_checkpoint_transfer_2026_10.guess_or_pass_rllib",
        "algorithm": "ppo",
        "cost": cost,
        "seed": context.seed,
        "smoke": context.smoke,
        "target_env_steps": target_steps,
        "completed_env_steps": _metric(
            metrics,
            "env_runners/num_env_steps_sampled_lifetime",
        ),
        "episode_return_mean": _metric(
            metrics,
            "env_runners/episode_return_mean",
        ),
        "episode_length_mean": _metric(
            metrics,
            "env_runners/episode_len_mean",
        ),
        "guess_correct": _metric(
            metrics,
            "env_runners/guess_or_pass/guess_correct",
        ),
        "guess_wrong": _metric(
            metrics,
            "env_runners/guess_or_pass/guess_wrong",
        ),
        "passed": _metric(metrics, "env_runners/guess_or_pass/passed"),
        "bayes_expected_reward": _metric(
            metrics,
            "env_runners/guess_or_pass/bayes_expected_reward",
        ),
        "transfer": transfer,
    }
    artifacts.write_json("summary.json", summary)
    return summary
