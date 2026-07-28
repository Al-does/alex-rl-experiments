"""One-factor RLlib Tune sweeps for token-guess cycle 2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

from ray import tune

from experiments.mess3_token_guess_cycle_2.learning import (
    KELLY_LOSS_COEFFICIENT_KEY,
)
from experiments.mess3_token_guess_cycle_2.shared import (
    ENV_CONFIG,
    SMOKE_ENV_STEPS,
    TOTAL_ENV_STEPS,
    build_config,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.runners import run_tune


PREDICTIVE_LOSS_COEFFICIENT_KEY = "next_token_aux/lambda"


@dataclass(frozen=True, slots=True)
class SweepSpec:
    """A controlled one-dimensional Tune grid."""

    name: str
    condition: str
    parameter: str
    values: tuple[float, ...]
    learner_config_key: str | None = None


SWEEP_SPECS = {
    "learning_rate": SweepSpec(
        name="learning_rate",
        condition="ppo",
        parameter="lr",
        values=(3e-5, 1e-4, 3e-4, 1e-3),
    ),
    "predictive_loss_coefficient": SweepSpec(
        name="predictive_loss_coefficient",
        condition="predictive_loss",
        parameter="predictive_loss_coefficient",
        values=(0.01, 0.03, 0.1, 0.3),
        learner_config_key=PREDICTIVE_LOSS_COEFFICIENT_KEY,
    ),
    "kelly_loss_coefficient": SweepSpec(
        name="kelly_loss_coefficient",
        condition="decoupled_kelly",
        parameter="kelly_loss_coefficient",
        values=(0.1, 0.3, 1.0, 3.0),
        learner_config_key=KELLY_LOSS_COEFFICIENT_KEY,
    ),
}


def sweep_spec(name: str) -> SweepSpec:
    try:
        return SWEEP_SPECS[name]
    except KeyError as error:
        raise ValueError(f"unknown token-guess sweep {name!r}") from error


def build_sweep_config(context: RunContext, name: str):
    """Build an RLlib config containing exactly one Tune grid search."""

    spec = sweep_spec(name)
    config = build_config(context, spec.condition)
    search = tune.grid_search(list(spec.values))
    if spec.learner_config_key is None:
        return config.training(lr=search)
    learner_config = dict(config.learner_config_dict)
    learner_config[spec.learner_config_key] = search
    return config.learners(learner_config_dict=learner_config)


def _nested_metric(metrics: Mapping[str, Any], path: str) -> float | None:
    value: Any = metrics
    parts = path.split("/")
    for index, part in enumerate(parts):
        if not isinstance(value, Mapping):
            return None
        remainder = "/".join(parts[index:])
        direct = value.get(remainder)
        if isinstance(direct, Real):
            return float(direct)
        if part not in value:
            return None
        value = value[part]
    return float(value) if isinstance(value, Real) else None


def _resolved_value(result: Any, spec: SweepSpec) -> float:
    value: Any = result.config
    if spec.learner_config_key is None:
        value = value["lr"]
    else:
        value = value["learner_config_dict"][spec.learner_config_key]
    if not isinstance(value, Real):
        raise TypeError(f"{spec.parameter} did not resolve to a scalar: {value!r}")
    return float(value)


def _trial_summary(result: Any, spec: SweepSpec) -> dict[str, Any]:
    metrics = result.metrics or {}
    episode_return = _nested_metric(metrics, "env_runners/episode_return_mean")
    episode_length = _nested_metric(metrics, "env_runners/episode_len_mean")
    if episode_return is None:
        raise KeyError("RLlib result omitted env_runners/episode_return_mean")
    if episode_length is None or episode_length <= 0.0:
        episode_length = float(ENV_CONFIG["episode_length"])
    summary = {
        spec.parameter: _resolved_value(result, spec),
        "trial_id": metrics.get("trial_id"),
        "training_iteration": int(metrics.get("training_iteration", 0)),
        "env_steps": int(
            _nested_metric(
                metrics,
                "env_runners/num_env_steps_sampled_lifetime",
            )
            or 0
        ),
        "episode_return_mean": episode_return,
        "episode_len_mean": episode_length,
        "token_accuracy_mean": episode_return / episode_length,
    }
    predictive_accuracy = _nested_metric(
        metrics,
        "learners/default_policy/next_token_aux/accuracy",
    )
    if predictive_accuracy is not None:
        summary["predictive_accuracy"] = predictive_accuracy
    predictive_ce = _nested_metric(
        metrics,
        "learners/default_policy/next_token_aux/ce",
    )
    if predictive_ce is not None:
        summary["predictive_cross_entropy"] = predictive_ce
    kelly_log_growth = _nested_metric(
        metrics,
        "learners/default_policy/token_guess_kelly/log_growth_mean",
    )
    if kelly_log_growth is not None:
        summary["kelly_log_growth_mean"] = kelly_log_growth
    kelly_wager = _nested_metric(
        metrics,
        "learners/default_policy/token_guess_kelly/wager_mean",
    )
    if kelly_wager is not None:
        summary["kelly_wager_mean"] = kelly_wager
    return summary


def run_sweep(context: RunContext, name: str) -> dict[str, Any]:
    """Run one four-point grid and select by final-window token accuracy."""

    if context.seed is None:
        raise ValueError("token-guess sweeps require a resolved seed")
    spec = sweep_spec(name)
    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json(
        "resolved_recipe.json",
        {
            "sweep": spec.name,
            "condition": spec.condition,
            "parameter": spec.parameter,
            "values": list(spec.values),
            "num_trials": len(spec.values),
            "sampling": "RLlib Tune grid_search",
            "one_factor_at_a_time": True,
            "total_env_steps_per_trial": target_steps,
            "seed_per_trial": context.seed,
            "selection_metric": "final_window_token_accuracy",
        },
    )
    result_grid = run_tune(
        build_sweep_config(context, name),
        context,
        stop={"env_runners/num_env_steps_sampled_lifetime": target_steps},
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(
                checkpoint_at_end=False,
            ),
        },
    )
    results = list(result_grid)
    if len(results) != len(spec.values):
        raise RuntimeError(
            f"{name} expected {len(spec.values)} trials, got {len(results)}"
        )
    failures = [result for result in results if result.error is not None]
    if failures:
        raise RuntimeError(
            f"{name} had {len(failures)} failed Tune trial(s)"
        ) from failures[0].error
    trials = sorted(
        (_trial_summary(result, spec) for result in results),
        key=lambda trial: trial[spec.parameter],
    )
    best = max(trials, key=lambda trial: trial["token_accuracy_mean"])
    summary = {
        "sweep": spec.name,
        "condition": spec.condition,
        "parameter": spec.parameter,
        "seed": context.seed,
        "smoke": context.smoke,
        "selection_metric": "final_window_token_accuracy",
        "best": best,
        "trials": trials,
    }
    outputs.write_json("sweep_summary.json", summary)
    return summary
