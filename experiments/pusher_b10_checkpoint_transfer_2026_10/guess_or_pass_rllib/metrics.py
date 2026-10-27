from __future__ import annotations

from collections.abc import Mapping

from ray.rllib.callbacks.callbacks import RLlibCallback

from experiments.storage import CurveField

METRIC_NAMES = (
    "guess_correct",
    "guess_wrong",
    "passed",
    "bayes_expected_reward",
)
TRAINING_CURVE_FIELDS: tuple[CurveField, ...] = tuple(
    (
        (f"env_runners/guess_or_pass/{name}", f"guess_or_pass/{name}"),
        name,
    )
    for name in METRIC_NAMES
)


class GuessOrPassMetricsCallback(RLlibCallback):
    def on_episode_step(self, *, episode, **kwargs) -> None:
        info = episode.get_infos(-1)
        components = info.get("reward_components")
        if not isinstance(components, Mapping):
            return
        totals = episode.custom_data.setdefault(
            "guess_or_pass_metric_totals",
            {name: 0.0 for name in METRIC_NAMES},
        )
        for name in METRIC_NAMES:
            totals[name] += float(components[name])
        episode.custom_data["guess_or_pass_metric_steps"] = (
            episode.custom_data.get("guess_or_pass_metric_steps", 0) + 1
        )

    def on_episode_end(self, *, episode, metrics_logger, **kwargs) -> None:
        totals = episode.custom_data.get("guess_or_pass_metric_totals")
        steps = episode.custom_data.get("guess_or_pass_metric_steps", 0)
        if not totals or not steps:
            return
        for name in METRIC_NAMES:
            metrics_logger.log_value(
                ("guess_or_pass", name),
                totals[name] / steps,
                reduce="mean",
                window=1000,
            )
