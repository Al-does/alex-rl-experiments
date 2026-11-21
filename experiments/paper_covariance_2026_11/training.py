"""Preserve the manuscript trainers and add a loss at the primary probe site."""

from dataclasses import asdict, dataclass, replace
from functools import partial

import jax

from experiments.belief_quotient_history_jax_2026_11 import reinforce
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import (
    shared as ladder,
)
from experiments.reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32_2026_11 import (
    shared as factors,
)
from experiments.rocksample_jax_2026_10 import sweep
from experiments.rocksample_jax_2026_10.ppo_bmax_r7 import experiment as r7
from experiments.rocksample_jax_2026_10.ppo_bmax_r7_cont330m import experiment as cont

STUDY = "paper_covariance_2026_11"
SEEDS = (42, 43)
WEIGHTS = (0.01, 0.1)


def validate(context, weight):
    if context.seed not in SEEDS or weight not in WEIGHTS:
        raise ValueError(f"expected seed in {SEEDS} and covariance weight in {WEIGHTS}")


@dataclass(frozen=True)
class HistoryConfig(reinforce.TrainConfig):
    covariance_weight: float = 0.0
    covariance_site: str = "post_final_layer_norm"


@dataclass(frozen=True)
class HistoryBackend:
    weight: float
    site: str

    def configure(self, config):
        return HistoryConfig(**asdict(config), covariance_weight=self.weight,
                             covariance_site=self.site)

    def train(self, spec, process, config, **kwargs):
        from losses.jax_covariance import off_diagonal_covariance_loss

        def activation_loss(residual, embedding):
            features = embedding if self.site == "post_final_layer_norm" else residual
            covariance = off_diagonal_covariance_loss(features)
            weighted = config.covariance_weight * covariance
            return weighted, {"covariance_loss": covariance,
                              "weighted_covariance_loss": weighted}

        loss = partial(reinforce.surrogate_loss, activation_loss=activation_loss)
        iteration = partial(reinforce.make_iteration, loss_fn=loss)
        return reinforce.train(spec, process, config, iteration_fn=iteration, **kwargs)


def run_ladder(context, *, variant, weight):
    from .history_analysis import analyze_ladder

    validate(context, weight)
    summary = ladder.run_condition(
        context, variant=variant, architecture="transformer", study=STUDY,
        training_backend=HistoryBackend(weight, "post_final_layer_norm"),
    )
    jax.clear_caches()
    analyze_ladder(context, variant, summary)
    return summary


def run_factors(context, *, condition, weight):
    from .history_analysis import analyze_factors

    validate(context, weight)
    summary = factors.run_condition(
        context, condition=condition, study=STUDY,
        training_backend=HistoryBackend(weight, "pre_final_layer_norm"),
    )
    jax.clear_caches()
    analyze_factors(context, condition, summary)
    return summary


def rocksample_recipe(weight, *, smoke=False):
    arm = cont.SMOKE_ARM if smoke else r7.ARM
    arm = replace(arm, ppo=replace(arm.ppo, covariance_weight=weight))
    if smoke:
        updates = 3
        checkpoints = (1, 2)
    else:
        updates = cont.TOTAL_UPDATES
        r7_schedule = r7.RECIPE.checkpoint_schedule(cont.SOURCE_UPDATES)
        continued = range(cont.SOURCE_UPDATES + cont.CHECKPOINT_EVERY_UPDATES,
                          updates, cont.CHECKPOINT_EVERY_UPDATES)
        checkpoints = tuple(sorted({*r7_schedule, cont.SOURCE_UPDATES, *continued}))
    return sweep.SweepRecipe(
        arms=(arm,), env_steps_per_seed=updates * arm.ppo.batch_size,
        num_seeds=1, num_keys=r7.RECIPE.num_keys, checkpoint_at=checkpoints,
        resumable_checkpoint_at=(1 if smoke else cont.SOURCE_UPDATES,),
        upload_checkpoints=not smoke,
    )


def run_rocksample(context, *, weight):
    from .rocksample_analysis import analyze

    validate(context, weight)
    if context.resume_from is not None:
        raise ValueError("start the covariance RockSample recipe from scratch")
    recipe = rocksample_recipe(weight, smoke=context.smoke)
    summary = sweep.run(context, recipe)
    arm = recipe.arms[0]
    result = summary["arms"][arm.name]
    if len(result["seeds"]) != 1:
        raise RuntimeError(f"RockSample training did not finish: {result}")
    analyze(context, recipe, result["seeds"][0])
    return summary
