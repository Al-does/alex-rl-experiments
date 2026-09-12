from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import partial
import math
from pathlib import Path
from typing import Any

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleSpec

from envs.hmm import HMMEnv
from experiments.factored_representations_reproduction_PPO_2026_08.model import (
    FactoredReproductionActorCritic,
    FactoredReproductionModelConfig,
)
from experiments.factored_representations_reproduction_PPO_2026_08.shared import (
    _save_initial_checkpoint,
    _save_log_spaced_checkpoint,
    checkpoint_records,
)
from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.process import (
    COMPONENT_PARAMETERS,
    CONTEXT_LENGTH,
    EFFECT_SIZE,
    EPISODE_LENGTH,
    environment_config,
)
from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.task import (
    DIRECTIONS,
)
from experiments.storage.training_curves import write_training_curves
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.env_runners import FreshEpisodeSingleAgentEnvRunner
from harness.hardware import HardwareProfile, PROFILES, resolve_env_runners
from harness.runners import run_tune


ARTICLE_URL = "https://simplex.pub/nonergodic-geometry/"
TOTAL_ENV_STEPS = 15_000_000
SMOKE_ENV_STEPS = 1_024
TRAIN_BATCH_SIZE = 262_144
SMOKE_BATCH_SIZE = 512
MINIBATCH_SIZE = 8_192
SMOKE_MINIBATCH_SIZE = 128
LEARNING_RATE = 4.2e-4
SMOKE_LEARNING_RATE = 3e-4
DEFAULT_ENTROPY_COEFF = 0.01
NUM_EPOCHS = 6
MIN_EPISODES_PER_TRAIN_BATCH = math.ceil(
    TRAIN_BATCH_SIZE / EPISODE_LENGTH
)
MODEL_CONFIG = FactoredReproductionModelConfig(
    d_model=128,
    n_layers=4,
    n_heads=4,
    d_mlp=512,
    context_length=CONTEXT_LENGTH,
    max_seq_len=CONTEXT_LENGTH,
    activation="gated_gelu",
    normalization="rms_norm",
    positional_embedding="rope",
    attention_implementation="sdpa",
    training_sequence_mode="complete_episode",
).to_dict()


def _init_algorithm(
    *,
    algorithm,
    checkpoint_path: str,
    warm_start_path: str | None = None,
    **kwargs: Any,
) -> None:
    _save_initial_checkpoint(
        algorithm=algorithm, checkpoint_path=checkpoint_path, **kwargs
    )
    if warm_start_path is not None:
        # Weights-only warm start: a full restore_from_path also restores
        # counters, which would instantly satisfy step-based stop rules and
        # schedules. Load just the trained module instead.
        module_checkpoint = (
            Path(warm_start_path)
            / "learner_group"
            / "learner"
            / "rl_module"
            / "default_policy"
        )
        warm_module = RLModule.from_checkpoint(module_checkpoint)
        algorithm.get_module().set_state(warm_module.get_state())


def _sampling_layout(
    context: RunContext,
    profile: HardwareProfile,
) -> tuple[int, int]:
    if context.smoke:
        return 0, 1
    num_env_runners = resolve_env_runners(profile, default=16)
    return (
        num_env_runners,
        math.ceil(MIN_EPISODES_PER_TRAIN_BATCH / num_env_runners),
    )


def build_config(
    context: RunContext,
    variant: int,
    *,
    component_parameters: Sequence[Mapping[str, object]] | None = None,
    entropy_coeff: float | Sequence[Sequence[float]] = DEFAULT_ENTROPY_COEFF,
) -> PPOConfig:
    profile = context.hardware or PROFILES["cpu"]
    num_env_runners, num_envs_per_env_runner = _sampling_layout(
        context,
        profile,
    )
    return (
        PPOConfig()
        .environment(
            HMMEnv,
            env_config=environment_config(
                variant,
                component_parameters=component_parameters,
            ),
        )
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
        .training(
            lr=SMOKE_LEARNING_RATE if context.smoke else LEARNING_RATE,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            use_critic=True,
            use_gae=True,
            use_kl_loss=False,
            vf_loss_coeff=0.5,
            entropy_coeff=entropy_coeff,
            train_batch_size_per_learner=(
                SMOKE_BATCH_SIZE if context.smoke else TRAIN_BATCH_SIZE
            ),
            minibatch_size=(
                SMOKE_MINIBATCH_SIZE if context.smoke else MINIBATCH_SIZE
            ),
            num_epochs=NUM_EPOCHS,
            shuffle_batch_per_epoch=True,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=FactoredReproductionActorCritic,
                model_config=dict(MODEL_CONFIG),
            )
        )
        .callbacks(
            on_algorithm_init=partial(
                _init_algorithm,
                checkpoint_path=str(
                    context.artifacts_dir / "initial_checkpoint"
                ),
                warm_start_path=(
                    str(context.resume_from)
                    if context.resume_from is not None
                    else None
                ),
            ),
            on_train_result=partial(
                _save_log_spaced_checkpoint,
                checkpoint_root=str(
                    context.artifacts_dir / "log_spaced_checkpoints"
                ),
            ),
        )
        .debugging(seed=context.seed)
        .env_runners(
            env_runner_cls=FreshEpisodeSingleAgentEnvRunner,
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=num_envs_per_env_runner,
            num_gpus_per_env_runner=0,
            rollout_fragment_length="auto",
            batch_mode="complete_episodes",
            sample_timeout_s=600.0,
        )
        .learners(
            num_gpus_per_learner=(
                1 if profile.learner_device == "cuda" else 0
            )
        )
    )


def resolved_recipe(
    context: RunContext,
    variant: int,
    *,
    component_parameters: Sequence[Mapping[str, object]] | None = None,
    entropy_coeff: float | Sequence[Sequence[float]] = DEFAULT_ENTROPY_COEFF,
    total_env_steps: int = TOTAL_ENV_STEPS,
    recipe_extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    profile = context.hardware or PROFILES["cpu"]
    num_env_runners, num_envs_per_env_runner = _sampling_layout(
        context,
        profile,
    )
    resolved_components = (
        COMPONENT_PARAMETERS
        if component_parameters is None
        else component_parameters
    )
    recipe = {
        "study": "nonergodic_mess3_reward_state_action_symmetry_cycle_5",
        "condition": f"variant_{variant}",
        "seed": context.seed,
        "smoke": context.smoke,
        "warm_start_from": (
            str(context.resume_from) if context.resume_from else None
        ),
        "source": ARTICLE_URL,
        "hypothesis": (
            "The action-symmetry ladder changes which coordinates of the "
            "six-state non-ergodic belief are useful for reward control."
        ),
        "components": [
            dict(parameters) for parameters in resolved_components
        ],
        "component_prior": [0.5, 0.5],
        "component_sampling": (
            "one component is selected at reset and remains fixed for the "
            "complete 127-decision episode"
        ),
        "variant_directions": DIRECTIONS[variant].tolist(),
        "effect_size": EFFECT_SIZE,
        "effect_size_selection": (
            "increased from cycle 5's 1.5 to 3.0 because an exact-filter "
            "expected-reward controller collapses to constant positive action "
            "in variant 2 at 1.5; at 3.0 a feasible belief-conditioned policy "
            "strictly beats every constant action"
        ),
        "action_control": (
            "exponential tilt of each component's local reward-state "
            "destination probability with the original transition support "
            "preserved; edge kernels retain P(token | source, destination)"
        ),
        "support_note": (
            "under the default component set, mess3_b has zero "
            "self-transition probability, so finite exponential tilts "
            "preserve that zero exactly"
        ),
        "reward": (
            "one when the pre-transition local state is state 2 in either "
            "component, else zero"
        ),
        "environment": environment_config(
            variant,
            component_parameters=component_parameters,
        ),
        "previous_action_in_observation": True,
        "previous_reward_in_observation": False,
        "environment_seed_semantics": (
            "the fixed worker seed initializes each vector environment once; "
            "later complete-episode resets advance the same RNG stream"
        ),
        "env_runner": (
            "harness.env_runners:FreshEpisodeSingleAgentEnvRunner"
        ),
        "sampling_layout": {
            "num_env_runners": num_env_runners,
            "num_envs_per_env_runner": num_envs_per_env_runner,
            "episodes_per_sampling_round": (
                num_env_runners * num_envs_per_env_runner
            ),
        },
        "algorithm": "clipped PPO",
        "learning_rate": (
            SMOKE_LEARNING_RATE if context.smoke else LEARNING_RATE
        ),
        "gamma": 0.99,
        "lambda": 0.95,
        "clip_param": 0.2,
        "use_kl_loss": False,
        "value_loss_coeff": 0.5,
        "entropy_coeff": entropy_coeff,
        "train_batch_size_per_learner": (
            SMOKE_BATCH_SIZE if context.smoke else TRAIN_BATCH_SIZE
        ),
        "minibatch_size": (
            SMOKE_MINIBATCH_SIZE if context.smoke else MINIBATCH_SIZE
        ),
        "num_epochs": NUM_EPOCHS,
        "model": dict(MODEL_CONFIG),
        "context_semantics": (
            "one learned BOS position plus 127 delayed token/action decisions "
            "forms one 128-position complete learner sequence"
        ),
        "episode_length": EPISODE_LENGTH,
        "total_env_steps": (
            SMOKE_ENV_STEPS if context.smoke else total_env_steps
        ),
        "stopping_metric": "env_runners/num_env_steps_sampled_lifetime",
        "checkpoint_schedule": "initial, powers of two iterations, final",
        "analysis": [
            "exact-filter expected-reward policy audit against exact "
            "constant-action returns; this is a feasible Bayes-observer "
            "certificate, not a claim of solving the continuous-belief POMDP",
            "action-conditioned edge-transducer reconstruction of the exact "
            "six-state decision-time belief",
            "held-out layerwise affine probes of weighted belief, component "
            "posterior, reward-state belief, and antisymmetric belief",
            "pending-token-distribution and observable token/action controls",
        ],
        "intended_hardware": (
            "CPU smoke; hardware-profile learner device for full training"
        ),
    }
    if recipe_extra:
        recipe.update(recipe_extra)
    return recipe


def run_condition(
    context: RunContext,
    variant: int,
    *,
    component_parameters: Sequence[Mapping[str, object]] | None = None,
    total_env_steps: int = TOTAL_ENV_STEPS,
    entropy_coeff: float | Sequence[Sequence[float]] = DEFAULT_ENTROPY_COEFF,
    continuation: bool = False,
    recipe_extra: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    from experiments.nonergodic_mess3_reward_state_action_symmetry_cycle_5.analysis import (
        analyze_checkpoint,
    )

    if context.seed is None:
        raise ValueError("non-ergodic action symmetry requires a resolved seed")
    if context.resume_from is not None and not continuation:
        raise ValueError("continuation is not defined for this experiment")
    if context.resume_from is None and continuation:
        raise ValueError("continuation runs require --resume-from")
    condition = f"variant_{variant}"
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json(
        "resolved_recipe.json",
        resolved_recipe(
            context,
            variant,
            component_parameters=component_parameters,
            entropy_coeff=entropy_coeff,
            total_env_steps=total_env_steps,
            recipe_extra=recipe_extra,
        ),
    )
    # Warm-start is handled by the on_algorithm_init callback inside
    # build_config; a Tune-level resume would demand a full trial directory.
    tune_context = (
        replace(context, resume_from=None)
        if context.resume_from is not None
        else context
    )
    result_grid = run_tune(
        build_config(
            context,
            variant,
            component_parameters=component_parameters,
            entropy_coeff=entropy_coeff,
        ),
        tune_context,
        stop={
            "env_runners/num_env_steps_sampled_lifetime": (
                SMOKE_ENV_STEPS if context.smoke else total_env_steps
            )
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
        raise RuntimeError(f"{condition} PPO training failed")
    write_training_curves(context)
    records: list[Mapping[str, Any]] = [
        {
            "checkpoint_path": context.artifacts_dir / "initial_checkpoint",
            "checkpoint_name": "initial_checkpoint",
            "training_iteration": 0,
            "agent_steps": 0,
        },
        *checkpoint_records(
            results[0],
            checkpoint_root=(
                context.artifacts_dir / "log_spaced_checkpoints"
            ),
        ),
    ]
    reports = []
    for record in records:
        reports.append(
            analyze_checkpoint(
                replace(
                    context,
                    results_dir=(
                        context.results_dir
                        / "checkpoint_probes"
                        / f"steps_{record['agent_steps']:09d}"
                    ),
                    resume_from=Path(record["checkpoint_path"]),
                ),
                variant=variant,
                checkpoint=Path(record["checkpoint_path"]),
                checkpoint_label=str(record["checkpoint_name"]),
                agent_steps=int(record["agent_steps"]),
                training_iteration=int(record["training_iteration"]),
                component_parameters=component_parameters,
            )
        )
    summary = {
        "condition": condition,
        "seed": context.seed,
        "smoke": context.smoke,
        "algorithm": "PPO",
        "checkpoint_probes": [
            {
                "checkpoint": report["checkpoint"],
                "agent_steps": report["agent_steps"],
                "training_iteration": report["training_iteration"],
                "probe_fits": report["probe_fits"],
                "controls": report["controls"],
                "policy": report["policy"],
            }
            for report in reports
        ],
        "initial_probe": reports[0],
        "final_probe": reports[-1],
    }
    outputs.write_json("condition_summary.json", summary)
    return summary
