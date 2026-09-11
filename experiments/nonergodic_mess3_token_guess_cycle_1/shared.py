from __future__ import annotations

from dataclasses import replace
from functools import partial
import math
from pathlib import Path

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

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
from experiments.nonergodic_mess3_token_guess_cycle_1.process import (
    COMPONENT_PARAMETERS,
    CONTEXT_LENGTH,
    EPISODE_LENGTH,
    environment_config,
)
from experiments.storage.training_curves import write_training_curves
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES, resolve_env_runners
from harness.runners import run_tune


ARTICLE_URL = "https://simplex.pub/nonergodic-geometry/"
TOTAL_ENV_STEPS = 2_500_000
SMOKE_ENV_STEPS = 1_024
TRAIN_BATCH_SIZE = 32_768
SMOKE_BATCH_SIZE = 512
MINIBATCH_SIZE = 1_024
SMOKE_MINIBATCH_SIZE = 128
LEARNING_RATE = 1e-4
NUM_EPOCHS = 6
MIN_EPISODES_PER_TRAIN_BATCH = math.ceil(TRAIN_BATCH_SIZE / EPISODE_LENGTH)
ALL_ONE_COMPONENT_BATCH_PROBABILITY = 2.0 ** (
    1 - MIN_EPISODES_PER_TRAIN_BATCH
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
).to_dict()


def build_config(context: RunContext) -> PPOConfig:
    profile = context.hardware or PROFILES["cpu"]
    return (
        PPOConfig()
        .environment(HMMEnv, env_config=environment_config())
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
            lr=LEARNING_RATE,
            gamma=0.0,
            lambda_=0.0,
            clip_param=0.2,
            use_critic=True,
            use_gae=True,
            use_kl_loss=False,
            vf_loss_coeff=0.5,
            entropy_coeff=0.0,
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
                _save_initial_checkpoint,
                checkpoint_path=str(
                    context.artifacts_dir / "initial_checkpoint"
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
            num_env_runners=(
                0 if context.smoke else resolve_env_runners(profile, default=16)
            ),
            num_envs_per_env_runner=(
                1 if context.smoke else profile.num_envs_per_env_runner
            ),
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


def resolved_recipe(context: RunContext) -> dict[str, object]:
    return {
        "study": "nonergodic_mess3_token_guess_cycle_1",
        "condition": "ppo",
        "seed": context.seed,
        "smoke": context.smoke,
        "source": ARTICLE_URL,
        "hypothesis": (
            "Correctness-only PPO can learn the next-token policy while its "
            "residual stream retains the higher-dimensional weighted belief "
            "(w_A eta_A, w_B eta_B), including information unavailable from "
            "the three-way next-token distribution alone."
        ),
        "components": [dict(parameters) for parameters in COMPONENT_PARAMETERS],
        "component_prior": [0.5, 0.5],
        "component_sampling": (
            "one component is selected at reset and remains fixed for all "
            "127 emissions"
        ),
        "training_batch_component_mix": {
            "sampling": "independent equal-probability draw per complete episode",
            "minimum_episodes_per_full_train_batch": (
                MIN_EPISODES_PER_TRAIN_BATCH
            ),
            "probability_full_train_batch_uses_one_component_only": (
                ALL_ONE_COMPONENT_BATCH_PROBABILITY
            ),
        },
        "environment": environment_config(),
        "action_semantics": "three categorical logits, one per pending token",
        "reward": "1 when the sampled action equals the pending token, else 0",
        "previous_reward_in_observation": False,
        "previous_action_in_observation": False,
        "algorithm": "clipped PPO",
        "objective": "sampled next-token correctness only; no cross-entropy loss",
        "learning_rate": LEARNING_RATE,
        "gamma": 0.0,
        "lambda": 0.0,
        "clip_param": 0.2,
        "use_kl_loss": False,
        "value_loss_coeff": 0.5,
        "entropy_coeff": 0.0,
        "train_batch_size_per_learner": (
            SMOKE_BATCH_SIZE if context.smoke else TRAIN_BATCH_SIZE
        ),
        "minibatch_size": (
            SMOKE_MINIBATCH_SIZE if context.smoke else MINIBATCH_SIZE
        ),
        "num_epochs": NUM_EPOCHS,
        "model": dict(MODEL_CONFIG),
        "article_architecture_matches": [
            "four decoder blocks",
            "d_model=128",
            "four attention heads",
            "MLP width 512",
            "rotary positions",
            "context length 128",
            "128-step maximum BPTT sequence length",
            "gated GELU MLP",
            "RMS normalization",
            "weight initialization standard deviation 0.02",
            "learned BOS embedding at every episode reset",
        ],
        "article_architecture_deviations": [
            "PPO actor-critic heads replace the language-model unembedding",
        ],
        "context_semantics": (
            "each complete 127-decision episode is one learner sequence; the "
            "first input is learned BOS and subsequent inputs are delayed emissions"
        ),
        "episode_length": EPISODE_LENGTH,
        "total_env_steps": (
            SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
        ),
        "stopping_metric": "env_runners/num_env_steps_sampled_lifetime",
        "checkpoint_schedule": "initial, powers of two iterations, final",
        "analysis": [
            "held-out layerwise affine probe of six weighted belief entries",
            "held-out component-posterior probe",
            "held-out next-token-distribution probe",
            "next-token-distribution-only control for weighted belief",
            "current-visible-token and shuffled-label controls",
        ],
        "intended_hardware": (
            "CPU smoke; hardware-profile learner device for full training"
        ),
    }


def run_condition(context: RunContext) -> dict[str, object]:
    from experiments.nonergodic_mess3_token_guess_cycle_1.analysis import (
        analyze_checkpoint,
    )

    if context.seed is None:
        raise ValueError("non-ergodic MESS3 PPO requires a resolved seed")
    if context.resume_from is not None:
        raise ValueError("continuation is not defined for this experiment")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json("resolved_recipe.json", resolved_recipe(context))
    result_grid = run_tune(
        build_config(context),
        context,
        stop={
            "env_runners/num_env_steps_sampled_lifetime": (
                SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
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
        raise RuntimeError("non-ergodic MESS3 token-guess PPO training failed")
    write_training_curves(context)
    records = [
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
                checkpoint=Path(record["checkpoint_path"]),
                checkpoint_label=record["checkpoint_name"],
                agent_steps=record["agent_steps"],
                training_iteration=record["training_iteration"],
            )
        )
    summary = {
        "condition": "ppo",
        "seed": context.seed,
        "smoke": context.smoke,
        "objective": "sampled next-token correctness only",
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
