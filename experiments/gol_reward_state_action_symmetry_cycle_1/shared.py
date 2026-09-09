from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ResultDict

from envs.hmm import HMMEnv
from experiments.factored_representations_reproduction_PPO_2026_08.shared import (
    _save_initial_checkpoint,
    _save_log_spaced_checkpoint,
    checkpoint_records,
)
from experiments.gol_reward_state_action_symmetry_cycle_1.design import analytic_design_summary
from experiments.gol_reward_state_action_symmetry_cycle_1.process import SPEED, environment_config
from experiments.storage.training_curves import write_training_curves
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES, resolve_env_runners
from harness.runners import run_tune
from learners.models.transformer import TransformerModel, TransformerModelConfig


TOTAL_ENV_STEPS = 2_500_000
SMOKE_ENV_STEPS = 2_048
TRAIN_BATCH_SIZE = 32_768
SMOKE_BATCH_SIZE = 1_024
MINIBATCH_SIZE = 2_048
SMOKE_MINIBATCH_SIZE = 128
MODEL_CONFIG = TransformerModelConfig(
    d_model=96, n_layers=3, n_heads=4, context_len=64,
).to_dict()


class ContinuingSingleAgentEnvRunner(SingleAgentEnvRunner):
    """SingleAgentEnvRunner that drops the ongoing-episode metrics cache for
    continuing tasks.

    RLlib's new API stack keeps every returned episode chunk in
    ``_ongoing_episodes_for_metrics`` so that, when an episode eventually ends,
    it can reconstruct full-episode metrics from the prior chunks.  For
    non-terminating (``episode_length=None``) environments the ending never
    arrives, so this cache grows without bound.  Stateful models that return
    ``state_out`` (e.g. transformer KV caches) make the leak severe: each step
    stores the full KV state, and after millions of steps the worker exhausts
    memory and the box becomes unresponsive.  For these environments the cache
    can be safely discarded because there are no done episodes to reconstruct
    metrics from.
    """

    @override(SingleAgentEnvRunner)
    def get_metrics(self) -> ResultDict:
        result = super().get_metrics()
        env_config = getattr(self.config, "env_config", None)
        if env_config is not None and env_config.get("episode_length", "unset") is None:
            self._ongoing_episodes_for_metrics.clear()
        return result


def build_config(context: RunContext, variant: int, *, speed: str = SPEED) -> PPOConfig:
    profile = context.hardware or PROFILES["cpu"]
    if profile.name == "cuda4090_gpuinfer":
        profile = PROFILES["cuda4090"]
    return (
        PPOConfig()
        .environment(HMMEnv, env_config=environment_config(variant, speed))
        .framework("torch", torch_compile_learner=False, torch_compile_worker=False)
        .training(
            lr=3e-4 if context.smoke else 4.2e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            use_critic=True,
            use_gae=True,
            use_kl_loss=False,
            vf_loss_coeff=0.5,
            vf_clip_param=1e9,
            grad_clip=0.5,
            grad_clip_by="global_norm",
            entropy_coeff=0.003,
            train_batch_size_per_learner=SMOKE_BATCH_SIZE if context.smoke else TRAIN_BATCH_SIZE,
            minibatch_size=SMOKE_MINIBATCH_SIZE if context.smoke else MINIBATCH_SIZE,
            num_epochs=6,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=TransformerModel,
                model_config=dict(MODEL_CONFIG),
            )
        )
        .callbacks(
            on_algorithm_init=partial(
                _save_initial_checkpoint,
                checkpoint_path=str(context.artifacts_dir / "initial_checkpoint"),
            ),
            on_train_result=partial(
                _save_log_spaced_checkpoint,
                checkpoint_root=str(context.artifacts_dir / "log_spaced_checkpoints"),
            ),
        )
        .debugging(seed=context.seed)
        .env_runners(
            env_runner_cls=ContinuingSingleAgentEnvRunner,
            batch_mode="truncate_episodes",
            num_env_runners=0 if context.smoke else resolve_env_runners(profile, default=16),
            num_envs_per_env_runner=1 if context.smoke else profile.num_envs_per_env_runner,
            num_gpus_per_env_runner=0 if context.smoke else profile.num_gpus_per_env_runner,
            sample_timeout_s=600.0,
        )
        .learners(num_gpus_per_learner=0 if context.smoke or profile.learner_device != "cuda" else 1)
    )


def resolved_recipe(context: RunContext, variant: int, *, speed: str = SPEED) -> dict[str, Any]:
    return {
        "study": "gol_reward_state_action_symmetry_cycle_1",
        "reference_recipe": "mess3_reward_state_action_symmetry_cycle_4 PPO",
        "variant": variant,
        "condition": f"variant_{variant}",
        "speed": speed,
        "seed": context.seed,
        "smoke": context.smoke,
        "environment": environment_config(variant, speed),
        "analytic_design": analytic_design_summary(speed),
        "algorithm": "PPO",
        "gamma": 0.99,
        "lambda": 0.95,
        "learning_rate": 3e-4 if context.smoke else 4.2e-4,
        "entropy_coeff": 0.003,
        "clip_param": 0.2,
        "use_kl_loss": False,
        "value_loss_coeff": 0.5,
        "value_clip_param": 1e9,
        "grad_clip_global_norm": 0.5,
        "num_epochs": 6,
        "train_batch_size_per_learner": SMOKE_BATCH_SIZE if context.smoke else TRAIN_BATCH_SIZE,
        "minibatch_size": SMOKE_MINIBATCH_SIZE if context.smoke else MINIBATCH_SIZE,
        "model": dict(MODEL_CONFIG),
        "context_semantics": "Banded causal transformer; 64 frames per layer, not a strict 64-frame total receptive field.",
        "reward": "1[destination_state=E]",
        "previous_reward_in_observation": False,
        "filter_conditions_on_reward": False,
        "reset": "Sample stationary uniform-action prior; zero-padded start observation; no emission or reward.",
        "rollout_boundaries": "Continuing task; truncate batches with bootstrap, not environments or filter/model contexts.",
        "total_env_steps": SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS,
        "budget_semantics": "Stop after crossing the threshold at a complete training iteration; full budget is an untuned starting point.",
        "checkpoint_schedule": "Exact trained initialization, powers of two training iterations, final.",
        "probe_target": "Decision-time destination posterior b_t from actions and tokens only.",
        "probe_controls": ["initial_untrained_network", "previous_action_latest_token", "next_token_reward_marginals", "shuffled_history"],
        "interpretation": "Smoke verifies execution only; belief decodability does not establish causal use or optimal control.",
    }


def run_condition(context: RunContext, variant: int, *, speed: str = SPEED) -> dict[str, Any]:
    from experiments.gol_reward_state_action_symmetry_cycle_1.analysis import analyze_checkpoint

    if context.seed is None:
        raise ValueError("gol PPO requires a resolved seed")
    if context.resume_from is not None:
        raise ValueError("continuation is not defined for this experiment")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    outputs.write_json("resolved_recipe.json", resolved_recipe(context, variant, speed=speed))
    results = list(run_tune(
        build_config(context, variant, speed=speed),
        context,
        stop={"env_runners/num_env_steps_sampled_lifetime": SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS},
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(num_to_keep=1, checkpoint_at_end=True),
        },
    ))
    if len(results) != 1 or results[0].error is not None:
        raise RuntimeError(f"gol variant {variant} ({speed}): PPO training failed")
    write_training_curves(context)
    records = [
        {
            "checkpoint_path": context.artifacts_dir / "initial_checkpoint",
            "checkpoint_name": "initial_checkpoint",
            "training_iteration": 0,
            "agent_steps": 0,
        },
        *checkpoint_records(results[0], checkpoint_root=context.artifacts_dir / "log_spaced_checkpoints"),
    ]
    if len(records) < 2:
        raise RuntimeError("gol retained no trained checkpoint")
    reports = []
    for record in records:
        checkpoint = Path(record["checkpoint_path"])
        reports.append(analyze_checkpoint(
            replace(
                context,
                results_dir=context.results_dir / "checkpoint_probes" / f"steps_{record['agent_steps']:09d}",
                resume_from=checkpoint,
            ),
            checkpoint=checkpoint,
            variant=variant,
            speed=speed,
            agent_steps=record["agent_steps"],
            checkpoint_label=record["checkpoint_name"],
            training_iteration=record["training_iteration"],
        ))
    summary = {
        "condition": f"variant_{variant}",
        "speed": speed,
        "seed": context.seed,
        "smoke": context.smoke,
        "checkpoint_reports": reports,
    }
    outputs.write_json("condition_summary.json", summary)
    return summary
