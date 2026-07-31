"""Shared recipe for guess-driven MESS3 feedback, cycle 1.

Every arm holds the algorithm fixed (gamma-zero clipped PPO on the paper-scale
residual stream, exactly the ``ppo`` arm of ``mess3_token_guess_cycle_2``) and
varies only the generator, along two axes:

``feedback_strength`` (kappa)
    how hard a token guess shoves the hidden process;
``register_noise`` (epsilon)
    how uninformative the register's own sub-token is, and therefore how much
    predictive fidelity a factored representation gives up.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from numbers import Real
import os
from pathlib import Path
from typing import Any

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from envs.hmm import HMMEnv
from experiments.mess3_feedback_factoring_cycle_1.analysis import (
    CONTEXT_LENGTH,
    ProbeBudget,
    ProbeResult,
    plot_contrast,
    plot_init_final,
    plot_probe_trajectory,
    plot_probe_triplet,
    probe_at,
)
from experiments.mess3_feedback_factoring_cycle_1.composition import (
    myopic_ceiling,
    single_hmm_report,
)
from experiments.mess3_token_guess_cycle_2.model import (
    PaperActorCriticConfig,
    PaperActorCriticModel,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES, resolve_env_runners
from harness.runners import run_tune


@dataclass(frozen=True, slots=True)
class Condition:
    """One point in the ``(kappa, epsilon, guess observability)`` design."""

    name: str
    feedback_strength: float
    register_noise: float
    observe_previous_guess: bool
    hypothesis: str


# kappa = 0.7 is the operating point of the epsilon sweep: strong enough that
# ignoring the guess costs nearly the whole target variance, far from the
# kappa = 0.5 argmax degeneracy.
OPERATING_STRENGTH = 0.7

CONDITIONS = (
    Condition(
        "factoring_free",
        OPERATING_STRENGTH,
        0.0,
        True,
        "The register reports itself exactly, so the token-labelled operator "
        "is a tensor product and a factored representation is lossless. This "
        "is where the Factored World Hypothesis predicts two orthogonal "
        "subspaces, for a factor the policy itself drives.",
    ),
    Condition(
        "factoring_cheap",
        OPERATING_STRENGTH,
        0.3,
        True,
        "Factoring costs about 0.011 nats per token, five percent of the "
        "maximum. The product representation is still nearly adequate.",
    ),
    Condition(
        "factoring_costly",
        OPERATING_STRENGTH,
        0.85,
        True,
        "Factoring costs about 0.143 nats per token, two thirds of the "
        "maximum, so dimensional efficiency and fidelity genuinely conflict.",
    ),
    Condition(
        "factoring_impossible",
        OPERATING_STRENGTH,
        1.0,
        True,
        "The register sub-token is pure noise, only the factor sum is ever "
        "observed, and both factor marginals sit at their uniform priors: a "
        "factored representation carries no predictive information at all.",
    ),
    Condition(
        "no_feedback",
        0.0,
        1.0,
        True,
        "The guess never moves the register, so the action-conditioned and "
        "action-blind targets coincide and the arm reproduces the passive "
        "token-guess study.",
    ),
    Condition(
        "deterministic_feedback",
        1.0,
        1.0,
        True,
        "Every guess rotates the register exactly, so past guesses pin the "
        "register without any report and the Bayes ceiling returns to the "
        "passive value.",
    ),
    Condition(
        "factoring_free_blind",
        OPERATING_STRENGTH,
        0.0,
        False,
        "Counterfactual for factoring_free: hiding the previous guess should "
        "cost little, because the register reports itself directly.",
    ),
    Condition(
        "factoring_impossible_blind",
        OPERATING_STRENGTH,
        1.0,
        False,
        "Counterfactual for factoring_impossible: with no report and no "
        "visible guess, the register is unrecoverable and action awareness "
        "should be impossible.",
    ),
)

ALPHA = 0.85
DELAY = 1
EPISODE_LENGTH = 512
TOTAL_ENV_STEPS = 2_500_000
SMOKE_ENV_STEPS = 4_096
TRAIN_BATCH_SIZE = 32_768
SMOKE_BATCH_SIZE = 2_048
MINIBATCH_SIZE = 4_096
SMOKE_MINIBATCH_SIZE = 256
CHECKPOINT_FREQUENCY = 10
LEARNING_RATE = 1e-4
BASE_MODEL_CONFIG = PaperActorCriticConfig().to_dict()
BUDGET_ENV_VARIABLE = "MESS3_FEEDBACK_FACTORING_C1_MAX_ENV_STEPS"


def condition_by_name(name: str) -> Condition:
    try:
        return next(item for item in CONDITIONS if item.name == name)
    except StopIteration as error:
        raise ValueError(f"unknown feedback condition {name!r}") from error


def env_config(condition: Condition) -> dict[str, Any]:
    """Build the delay-one composed environment for one condition."""

    return {
        "model": {
            "factory": "experiments.mess3_feedback_factoring_cycle_1.model:composed_model",
            "kwargs": {
                "alpha": ALPHA,
                "feedback_strength": condition.feedback_strength,
                "register_noise": condition.register_noise,
            },
        },
        "task": {
            "class": (
                "experiments.mess3_feedback_factoring_cycle_1.task:FeedbackTokenGuessTask"
            ),
            "kwargs": {"feedback_strength": condition.feedback_strength},
        },
        "observation": {
            "token": {"offset": 0, "depth": 1},
            "action": (
                {"offset": 0, "depth": 1}
                if condition.observe_previous_guess
                else None
            ),
        },
        "delay": DELAY,
        "episode_length": EPISODE_LENGTH,
        "randomize_first_episode_length": True,
    }


@lru_cache(maxsize=None)
def condition_ceiling(
    feedback_strength: float,
    register_noise: float,
) -> dict[str, float]:
    """Cache the finite-context myopic Bayes ceiling for one generator."""

    return myopic_ceiling(
        feedback_strength,
        register_noise,
        context_length=CONTEXT_LENGTH,
    )


def _apply_runtime_resources(
    config: PPOConfig,
    context: RunContext,
) -> PPOConfig:
    profile = context.hardware or PROFILES["cpu"]
    return config.env_runners(
        num_env_runners=(
            0 if context.smoke else resolve_env_runners(profile, default=16)
        ),
        num_envs_per_env_runner=(
            1 if context.smoke else profile.num_envs_per_env_runner
        ),
        # Keep rollout inference on CPU so one-GPU workers reserve the device
        # for the learner's forward/backward hot path.
        num_gpus_per_env_runner=0,
        sample_timeout_s=600.0,
    ).learners(
        num_gpus_per_learner=1 if profile.learner_device == "cuda" else 0,
    )


def build_config(
    context: RunContext,
    condition_name: str = "no_feedback",
) -> PPOConfig:
    """Build one fresh gamma-zero feedback condition."""

    condition = condition_by_name(condition_name)
    profile = context.hardware or PROFILES["cpu"]
    config = (
        PPOConfig()
        .environment(HMMEnv, env_config=env_config(condition))
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
            use_kl_loss=False,
            vf_loss_coeff=0.5,
            entropy_coeff=0.0,
            train_batch_size_per_learner=(
                SMOKE_BATCH_SIZE if context.smoke else TRAIN_BATCH_SIZE
            ),
            minibatch_size=(
                SMOKE_MINIBATCH_SIZE if context.smoke else MINIBATCH_SIZE
            ),
            num_epochs=6,
            shuffle_batch_per_epoch=True,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=PaperActorCriticModel,
                model_config=dict(BASE_MODEL_CONFIG),
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


def checkpoint_records(result: Any) -> list[dict[str, Any]]:
    """Return every retained checkpoint in sampled-step order."""

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    candidates = list(result.best_checkpoints or [])
    if result.checkpoint is not None:
        candidates.append((result.checkpoint, result.metrics or {}))
    for checkpoint, metrics in candidates:
        path = str(checkpoint.path)
        if path in seen:
            continue
        steps = _metric(metrics, "env_runners/num_env_steps_sampled_lifetime")
        iteration = _metric(metrics, "training_iteration")
        if steps is None or iteration is None:
            continue
        seen.add(path)
        records.append(
            {
                "checkpoint": checkpoint,
                "checkpoint_name": Path(path).name,
                "training_iteration": int(iteration),
                "agent_steps": int(steps),
            }
        )
    return sorted(records, key=lambda record: record["agent_steps"])


def _save_initial_checkpoint(config: PPOConfig, path: Path) -> Path:
    """Materialize the deterministic pre-training module for N-init probing."""

    path.parent.mkdir(parents=True, exist_ok=True)
    algorithm = config.build_algo()
    try:
        saved = algorithm.save_to_path(str(path))
    finally:
        algorithm.stop()
    return Path(saved)


def _run_schedule(
    context: RunContext,
    target_steps_override: int | None,
    *,
    preserve_checkpoint_cadence: bool = False,
) -> tuple[int, int]:
    """Return sampled-step budget and Tune checkpoint frequency."""

    if target_steps_override is not None:
        target_steps = target_steps_override
    elif context.smoke:
        target_steps = SMOKE_ENV_STEPS
    else:
        target_steps = TOTAL_ENV_STEPS
    if target_steps <= 0:
        raise ValueError("target steps must be positive")
    if preserve_checkpoint_cadence:
        checkpoint_frequency = CHECKPOINT_FREQUENCY
    elif context.smoke or target_steps_override is not None:
        checkpoint_frequency = 1
    else:
        checkpoint_frequency = CHECKPOINT_FREQUENCY
    return target_steps, checkpoint_frequency


def composition_report(
    condition: Condition,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    """Summarize the analytic closed loop for this condition's generator."""

    scale = {"n_chains": 64, "n_steps": 384, "burn_in": 64} if smoke else {}
    return single_hmm_report(
        condition.feedback_strength,
        condition.register_noise,
        policy="probability_matching",
        seed=17,
        **scale,
    )


def run_condition(
    context: RunContext,
    condition_name: str,
    *,
    target_steps_override: int | None = None,
    preserve_checkpoint_cadence: bool = False,
    probe_budget: ProbeBudget | None = None,
) -> dict[str, Any]:
    """Train one feedback condition and probe init plus every checkpoint."""

    if context.seed is None:
        raise ValueError("feedback cycle 1 requires a resolved seed")
    env_budget = os.environ.get(BUDGET_ENV_VARIABLE)
    if target_steps_override is None and env_budget:
        target_steps_override = int(env_budget)
        preserve_checkpoint_cadence = True
    condition = condition_by_name(condition_name)
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    target_steps, checkpoint_frequency = _run_schedule(
        context,
        target_steps_override,
        preserve_checkpoint_cadence=preserve_checkpoint_cadence,
    )
    config = build_config(context, condition.name)
    ceiling = condition_ceiling(
        condition.feedback_strength,
        condition.register_noise,
    )
    probe_budget = probe_budget or ProbeBudget.for_context(context)
    probe_kwargs = {
        "feedback_strength": condition.feedback_strength,
        "register_noise": condition.register_noise,
        "ceiling": ceiling["accuracy"],
        "budget": probe_budget,
    }
    recipe = {
        "condition": condition.name,
        "algorithm": "PPO",
        "objective": "clipped_correctness",
        "hypothesis": condition.hypothesis,
        "feedback_strength": condition.feedback_strength,
        "register_noise": condition.register_noise,
        "observe_previous_guess": condition.observe_previous_guess,
        "lr": LEARNING_RATE,
        "gamma": 0.0,
        "lambda": 0.0,
        "environment": env_config(condition),
        "model": dict(BASE_MODEL_CONFIG),
        "total_env_steps": target_steps,
        "train_batch_size_per_learner": config.train_batch_size_per_learner,
        "minibatch_size": config.minibatch_size,
        "num_epochs": config.num_epochs,
        "vf_loss_coeff": config.vf_loss_coeff,
        "checkpoint_frequency_iterations": checkpoint_frequency,
        "myopic_ceiling_context_10": ceiling,
        "probe_budget": asdict(probe_budget),
    }
    outputs.write_json("resolved_recipe.json", recipe)
    outputs.write_json(
        "composition_report.json",
        composition_report(condition, smoke=context.smoke),
    )

    initial_checkpoint = _save_initial_checkpoint(
        config,
        context.artifacts_dir / "initial_checkpoint",
    )
    initial_probe, initial_point = probe_at(
        context,
        checkpoint=initial_checkpoint,
        condition=f"{condition.name}_init",
        agent_steps=0,
        **probe_kwargs,
    )
    plot_probe_triplet(
        initial_probe,
        title=f"{condition.name} - init",
        path=context.results_dir / "belief_simplex_init.png",
    )

    result_grid = run_tune(
        config,
        context,
        stop={"env_runners/num_env_steps_sampled_lifetime": target_steps},
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(
                num_to_keep=None,
                checkpoint_frequency=checkpoint_frequency,
                checkpoint_at_end=True,
            )
        },
    )
    results = list(result_grid)
    if len(results) != 1:
        raise RuntimeError(
            f"{condition.name} expected one trial, got {len(results)}"
        )
    result = results[0]
    if result.error is not None:
        raise RuntimeError(f"{condition.name} training failed") from result.error
    checkpoints = checkpoint_records(result)
    if not checkpoints:
        raise RuntimeError(f"{condition.name} retained no checkpoints")

    trajectory = [initial_point]
    probes: list[ProbeResult] = []
    for record in checkpoints:
        probed, point = probe_at(
            context,
            checkpoint=Path(record["checkpoint"].path),
            condition=condition.name,
            agent_steps=record["agent_steps"],
            **probe_kwargs,
        )
        probes.append(probed)
        trajectory.append(
            {
                **point,
                "training_iteration": record["training_iteration"],
                "checkpoint_name": record["checkpoint_name"],
            }
        )
    final_probe = probes[-1]
    plot_probe_triplet(
        final_probe,
        title=f"{condition.name} - final",
        path=context.results_dir / "belief_simplex_final.png",
    )
    plot_init_final(
        initial_probe,
        final_probe,
        condition=condition.name,
        path=context.results_dir / "belief_simplex_init_vs_final.png",
    )
    plot_probe_trajectory(
        trajectory,
        condition=condition.name,
        ceiling=ceiling["accuracy"],
        path=context.results_dir / "probe_and_success_trajectory.png",
    )
    outputs.write_json(
        "checkpoint_probe_curve.json",
        {
            "condition": condition.name,
            "feedback_strength": condition.feedback_strength,
            "register_noise": condition.register_noise,
            "checkpoints": trajectory,
        },
    )
    summary = {
        "condition": condition.name,
        "seed": context.seed,
        "smoke": context.smoke,
        "gamma": 0.0,
        "feedback_strength": condition.feedback_strength,
        "register_noise": condition.register_noise,
        "observe_previous_guess": condition.observe_previous_guess,
        "myopic_ceiling_context_10": ceiling,
        "initial_probe": initial_probe.metrics,
        "final_probe": final_probe.metrics,
        "training_change": {
            "mse_delta": (
                float(final_probe.metrics["mse"])
                - float(initial_probe.metrics["mse"])
            ),
            "action_awareness_ratio_delta": (
                float(final_probe.metrics["action_awareness_ratio"])
                - float(initial_probe.metrics["action_awareness_ratio"])
            ),
            "factor_subspace_overlap_delta": (
                float(final_probe.metrics["geometry"]["factor_subspace_overlap"])
                - float(
                    initial_probe.metrics["geometry"]["factor_subspace_overlap"]
                )
            ),
            "task_success_delta": (
                float(final_probe.metrics["token_accuracy_greedy"])
                - float(initial_probe.metrics["token_accuracy_greedy"])
            ),
        },
        "checkpoint_probes": trajectory,
        "figures": {
            "init": str(context.results_dir / "belief_simplex_init.png"),
            "final": str(context.results_dir / "belief_simplex_final.png"),
            "init_vs_final": str(
                context.results_dir / "belief_simplex_init_vs_final.png"
            ),
            "trajectory": str(
                context.results_dir / "probe_and_success_trajectory.png"
            ),
        },
    }
    outputs.write_json("condition_summary.json", summary)
    return summary


def _contrast_row(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        "joint_mse_ratio": float(metrics["targets"]["joint"]["global_mse_ratio"]),
        "blind_mse_ratio": float(metrics["targets"]["blind"]["global_mse_ratio"]),
        "action_awareness_ratio": float(metrics["action_awareness_ratio"]),
        "factor_subspace_overlap": float(
            metrics["geometry"]["factor_subspace_overlap"]
        ),
        "token_accuracy_greedy": float(metrics["token_accuracy_greedy"]),
    }


def run_conditions(
    context: RunContext,
    condition_names: tuple[str, ...],
    *,
    target_steps: int | None = None,
    probe_budget: ProbeBudget | None = None,
    label: str = "battery",
) -> dict[str, Any]:
    """Train several conditions in one run and tabulate them side by side."""

    summaries = {}
    for name in condition_names:
        condition_context = replace(
            context,
            results_dir=context.results_dir / name,
            artifacts_dir=context.artifacts_dir / name,
            resume_from=None,
        )
        summaries[name] = run_condition(
            condition_context,
            name,
            target_steps_override=target_steps,
            probe_budget=probe_budget,
        )
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    summary = {
        "label": label,
        "seed": context.seed,
        "smoke": context.smoke,
        "target_env_steps": target_steps,
        "contrast": {
            name: {
                "feedback_strength": item["feedback_strength"],
                "register_noise": item["register_noise"],
                "observe_previous_guess": item["observe_previous_guess"],
                "initial": _contrast_row(item["initial_probe"]),
                "final": _contrast_row(item["final_probe"]),
            }
            for name, item in summaries.items()
        },
        "reading": (
            "action_awareness_ratio below one means the residual stream "
            "decodes the guess-conditioned belief better than the belief of "
            "an agent that ignores its own guesses."
        ),
        "conditions": summaries,
    }
    outputs.write_json(f"{label}_summary.json", summary)
    plot_contrast(summaries, path=context.results_dir / "action_awareness.png")
    return summary


def run_battery(context: RunContext) -> dict[str, Any]:
    """Run every condition once; intended for smoke validation."""

    return run_conditions(
        context,
        tuple(condition.name for condition in CONDITIONS),
        label="battery",
    )
