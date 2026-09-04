"""Complete PPO recipe shared by explicit factored-MESS3 experiment leaves."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from envs.hmm import HMMEnv
from experiments.mess3_belief_geometry_2026_07.shared import (
    apply_runtime_resources,
)
from experiments.mess3_factored_cycle_1.dynamics import BASE_TRANSITION
from experiments.mess3_factored_cycle_1.observation import (
    FactoredObservationHMMEnv,
)
from experiments.mess3_factored_cycle_1.probe import probe_checkpoint
from experiments.mess3_factored_cycle_1.prediction import (
    train_prediction_twin,
    twin_context,
)
from experiments.mess3_factored_cycle_1.reference import (
    structural_audit_report,
)
from experiments.storage.training_curves import write_training_curves
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.runners import run_tune
from learners.models.transformer import TransformerModel, TransformerModelConfig


TOTAL_ENV_STEPS = 700_000
SMOKE_ENV_STEPS = 2_048
TRAIN_BATCH_SIZE = 32_768
SMOKE_BATCH_SIZE = 1_024
MINIBATCH_SIZE = 2_048
SMOKE_MINIBATCH_SIZE = 256
DEFAULT_ENTROPY_COEFF = 0.003
BASE_MODEL_CONFIG = TransformerModelConfig(
    d_model=64,
    n_layers=4,
    n_heads=1,
    context_len=10,
).to_dict()
FULL_REFERENCE_AUDIT_PATH = (
    Path(__file__).parent / "results" / "reference_audits.json"
)


@dataclass(frozen=True, slots=True)
class Condition:
    """One explicit scientific cell; leaves construct this directly."""

    name: str
    experiment: str
    action_kind: str
    reward_kind: str
    alpha1: float
    alpha2: float
    token_encoding: str = "factored"
    action_encoding: str = "factored"
    coupling_lambda: float = 0.0
    expected_quotient_dimension: int | None = None
    campaign_role: str = "primary"
    hypothesis: str = ""

    def __post_init__(self) -> None:
        if self.experiment not in {"E1", "E2", "E3a", "E3b", "E3c", "E4"}:
            raise ValueError("unknown experiment label")
        if self.action_kind not in {"product", "diagonal", "e2_tilt", "e4_gauge"}:
            raise ValueError("unknown action kind")
        if self.reward_kind not in {
            "f1_goal",
            "f2_goal",
            "additive",
            "conjunctive",
        }:
            raise ValueError("unknown reward kind")
        for name, value in (("alpha1", self.alpha1), ("alpha2", self.alpha2)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.token_encoding not in {"factored", "joint"}:
            raise ValueError("token_encoding must be factored or joint")
        if self.action_encoding not in {"factored", "joint"}:
            raise ValueError("action_encoding must be factored or joint")
        if self.coupling_lambda < 0.0:
            raise ValueError("coupling_lambda must be non-negative")


def environment_config(
    condition: Condition,
    *,
    fixed_episode_length: bool = False,
) -> dict[str, Any]:
    """Build one PR-35-composed two-factor HMM configuration."""

    transition = BASE_TRANSITION.tolist()
    return {
        "model": {
            "factory": "envs.hmm:factored_model",
            "kwargs": {
                "factors": [
                    {
                        "factory": "envs.mess3.model:control_model",
                        "kwargs": {
                            "alpha": condition.alpha1,
                            "transition_matrix": transition,
                        },
                    },
                    {
                        "factory": "envs.mess3.model:control_model",
                        "kwargs": {
                            "alpha": condition.alpha2,
                            "transition_matrix": transition,
                        },
                    },
                ],
            },
        },
        "task": {
            "class": (
                "experiments.mess3_factored_cycle_1.task:FactoredControlTask"
            ),
            "kwargs": {
                "action_kind": condition.action_kind,
                "reward_kind": condition.reward_kind,
                "coupling_lambda": condition.coupling_lambda,
                "action_encoding": condition.action_encoding,
            },
        },
        "observation": {
            "token": {"offset": 0, "depth": 1},
            "action": {"offset": 0, "depth": 1},
        },
        "delay": 0,
        "episode_length": 1024,
        "randomize_first_episode_length": not fixed_episode_length,
    }


def environment_class(condition: Condition):
    return (
        FactoredObservationHMMEnv
        if condition.token_encoding == "factored"
        else HMMEnv
    )


def make_environment(
    condition: Condition,
    *,
    fixed_episode_length: bool = False,
):
    return environment_class(condition)(
        environment_config(
            condition,
            fixed_episode_length=fixed_episode_length,
        )
    )


def _single_gpu_context(context: RunContext) -> RunContext:
    profile = context.hardware
    if (
        not context.smoke
        and profile is not None
        and profile.name == "cuda4090_gpuinfer"
    ):
        return replace(context, hardware=PROFILES["cuda4090"])
    return context


def build_config(
    context: RunContext,
    condition: Condition,
) -> PPOConfig:
    """Build a fresh cycle-5-scale transformer PPO configuration."""

    config = (
        PPOConfig()
        .environment(
            environment_class(condition),
            env_config=environment_config(condition),
        )
        .framework(
            "torch",
            torch_compile_learner=False,
            torch_compile_worker=False,
        )
        .training(
            lr=3e-4 if context.smoke else 4.2e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=DEFAULT_ENTROPY_COEFF,
            train_batch_size_per_learner=(
                SMOKE_BATCH_SIZE if context.smoke else TRAIN_BATCH_SIZE
            ),
            minibatch_size=(
                SMOKE_MINIBATCH_SIZE if context.smoke else MINIBATCH_SIZE
            ),
            num_epochs=6,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=TransformerModel,
                model_config=dict(BASE_MODEL_CONFIG),
            )
        )
        .debugging(seed=context.seed)
    )
    return apply_runtime_resources(
        config,
        _single_gpu_context(context),
        default_env_runners=16,
    )


def _pretraining_audits(context: RunContext) -> dict[str, Any]:
    """Permit wiring smokes, but gate every research run on A1-A6 evidence."""

    structural = structural_audit_report()
    if structural["status"] != "passed":
        raise RuntimeError("structural pre-training audits failed")
    if context.smoke:
        return {
            **structural,
            "training_authorization": "smoke_wiring_only",
        }
    if not FULL_REFERENCE_AUDIT_PATH.is_file():
        raise RuntimeError(
            "full factored-MESS3 training is blocked until the registered "
            f"A1-A6 campaign is recorded at {FULL_REFERENCE_AUDIT_PATH}"
        )
    reference = json.loads(FULL_REFERENCE_AUDIT_PATH.read_text())
    if (
        reference.get("status") != "passed"
        or reference.get("protocol", {}).get("n_chains", 0) < 4096
        or reference.get("protocol", {}).get("n_steps", 0) < 6000
        or reference.get("protocol", {}).get("burn_in", 0) < 500
        or reference.get("max_standard_error", float("inf")) > 5e-4
    ):
        raise RuntimeError("registered A1-A6 reference campaign has not passed")
    return {
        **structural,
        "training_authorization": "registered_A1_A6_passed",
        "reference_campaign": reference,
    }


def run_condition(
    context: RunContext,
    condition: Condition,
) -> dict[str, Any]:
    """Train PPO, then train its matched next-symbol prediction twin."""

    if context.seed is None:
        raise ValueError("factored MESS3 requires a resolved seed")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    audit_report = _pretraining_audits(context)
    outputs.write_json("audit_status.json", audit_report)
    outputs.write_json(
        "resolved_recipe.json",
        {
            "condition": asdict(condition),
            "algorithm": "PPO",
            "architecture_source": (
                "mess3_reward_state_action_symmetry_cycle_5"
            ),
            "model_config": BASE_MODEL_CONFIG,
            "gamma": 0.99,
            "lambda": 0.95,
            "entropy_coeff": DEFAULT_ENTROPY_COEFF,
            "total_env_steps": target_steps,
            "episode_length": 1024,
            "initial_state_distribution": "uniform_product",
            "reward_timing": "decision_state_before_transition",
            "prediction_twin": {
                "trajectory_source": "final_stochastic_RL_policy",
                "target": "next_joint_symbol",
                "epochs": 1 if context.smoke else 6,
                "learning_rate": 3e-4,
            },
        },
    )
    config = build_config(context, condition)
    result_grid = run_tune(
        config,
        context,
        stop={"env_runners/num_env_steps_sampled_lifetime": target_steps},
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(
                num_to_keep=1,
                checkpoint_at_end=True,
            )
        },
    )
    results = list(result_grid)
    if len(results) != 1:
        raise RuntimeError(
            f"{condition.name} expected one Tune trial, got {len(results)}"
        )
    result = results[0]
    if result.error is not None:
        raise RuntimeError(f"{condition.name} training failed") from result.error
    if result.checkpoint is None:
        raise RuntimeError(f"{condition.name} produced no final checkpoint")
    write_training_curves(context)
    final_probe = probe_checkpoint(
        context,
        Path(result.checkpoint.path),
        condition,
    )
    outputs.write_json("final_probe.json", final_probe)

    def twin_environment():
        return make_environment(condition, fixed_episode_length=True)

    twin_summary = train_prediction_twin(
        twin_context(context),
        checkpoint=Path(result.checkpoint.path),
        env_factory=twin_environment,
        model_config=dict(BASE_MODEL_CONFIG),
        token_encoding=condition.token_encoding,
        data_steps=1024 if context.smoke else TOTAL_ENV_STEPS,
        epochs=1 if context.smoke else 6,
        learning_rate=3e-4,
    )
    summary = {
        "condition": asdict(condition),
        "seed": context.seed,
        "smoke": context.smoke,
        "target_env_steps": target_steps,
        "final_checkpoint": str(result.checkpoint.path),
        "prediction_twin": twin_summary,
        "final_probe": {
            "path": str(context.results_dir / "final_probe.json"),
            "representation": final_probe["representation"],
        },
        "structural_audits": audit_report["status"],
    }
    outputs.write_json("condition_summary.json", summary)
    return summary
