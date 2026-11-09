from dataclasses import replace
from functools import partial
import json
from pathlib import Path

from ray import tune

from envs.hmm import HMMEnv
from experiments.factored_representations_reproduction_PPO_2026_08.shared import (
    _metric,
    _save_initial_checkpoint,
    _save_log_spaced_checkpoint,
    checkpoint_records,
)
from experiments.storage.training_curves import write_training_curves
from experiments.two_factor_reward_state_REINFORCE_cycle_4.shared import (
    _step_checkpoint_records,
)
from harness.artifacts import RunArtifacts
from harness.runners import run_tune


SMOKE_ENV_STEPS = 4_096
STEP_CHECKPOINT_INTERVAL = 2_000_000


def _save_training_checkpoints(*, algorithm, result, context, **kwargs):
    _save_log_spaced_checkpoint(
        algorithm=algorithm,
        result=result,
        checkpoint_root=str(context.artifacts_dir / "log_spaced_checkpoints"),
    )
    steps_value = _metric(result, "env_runners/num_env_steps_sampled_lifetime")
    iteration_value = _metric(result, "training_iteration")
    if steps_value is None or iteration_value is None:
        return
    steps = int(steps_value)
    threshold = (steps // STEP_CHECKPOINT_INTERVAL) * STEP_CHECKPOINT_INTERVAL
    if threshold == 0:
        return
    root = context.artifacts_dir / "step_checkpoints"
    index_path = root / "index.json"
    records = json.loads(index_path.read_text())["checkpoints"] if index_path.exists() else []
    if records and threshold <= int(records[-1]["threshold_env_steps"]):
        return
    root.mkdir(parents=True, exist_ok=True)
    saved = Path(algorithm.save_to_path(str(root / f"steps_{steps:09d}")))
    records.append(
        {
            "path": str(saved),
            "checkpoint_name": saved.name,
            "training_iteration": int(iteration_value),
            "agent_steps": steps,
            "threshold_env_steps": threshold,
        }
    )
    temporary = index_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"checkpoints": records}, indent=2) + "\n")
    temporary.replace(index_path)


def run_study(context, config, *, recipe, analyze_checkpoint):
    if context.seed is None:
        raise ValueError("history-dependent studies require a resolved seed")
    if context.resume_from is not None:
        raise ValueError(
            "these fresh-comparison recipes do not accept --resume-from; "
            "do not restore a historical environment into the new study"
        )
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    if (context.artifacts_dir / "initial_checkpoint").exists():
        raise ValueError("initial checkpoint already exists; use a fresh run ID")
    spec = config.get_rl_module_spec()
    environment = HMMEnv(config.env_config)
    try:
        module = spec.module_class(
            observation_space=environment.observation_space,
            action_space=environment.action_space,
            model_config=spec.model_config,
        )
        parameter_count = sum(parameter.numel() for parameter in module.parameters())
    finally:
        environment.close()
    del module
    target_steps = SMOKE_ENV_STEPS if context.smoke else recipe["total_env_steps"]
    resolved = {
        **recipe,
        "seed": context.seed,
        "smoke": context.smoke,
        "total_env_steps": target_steps,
        "environment": config.env_config,
        "model_config": spec.model_config,
        "module_class": f"{spec.module_class.__module__}.{spec.module_class.__name__}",
        "parameter_count": parameter_count,
        "algorithm": "REINFORCE",
        "rllib_engine": "PPOConfig/PPOTorchLearner",
        "gamma": config.gamma,
        "lambda": config.lambda_,
        "learning_rate": config.lr,
        "entropy_coeff": config.entropy_coeff,
        "use_critic": config.use_critic,
        "use_gae": config.use_gae,
        "value_loss_coeff": config.vf_loss_coeff,
        "train_batch_size_per_learner": config.train_batch_size_per_learner,
        "num_epochs": config.num_epochs,
        "checkpoint_schedule": "actual initialization, powers of two iterations, 2M steps, final",
        "policy_evaluation_mode": "greedy",
        "probe_sampling_distribution": "each checkpoint's own-policy histories",
        "interpretation": (
            "compare probe fits from initialization to final within architecture; "
            "own-policy changes also change visitation; accessibility is not causal use"
        ),
        "minimum_harness_model_support": "RL-Harness PR 87, merged as 9e4fc16",
    }
    outputs.write_json("resolved_recipe.json", resolved)
    config = config.callbacks(
        on_algorithm_init=partial(
            _save_initial_checkpoint,
            checkpoint_path=str(context.artifacts_dir / "initial_checkpoint"),
        ),
        on_train_result=partial(_save_training_checkpoints, context=context),
    )
    result_grid = run_tune(
        config,
        context,
        stop={"env_runners/num_env_steps_sampled_lifetime": target_steps},
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(
                num_to_keep=1, checkpoint_at_end=True
            )
        },
    )
    results = list(result_grid)
    if len(results) != 1 or results[0].error is not None:
        raise RuntimeError(f"{recipe['condition']} training failed; inspect tune_summary.json")
    result = results[0]
    write_training_curves(context)
    records = [
        {
            "checkpoint_path": context.artifacts_dir / "initial_checkpoint",
            "checkpoint_name": "initial_checkpoint",
            "training_iteration": 0,
            "agent_steps": 0,
        },
        *_step_checkpoint_records(context.artifacts_dir / "step_checkpoints"),
        *checkpoint_records(
            result,
            checkpoint_root=context.artifacts_dir / "log_spaced_checkpoints",
        ),
    ]
    by_steps = {int(record["agent_steps"]): record for record in records}
    reports = []
    for steps, record in sorted(by_steps.items()):
        checkpoint = Path(record["checkpoint_path"])
        reports.append(
            analyze_checkpoint(
                replace(
                    context,
                    results_dir=context.results_dir / "checkpoint_probes" / f"steps_{steps:09d}",
                    resume_from=checkpoint,
                ),
                checkpoint=checkpoint,
                condition=recipe["condition"],
                checkpoint_label=record["checkpoint_name"],
                agent_steps=steps,
                training_iteration=record["training_iteration"],
            )
        )
    if len(reports) < 2 or reports[0]["agent_steps"] != 0:
        raise RuntimeError("initialization and trained-checkpoint probes are required")
    summary = {
        "study": recipe["study"],
        "condition": recipe["condition"],
        "architecture": recipe["architecture"],
        "seed": context.seed,
        "smoke": context.smoke,
        "algorithm": "REINFORCE",
        "target_env_steps": target_steps,
        "checkpoint_reports": reports,
    }
    outputs.write_json("condition_summary.json", summary)
    return summary
