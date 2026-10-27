"""Pure-JAX PPO for the b10 guess-or-pass transfer campaign."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from ray.rllib.core.rl_module.rl_module import RLModule

from experiments.pusher_b_jax_2026_10 import model as tm
from experiments.pusher_b_jax_2026_10.ppo import (
    Batch,
    PPOConfig,
    RunnerState,
    _loss,
    _optimizer,
    collect,
    gae,
    schedule_value,
)

from .convert import torch_encoder_to_jax
from .process import NUM_ACTIONS, make_env

TOTAL_ENV_STEPS = 10_000_000
LR_SCHEDULE = ((0, 5e-5), (10_000_000, 5e-5))
OVERRIDES_ENV = "PUSHER_B_GUESS_OR_PASS_JAX_TRANSFER"
TRUNK_KEYS = ("embed", "blocks", "final_norm")
HEAD_KEYS = ("policy", "value")
CONFIG = PPOConfig(total_env_steps=TOTAL_ENV_STEPS, lr_schedule=LR_SCHEDULE)


def init(
    config: PPOConfig,
    spec: tm.ModelSpec,
    key: jax.Array,
    trunk_mode: str = "scratch",
) -> RunnerState:
    if trunk_mode not in {"scratch", "full", "frozen"}:
        raise ValueError(f"unsupported trunk mode: {trunk_mode}")
    state_key, param_key, head_key = jax.random.split(key, 3)
    params = tm.init_params(spec, param_key, actor_critic=True)
    params["policy"] = tm._torch_default_linear(head_key, spec.d_model, NUM_ACTIONS)
    trainable = (
        {name: params[name] for name in HEAD_KEYS}
        if trunk_mode == "frozen"
        else params
    )
    optimizer_state = _optimizer(config).init(trainable)
    return RunnerState(params, optimizer_state, state_key, jnp.int32(0))


def transfer_overrides() -> dict[str, Any] | None:
    value = os.environ.get(OVERRIDES_ENV)
    if not value:
        return None
    metadata = json.loads(value)
    if not isinstance(metadata, dict):
        raise TypeError(f"{OVERRIDES_ENV} must contain a JSON object")
    required = {
        "source",
        "run_id",
        "checkpoint_path",
        "source_env_steps",
        "source_probe_1_minus_r2",
        "pretraining_objective",
        "trunk_mode",
        "source_module_directory",
        "encoder_parity_max_abs_diff",
        "encoder_keys",
        "ignored_source_keys",
    }
    missing = required.difference(metadata)
    if missing:
        raise ValueError(f"transfer metadata is missing {sorted(missing)}")
    if metadata["trunk_mode"] not in {"full", "frozen"}:
        raise ValueError("transfer trunk_mode must be 'full' or 'frozen'")
    return metadata


def _merge_heads(trunk: dict, heads: dict) -> dict:
    return {**trunk, **heads}


def _head_params(params: dict) -> dict:
    return {name: params[name] for name in HEAD_KEYS}


def _trunk_params(params: dict) -> dict:
    return {name: params[name] for name in TRUNK_KEYS}


def _tree_equal(left, right) -> jax.Array:
    leaves = jax.tree.leaves(jax.tree.map(jnp.array_equal, left, right))
    return jnp.all(jnp.stack(leaves))


def make_update(
    config: PPOConfig,
    env,
    spec: tm.ModelSpec,
    cost: float,
    trunk_mode: str,
):
    if trunk_mode not in {"scratch", "full", "frozen"}:
        raise ValueError(f"unsupported trunk mode: {trunk_mode}")
    optimizer = _optimizer(config)
    num_minibatches = config.num_envs // config.minibatch_episodes
    batch_size = config.num_envs * env.episode_length
    if trunk_mode == "frozen":
        head_loss = jax.value_and_grad(
            lambda heads, entropy_coeff, mb, trunk: _loss(
                config,
                spec,
                _merge_heads(trunk, heads),
                entropy_coeff,
                mb,
            ),
            has_aux=True,
        )
    else:
        full_loss = jax.value_and_grad(
            lambda params, entropy_coeff, mb: _loss(
                config, spec, params, entropy_coeff, mb
            ),
            has_aux=True,
        )

    def update(state: RunnerState) -> tuple[RunnerState, dict]:
        key, collect_key, shuffle_key = jax.random.split(state.key, 3)
        traj: Batch = collect(env, spec, config.num_envs, state.params, collect_key)
        env_steps = state.env_steps + batch_size
        lr = schedule_value(config.lr_schedule, env_steps)
        entropy_coeff = schedule_value(config.entropy_schedule, env_steps)
        advantages, targets = gae(config, traj.reward, traj.value)
        advantages = (advantages - advantages.mean()) / jnp.maximum(
            advantages.std(), 1e-4
        )
        batch = (
            traj.tokens,
            traj.action,
            traj.log_prob,
            advantages,
            targets,
            traj.raw_token,
        )
        opt_state = state.opt_state
        opt_state.hyperparams["learning_rate"] = lr
        before_heads = _head_params(state.params)
        before_trunk = _trunk_params(state.params)
        if trunk_mode == "frozen":
            trainable_params = before_heads
            loss_gradient = lambda heads, mb: head_loss(
                heads, entropy_coeff, mb, before_trunk
            )
        else:
            trainable_params = state.params
            loss_gradient = lambda params, mb: full_loss(
                params, entropy_coeff, mb
            )

        def epoch(carry, epoch_key):
            order = jax.random.permutation(epoch_key, config.num_envs)
            minibatches = jax.tree.map(
                lambda x: x[order].reshape(
                    (num_minibatches, -1) + x.shape[1:]
                ),
                batch,
            )

            def minibatch(carry, mb):
                params, opt_state = carry
                (_, aux), grads = loss_gradient(params, mb)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), opt_state), aux

            return jax.lax.scan(minibatch, carry, minibatches)

        (trainable_params, opt_state), aux = jax.lax.scan(
            epoch,
            (trainable_params, opt_state),
            jax.random.split(shuffle_key, config.num_epochs),
        )
        params = (
            _merge_heads(before_trunk, trainable_params)
            if trunk_mode == "frozen"
            else trainable_params
        )
        returns = traj.reward.sum(1)
        expected_reward = bayes_expected_reward(traj.raw_token, env, cost)
        metrics = {
            **jax.tree.map(jnp.mean, aux),
            "steps": env_steps,
            "return_mean": returns.mean(),
            "return_min": returns.min(),
            "return_max": returns.max(),
            "reward_per_step": traj.reward.mean(),
            "guess_correct": jnp.mean(
                (traj.action != 2) & (traj.action == traj.raw_token)
            ),
            "guess_wrong": jnp.mean(
                (traj.action != 2) & (traj.action != traj.raw_token)
            ),
            "passed": jnp.mean(traj.action == 2),
            "bayes_expected_reward": expected_reward.mean(),
            "lr": lr,
            "entropy_coeff": entropy_coeff,
        }
        if trunk_mode == "frozen":
            metrics["trunk_bit_identical"] = _tree_equal(
                before_trunk, _trunk_params(params)
            )
            metrics["heads_changed"] = ~_tree_equal(before_heads, _head_params(params))
        return RunnerState(params, opt_state, key, env_steps), metrics

    return update


def bayes_expected_reward(tokens: jax.Array, env, cost: float) -> jax.Array:
    max_probability = env.predictive_distributions(tokens).max(-1)
    return jnp.maximum(0.0, max_probability - cost * (1.0 - max_probability))


def _save_params(path: Path, params: dict) -> None:
    np.savez(
        path,
        **{
            jax.tree_util.keystr(key): np.asarray(value)
            for key, value in jax.tree_util.tree_leaves_with_path(params)
        },
    )


def _load_transfer_trunk(path: Path, spec: tm.ModelSpec) -> tuple[dict, list[str]]:
    module = RLModule.from_checkpoint(str(path))
    if hasattr(module, "to"):
        module.to("cpu")
    source_state = module.get_state()
    trunk = torch_encoder_to_jax(source_state, spec)
    ignored = sorted(key for key in source_state if not key.startswith("encoder."))
    return trunk, ignored


def _resolved_recipe(
    context: RunContext,
    *,
    cost: float,
    config: PPOConfig,
    spec: tm.ModelSpec,
    transfer: dict[str, Any] | None,
) -> dict[str, Any]:
    recipe = {
        "study": "pusher_b10_guess_or_pass_jax_2026_10",
        "framework": "jax",
        "legacy_recipe": "experiments/pusher_b10_guess_or_pass_2026_10/ppo.py",
        "cost": cost,
        "seed": context.seed,
        "smoke": context.smoke,
        "target_env_steps": config.total_env_steps,
        "action_semantics": "0/1 guess the pending token; 2 passes",
        "reward": "1 for a correct guess, -cost for a wrong guess, 0 for pass",
        "model": asdict(spec),
        "ppo": asdict(config),
        "pretraining_objective": (
            transfer["pretraining_objective"] if transfer is not None else "scratch"
        ),
        "transfer": None,
    }
    if transfer is not None:
        recipe["transfer"] = {
            **transfer,
            "frozen_encoder_update_verified": (
                False if transfer["trunk_mode"] == "frozen" else None
            ),
            "fresh_heads_changed_after_update": (
                False if transfer["trunk_mode"] == "frozen" else None
            ),
        }
    return recipe


def run_ppo(context: RunContext, *, cost: float) -> dict[str, Any]:
    if context.seed is None:
        raise ValueError("guess-or-pass PPO requires a resolved seed")
    transfer = transfer_overrides()
    if (context.resume_from is None) != (transfer is None):
        raise ValueError("warm-start checkpoint and transfer metadata must be paired")
    trunk_mode = transfer["trunk_mode"] if transfer is not None else "scratch"
    config = (
        PPOConfig.smoke(lr_schedule=LR_SCHEDULE)
        if context.smoke
        else CONFIG
    )
    spec = tm.ModelSpec()
    env = make_env(cost)
    state = init(config, spec, jax.random.key(context.seed), trunk_mode)
    if transfer is not None:
        trunk, ignored = _load_transfer_trunk(context.resume_from, spec)
        for key in TRUNK_KEYS:
            state.params[key] = trunk[key]
        if sorted(ignored) != sorted(transfer["ignored_source_keys"]):
            raise ValueError("source ignored-key report changed since transfer setup")
    artifacts = RunArtifacts.from_context(context)
    artifacts.prepare()
    recipe = _resolved_recipe(
        context, cost=cost, config=config, spec=spec, transfer=transfer
    )
    artifacts.write_json("resolved_recipe.json", recipe)
    _save_params(context.artifacts_dir / "initial_params.npz", state.params)

    update = jax.jit(make_update(config, env, spec, cost, trunk_mode), donate_argnums=0)
    compile_started = time.perf_counter()
    compiled = update.lower(state).compile()
    compile_seconds = time.perf_counter() - compile_started
    batch_size = config.num_envs * env.episode_length
    updates = math.ceil(config.total_env_steps / batch_size)
    active_started = time.perf_counter()
    active_seconds = 0.0
    curves = []
    for iteration in range(1, updates + 1):
        started = time.perf_counter()
        state, raw_metrics = compiled(state)
        metrics = {key: float(value) for key, value in raw_metrics.items()}
        duration = time.perf_counter() - started
        active_seconds += duration
        record = {
            "iteration": iteration,
            **metrics,
            "time_iter_s": duration,
            "time_total_s": time.perf_counter() - active_started,
            "env_steps_per_s": batch_size / duration,
        }
        curves.append(record)
        artifacts.append_result(record)
        if trunk_mode == "frozen" and iteration == 1:
            if not metrics["trunk_bit_identical"]:
                raise RuntimeError("frozen encoder changed during its first update")
            if not metrics["heads_changed"]:
                raise RuntimeError("fresh transfer heads did not change on first update")
            recipe["transfer"]["frozen_encoder_update_verified"] = True
            recipe["transfer"]["fresh_heads_changed_after_update"] = True
            artifacts.write_json("resolved_recipe.json", recipe)
    with (context.results_dir / "training_curves.jsonl").open("w") as handle:
        handle.writelines(json.dumps(row, sort_keys=True) + "\n" for row in curves)
    _save_params(context.artifacts_dir / "final_params.npz", state.params)
    final = curves[-1]
    summary = {
        "study": "pusher_b10_guess_or_pass_jax_2026_10",
        "algorithm": "ppo",
        "cost": cost,
        "seed": context.seed,
        "smoke": context.smoke,
        "target_env_steps": config.total_env_steps,
        "completed_env_steps": int(state.env_steps),
        "episode_return_mean": final["return_mean"],
        "guess_correct": final["guess_correct"],
        "guess_wrong": final["guess_wrong"],
        "passed": final["passed"],
        "bayes_expected_reward": final["bayes_expected_reward"],
        "transfer": recipe.get("transfer"),
        "reward_per_step": final["reward_per_step"],
        "compile_seconds": compile_seconds,
        "env_steps_per_second_active": int(state.env_steps)
        / max(active_seconds, 1e-9),
        "device": str(jax.devices()[0]),
        "parameter_count": tm.parameter_count(state.params),
    }
    artifacts.write_json("summary.json", summary)
    return summary
