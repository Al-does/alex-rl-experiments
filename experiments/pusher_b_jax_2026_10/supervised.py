"""Pusher-B next-token cross-entropy ("supervised") in pure JAX.

Recipe of ``experiments/pusher_b/supervised.py``: BOS + 127 emissions,
shifted next-token CE, AdamW(lr 1e-3, betas 0.9/0.999, wd 0), batch 512,
10,000 updates, 4,096 held-out sequences scored against the exact Bayesian
predictive distribution. Sequences are sampled on device inside the jitted
update (``JaxHMMEnv.sample_tokens``), ``log_every`` updates per jit call.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from harness.artifacts import RunArtifacts

from experiments.pusher_b_jax_2026_10 import model as tm
from experiments.pusher_b_jax_2026_10.process import (
    BOS_TOKEN,
    EPISODE_LENGTH,
    PRESETS,
    TOKEN_COUNT,
    make_env,
    pusher_b_model,
)

CHECKPOINT_STEPS = (1, 3, 10, 30, 100, 300, 1_000, 3_000, 10_000)

# Direct Kelly wager constants, mirroring
# ``experiments/mess3_token_guess_cycle_2/learning.py``: a fair N-way bet
# returns Nx the stake on a win (NET_WIN_ODDS = N - 1; Pusher guesses two
# tokens, so 1.0) and wagers are capped just below all-in.
KELLY_NET_WIN_ODDS = float(TOKEN_COUNT - 1)
KELLY_MAX_WAGER = 1.0 - 1e-4


@dataclass(frozen=True)
class SupervisedConfig:
    total_steps: int = 10_000
    eval_steps: tuple[int, ...] = CHECKPOINT_STEPS
    batch_size: int = 512
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    log_every: int = 100
    validation_batch_size: int = 4_096
    validation_minibatch_size: int = 512
    # When set, the update budget is ceil(target_env_steps / tokens per
    # update); each emitted token counts as one env step.
    target_env_steps: int | None = None
    # Decoupled Kelly aux weight: 0.0 is plain next-token CE. Positive adds
    # a per-class wager head trained on realized log-growth of the sampled
    # next-token guess, exactly the token-guess Kelly loss off the shared
    # trunk.
    kelly_weight: float = 0.0
    # Keep the params seen at each eval boundary for later probing.
    param_checkpoints: bool = False

    @classmethod
    def smoke(cls) -> SupervisedConfig:
        return cls(
            total_steps=4,
            eval_steps=(2, 4),
            batch_size=4,
            log_every=2,
            validation_batch_size=8,
            validation_minibatch_size=4,
        )

    @property
    def resolved_total_steps(self) -> int:
        if self.target_env_steps is None:
            return self.total_steps
        tokens_per_update = self.batch_size * EPISODE_LENGTH
        return -(-self.target_env_steps // tokens_per_update)


def with_bos(tokens: jax.Array) -> jax.Array:
    bos = jnp.full(tokens.shape[:-1] + (1,), BOS_TOKEN, tokens.dtype)
    return jnp.concatenate([bos, tokens], axis=-1)


def _losses(spec, config: SupervisedConfig, params, tokens, act_key):
    """Next-token CE plus the optional decoupled Kelly wager loss."""

    embedding = tm.encode(spec, params, tokens[:, :-1])
    logits = tm.lm_logits(params, embedding)
    targets = tokens[:, 1:]
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    if config.kelly_weight <= 0.0:
        return ce, {}
    wager_logits = tm.kelly_logits(params, embedding)
    # The supervised analogue of the RL arm's sampled action: the guessable
    # classes are the emitted tokens {0, 1}, sampled from the model's own
    # next-token distribution restricted to them.
    action = jax.random.categorical(act_key, logits[..., :TOKEN_COUNT])
    correct = action == targets
    wager = jnp.minimum(
        jax.nn.sigmoid(
            jnp.take_along_axis(wager_logits, action[..., None], axis=-1)[..., 0]
        ),
        KELLY_MAX_WAGER,
    )
    growth = jnp.where(
        correct, jnp.log1p(wager * KELLY_NET_WIN_ODDS), jnp.log1p(-wager)
    )
    kelly_loss = -growth.mean()
    return ce + config.kelly_weight * kelly_loss, {
        "kelly_loss": kelly_loss,
        "kelly_log_growth": growth.mean(),
        "kelly_wager_mean": wager.mean(),
        "kelly_guess_correct": correct.astype(jnp.float32).mean(),
    }


def make_trainer(preset: str, config: SupervisedConfig, spec: tm.ModelSpec):
    env = make_env(preset)
    optimizer = optax.adamw(
        config.learning_rate,
        b1=config.beta1,
        b2=config.beta2,
        eps=1e-8,
        weight_decay=config.weight_decay,
    )
    grad_fn = jax.value_and_grad(partial(_losses, spec, config), has_aux=True)

    def update(carry, step_keys):
        params, opt_state = carry
        data_key, act_key = step_keys
        tokens = with_bos(env.sample_tokens(data_key, config.batch_size, EPISODE_LENGTH))
        (loss, aux), grads = grad_fn(params, tokens, act_key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state), {
            "training_loss_nats": loss,
            **aux,
        }

    @partial(jax.jit, static_argnames="num_updates", donate_argnums=(0, 1))
    def train_chunk(params, opt_state, data_key, act_key, num_updates):
        xs = (
            jax.random.split(data_key, num_updates),
            jax.random.split(act_key, num_updates),
        )
        (params, opt_state), losses = jax.lax.scan(update, (params, opt_state), xs)
        return params, opt_state, jax.tree.map(jnp.mean, losses)

    @jax.jit
    def eval_chunk(params, tokens, floor_probs):
        logits = tm.lm_logits(params, tm.encode(spec, params, tokens[:, :-1]))
        targets = tokens[:, 1:]
        nll = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        floor = -jnp.log(
            jnp.take_along_axis(floor_probs, targets[..., None], axis=-1)[..., 0]
        )
        greedy = logits[..., :TOKEN_COUNT].argmax(-1) == targets
        return (
            nll.sum(),
            floor.sum(),
            greedy.astype(jnp.float32).sum(),
            floor_probs.max(-1).sum(),
        )

    return env, optimizer, train_chunk, eval_chunk


def train(
    preset: str,
    *,
    seed: int,
    config: SupervisedConfig | None = None,
    spec: tm.ModelSpec | None = None,
    log=print,
) -> dict[str, Any]:
    config = config or SupervisedConfig()
    spec = spec or tm.ModelSpec()
    env, optimizer, train_chunk, eval_chunk = make_trainer(preset, config, spec)
    init_key, train_key, valid_key = jax.random.split(jax.random.key(seed), 3)
    # Separate stream for the Kelly arm's action sampling so the data stream
    # (and init, via init_params key order) is identical across arms.
    act_key = jax.random.fold_in(jax.random.key(seed), 0x6B65)
    params = tm.init_params(
        spec, init_key, lm_head=True, kelly_head=config.kelly_weight > 0.0
    )
    opt_state = optimizer.init(params)
    raw = env.sample_tokens(valid_key, config.validation_batch_size, EPISODE_LENGTH)
    validation = with_bos(raw)
    floor_probs = env.predictive_distributions(raw)

    def evaluate(step, params):
        nll = floor = acc = oracle = 0.0
        mb = config.validation_minibatch_size
        for start in range(0, config.validation_batch_size, mb):
            a, b, c, d = eval_chunk(
                params, validation[start : start + mb], floor_probs[start : start + mb]
            )
            nll, floor, acc, oracle = (
                nll + float(a),
                floor + float(b),
                acc + float(c),
                oracle + float(d),
            )
        count = config.validation_batch_size * EPISODE_LENGTH
        record = {
            "kind": "validation",
            "step": step,
            "validation_loss_nats": nll / count,
            "bayesian_floor_nats": floor / count,
            "excess_loss_nats": (nll - floor) / count,
            "greedy_accuracy": acc / count,
            "bayesian_optimal_accuracy": oracle / count,
        }
        log(record)
        return record

    total_steps = config.resolved_total_steps
    history = [evaluate(0, params)]
    # Snapshots live on the host: train_chunk donates its input buffers, so a
    # stored device pytree is deleted by the next compiled call.
    eval_params: dict[int, Any] = (
        {0: jax.device_get(params)} if config.param_checkpoints else {}
    )
    boundaries = sorted(
        set(range(config.log_every, total_steps + 1, config.log_every))
        | {s for s in config.eval_steps if s <= total_steps}
        | {total_steps}
    )
    started = time.perf_counter()
    compile_s = 0.0
    active_s = 0.0
    active_steps = 0
    done = 0
    compiled: dict[int, Any] = {}
    for boundary in boundaries:
        n = boundary - done
        data_key = jax.random.fold_in(train_key, boundary)
        chunk_act_key = jax.random.fold_in(act_key, boundary)
        if n not in compiled:
            t0 = time.perf_counter()
            compiled[n] = train_chunk.lower(
                params, opt_state, data_key, chunk_act_key, num_updates=n
            ).compile()
            compile_s += time.perf_counter() - t0
        t0 = time.perf_counter()
        params, opt_state, chunk = compiled[n](
            params, opt_state, data_key, chunk_act_key
        )
        active_s += time.perf_counter() - t0
        active_steps += n
        done = boundary
        rate = active_steps / max(active_s, 1e-9)
        if done % config.log_every == 0 or done == total_steps:
            record = {
                "kind": "training",
                "step": done,
                "env_steps": done * config.batch_size * EPISODE_LENGTH,
                **{k: float(v) for k, v in chunk.items()},
                "updates_per_second_active": rate,
                "sequences_per_second_active": rate * config.batch_size,
                "end_to_end_wall_seconds": time.perf_counter() - started,
            }
            history.append(record)
            log(record)
        if done in config.eval_steps or done == total_steps:
            history.append(evaluate(done, params))
            if config.param_checkpoints:
                eval_params[done] = jax.device_get(params)
    final = history[-1]
    rate = active_steps / max(active_s, 1e-9)
    return {
        "completed_step": done,
        "completed_env_steps": done * config.batch_size * EPISODE_LENGTH,
        "validation_loss_nats": final["validation_loss_nats"],
        "bayesian_floor_nats": final["bayesian_floor_nats"],
        "excess_loss_nats": final["excess_loss_nats"],
        "greedy_accuracy": final["greedy_accuracy"],
        "bayesian_optimal_accuracy": final["bayesian_optimal_accuracy"],
        "active_optimization_wall_seconds": active_s,
        "compile_seconds": compile_s,
        "updates_per_second_active": rate,
        "sequences_per_second_active": rate * config.batch_size,
        "end_to_end_training_wall_seconds": time.perf_counter() - started,
        "device": str(jax.devices()[0]),
        "parameter_count": tm.parameter_count(params),
        "history": history,
        "params": params,
        "eval_params": eval_params,
    }


def _save_params(path, params) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **{
            jax.tree_util.keystr(k): np.asarray(v)
            for k, v in jax.tree_util.tree_leaves_with_path(params)
        },
    )


def run_supervised(
    context,
    *,
    preset: str,
    kelly: bool = False,
    target_env_steps: int | None = None,
    param_checkpoints: bool = False,
) -> dict[str, Any]:
    if preset not in PRESETS:
        raise ValueError(f"unknown Pusher-B preset: {preset!r}")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    config = SupervisedConfig.smoke() if context.smoke else SupervisedConfig()
    config = replace(
        config,
        kelly_weight=(1.0 if kelly else 0.0),
        target_env_steps=None if context.smoke else target_env_steps,
        param_checkpoints=param_checkpoints,
    )
    spec = tm.ModelSpec()
    hmm = pusher_b_model(preset)
    condition = (
        "supervised_next_token_prediction_kelly"
        if kelly
        else "supervised_next_token_prediction"
    )
    outputs.write_json(
        "resolved_recipe.json",
        {
            "study": "pusher_b_jax",
            "condition": condition,
            "preset": preset,
            "parameters": PRESETS[preset],
            "seed": context.seed,
            "smoke": context.smoke,
            "legacy_recipe": "experiments/pusher_b/supervised.py",
            "edge_transition_matrices": hmm.edge_transition_matrices.tolist(),
            "model": asdict(spec),
            "objective": (
                "shifted next-token cross-entropy + decoupled Kelly wager loss"
                if kelly
                else "shifted next-token cross-entropy"
            ),
            "kelly": {
                "loss_weight": config.kelly_weight,
                "head_logits": TOKEN_COUNT,
                "net_win_odds": KELLY_NET_WIN_ODDS,
                "max_wager": KELLY_MAX_WAGER,
                "action": "sampled from softmax over guessable tokens",
                "reference": (
                    "experiments/mess3_token_guess_cycle_2/learning.py "
                    "(decoupled_kelly arm)"
                ),
            },
            "training": {
                **asdict(config),
                "resolved_total_updates": config.resolved_total_steps,
                "resolved_env_steps": (
                    config.resolved_total_steps
                    * config.batch_size
                    * EPISODE_LENGTH
                ),
            },
        },
    )
    result = train(preset, seed=context.seed, config=config, spec=spec, log=outputs.append_result)
    params = result.pop("params")
    eval_params = result.pop("eval_params")
    result.pop("history")
    _save_params(context.artifacts_dir / "final_params.npz", params)
    for step, eval_param in eval_params.items():
        _save_params(
            context.artifacts_dir / "param_checkpoints" / f"update_{step:07d}.npz",
            eval_param,
        )
    summary = {
        "study": "pusher_b_jax",
        "condition": condition,
        "preset": preset,
        "seed": context.seed,
        "smoke": context.smoke,
        "training": result,
    }
    outputs.write_json("summary.json", summary)
    return summary
