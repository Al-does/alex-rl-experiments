"""Pusher-B next-token cross-entropy ("supervised") in pure JAX.

Recipe of ``experiments/pusher_b/supervised.py``: BOS + 127 emissions,
shifted next-token CE, AdamW(lr 1e-3, betas 0.9/0.999, wd 0), batch 512,
10,000 updates, 4,096 held-out sequences scored against the exact Bayesian
predictive distribution. Sequences are sampled on device inside the jitted
update (``JaxHMMEnv.sample_tokens``), ``log_every`` updates per jit call.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
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
    make_env,
    pusher_b_model,
)

CHECKPOINT_STEPS = (1, 3, 10, 30, 100, 300, 1_000, 3_000, 10_000)


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


def with_bos(tokens: jax.Array) -> jax.Array:
    bos = jnp.full(tokens.shape[:-1] + (1,), BOS_TOKEN, tokens.dtype)
    return jnp.concatenate([bos, tokens], axis=-1)


def _loss(spec, params, tokens):
    logits = tm.lm_logits(params, tm.encode(spec, params, tokens[:, :-1]))
    return optax.softmax_cross_entropy_with_integer_labels(logits, tokens[:, 1:]).mean()


def make_trainer(preset: str, config: SupervisedConfig, spec: tm.ModelSpec):
    env = make_env(preset)
    optimizer = optax.adamw(
        config.learning_rate,
        b1=config.beta1,
        b2=config.beta2,
        eps=1e-8,
        weight_decay=config.weight_decay,
    )
    grad_fn = jax.value_and_grad(partial(_loss, spec))

    def update(carry, step_key):
        params, opt_state = carry
        tokens = with_bos(env.sample_tokens(step_key, config.batch_size, EPISODE_LENGTH))
        loss, grads = grad_fn(params, tokens)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state), loss

    @partial(jax.jit, static_argnames="num_updates", donate_argnums=(0, 1))
    def train_chunk(params, opt_state, key, num_updates):
        (params, opt_state), losses = jax.lax.scan(
            update, (params, opt_state), jax.random.split(key, num_updates)
        )
        return params, opt_state, losses.mean()

    @jax.jit
    def eval_chunk(params, tokens, floor_probs):
        logits = tm.lm_logits(params, tm.encode(spec, params, tokens[:, :-1]))
        nll = optax.softmax_cross_entropy_with_integer_labels(logits, tokens[:, 1:])
        floor = -jnp.log(
            jnp.take_along_axis(floor_probs, tokens[:, 1:, None], axis=-1)[..., 0]
        )
        return nll.sum(), floor.sum()

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
    params = tm.init_params(spec, init_key, lm_head=True)
    opt_state = optimizer.init(params)
    raw = env.sample_tokens(valid_key, config.validation_batch_size, EPISODE_LENGTH)
    validation = with_bos(raw)
    floor_probs = env.predictive_distributions(raw)

    def evaluate(step, params):
        nll = floor = 0.0
        mb = config.validation_minibatch_size
        for start in range(0, config.validation_batch_size, mb):
            a, b = eval_chunk(
                params, validation[start : start + mb], floor_probs[start : start + mb]
            )
            nll, floor = nll + float(a), floor + float(b)
        count = config.validation_batch_size * EPISODE_LENGTH
        record = {
            "kind": "validation",
            "step": step,
            "validation_loss_nats": nll / count,
            "bayesian_floor_nats": floor / count,
            "excess_loss_nats": (nll - floor) / count,
        }
        log(record)
        return record

    history = [evaluate(0, params)]
    boundaries = sorted(
        set(range(config.log_every, config.total_steps + 1, config.log_every))
        | {s for s in config.eval_steps if s <= config.total_steps}
        | {config.total_steps}
    )
    started = time.perf_counter()
    compile_s = 0.0
    active_s = 0.0
    active_steps = 0
    done = 0
    compiled: dict[int, Any] = {}
    for boundary in boundaries:
        n = boundary - done
        chunk_key = jax.random.fold_in(train_key, boundary)
        if n not in compiled:
            t0 = time.perf_counter()
            compiled[n] = train_chunk.lower(
                params, opt_state, chunk_key, num_updates=n
            ).compile()
            compile_s += time.perf_counter() - t0
        t0 = time.perf_counter()
        params, opt_state, loss = compiled[n](params, opt_state, chunk_key)
        loss = float(loss)
        active_s += time.perf_counter() - t0
        active_steps += n
        done = boundary
        rate = active_steps / max(active_s, 1e-9)
        if done % config.log_every == 0 or done == config.total_steps:
            record = {
                "kind": "training",
                "step": done,
                "training_loss_nats": loss,
                "updates_per_second_active": rate,
                "sequences_per_second_active": rate * config.batch_size,
                "end_to_end_wall_seconds": time.perf_counter() - started,
            }
            history.append(record)
            log(record)
        if done in config.eval_steps or done == config.total_steps:
            history.append(evaluate(done, params))
    final = history[-1]
    rate = active_steps / max(active_s, 1e-9)
    return {
        "completed_step": done,
        "validation_loss_nats": final["validation_loss_nats"],
        "bayesian_floor_nats": final["bayesian_floor_nats"],
        "excess_loss_nats": final["excess_loss_nats"],
        "active_optimization_wall_seconds": active_s,
        "compile_seconds": compile_s,
        "updates_per_second_active": rate,
        "sequences_per_second_active": rate * config.batch_size,
        "end_to_end_training_wall_seconds": time.perf_counter() - started,
        "device": str(jax.devices()[0]),
        "parameter_count": tm.parameter_count(params),
        "history": history,
        "params": params,
    }



def run_supervised(context, *, preset: str) -> dict[str, Any]:
    if preset not in PRESETS:
        raise ValueError(f"unknown Pusher-B preset: {preset!r}")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    config = SupervisedConfig.smoke() if context.smoke else SupervisedConfig()
    spec = tm.ModelSpec()
    hmm = pusher_b_model(preset)
    outputs.write_json(
        "resolved_recipe.json",
        {
            "study": "pusher_b_jax",
            "condition": "supervised_next_token_prediction",
            "preset": preset,
            "parameters": PRESETS[preset],
            "seed": context.seed,
            "smoke": context.smoke,
            "legacy_recipe": "experiments/pusher_b/supervised.py",
            "edge_transition_matrices": hmm.edge_transition_matrices.tolist(),
            "model": asdict(spec),
            "objective": "shifted next-token cross-entropy",
            "training": asdict(config),
        },
    )
    result = train(preset, seed=context.seed, config=config, spec=spec, log=outputs.append_result)
    params = result.pop("params")
    result.pop("history")
    np.savez(
        context.artifacts_dir / "final_params.npz",
        **{jax.tree_util.keystr(k): np.asarray(v) for k, v in jax.tree_util.tree_leaves_with_path(params)},
    )
    summary = {
        "study": "pusher_b_jax",
        "condition": "supervised_next_token_prediction",
        "preset": preset,
        "seed": context.seed,
        "smoke": context.smoke,
        "training": result,
    }
    outputs.write_json("summary.json", summary)
    return summary
