"""Finite-pool next-token CE (+ optional Kelly) with a held-out split.

The ``supervised.py`` recipe samples a fresh batch every update, so its
validation set is just more draws from the training distribution. Here one
fixed pool of HMM sequences is drawn per seed and split by sequence: the
first ``train_sequences`` rows are the only data ever trained on (batches
are uniform draws with replacement from them); the rest are held out and
only scored. Rungs of the ladder differ only in ``heldout_fraction``, and
the pool order is shared, so smaller training sets are prefixes of larger
ones and every rung scores the same fixed train/held-out evaluation rows.

At every checkpoint (log-spaced early, then every ``eval_every`` updates)
the run records CE, excess loss over the exact Bayesian predictive, and
greedy accuracy on both splits, plus an affine belief probe: post-final-norm
embeddings are regressed onto the exact Bayesian filtering belief over the
three hidden states, fit on one half of a split's probe sequences and scored
(``1 - R^2``) on the other half. Parameters at each checkpoint are written
and uploaded to B2 on a background thread so training never waits on I/O.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from analysis.probes import (
    fit_affine_probe,
    global_mse_metrics,
    predictive_belief_sequence,
    probe_predict,
)
from harness.artifacts import RunArtifacts

from experiments.pusher_b_jax_2026_10 import model as tm
from experiments.pusher_b_jax_2026_10.process import (
    EPISODE_LENGTH,
    PRESETS,
    TOKEN_COUNT,
    make_env,
    pusher_b_model,
)
from experiments.pusher_b_jax_2026_10.supervised import (
    KELLY_MAX_WAGER,
    KELLY_NET_WIN_ODDS,
    _losses,
    _save_params,
    with_bos,
)

HELDOUT_LADDER = (0.20, 0.60, 0.80, 0.90)
EARLY_EVAL_STEPS = (1, 5, 20, 50)


@dataclass(frozen=True)
class HeldoutConfig:
    heldout_fraction: float = 0.5
    pool_sequences: int = 131_072
    target_env_steps: int = 400_000_000
    batch_size: int = 512
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    kelly_weight: float = 0.0
    # Checkpoints: EARLY_EVAL_STEPS, then every eval_every updates, then the
    # final update. 400M env steps = 6,152 updates -> 36 checkpoints incl. init.
    early_eval_steps: tuple[int, ...] = EARLY_EVAL_STEPS
    eval_every: int = 200
    # Fixed scoring rows. Train rows are the first ``eval_sequences`` of the
    # pool and held-out rows the last, so both are in their split on every
    # rung (train >= 10% of the pool, held-out >= 20%).
    eval_sequences: int = 4_096
    eval_minibatch: int = 512
    # Probe rows per split (next block in from each end); half fit, half score.
    probe_sequences: int = 1_024
    probe_ridge: float = 1e-6
    save_checkpoints: bool = True

    @classmethod
    def smoke(cls, **overrides) -> HeldoutConfig:
        base = cls(
            pool_sequences=1_024,
            target_env_steps=4 * 4 * EPISODE_LENGTH,
            batch_size=4,
            early_eval_steps=(1, 2),
            eval_every=2,
            eval_sequences=16,
            eval_minibatch=8,
            probe_sequences=16,
        )
        return replace(base, **overrides)

    @property
    def train_sequences(self) -> int:
        return round((1.0 - self.heldout_fraction) * self.pool_sequences)

    @property
    def total_updates(self) -> int:
        return -(-self.target_env_steps // (self.batch_size * EPISODE_LENGTH))

    @property
    def eval_steps(self) -> tuple[int, ...]:
        total = self.total_updates
        steps = {0, total, *range(self.eval_every, total, self.eval_every)}
        steps |= {s for s in self.early_eval_steps if s < total}
        return tuple(sorted(steps))

    def validate(self) -> None:
        if not 0.0 < self.heldout_fraction < 1.0:
            raise ValueError("heldout_fraction must be in (0, 1)")
        reserved = self.eval_sequences + self.probe_sequences
        if self.train_sequences < reserved:
            raise ValueError(
                f"train split ({self.train_sequences}) smaller than its "
                f"eval+probe rows ({reserved})"
            )
        if self.pool_sequences - self.train_sequences < reserved:
            raise ValueError("held-out split smaller than its eval+probe rows")
        if self.probe_sequences < 2:
            raise ValueError("probe_sequences must allow a fit/score split")


def filtering_beliefs(edges: np.ndarray, initial: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Exact float64 belief over hidden states at each model input position.

    Position ``t`` of the shifted CE input has seen BOS and ``x_1..x_t``, so
    its target is the belief after ``t`` edge updates (stationary at BOS).
    Returns ``(n, EPISODE_LENGTH, S)``.
    """

    return np.stack(
        [
            predictive_belief_sequence(initial, (edges[x] for x in row[:-1]))
            for row in np.asarray(raw)
        ]
    )


class AsyncCheckpointWriter:
    """Write ``.npz`` param checkpoints and push them to B2 off-thread.

    The caller hands over host arrays; serialization and upload both run on
    one worker thread, so training only pays the device-to-host copy.
    Upload failures are logged and never abort training; the end-of-run
    artifact upload is the durability backstop.
    """

    def __init__(self, context, root: Path, *, upload: bool, log=print) -> None:
        self.context = context
        self.root = root
        self.upload = upload
        self.log = log
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ckpt")
        self.futures: list[Future] = []
        self.failures = 0

    def submit(self, step: int, host_params) -> None:
        self.futures.append(self.pool.submit(self._write, step, host_params))

    def _write(self, step: int, host_params) -> dict[str, Any]:
        directory = self.root / f"update_{step:07d}"
        _save_params(directory / "params.npz", host_params)
        if not self.upload:
            return {"step": step, "uploaded": False}
        from harness.storage import upload_artifact_directory

        try:
            summary = upload_artifact_directory(self.context, directory)
        except Exception as error:  # noqa: BLE001 - never kill training
            self.failures += 1
            print(f"[checkpoint-upload] FAILED update {step}: {error}", flush=True)
            return {"step": step, "uploaded": False, "error": str(error)}
        return {"step": step, "uploaded": True, "uri": summary["base_uri"]}

    def close(self) -> list[dict[str, Any]]:
        results = [future.result() for future in self.futures]
        self.pool.shutdown()
        return results


def _upload_policy(context) -> bool:
    from harness.storage import is_b2_configured

    if context.upload_artifacts is False:
        return False
    if context.smoke and not context.publish_smoke and context.upload_artifacts is None:
        return False
    return is_b2_configured()


def train(
    preset: str,
    *,
    seed: int,
    config: HeldoutConfig,
    spec: tm.ModelSpec | None = None,
    log=print,
    on_checkpoint=None,
) -> dict[str, Any]:
    config.validate()
    spec = spec or tm.ModelSpec()
    env = make_env(preset)
    hmm = pusher_b_model(preset)
    init_key, train_key, pool_key = jax.random.split(jax.random.key(seed), 3)
    act_key = jax.random.fold_in(jax.random.key(seed), 0x6B65)
    params = tm.init_params(
        spec, init_key, lm_head=True, kelly_head=config.kelly_weight > 0.0
    )
    optimizer = optax.adamw(
        config.learning_rate,
        b1=config.beta1,
        b2=config.beta2,
        eps=1e-8,
        weight_decay=config.weight_decay,
    )
    opt_state = optimizer.init(params)

    raw_pool = env.sample_tokens(pool_key, config.pool_sequences, EPISODE_LENGTH)
    n_train = config.train_sequences
    train_pool = with_bos(raw_pool[:n_train])
    e, p = config.eval_sequences, config.probe_sequences
    splits = {
        "train": raw_pool[:e],
        "heldout": raw_pool[-e:],
    }
    probe_raw = {
        "train": np.asarray(raw_pool[e : e + p]),
        "heldout": np.asarray(raw_pool[-(e + p) : -e]),
    }
    floor_probs = {k: env.predictive_distributions(v) for k, v in splits.items()}
    splits = {k: with_bos(v) for k, v in splits.items()}
    probe_tokens = {k: with_bos(jnp.asarray(v)) for k, v in probe_raw.items()}
    edges = np.asarray(hmm.edge_transition_matrices, dtype=np.float64)
    initial = np.asarray(hmm.initial_distribution, dtype=np.float64)
    probe_targets = {
        k: filtering_beliefs(edges, initial, v) for k, v in probe_raw.items()
    }
    half = p // 2

    grad_fn = jax.value_and_grad(partial(_losses, spec, config), has_aux=True)

    def update(carry, step_keys):
        params, opt_state = carry
        data_key, step_act_key = step_keys
        rows = jax.random.randint(data_key, (config.batch_size,), 0, n_train)
        (loss, aux), grads = grad_fn(params, train_pool[rows], step_act_key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state), {
            "training_loss_nats": loss,
            **aux,
        }

    @partial(jax.jit, static_argnames="num_updates", donate_argnums=(0, 1))
    def train_chunk(params, opt_state, data_key, chunk_act_key, num_updates):
        xs = (
            jax.random.split(data_key, num_updates),
            jax.random.split(chunk_act_key, num_updates),
        )
        (params, opt_state), losses = jax.lax.scan(update, (params, opt_state), xs)
        return params, opt_state, jax.tree.map(jnp.mean, losses)

    @jax.jit
    def eval_chunk(params, tokens, floor):
        logits = tm.lm_logits(params, tm.encode(spec, params, tokens[:, :-1]))
        targets = tokens[:, 1:]
        nll = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        floor_nll = -jnp.log(
            jnp.take_along_axis(floor, targets[..., None], axis=-1)[..., 0]
        )
        greedy = logits[..., :TOKEN_COUNT].argmax(-1) == targets
        return nll.sum(), floor_nll.sum(), greedy.astype(jnp.float32).sum()

    @jax.jit
    def embed(params, tokens):
        return tm.encode(spec, params, tokens[:, :-1])

    def score(split: str, params) -> dict[str, float]:
        tokens, floor = splits[split], floor_probs[split]
        totals = np.zeros(3)
        mb = config.eval_minibatch
        for start in range(0, len(tokens), mb):
            totals += np.asarray(
                eval_chunk(params, tokens[start : start + mb], floor[start : start + mb])
            )
        nll, floor_nll, acc = totals / (len(tokens) * EPISODE_LENGTH)
        return {
            f"{split}_loss_nats": nll,
            f"{split}_bayesian_floor_nats": floor_nll,
            f"{split}_excess_loss_nats": nll - floor_nll,
            f"{split}_greedy_accuracy": acc,
        }

    def probe(split: str, params) -> dict[str, float]:
        features = np.asarray(embed(params, probe_tokens[split]), dtype=np.float64)
        target = probe_targets[split]
        flat = lambda a: a.reshape(-1, a.shape[-1])
        weight, bias = fit_affine_probe(
            flat(features[:half]), flat(target[:half]), ridge=config.probe_ridge
        )
        metrics = global_mse_metrics(
            probe_predict(weight, bias, flat(features[half:])), flat(target[half:])
        )
        return {
            f"{split}_probe_mse": metrics["mse"],
            f"{split}_probe_target_variance": metrics["target_variance"],
            f"{split}_probe_1_minus_r2": metrics["global_mse_ratio"],
        }

    def evaluate(step: int, params) -> dict[str, Any]:
        t0 = time.perf_counter()
        record: dict[str, Any] = {
            "kind": "checkpoint",
            "step": step,
            "env_steps": step * config.batch_size * EPISODE_LENGTH,
            "epochs_over_train": step * config.batch_size / n_train,
        }
        for split in ("train", "heldout"):
            record.update(score(split, params))
            record.update(probe(split, params))
        record["generalization_gap_nats"] = (
            record["heldout_loss_nats"] - record["train_loss_nats"]
        )
        record["eval_seconds"] = time.perf_counter() - t0
        log(record)
        return record

    boundaries = config.eval_steps
    history = [evaluate(0, params)]
    if on_checkpoint is not None:
        on_checkpoint(0, jax.device_get(params))
    started = time.perf_counter()
    compile_s = active_s = eval_s = 0.0
    done = 0
    compiled: dict[int, Any] = {}
    for boundary in boundaries[1:]:
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
        params, opt_state, chunk = compiled[n](params, opt_state, data_key, chunk_act_key)
        chunk = {k: float(v) for k, v in chunk.items()}
        active_s += time.perf_counter() - t0
        done = boundary
        record = {
            "kind": "training",
            "step": done,
            "env_steps": done * config.batch_size * EPISODE_LENGTH,
            **chunk,
            "updates_per_second_active": done / max(active_s, 1e-9),
            "end_to_end_wall_seconds": time.perf_counter() - started,
        }
        history.append(record)
        log(record)
        t0 = time.perf_counter()
        history.append(evaluate(done, params))
        if on_checkpoint is not None:
            on_checkpoint(done, jax.device_get(params))
        eval_s += time.perf_counter() - t0
    final = history[-1]
    return {
        "completed_step": done,
        "completed_env_steps": done * config.batch_size * EPISODE_LENGTH,
        "train_sequences": n_train,
        "heldout_sequences": config.pool_sequences - n_train,
        "checkpoint_count": len(boundaries),
        **{k: v for k, v in final.items() if k not in ("kind", "step", "env_steps")},
        "active_optimization_wall_seconds": active_s,
        "checkpoint_eval_wall_seconds": eval_s,
        "compile_seconds": compile_s,
        "updates_per_second_active": done / max(active_s, 1e-9),
        "end_to_end_training_wall_seconds": time.perf_counter() - started,
        "device": str(jax.devices()[0]),
        "parameter_count": tm.parameter_count(params),
        "history": history,
        "params": params,
    }


def plot_curves(series: dict[str, list[dict[str, Any]]], path: Path, *, title: str) -> Path:
    """Held-out probe 1 - R^2 (left axis, solid) and held-out excess CE
    (right axis, dashed) vs env steps; one color per arm (CE / CE+Kelly)."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"CE": "tab:blue", "CE+Kelly": "tab:orange"}
    fig, ax_probe = plt.subplots(figsize=(7, 4.2))
    ax_loss = ax_probe.twinx()
    for label, curve in series.items():
        color = colors["CE+Kelly" if "Kelly" in label else "CE"]
        steps = [r["env_steps"] for r in curve]
        ax_probe.plot(
            steps, [r["heldout_probe_1_minus_r2"] for r in curve], "-",
            color=color, alpha=0.9, label=f"{label} probe 1-R²",
        )
        ax_loss.plot(
            steps, [r["heldout_excess_loss_nats"] for r in curve], "--",
            color=color, alpha=0.9, label=f"{label} excess CE",
        )
    ax_probe.set_xlabel("training steps (tokens)")
    ax_probe.set_ylabel("held-out belief probe 1 - R² (solid)")
    ax_loss.set_ylabel("held-out CE above Bayes floor, nats (dashed)")
    ax_probe.set_xscale("log")
    ax_probe.set_yscale("log")
    ax_loss.set_yscale("log")
    handles = [h for ax in (ax_probe, ax_loss) for h in ax.get_legend_handles_labels()[0]]
    ax_probe.legend(handles=handles, fontsize=7, loc="upper right")
    ax_probe.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def run_heldout(
    context,
    *,
    heldout_fraction: float,
    kelly: bool = False,
    preset: str = "b10",
) -> dict[str, Any]:
    if preset not in PRESETS:
        raise ValueError(f"unknown Pusher-B preset: {preset!r}")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    overrides = {
        "heldout_fraction": heldout_fraction,
        "kelly_weight": 1.0 if kelly else 0.0,
    }
    config = (
        HeldoutConfig.smoke(**overrides)
        if context.smoke
        else replace(HeldoutConfig(), **overrides)
    )
    config.validate()
    spec = tm.ModelSpec()
    condition = "heldout_ntp_kelly" if kelly else "heldout_ntp_ce"
    outputs.write_json(
        "resolved_recipe.json",
        {
            "study": "pusher_b_jax",
            "condition": condition,
            "preset": preset,
            "parameters": PRESETS[preset],
            "seed": context.seed,
            "smoke": context.smoke,
            "model": asdict(spec),
            "objective": (
                "shifted next-token CE + decoupled Kelly wager loss"
                if kelly
                else "shifted next-token CE"
            ),
            "kelly": {
                "loss_weight": config.kelly_weight,
                "net_win_odds": KELLY_NET_WIN_ODDS,
                "max_wager": KELLY_MAX_WAGER,
            },
            "data": {
                "pool_sequences": config.pool_sequences,
                "heldout_fraction": config.heldout_fraction,
                "train_sequences": config.train_sequences,
                "heldout_sequences": config.pool_sequences - config.train_sequences,
                "sampling": "uniform with replacement from the train split",
                "train_eval_rows": "pool[:eval_sequences]",
                "heldout_eval_rows": "pool[-eval_sequences:]",
            },
            "probe": {
                "representation": "post_final_layer_norm",
                "target": "exact Bayesian filtering belief over hidden states (float64)",
                "positions": "all input positions incl. BOS",
                "fit_score_split": "first/second half of probe sequences, per split",
                "metric": "1 - R^2 (global_mse_ratio)",
                "ridge": config.probe_ridge,
            },
            "training": {
                **asdict(config),
                "resolved_total_updates": config.total_updates,
                "resolved_env_steps": config.total_updates
                * config.batch_size
                * EPISODE_LENGTH,
                "checkpoint_steps": list(config.eval_steps),
            },
        },
    )
    writer = AsyncCheckpointWriter(
        context,
        Path(context.artifacts_dir) / "param_checkpoints",
        upload=_upload_policy(context),
    )
    try:
        result = train(
            preset,
            seed=context.seed,
            config=config,
            spec=spec,
            log=outputs.append_result,
            on_checkpoint=writer.submit if config.save_checkpoints else None,
        )
    finally:
        uploads = writer.close()
    params = result.pop("params")
    curve = [r for r in result.pop("history") if r["kind"] == "checkpoint"]
    label = f"h{round(100 * heldout_fraction)} {'CE+Kelly' if kelly else 'CE'}"
    outputs.write_json("curve.json", {"label": label, "checkpoints": curve})
    plot_curves({label: curve}, outputs.results_dir / "curve.png", title=label)
    _save_params(Path(context.artifacts_dir) / "final_params.npz", params)
    result["checkpoint_uploads"] = {
        "written": len(uploads),
        "uploaded": sum(bool(u.get("uploaded")) for u in uploads),
        "failed": writer.failures,
    }
    summary = {
        "study": "pusher_b_jax",
        "condition": condition,
        "preset": preset,
        "heldout_fraction": heldout_fraction,
        "seed": context.seed,
        "smoke": context.smoke,
        "training": result,
    }
    outputs.write_json("summary.json", summary)
    return summary
