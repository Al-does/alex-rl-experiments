"""Held-out ladder of next-token CE vs CE + decoupled Kelly on the sum channel.

Same protocol as ``experiments/pusher_b_jax_2026_10/heldout.py`` (one fixed
per-seed sequence pool, train only on its first ``1 - h`` fraction, 400M env
steps, checkpoints at init, 1/5/20/50, then every 200 updates and the final
update, params uploaded off-thread), on :class:`process.SumChannel`.

Every checkpoint records on fixed train and held-out rows:

- CE, excess CE over the exact Bayes floor, perplexity (model, Bayes and the
  factored assumed-density filter), and ``factored_gap_closed`` =
  ``(CE_factored - CE_model) / (CE_factored - CE_Bayes)``: above 0 means the
  model beats the best purely factored belief tracker, 1 is Bayes-optimal.
- Affine probes (post-final-norm embedding, fit on one half of a split's probe
  rows, ``1 - R^2`` on the other) onto three targets: the joint belief over
  ``3**N`` states, the concatenated factor marginals (``3N``), and the
  correlation residual ``joint - product(marginals)`` (zero when ``eps = 0``).
- Effective dimension: principal components for 95% of embedding variance
  (BOS position excluded), against the factored ``2N`` and joint ``3**N - 1``.
- Guess hit rate ``E[p_model(next token)]`` (the rate at which a guess
  sampled from the model's own predictive is right; Bayes reference
  ``E[p_Bayes]``), and for the Kelly arm the expected wager and expected Kelly
  log growth of that sampled guess.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from analysis.probes import fit_affine_probe, global_mse_metrics, probe_predict
from harness.artifacts import RunArtifacts

from experiments.factored_sum_channel_n3_2026_10.process import (
    CONTEXT_LENGTH,
    EPISODE_LENGTH,
    EPSILON,
    MESS3_ALPHA,
    MESS3_X,
    SumChannel,
)
from experiments.pusher_b_jax_2026_10 import model as tm
from experiments.pusher_b_jax_2026_10.heldout import (
    EARLY_EVAL_STEPS,
    AsyncCheckpointWriter,
    _upload_policy,
)
from experiments.pusher_b_jax_2026_10.supervised import KELLY_MAX_WAGER, _save_params

HELDOUT_LADDER = (0.20, 0.40, 0.80)
PROBE_TARGETS = ("joint", "marginals", "residual")
CEV_THRESHOLD = 0.95


@dataclass(frozen=True)
class HeldoutConfig:
    heldout_fraction: float = 0.2
    pool_sequences: int = 131_072
    target_env_steps: int = 400_000_000
    batch_size: int = 512
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    kelly_weight: float = 0.0
    early_eval_steps: tuple[int, ...] = EARLY_EVAL_STEPS
    eval_every: int = 200
    eval_sequences: int = 4_096
    eval_minibatch: int = 512
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


def model_spec(channel: SumChannel) -> tm.ModelSpec:
    """The Pusher-B JAX transformer (4 layers, d128) with this vocabulary."""

    return tm.ModelSpec(vocab=channel.vocab_size, context_length=CONTEXT_LENGTH)


def init_params(spec: tm.ModelSpec, channel: SumChannel, key, *, kelly: bool) -> dict:
    """LM params; the Kelly head (one wager logit per emitted token) uses its
    own key fold so trunk init is identical across the CE and Kelly arms."""

    params = tm.init_params(spec, key, lm_head=True)
    if kelly:
        params["kelly"] = tm._torch_default_linear(
            jax.random.fold_in(key, 0x4B), spec.d_model, channel.token_count
        )
    return params


def with_bos(tokens: jax.Array, bos: int) -> jax.Array:
    pad = jnp.full(tokens.shape[:-1] + (1,), bos, tokens.dtype)
    return jnp.concatenate([pad, tokens], axis=-1)


def _losses(spec, channel: SumChannel, kelly_weight: float, params, tokens, act_key):
    """Next-token CE plus the decoupled Kelly loss on a sampled guess.

    The guess is sampled from the model's predictive over the emitted tokens
    (BOS excluded); the bet pays fair odds for a uniform guess
    (``token_count - 1``), as in ``pusher_b_jax_2026_10/supervised.py``.
    """

    embedding = tm.encode(spec, params, tokens[:, :-1])
    logits = tm.lm_logits(params, embedding)
    targets = tokens[:, 1:]
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()
    if kelly_weight <= 0.0:
        return ce, {}
    odds = float(channel.token_count - 1)
    wager_logits = tm.kelly_logits(params, embedding)
    action = jax.random.categorical(act_key, logits[..., : channel.token_count])
    correct = action == targets
    wager = jnp.minimum(
        jax.nn.sigmoid(
            jnp.take_along_axis(wager_logits, action[..., None], axis=-1)[..., 0]
        ),
        KELLY_MAX_WAGER,
    )
    growth = jnp.where(correct, jnp.log1p(wager * odds), jnp.log1p(-wager))
    kelly_loss = -growth.mean()
    return ce + kelly_weight * kelly_loss, {
        "kelly_loss": kelly_loss,
        "kelly_log_growth": growth.mean(),
        "kelly_wager_mean": wager.mean(),
        "kelly_guess_correct": correct.astype(jnp.float32).mean(),
    }


def effective_dimension(features: np.ndarray, threshold: float = CEV_THRESHOLD) -> int:
    """Principal components needed for ``threshold`` of the variance."""

    flat = features.reshape(-1, features.shape[-1])
    flat = flat - flat.mean(axis=0)
    variance = np.linalg.svd(flat, compute_uv=False) ** 2
    if variance.sum() <= 0.0:
        return 0
    return int(np.searchsorted(np.cumsum(variance) / variance.sum(), threshold) + 1)


def _probe_targets(channel: SumChannel, beliefs: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "joint": beliefs,
        "marginals": channel.marginals(beliefs),
        "residual": beliefs - channel.product_of_marginals(beliefs),
    }


def train(
    channel: SumChannel,
    *,
    seed: int,
    config: HeldoutConfig,
    spec: tm.ModelSpec | None = None,
    log=print,
    on_checkpoint=None,
) -> dict[str, Any]:
    config.validate()
    spec = spec or model_spec(channel)
    if spec.vocab != channel.vocab_size:
        raise ValueError("model vocab must match the channel vocabulary")
    kelly = config.kelly_weight > 0.0
    V, bos = channel.token_count, channel.bos_token
    env = channel.make_env()
    init_key, train_key, pool_key = jax.random.split(jax.random.key(seed), 3)
    act_key = jax.random.fold_in(jax.random.key(seed), 0x6B65)
    params = init_params(spec, channel, init_key, kelly=kelly)
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
    train_pool = with_bos(raw_pool[:n_train], bos)
    e, p = config.eval_sequences, config.probe_sequences
    eval_raw = {"train": np.asarray(raw_pool[:e]), "heldout": np.asarray(raw_pool[-e:])}
    probe_raw = {
        "train": np.asarray(raw_pool[e : e + p]),
        "heldout": np.asarray(raw_pool[-(e + p) : -e]),
    }
    # Per-position reference NLLs of the realized next token on the eval rows.
    bayes_nll = {k: channel.run_filter(v)[1] for k, v in eval_raw.items()}
    factored_nll = {
        k: channel.run_filter(v, project=True)[1] for k, v in eval_raw.items()
    }
    references = {}
    for split in eval_raw:
        floor, fact = bayes_nll[split].mean(), factored_nll[split].mean()
        references.update(
            {
                f"{split}_bayesian_floor_nats": floor,
                f"{split}_factored_floor_nats": fact,
                f"{split}_factored_gap_nats": fact - floor,
                f"{split}_bayesian_perplexity": float(np.exp(floor)),
                f"{split}_factored_perplexity": float(np.exp(fact)),
                f"{split}_bayes_guess_hit_rate": float(
                    np.exp(-bayes_nll[split]).mean()
                ),
            }
        )
    eval_tokens = {k: with_bos(jnp.asarray(v), bos) for k, v in eval_raw.items()}
    eval_floor = {k: jnp.asarray(v, jnp.float32) for k, v in bayes_nll.items()}
    probe_tokens = {k: with_bos(jnp.asarray(v), bos) for k, v in probe_raw.items()}
    probe_targets = {
        k: _probe_targets(channel, channel.run_filter(v)[0])
        for k, v in probe_raw.items()
    }
    half = p // 2
    joint_heldout = probe_targets["heldout"]["joint"][:, 1:]
    references["heldout_belief_effective_dimension"] = {
        "exact_joint": effective_dimension(joint_heldout),
        "product_of_marginals": effective_dimension(
            channel.product_of_marginals(joint_heldout)
        ),
        "factored_prediction": 2 * channel.n_factors,
        "joint_prediction": channel.n_states - 1,
    }
    log({"kind": "references", **references})

    grad_fn = jax.value_and_grad(
        partial(_losses, spec, channel, config.kelly_weight), has_aux=True
    )

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
    def eval_chunk(params, tokens, floor_nll):
        embedding = tm.encode(spec, params, tokens[:, :-1])
        logits = tm.lm_logits(params, embedding)
        targets = tokens[:, 1:]
        nll = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        greedy = logits.argmax(-1) == targets
        guess_probs = jax.nn.softmax(logits[..., :V], axis=-1)
        hit = jnp.take_along_axis(guess_probs, targets[..., None], axis=-1)[..., 0]
        sums = [nll.sum(), floor_nll.sum(), greedy.astype(jnp.float32).sum(), hit.sum()]
        if kelly:
            odds = float(V - 1)
            wager = jnp.minimum(
                jax.nn.sigmoid(tm.kelly_logits(params, embedding)), KELLY_MAX_WAGER
            )
            right = jax.nn.one_hot(targets, V, dtype=bool)
            growth = jnp.where(right, jnp.log1p(wager * odds), jnp.log1p(-wager))
            sums += [
                (guess_probs * wager).sum(),
                (guess_probs * growth).sum(),
                (
                    hit * jnp.take_along_axis(wager, targets[..., None], -1)[..., 0]
                ).sum(),
            ]
        return jnp.stack(sums)

    @jax.jit
    def embed(params, tokens):
        return tm.encode(spec, params, tokens[:, :-1])

    def score(split: str, params) -> dict[str, float]:
        tokens, floor = eval_tokens[split], eval_floor[split]
        mb = config.eval_minibatch
        totals = sum(
            np.asarray(
                eval_chunk(params, tokens[s : s + mb], floor[s : s + mb]), np.float64
            )
            for s in range(0, len(tokens), mb)
        )
        totals = totals / (len(tokens) * EPISODE_LENGTH)
        nll, floor_nll, acc, hit = totals[:4]
        fact = references[f"{split}_factored_floor_nats"]
        gap = fact - floor_nll
        out = {
            f"{split}_loss_nats": nll,
            f"{split}_excess_loss_nats": nll - floor_nll,
            f"{split}_perplexity": float(np.exp(nll)),
            f"{split}_factored_gap_closed": (fact - nll) / gap
            if gap > 1e-9
            else float("nan"),
            f"{split}_greedy_accuracy": acc,
            f"{split}_guess_hit_rate": hit,
        }
        if kelly:
            out[f"{split}_kelly_wager_mean"] = totals[4]
            out[f"{split}_kelly_log_growth"] = totals[5]
            out[f"{split}_kelly_wager_when_right"] = totals[6] / max(hit, 1e-12)
        return out

    def probe(split: str, params) -> dict[str, float]:
        features = np.asarray(embed(params, probe_tokens[split]), dtype=np.float64)
        flat = lambda a: a.reshape(-1, a.shape[-1])
        out: dict[str, float] = {
            f"{split}_effective_dimension": effective_dimension(features[:, 1:])
        }
        for name, target in probe_targets[split].items():
            weight, bias = fit_affine_probe(
                flat(features[:half]), flat(target[:half]), ridge=config.probe_ridge
            )
            metrics = global_mse_metrics(
                probe_predict(weight, bias, flat(features[half:])), flat(target[half:])
            )
            ratio = (
                metrics["global_mse_ratio"]
                if metrics["target_variance"] > 1e-12
                else float("nan")
            )
            out[f"{split}_probe_{name}_target_variance"] = metrics["target_variance"]
            out[f"{split}_probe_{name}_1_minus_r2"] = ratio
        return out

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
        params, opt_state, chunk = compiled[n](
            params, opt_state, data_key, chunk_act_key
        )
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
        "references": references,
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


ARM_COLORS = {"CE": "tab:blue", "CE+Kelly": "tab:orange"}


def _arm_color(label: str) -> str:
    return ARM_COLORS["CE+Kelly" if "Kelly" in label else "CE"]


def plot_curves(
    series: dict[str, list[dict[str, Any]]],
    references: dict[str, Any],
    path: Path,
    *,
    title: str,
) -> Path:
    """Held-out joint-belief probe 1 - R^2 (left, solid) and excess CE (right,
    dashed) vs env steps, one color per arm; grey dotted line is the factored
    filter's excess CE (what a purely factored representation can reach)."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax_probe = plt.subplots(figsize=(7, 4.2))
    ax_loss = ax_probe.twinx()
    for label, curve in series.items():
        color = _arm_color(label)
        steps = [r["env_steps"] for r in curve]
        ax_probe.plot(
            steps,
            [r["heldout_probe_joint_1_minus_r2"] for r in curve],
            "-",
            color=color,
            alpha=0.9,
            label=f"{label} joint probe 1-R²",
        )
        ax_loss.plot(
            steps,
            [r["heldout_excess_loss_nats"] for r in curve],
            "--",
            color=color,
            alpha=0.9,
            label=f"{label} excess CE",
        )
    gap = references.get("heldout_factored_gap_nats", 0.0)
    if gap > 1e-9:
        ax_loss.axhline(gap, color="grey", ls=":", label="factored filter excess CE")
    ax_probe.set_xlabel("training steps (tokens)")
    ax_probe.set_ylabel("held-out joint belief probe 1 - R² (solid)")
    ax_loss.set_ylabel("held-out CE above Bayes floor, nats (dashed)")
    ax_probe.set_ylim(bottom=0)
    ax_loss.set_ylim(bottom=0)
    handles = [
        h for ax in (ax_probe, ax_loss) for h in ax.get_legend_handles_labels()[0]
    ]
    ax_probe.legend(handles=handles, fontsize=7, loc="upper right")
    ax_probe.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_diagnostics(
    series: dict[str, list[dict[str, Any]]],
    references: dict[str, Any],
    path: Path,
    *,
    title: str,
) -> Path:
    """2x4 held-out panels: the three probes, effective dimension, excess CE,
    perplexity, guess hit rate and Kelly wager; grey lines are references."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("heldout_probe_joint_1_minus_r2", "joint belief probe 1-R²", []),
        ("heldout_probe_marginals_1_minus_r2", "factor marginals probe 1-R²", []),
        ("heldout_probe_residual_1_minus_r2", "correlation residual probe 1-R²", []),
        (
            "heldout_effective_dimension",
            "dims for 95% variance",
            [
                (references.get("heldout_belief_effective_dimension", {}).get(k), name)
                for k, name in (
                    ("factored_prediction", "factored 2N"),
                    ("exact_joint", "exact beliefs (95%)"),
                )
            ],
        ),
        (
            "heldout_excess_loss_nats",
            "excess CE over Bayes (nats)",
            [(references.get("heldout_factored_gap_nats"), "factored filter")],
        ),
        (
            "heldout_perplexity",
            "perplexity",
            [
                (references.get("heldout_bayesian_perplexity"), "Bayes"),
                (references.get("heldout_factored_perplexity"), "factored filter"),
            ],
        ),
        (
            "heldout_guess_hit_rate",
            "sampled-guess hit rate",
            [(references.get("heldout_bayes_guess_hit_rate"), "Bayes")],
        ),
        ("heldout_kelly_wager_mean", "Kelly expected wager", []),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    styles = [":", "-."]
    for ax, (key, name, refs) in zip(axes.flat, panels):
        for label, curve in series.items():
            points = [(r["env_steps"], r[key]) for r in curve if key in r]
            if points:
                ax.plot(*zip(*points), color=_arm_color(label), alpha=0.9, label=label)
        for (value, ref_name), style in zip(refs, styles):
            if value is not None and np.isfinite(value):
                ax.axhline(value, color="grey", ls=style, label=ref_name)
        if key == "heldout_perplexity":
            ax.set_yscale("log")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("training steps (tokens)", fontsize=8)
        ax.tick_params(labelsize=7)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def run_label(heldout_fraction: float, epsilon: float, kelly: bool) -> str:
    eps = "" if epsilon == EPSILON else f" eps={epsilon:g}"
    return f"h{round(100 * heldout_fraction)}{eps} {'CE+Kelly' if kelly else 'CE'}"


def run_heldout(
    context,
    *,
    n_factors: int,
    heldout_fraction: float,
    kelly: bool = False,
    epsilon: float = EPSILON,
) -> dict[str, Any]:
    channel = SumChannel(n_factors=n_factors, epsilon=epsilon)
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
    spec = model_spec(channel)
    study = f"factored_sum_channel_n{n_factors}"
    condition = "heldout_ntp_kelly" if kelly else "heldout_ntp_ce"
    outputs.write_json(
        "resolved_recipe.json",
        {
            "study": study,
            "condition": condition,
            "seed": context.seed,
            "smoke": context.smoke,
            "process": {
                "factors": n_factors,
                "factor": "Mess3 (edge-emitting)",
                "alpha": MESS3_ALPHA,
                "x": MESS3_X,
                "channel": "w.p. 1-eps the sub-token tuple, w.p. eps only sum mod 3",
                "epsilon": epsilon,
                "joint_states": channel.n_states,
                "emitted_tokens": channel.token_count,
                "vocab_with_bos": channel.vocab_size,
            },
            "model": asdict(spec),
            "objective": (
                "shifted next-token CE + decoupled Kelly wager loss"
                if kelly
                else "shifted next-token CE"
            ),
            "kelly": {
                "loss_weight": config.kelly_weight,
                "net_win_odds": channel.token_count - 1,
                "max_wager": KELLY_MAX_WAGER,
                "guess": "sampled from model predictive over emitted tokens",
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
                "targets": {
                    "joint": "exact Bayesian filtering belief over 3**N joint states",
                    "marginals": "its N factor marginals, concatenated",
                    "residual": "joint - product of its marginals",
                },
                "positions": "all input positions incl. BOS",
                "fit_score_split": "first/second half of probe sequences, per split",
                "metric": "1 - R^2 (global_mse_ratio); NaN when target variance is 0",
                "ridge": config.probe_ridge,
                "effective_dimension": "PCs for 95% embedding variance, BOS excluded",
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
            channel,
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
    label = run_label(heldout_fraction, epsilon, kelly)
    references = result["references"]
    outputs.write_json(
        "curve.json", {"label": label, "references": references, "checkpoints": curve}
    )
    title = f"N={n_factors} {label}"
    plot_curves(
        {label: curve}, references, outputs.results_dir / "curve.png", title=title
    )
    plot_diagnostics(
        {label: curve}, references, outputs.results_dir / "diagnostics.png", title=title
    )
    _save_params(Path(context.artifacts_dir) / "final_params.npz", params)
    result["checkpoint_uploads"] = {
        "written": len(uploads),
        "uploaded": sum(bool(u.get("uploaded")) for u in uploads),
        "failed": writer.failures,
    }
    summary = {
        "study": study,
        "condition": condition,
        "n_factors": n_factors,
        "epsilon": epsilon,
        "heldout_fraction": heldout_fraction,
        "seed": context.seed,
        "smoke": context.smoke,
        "training": result,
    }
    outputs.write_json("summary.json", summary)
    return summary
