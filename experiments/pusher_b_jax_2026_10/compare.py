"""Plot and tabulate short legacy-vs-JAX runs.

    python -m experiments.pusher_b_jax_2026_10.compare \
        --legacy-ce legacy_ce/metrics.jsonl --legacy-ppo legacy_ppo/result.json \
        --jax-ce jax_ce.jsonl --jax-ppo jax_ppo.jsonl --out comparison

``--legacy-ce`` is the legacy run's ``metrics.jsonl``; ``--legacy-ppo`` is
the Tune trial's ``result.json``. Writes ``<out>.png`` and ``<out>.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _lines(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]


def legacy_ce(path):
    rows = _lines(path)
    valid = [
        (r["step"], r["excess_loss_nats"])
        for r in rows
        if r.get("kind") == "validation"
    ]
    train = [r for r in rows if r.get("kind") == "training"]
    return valid, train[-1]["updates_per_second_active"]


def jax_ce(path):
    rows = _lines(path)
    valid = [
        (r["step"], r["excess_loss_nats"])
        for r in rows
        if r.get("kind") == "validation"
    ]
    summary = next(r for r in rows if r["kind"] == "summary")
    return valid, summary["updates_per_second_active"]


def legacy_ppo(path):
    rows = _lines(path)
    curve = [
        (
            r["env_runners"]["num_env_steps_sampled_lifetime"],
            r["env_runners"]["episode_return_mean"],
        )
        for r in rows
    ]
    later = rows[2:] or rows
    steps = sum(r["env_runners"]["num_env_steps_sampled"] for r in later)
    seconds = sum(r["time_this_iter_s"] for r in later)
    return curve, steps / seconds


def jax_ppo(path):
    rows = _lines(path)
    curve = [(r["steps"], r["return_mean"]) for r in rows if r.get("iteration")]
    summary = next(r for r in rows if r["kind"] == "summary")
    return curve, summary["env_steps_per_second_active"]


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("legacy-ce", "legacy-ppo", "jax-ce", "jax-ppo"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lce, lce_rate = legacy_ce(args.legacy_ce)
    jce, jce_rate = jax_ce(args.jax_ce)
    lppo, lppo_rate = legacy_ppo(args.legacy_ppo)
    jppo, jppo_rate = jax_ppo(args.jax_ppo)

    fig, (ax_ce, ax_ppo) = plt.subplots(1, 2, figsize=(11, 4))
    for curve, label in (
        (lce, f"PyTorch ({lce_rate:.1f} upd/s)"),
        (jce, f"JAX ({jce_rate:.1f} upd/s)"),
    ):
        steps, excess = zip(*[(s, e) for s, e in curve if s > 0], strict=True)
        ax_ce.loglog(steps, excess, "o-", label=label)
    ax_ce.set(
        xlabel="updates (batch 512)",
        ylabel="excess CE over Bayes floor (nats)",
        title="b10 next-token CE",
    )
    ax_ce.legend()
    for curve, label in (
        (lppo, f"RLlib ({lppo_rate:,.0f} steps/s)"),
        (jppo, f"JAX ({jppo_rate:,.0f} steps/s)"),
    ):
        steps, returns = zip(*curve, strict=True)
        ax_ppo.plot(steps, returns, "o-", label=label)
    ax_ppo.set(
        xlabel="env steps",
        ylabel="episode return (127 steps)",
        title="b10 token-guess PPO",
    )
    ax_ppo.legend()
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=130)

    table = {
        "ce": {
            "legacy_updates_per_s": lce_rate,
            "jax_updates_per_s": jce_rate,
            "speedup": jce_rate / lce_rate,
            "legacy_excess": lce,
            "jax_excess": jce,
        },
        "ppo": {
            "legacy_env_steps_per_s": lppo_rate,
            "jax_env_steps_per_s": jppo_rate,
            "speedup": jppo_rate / lppo_rate,
            "legacy_return": lppo,
            "jax_return": jppo,
        },
    }
    Path(f"{args.out}.json").write_text(json.dumps(table, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: {
                    kk: vv
                    for kk, vv in v.items()
                    if "rate" in kk or "per_s" in kk or kk == "speedup"
                }
                for k, v in table.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
