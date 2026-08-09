"""Held-out init belief probes over untrained transformer architectures."""

from __future__ import annotations

import json
import statistics as stats
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from learners.models.transformer import TransformerModelConfig

from experiments.mess3_reward_state_action_symmetry_cycle_5.analysis import (
    probe_checkpoint,
)
from experiments.mess3_reward_state_action_symmetry_cycle_5.shared import (
    build_config,
)
from harness.context import RunContext


DEFAULT_SEEDS = (42, 43, 44, 45, 46)
# Internal rollout env wiring only; not part of the ablation comparison.
_ROLLOUT_ENV_VARIANT = 2


@dataclass(frozen=True, slots=True)
class ArchitectureSpec:
    key: str
    label: str
    d_model: int
    n_layers: int
    n_heads: int
    context_len: int

    def to_model_config(self) -> dict[str, Any]:
        return TransformerModelConfig(
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            context_len=self.context_len,
        ).to_dict()


ARCHITECTURE_SPECS = (
    ArchitectureSpec(
        key="small_baseline",
        label="small baseline (64/4/1/10)",
        d_model=64,
        n_layers=4,
        n_heads=1,
        context_len=10,
    ),
    ArchitectureSpec(
        key="width96_small_style",
        label="96-wide, small depth/heads/context",
        d_model=96,
        n_layers=4,
        n_heads=1,
        context_len=10,
    ),
    ArchitectureSpec(
        key="ablate_layers",
        label="3 layers (large depth, else small)",
        d_model=64,
        n_layers=3,
        n_heads=1,
        context_len=10,
    ),
    ArchitectureSpec(
        key="ablate_heads",
        label="4 heads (large head count, else small)",
        d_model=64,
        n_layers=4,
        n_heads=4,
        context_len=10,
    ),
    ArchitectureSpec(
        key="ablate_context",
        label="context 64 (large band, else small)",
        d_model=64,
        n_layers=4,
        n_heads=1,
        context_len=64,
    ),
    ArchitectureSpec(
        key="large_full",
        label="large full (96/3/4/64)",
        d_model=96,
        n_layers=3,
        n_heads=4,
        context_len=64,
    ),
)


def _save_initial_checkpoint(config, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    algorithm = config.build_algo()
    try:
        saved = algorithm.save_to_path(str(path))
    finally:
        algorithm.stop()
    return Path(saved)


def _probe_init(
    context: RunContext,
    *,
    spec: ArchitectureSpec,
) -> dict[str, Any]:
    if context.seed is None:
        raise ValueError("init architecture ablation requires a resolved seed")
    model_config = spec.to_model_config()
    build_context = replace(context, smoke=True)
    config = build_config(
        build_context,
        _ROLLOUT_ENV_VARIANT,
        model_config=model_config,
    )
    checkpoint_dir = (
        context.artifacts_dir.resolve()
        / f"init_ablation_seed{context.seed}_{spec.key}"
    )
    checkpoint = _save_initial_checkpoint(config, checkpoint_dir)
    probe_dir = context.results_dir / spec.key / f"seed_{context.seed}"
    probe_dir.mkdir(parents=True, exist_ok=True)
    result = probe_checkpoint(
        replace(context, results_dir=probe_dir),
        checkpoint=checkpoint,
        condition=f"{spec.key}_init",
        agent_steps=0,
    )
    metrics = result.metrics
    return {
        "architecture": spec.key,
        "architecture_label": spec.label,
        "seed": context.seed,
        "model_config": model_config,
        "mse": float(metrics["mse"]),
        "target_variance": float(metrics["target_variance"]),
        "global_mse_ratio": float(metrics["global_mse_ratio"]),
        "branch_baseline_mse": float(metrics["branch_baseline_mse"]),
        "fine_mse_ratio": float(metrics["fine_mse_ratio"]),
        "r_squared": float(metrics["r_squared"]),
        "reward_state_2_fraction_greedy": float(
            metrics["reward_state_2_fraction_greedy"]
        ),
        "greedy_action_fractions": metrics["greedy_action_fractions"],
        "checkpoint": str(checkpoint),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    by_arch: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_arch.setdefault(row["architecture"], []).append(row)
    for key, group in sorted(by_arch.items()):
        mses = [row["mse"] for row in group]
        summary[key] = {
            "label": group[0]["architecture_label"],
            "n": len(group),
            "mse_mean": stats.mean(mses),
            "mse_sd": stats.pstdev(mses) if len(mses) > 1 else 0.0,
            "global_mse_ratio_mean": stats.mean(
                row["global_mse_ratio"] for row in group
            ),
            "fine_mse_ratio_mean": stats.mean(
                row["fine_mse_ratio"] for row in group
            ),
            "per_seed_mse": {
                str(row["seed"]): row["mse"]
                for row in sorted(group, key=lambda item: item["seed"])
            },
        }
    return summary


def run_ablation(
    context: RunContext,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    specs: tuple[ArchitectureSpec, ...] = ARCHITECTURE_SPECS,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for spec in specs:
            row_context = replace(context, seed=seed)
            print(
                f"probing seed={seed} arch={spec.key} "
                f"({'smoke' if context.smoke else 'full'})",
                flush=True,
            )
            rows.append(_probe_init(row_context, spec=spec))
    payload = {
        "study": "mess3_untrained_transformer_probe_ablation_2026_08",
        "smoke": context.smoke,
        "seeds": list(seeds),
        "architectures": [asdict(spec) for spec in specs],
        "rows": rows,
        "summary": _summarize(rows),
    }
    context.results_dir.mkdir(parents=True, exist_ok=True)
    output = context.results_dir / "init_architecture_ablation.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
