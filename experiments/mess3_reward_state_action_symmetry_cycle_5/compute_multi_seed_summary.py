"""Aggregate five-seed Vast results for cycle 5 action-symmetry variants."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

RUN_PATTERN = re.compile(r"mess3-rsa-c5-v([123])-seed(\d+)$")
ACTION_LABELS = ("noop", "positive", "negative")


def _mean_std(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": float(statistics.mean(values)),
        "stdev": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
        "n": len(values),
    }


def _action_dict(fractions: list[float]) -> dict[str, float]:
    return {
        label: float(value)
        for label, value in zip(ACTION_LABELS, fractions, strict=True)
    }


def load_run(run_dir: Path) -> dict[str, Any]:
    summary = json.loads((run_dir / "condition_summary.json").read_text())
    tune = json.loads((run_dir / "tune_summary.json").read_text())
    final_probe = summary["final_probe"]
    metrics = tune["trials"][0]["metrics"]
    return {
        "seed": int(summary["seed"]),
        "run_id": run_dir.name,
        "final_mse": float(final_probe["mse"]),
        "global_mse_ratio": float(final_probe["global_mse_ratio"]),
        "fine_mse_ratio": float(final_probe["fine_mse_ratio"]),
        "reward_state_2_fraction_greedy": float(
            final_probe["reward_state_2_fraction_greedy"]
        ),
        "reward_state_2_fraction_greedy_pct": round(
            100.0 * float(final_probe["reward_state_2_fraction_greedy"]), 3
        ),
        "greedy_action_fractions": [
            float(x) for x in final_probe["greedy_action_fractions"]
        ],
        "greedy_action_fractions_pct": [
            round(100.0 * float(x), 2)
            for x in final_probe["greedy_action_fractions"]
        ],
        "greedy_action_distribution": _action_dict(
            final_probe["greedy_action_fractions"]
        ),
        "mean_episode_return": float(metrics["env_runners/episode_return_mean"]),
        "num_env_steps_sampled_lifetime": int(
            metrics["env_runners/num_env_steps_sampled_lifetime"]
        ),
    }


def aggregate(study_root: Path) -> dict[str, Any]:
    arms: dict[str, dict[str, Any]] = {}
    for results_dir in sorted(study_root.glob("variant_*/results/mess3-rsa-c5-*")):
        match = RUN_PATTERN.fullmatch(results_dir.name)
        if match is None:
            continue
        variant = f"variant_{match.group(1)}"
        arms.setdefault(
            variant,
            {"run_name_prefix": f"mess3-rsa-c5-v{match.group(1)}", "per_seed": []},
        )
        arms[variant]["per_seed"].append(load_run(results_dir))

    expected = {f"variant_{index}" for index in (1, 2, 3)}
    if set(arms) != expected:
        raise ValueError(f"expected {sorted(expected)}, found {sorted(arms)}")

    aggregate_stats: dict[str, Any] = {}
    for variant, payload in sorted(arms.items()):
        per_seed = sorted(payload["per_seed"], key=lambda item: item["seed"])
        if [item["seed"] for item in per_seed] != [42, 43, 44, 45, 46]:
            raise ValueError(
                f"{variant} has unexpected seeds: {[item['seed'] for item in per_seed]}"
            )
        payload["run_ids"] = {
            str(item["seed"]): item["run_id"] for item in per_seed
        }
        payload["per_seed"] = per_seed
        aggregate_stats[variant] = {
            "final_mse": _mean_std([item["final_mse"] for item in per_seed]),
            "mean_episode_return": _mean_std(
                [item["mean_episode_return"] for item in per_seed]
            ),
            "reward_state_2_fraction_greedy": _mean_std(
                [item["reward_state_2_fraction_greedy"] for item in per_seed]
            ),
            "reward_state_2_fraction_greedy_pct": _mean_std(
                [item["reward_state_2_fraction_greedy_pct"] for item in per_seed]
            ),
            "greedy_action_fractions": {
                label: _mean_std(
                    [item["greedy_action_distribution"][label] for item in per_seed]
                )
                for label in ACTION_LABELS
            },
            "greedy_action_fractions_pct": {
                label: _mean_std(
                    [
                        100.0 * item["greedy_action_distribution"][label]
                        for item in per_seed
                    ]
                )
                for label in ACTION_LABELS
            },
        }

    return {
        "study": "mess3_reward_state_action_symmetry_cycle_5",
        "provider": "vast",
        "seeds": [42, 43, 44, 45, 46],
        "experiment_ref": "1903c88dcb619f95394ee53de2df32b017f5de3a",
        "library_ref": "2d2c2ff4cd57b5e10d08a18eaa76ffc4c4c73d2c",
        "design_note": (
            "Sticky-state baseline row s2=[0.30,0.30,0.40]; transformer d=64, "
            "4 layers, 1 head, context 10; otherwise same protocol as cycle 4."
        ),
        "vast_instances": {
            "variant_1": {"id": 46259576, "run_name_prefix": "mess3-rsa-c5-v1"},
            "variant_2": {"id": 46260515, "run_name_prefix": "mess3-rsa-c5-v2"},
            "variant_3": {"id": 46260749, "run_name_prefix": "mess3-rsa-c5-v3"},
        },
        "note": (
            "One Vast RTX 4090 box per variant; seeds 42-46 sequential. "
            "Compact results recovered via SCP after manual tmux launch."
        ),
        "action_labels": list(ACTION_LABELS),
        "arms": arms,
        "aggregate": aggregate_stats,
    }


def render_findings(payload: dict[str, Any]) -> str:
    lines = [
        "# Cycle 5 action-symmetry — Vast multi-seed findings",
        "",
        "Five seeded runs per variant (`seeds 42–46`) on separate Vast RTX 4090 "
        "boxes (`delay=0`, `700k` env steps, experiment SHA "
        f"`{payload['experiment_ref'][:7]}`, harness `{payload['library_ref'][:7]}`).",
        "",
        "## Mean episode return (training)",
        "",
        "| variant | mean return | std |",
        "|---------|------------:|----:|",
    ]
    for variant in ("variant_1", "variant_2", "variant_3"):
        stats = payload["aggregate"][variant]["mean_episode_return"]
        lines.append(
            f"| {variant} | {stats['mean']:.2f} | {stats['stdev']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Final greedy reward-state-2 occupancy (held-out probe rollout)",
            "",
            "Fraction of greedy steps while the hidden state is reward state 2.",
            "",
            "| seed | variant_1 | variant_2 | variant_3 |",
            "|-----:|----------:|----------:|----------:|",
        ]
    )
    by_variant = {
        variant: {
            item["seed"]: item["reward_state_2_fraction_greedy_pct"]
            for item in payload["arms"][variant]["per_seed"]
        }
        for variant in ("variant_1", "variant_2", "variant_3")
    }
    for seed in payload["seeds"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(seed),
                    f"{by_variant['variant_1'][seed]:.1f}%",
                    f"{by_variant['variant_2'][seed]:.1f}%",
                    f"{by_variant['variant_3'][seed]:.1f}%",
                ]
            )
            + " |"
        )
    lines.append("")
    for variant in ("variant_1", "variant_2", "variant_3"):
        stats = payload["aggregate"][variant]["reward_state_2_fraction_greedy_pct"]
        lines.append(
            f"- **{variant}**: {stats['mean']:.1f}% ± {stats['stdev']:.1f}% "
            f"(range {stats['min']:.1f}–{stats['max']:.1f}%)"
        )

    lines.extend(
        [
            "",
            "## Final greedy action mix",
            "",
            "Percentages are `[noop, positive, negative]` from final checkpoint probes.",
            "",
            "| variant | mean noop | mean pos | mean neg |",
            "|---------|----------:|---------:|---------:|",
        ]
    )
    for variant in ("variant_1", "variant_2", "variant_3"):
        action_stats = payload["aggregate"][variant]["greedy_action_fractions_pct"]
        lines.append(
            "| "
            + " | ".join(
                [
                    variant,
                    f"{action_stats['noop']['mean']:.1f}%",
                    f"{action_stats['positive']['mean']:.1f}%",
                    f"{action_stats['negative']['mean']:.1f}%",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Final probe MSE",
            "",
            "| seed | variant_1 | variant_2 | variant_3 |",
            "|-----:|----------:|----------:|----------:|",
        ]
    )
    mse_by_variant = {
        variant: {
            item["seed"]: item["final_mse"]
            for item in payload["arms"][variant]["per_seed"]
        }
        for variant in ("variant_1", "variant_2", "variant_3")
    }
    for seed in payload["seeds"]:
        lines.append(
            "| "
            + " | ".join(
                [str(seed)]
                + [f"{mse_by_variant[v][seed]:.6f}" for v in ("variant_1", "variant_2", "variant_3")]
            )
            + " |"
        )
    lines.extend(["", "See `multi_seed_summary.json` for machine-readable aggregates."])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    args = parser.parse_args()
    payload = aggregate(args.study_root)
    summary_path = args.study_root / "multi_seed_summary.json"
    findings_path = args.study_root / "findings.md"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")
    findings_path.write_text(render_findings(payload))


if __name__ == "__main__":
    main()
