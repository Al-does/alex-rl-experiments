"""Cross-agent block-4 summary, plus the strength-tuning curves averaged over agents."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.two_factor_coupled_moving_reward_steering_v2_jax_2026_11.summarize import summarize

RESULTS = Path(__file__).resolve().parent / "results"


def tuning_curves(reports: list[dict]) -> dict:
    """Mean tuning reward and action-change fraction per strength (block 4, 64 episodes per agent)."""
    out = {}
    for reward in ("xor", "moving"):
        grids = [r["rewards"][reward]["strength_tuning"]["grid"] for r in reports]
        out[reward] = {}
        for key in grids[0]:
            rows = {}
            for strength in grids[0][key]:
                cells = [g[key][strength] for g in grids]
                gain = np.array([g[key][strength]["total_reward"] - g[key]["0.0"]["total_reward"]
                                 for g in grids])
                rows[strength] = {
                    "total_reward_mean": round(float(np.mean([c["total_reward"] for c in cells])), 4),
                    "gain_over_unsteered_mean": round(float(gain.mean()), 4),
                    "gain_over_unsteered_se_across_agents": round(
                        float(gain.std(ddof=1) / np.sqrt(len(gain))), 4),
                    "factor_action_change_fraction_mean": np.mean(
                        [c["factor_action_change_fraction"] for c in cells], 0).round(4).tolist()}
            out[reward][key] = rows
    return out


def main() -> None:
    out = summarize(RESULTS)
    reports = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("eps02_ctx32__*seed*.json"))]
    out["strength_tuning_mean_over_agents"] = tuning_curves(reports)
    out["selected_strength"] = {
        reward: {key: [r["rewards"][reward]["strength_tuning"]["selected_strength"][key] for r in reports]
                 for key in reports[0]["rewards"][reward]["strength_tuning"]["selected_strength"]}
        for reward in ("xor", "moving")}
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
