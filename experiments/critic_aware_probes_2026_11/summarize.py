"""Build compact git-tracked summary JSON for critic-aware probe results."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = STUDY_DIR / "artifacts"
RESULTS_DIR = STUDY_DIR / "results"


def _git_rev(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except Exception:
        return "unknown"


def _mean_or_none(vals: list[float | None]) -> float | None:
    clean = [float(v) for v in vals if v is not None]
    return float(np.mean(clean)) if clean else None


def _summarize_ladder() -> dict[str, Any]:
    files = sorted((ARTIFACTS_DIR / "ladder").glob("*.json"))
    rows = [json.loads(p.read_text()) for p in files]
    out: dict[str, Any] = {}
    targets = ("rho", "delta", "b", "V_star", "dV_nl", "Q_star", "dQ_nl", "A_star", "dA_nl")
    for variant, vname in ((1, "trivial_quotient"), (2, "reward_state_quotient"), (3, "full_state_quotient")):
        out[vname] = {}
        for alg, block, label in (
            ("reinforce", "matched_init_probes", "init_reinforce"),
            ("reinforce", "actor_probes", "reinforce"),
            ("ppo", "actor_probes", "ppo_shared"),
            ("sac_autoalpha", "actor_probes", "sac_actor"),
            ("sac_autoalpha", "critic_0", "sac_critic_q1"),
            ("sac_autoalpha", "critic_1", "sac_critic_q2"),
        ):
            sub = [r for r in rows if r["variant"] == variant and r["algorithm"] == alg]
            if not sub:
                continue
            entry: dict[str, Any] = {
                "seeds": sorted(int(r["seed"]) for r in sub),
                "greedy_reward_full_mean": float(np.mean([r["greedy_reward_full"] for r in sub])),
                "v_nl_var_frac_mean": float(np.mean([r["target_nonlinear_var_frac"]["v_nl_var_frac"] for r in sub])),
                "probe_error": {},
            }
            for k in targets:
                vals = []
                for r in sub:
                    d = r["critic_probes"][block] if block.startswith("critic_") else r[block]
                    vals.append(d[k]["error"])
                entry["probe_error"][k] = _mean_or_none(vals)
            if label == "ppo_shared":
                entry["direct_v_head_corr_dV_nl"] = _mean_or_none(
                    [r["head_fidelity"]["ppo_v_head_corr_with_dV_nl"] for r in sub]
                )
            elif label == "sac_critic_q1":
                entry["direct_q0_nl_r2_vs_dQ_nl"] = _mean_or_none(
                    [r["head_fidelity"]["sac_q0_nl_vs_dQ_nl"]["r2"] for r in sub]
                )
            out[vname][label] = entry
    return out


def _summarize_reward_factors() -> dict[str, Any]:
    files = sorted((ARTIFACTS_DIR / "reward_factors").glob("*.json"))
    rows = [json.loads(p.read_text()) for p in files]
    out: dict[str, Any] = {}
    targets = ("b1", "b2", "V1", "dV1_nl", "V2", "dV2_nl", "V_star", "dV_nl", "Q_star", "dQ_nl")
    for cond, cname in (("reward_both", "joint_reward"), ("reward_factor_1", "marginal_reward")):
        out[cname] = {}
        for alg, block, label in (
            ("reinforce", "matched_init_probes", "init_reinforce"),
            ("reinforce", "actor_probes", "reinforce"),
            ("ppo", "actor_probes", "ppo_shared"),
            ("sac_autoalpha", "actor_probes", "sac_actor"),
            ("sac_autoalpha", "critic_0", "sac_critic_q1"),
            ("sac_autoalpha", "critic_1", "sac_critic_q2"),
        ):
            sub = [r for r in rows if r["condition"] == cond and r["algorithm"] == alg]
            if not sub:
                continue
            entry: dict[str, Any] = {
                "seeds": sorted(int(r["seed"]) for r in sub),
                "greedy_reward_full_mean": float(np.mean([r["greedy_reward_full"] for r in sub])),
                "probe_error": {},
            }
            for k in targets:
                vals = []
                for r in sub:
                    d = r["critic_probes"][block] if block.startswith("critic_") else r[block]
                    vals.append(d[k]["error"])
                entry["probe_error"][k] = _mean_or_none(vals)
            out[cname][label] = entry
    return out


def _summarize_mess3_passive() -> dict[str, Any]:
    p = ARTIFACTS_DIR / "mess3_passive" / "mess3_passive_seed45.json"
    return json.loads(p.read_text())


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "provenance": {
            "alex_rl_experiments_rev": _git_rev(REPO_ROOT),
            "rl_harness_rev": _git_rev(REPO_ROOT.parent / "rl-harness"),
            "deep_beliefs_rev": _git_rev(REPO_ROOT.parent / "deep-beliefs"),
        },
        "belief_quotient_ladder": _summarize_ladder(),
        "reward_relevance": _summarize_reward_factors(),
        "mess3_passive": _summarize_mess3_passive(),
    }
    out_path = RESULTS_DIR / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
