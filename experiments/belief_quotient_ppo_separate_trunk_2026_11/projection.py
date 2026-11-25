"""Lossless report fields from the full checkpoint summaries and training curves."""
import json

PROJECTION = "checkpoint_projection.jsonl"
TARGETS = ("rho", "delta", "v_star")


def project(summary, curves):
    reports = summary["checkpoint_reports"]
    if ([row["checkpoint_step"] for row in reports] != [0, *(row["env_steps"] for row in curves)]
            or curves[-1]["reward_per_step"] != summary["training"]["final_reward_per_step"]):
        raise ValueError("checkpoint and sampled training trajectories do not align")
    trunks = sorted(reports[0]["trunks"])
    header = {"seed": summary["seed"], "batch_steps": summary["training"]["batch_steps"],
              "env_steps": summary["training"]["env_steps"], "trunks": trunks}
    rows = [header]
    for report, sample in zip(reports, [None, *(row["reward_per_step"] for row in curves)], strict=True):
        policy = report["policy"]
        rows.append([
            report["checkpoint_step"], sample, policy["full_episode_reward_per_step"],
            policy["full_episode_reward_se"],
            *[[report["trunks"][trunk][target]["error"] for target in TARGETS] for trunk in trunks],
        ])
    return "".join(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n" for row in rows)


def load_projection(path):
    with path.open() as handle:
        header = json.loads(next(handle))
        reports = []
        for line in handle:
            step, sample, reward, reward_se, *trunks = json.loads(line)
            reports.append({
                "checkpoint_step": step,
                "sampled_reward_per_step": sample,
                "policy": {"full_episode_reward_per_step": reward, "full_episode_reward_se": reward_se},
                "trunks": {
                    trunk: {target: {"error": error} for target, error in zip(TARGETS, values, strict=True)}
                    for trunk, values in zip(header["trunks"], trunks, strict=True)
                },
            })
    return {
        "seed": header["seed"],
        "training": {"batch_steps": header["batch_steps"], "env_steps": header["env_steps"],
                     "final_reward_per_step": reports[-1]["sampled_reward_per_step"] if reports else None},
        "checkpoint_reports": reports,
    }
