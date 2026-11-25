"""Summarize the held-out audit across the three frozen training seeds."""

import csv
import json
from itertools import product
from math import sqrt
from statistics import mean
import subprocess

from .audit import ALGORITHMS, ROOT, SEEDS
from . import audit


CONDITIONS = (1, 2, 3)
TRUNKS = (*ALGORITHMS, "q1", "q2")


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_suite(artifact_root, mode):
    cohort = {}
    for variant, seed in product(CONDITIONS, SEEDS):
        with (artifact_root / mode / f"variant{variant}_seed{seed}.json").open() as handle:
            data = json.load(handle)
        if (data["variant"], data["seed"]) != (variant, seed):
            raise ValueError(f"Incorrect metadata for {mode=} {variant=} {seed=}")
        cohort[variant, seed] = data
    return cohort


def avg(values):
    values = list(values)
    if any(x is None for x in values):
        return None
    return mean(values)


def belief_table(cohort):
    rows = []
    for (variant, seed), per_run in cohort.items():
        for donor, protocols in per_run["matched"].items():
            for window, trunks in protocols.items():
                for trunk, scores in trunks.items():
                    for index, coordinate in enumerate(per_run["coordinates"]):
                        rows.append(dict(variant=variant, seed=seed, donor=donor, window=window,
                                         trunk=trunk, coordinate=coordinate,
                                         error=scores["error"][index], sse=scores["sse"][index],
                                         sst=scores["sst"][index], test_rows=scores["test_rows"]))
    return rows


def value_table(cohort):
    rows = []
    for (variant, seed), per_run in cohort.items():
        for donor, windows in per_run["hard_reference"]["matched"].items():
            interior = windows["64:600"]
            for trunk, targets in interior["scores"].items():
                for target, scores in targets.items():
                    rows.append(dict(variant=variant, seed=seed, donor=donor, trunk=trunk, target=target,
                                     **scores))
    return rows


def soft_table(cohort):
    rows = []
    for (variant, seed), per_run in cohort.items():
        soft = per_run["soft"]
        for field in ("q", "policy_relative_q", "q_while_omitting_entropy",
                      "v_from_min_q", "v_from_min_q_while_omitting_entropy_in_target"):
            by_head = soft[field] if not field.startswith("v_from_") else {"min_q": soft[field]}
            for head, scores in by_head.items():
                rows.append(dict(variant=variant, seed=seed, field=field, head=head,
                                 alpha=soft["alpha"], **scores))
    return rows


def target_lag_table(cohort):
    rows = []
    for (variant, seed), per_run in cohort.items():
        path = audit.find_runs(variant, seed)["sac_autoalpha"]
        if path.name != per_run["checkpoints"]["sac_autoalpha"]["run_id"]:
            raise ValueError(f"Training curve does not match the checkpoint for {variant=} {seed=}")
        config = audit.archive.read(path / "resolved_recipe.json")["training"]
        if config["q_head_init"] != "zeros":
            raise ValueError("Scalar reference requires zero Q initialization")
        gamma, tau = config["gamma"], config["tau"]
        steps = config["gradient_steps"]
        with (path / "training_curves.jsonl").open() as handle:
            curves = [json.loads(line) for line in handle]
        target = 0.0
        differences = []
        for real in curves:
            # Approximate replay rewards by the rollout mean for each iteration.
            reward = real["reward_per_step"] + gamma * real["alpha"] * real["sampled_entropy"]
            online = []
            for _ in range(steps):
                estimate = (1 - gamma) * reward + gamma * target
                online.append(estimate)
                target += tau * (estimate - target)
            logged = (1 - gamma) * real["mean_policy_q"]
            scalar = mean(online)
            differences.append(logged - scalar)
        rows.append(dict(variant=variant, seed=seed, gamma=gamma, tau=tau,
                         gradient_updates=steps * len(curves), final_stationary_q_proxy=reward,
                         final_logged_q=logged, final_scalar_q=scalar,
                         trajectory_rmse=sqrt(mean(error**2 for error in differences))))
    return rows


def aggregate(artifact_root, output):
    belief, hard, soft = (load_suite(artifact_root, mode) for mode in ("matched", "hard", "soft"))
    for key in belief:
        identities = [suite[key]["checkpoints"] for suite in (belief, hard, soft)]
        if not all(x == identities[0] for x in identities):
            raise ValueError(f"Checkpoint mismatch for {key}")
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("belief_per_seed", belief_table(belief)),
        ("optimal_value_per_seed", value_table(hard)),
        ("sac_soft_value_per_seed", soft_table(soft)),
        ("sac_target_lag_per_seed", target_lag_table(soft)),
    ):
        write_csv(output / f"{name}.csv", rows)

    summary = {"seeds": SEEDS, "variants": {}}
    for variant in CONDITIONS:
        b_runs = [belief[variant, seed]["matched"] for seed in SEEDS]
        h_runs = [hard[variant, seed]["hard_reference"]["matched"] for seed in SEEDS]
        s_runs = [soft[variant, seed]["soft"] for seed in SEEDS]
        b_means = {}
        h_means = {}
        for donor in ALGORITHMS:
            b_means[donor] = {
                trunk: {coord: mean(r[donor]["64:600"][trunk]["error"][index] for r in b_runs)
                        for index, coord in enumerate(("rho", "delta"))}
                for trunk in TRUNKS
            }
            h_means[donor] = {
                trunk: {target: avg(r[donor]["64:600"]["scores"][trunk][target]["error"] for r in h_runs)
                        for target in h_runs[0][donor]["64:600"]["scores"][trunk]}
                for trunk in TRUNKS
            }
        s_means = {}
        fields = {
            "soft_q": ("q", "min_q"),
            "reward_only_q": ("q_while_omitting_entropy", "min_q"),
            "policy_relative_q": ("policy_relative_q", "min_q"),
            "soft_v": ("v_from_min_q",),
            "reward_only_v": ("v_from_min_q_while_omitting_entropy_in_target",),
        }
        for metric, path in fields.items():
            head = [r[path[0]] if len(path) == 1 else r[path[0]][path[1]] for r in s_runs]
            s_means[metric] = {field: avg(scores[field] for scores in head)
                               for field in ("r2", "noise_adjusted_r2", "bias", "rmse")}
        summary["variants"][str(variant)] = {
            "belief_error_by_donor": b_means,
            "optimal_value_error_by_donor": h_means,
            "nonlinear_v_variance_share_by_donor": {
                donor: mean(r[donor]["64:600"]["nonlinear_variance_shares"]["v_star"] for r in h_runs)
                for donor in ALGORITHMS
            },
            "sac_mean_learned_alpha": mean(r["alpha"] for r in s_runs),
            "sac_soft_head_scores": s_means,
        }
    summary["sac_reward_state_delta_reproduction"] = {
        window: [belief[2, seed]["matched"]["sac_autoalpha"][window]["sac_autoalpha"]["error"][1]
                 for seed in SEEDS]
        for window in ("64:600", "64:1024")
    }
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    provenance = {
        "analysis_base": revision,
        "jax_version": belief[1, SEEDS[0]]["jax_version"],
        "numpy_version": belief[1, SEEDS[0]]["numpy_version"],
        "frozen_checkpoints": [
            {"variant": v, "seed": s, "checkpoints": data["checkpoints"]}
            for (v, s), data in belief.items()
        ],
        "optimal_value_references": {
            str(v): {k: val for k, val in hard[v, SEEDS[0]]["hard_reference"].items() if k != "matched"}
            for v in CONDITIONS
        },
        "sac_on_policy_prefixes": [
            {"variant": v, "seed": s, **{k: val for k, val in data["soft"].items() if not isinstance(val, dict)}}
            for (v, s), data in soft.items()
        ],
    }
    audit.archive.write(output / "summary.json", summary)
    audit.archive.write(output / "provenance.json", provenance)


if __name__ == "__main__":
    aggregate(ROOT / "artifacts", ROOT / "results")
