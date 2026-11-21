"""Extract archived B2 controls without changing or rerunning their studies."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def read(revision, path):
    raw = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)
    return json.loads(raw), dict(revision=revision, path=path, sha256=hashlib.sha256(raw).hexdigest())


def extract(seed):
    result = {}
    for variant, test in ((2, "ladder"), (3, "ladder_full")):
        data, source = read("45517b13a67d746048a002f9db52ea55a654b219",
                            f"experiments/belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11/variant_{variant}_transformer/results/belief_steering_v2_seed{seed}_stable.json")
        rows = {r["condition"]: r for r in data["confirmation"] if r["site"] == 1}
        rho, mirror = (rows[k] for k in ("selected_rho_flip", "selected_mirror"))
        result[test] = dict(source=source, protocol=data["protocol"],
                            occupancy=rho["metrics"]["occupancy"], mirror_changed=mirror["metrics"]["action_change"],
                            strength=rho["strength"], action_change_reference="intact_same_executed_history")
    for cohort, tests in (("eps02_ctx32_2026_11", ("moving", "xor_eps02")), ("ctx32_2026_11", ("xor_v2",))):
        data, source = read("3504bfb72455a411b725ef368a18ff892760555f",
                            f"experiments/two_factor_coupled_moving_reward_steering_v2_jax_2026_11/results/{cohort}__reward_both__seed{seed}.json")
        for test in tests:
            task = "moving" if test == "moving" else "xor"
            confirmation = data["rewards"][task]["confirmation"]
            intact, reference = (confirmation[k]["total_reward"] for k in ("intact", "exact_belief_discounted_map"))
            result[test] = dict(source=source, tuning_episodes=data["tune_episodes"], confirmation_episodes=data["confirm_episodes"],
                                horizon=data["episode_horizon"], selection=data["selection"],
                                intact_reward=intact, map_reward=reference,
                                action_change_reference="local_current_write_only", representations={})
            for representation in ("marginal", "joint"):
                row = data["rewards"][task]["confirmation"]["comparisons"][f"discounted_{representation}_block_2"]
                result[test]["representations"][representation] = dict(strength=row["strength"], reward=row["total_reward"],
                                                                        gap_closed=(row["total_reward"]-intact)/(reference-intact),
                                                                        f1_changed=row["factor_action_change_fraction"][0])
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = extract(args.seed)
    output = Path(__file__).resolve().parent / "results" / f"historical_B2_seed{args.seed}.json"
    output.write_text(json.dumps(dict(seed=args.seed, site=1, tests=result), indent=2, sort_keys=True) + "\n")
    print(f"Extracted {len(result)} archived B2 test controls: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
