"""Run a named suite of counterfactual-twin steering variants on one r7 checkpoint."""

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

from experiments.rocksample_jax_2026_10 import belief_steering_cf as cf
from experiments.rocksample_jax_2026_10 import belief_steering_continuation as continuation

FIT_SEED = 41101
TUNE_SEED = 61307
DIRECTIONS = ((1, "Good→Bad"), (0, "Bad→Good"))


def explore1():
    variants = dict(intact={}, random=dict(random_policy=True),
                    twin_policy=dict(patch="full", patch_sites=(3,)))
    for s in range(3):
        variants[f"patch_belief_s{s}"] = dict(patch="full", patch_sites=(s,), gate_patch=(0, 1))
        variants[f"patch_evidence_s{s}"] = dict(patch="full", patch_sites=(s,), gate_patch=(1, 0))
    for encoder in ("twin_prob", "twinbelief_prob", "nuisance_prob", "plain_prob",
                    "twin_logit", "nuisance_logit"):
        space = encoder.split("_")[1]
        for s in range(cf.SITES):
            variants[f"clamp_{encoder}_s{s}"] = dict(encoder=encoder, space=space, sites=(s,))
    variants["clamp_twin_prob_s012"] = dict(encoder="twin_prob", sites=(0, 1, 2))
    variants["control_twin_prob_s0"] = dict(encoder="twin_prob", sites=(0,), control=True)
    return variants


def explore2():
    variants = {}
    for s in range(3):
        for rank in (1, 2, 4, 8):
            variants[f"pca{rank}_belief_s{s}"] = dict(patch="pca", rank=rank, patch_sites=(s,),
                                                      gate_patch=(0, 1))
            variants[f"pca{rank}_all_s{s}"] = dict(patch="pca", rank=rank, patch_sites=(s,))
        variants[f"dir_belief_s{s}"] = dict(patch="encoder", encoder="twin_prob", patch_sites=(s,),
                                            gate_patch=(0, 1))
        for strength in (.5, .75, 1.25, 1.5):
            variants[f"clamp_twin_prob_s{s}_l{strength}"] = dict(
                encoder="twin_prob", sites=(s,), strength=strength)
        for add in (.5, 1., 2.):
            variants[f"add_twin_prob_s{s}_a{add}"] = dict(
                encoder="twin_prob", sites=(s,), strength=0., add=add)
        variants[f"clamp_twin_prob_s{s}_belieftokens"] = dict(
            encoder="twin_prob", sites=(s,), gate_steer=(0, 1))
    return variants


SUITES = dict(explore1=explore1, explore2=explore2)


def jsonable(fitted):
    return dict(rows=fitted["rows"], r2=fitted["r2"], diagnostics=fitted["diagnostics"],
                encoder_norms={k: np.linalg.norm(v, axis=-1).tolist()
                               for k, v in fitted["encoder"].items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=int, required=True)
    parser.add_argument("--update", type=int, required=True, choices=(29, 316))
    parser.add_argument("--suite", required=True, choices=sorted(SUITES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-episodes", type=int, default=512)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=TUNE_SEED)
    parser.add_argument("--rocks", type=int, nargs="*")
    args = parser.parse_args()
    mapping = continuation.mappings()[args.agent]
    params, env, spec, provenance = continuation.load(mapping, args.update)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{args.suite}_agent{args.agent}_update{args.update}.json"
    state = json.loads(path.read_text()) if path.exists() else dict(
        suite=args.suite, agent=args.agent, update=args.update, run=mapping["source_run"],
        seed_index=mapping["seed"], checkpoint_sha256=provenance["sha256"],
        fit_seed=FIT_SEED, fit_episodes=args.fit_episodes, eval_seed=args.seed,
        episodes=args.episodes, revision=cf.bp.revision(cf.bp.ROOT), cells={})
    fitted = cf.fit(spec, env, params, args.fit_episodes, FIT_SEED)
    state["fit"] = jsonable(fitted)
    variants = SUITES[args.suite]()
    start = time.time()
    for rock in (args.rocks or range(env.k)):
        for good, direction in DIRECTIONS:
            for name, options in variants.items():
                key = f"{rock}|{direction}|{name}"
                if key in state["cells"]:
                    continue
                options = dict(options)
                random_policy = options.pop("random_policy", False)
                iv = cf.intervention(fitted, rock, good, **options)
                values = cf.rollout(spec, env, params, iv, episodes=args.episodes, seed=args.seed,
                                    rock=rock, good=good, random_policy=random_policy)
                state["cells"][key] = cf.summarize(values, rock)
            path.write_text(json.dumps(state, allow_nan=False) + "\n")
            print(f"rock {rock} {direction} done {time.time() - start:.0f}s", flush=True)
    print("complete", path, jax.devices(), flush=True)


if __name__ == "__main__":
    main()
