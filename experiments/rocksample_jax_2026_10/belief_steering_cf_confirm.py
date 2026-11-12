"""Held-out confirmation of the frozen counterfactual-twin belief-steering protocol.

The protocol (after block 1, rank 4, strength 1.25, one affine displacement map per
rock applied at every step until the rock is sampled) was selected on fit seed 41101
and rollout seed 61307. This driver refits every map on new episodes and evaluates
on new environment seeds, alongside baselines, controls, and single-factor sweeps.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.rocksample_jax_2026_10 import belief_steering_cf as cf
from experiments.rocksample_jax_2026_10 import belief_steering_continuation as continuation
from experiments.rocksample_jax_2026_10.belief_steering_cf_explore import DIRECTIONS, jsonable

FIT_SEED = 77003
EVAL_SEED = 88019
SITE, RANK, STRENGTH = 0, 4, 1.25
SITE_NAMES = ("block1", "block2", "block3", "final_norm")


def steer(site=SITE, rank=RANK, strength=STRENGTH, **options):
    return dict(reflect="pca_all", rank=rank, reflect_sites=(site,), reflect_scale=strength,
                **options)


def variants():
    out = dict(intact={}, random=dict(random_policy=True),
               twin_policy=dict(patch="full", patch_sites=(cf.SITES - 1,)),
               steer=steer(), control=steer(control=True),
               steer_evidence_tokens=steer(gate_steer=(1., 0.)),
               steer_belief_tokens=steer(gate_steer=(0., 1.)),
               steer_typed=steer(typed=True),
               interchange=dict(patch="pca_all", rank=RANK, patch_sites=(SITE,)),
               legacy_clamp=dict(encoder="nuisance_prob", sites=(SITE,)))
    for site in range(cf.SITES):
        if site != SITE:
            out[f"steer_{SITE_NAMES[site]}"] = steer(site=site)
            out[f"control_{SITE_NAMES[site]}"] = steer(site=site, control=True)
    for rank in (1, 2, 8, 128):
        out[f"steer_rank{rank}"] = steer(rank=rank)
    for strength in (0.5, 0.75, 1., 1.5):
        out[f"steer_strength{strength}"] = steer(strength=strength)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=int, required=True)
    parser.add_argument("--update", type=int, required=True, choices=(29, 316))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-episodes", type=int, default=512)
    parser.add_argument("--episodes", type=int, default=1024)
    args = parser.parse_args()
    mapping = continuation.mappings()[args.agent]
    params, env, spec, provenance = continuation.load(mapping, args.update)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"agent{args.agent}_update{args.update}.json"
    state = json.loads(path.read_text()) if path.exists() else dict(
        agent=args.agent, update=args.update, run=mapping["source_run"],
        seed_index=mapping["seed"], checkpoint_sha256=provenance["sha256"],
        protocol=dict(site=SITE_NAMES[SITE], rank=RANK, strength=STRENGTH),
        fit_seed=FIT_SEED, fit_episodes=args.fit_episodes, eval_seed=EVAL_SEED,
        episodes=args.episodes, revision=cf.bp.revision(cf.bp.ROOT), cells={})
    fitted = cf.fit(spec, env, params, args.fit_episodes, FIT_SEED)
    state["fit"] = jsonable(fitted)
    start = time.time()
    for rock in range(env.k):
        for good, direction in DIRECTIONS:
            for name, options in variants().items():
                key = f"{rock}|{direction}|{name}"
                if key in state["cells"]:
                    continue
                options = dict(options)
                random_policy = options.pop("random_policy", False)
                iv = cf.intervention(fitted, rock, good, **options)
                values = cf.rollout(spec, env, params, iv, episodes=args.episodes,
                                    seed=EVAL_SEED, rock=rock, good=good,
                                    random_policy=random_policy)
                state["cells"][key] = cf.summarize(values, rock)
            path.write_text(json.dumps(state, allow_nan=False) + "\n")
            print(f"rock {rock} {direction} done {time.time() - start:.0f}s", flush=True)
    print(f"complete {path}")


if __name__ == "__main__":
    main()
