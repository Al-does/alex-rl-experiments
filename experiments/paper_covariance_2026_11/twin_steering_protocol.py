# RockSample manuscript protocol extracted from alex-rl-experiments@372b3f886ad4b69e1fcdfb15f6d9350ec4a90568.
import numpy as np

from . import twin_steering as cf

FIT_SEED = 77003
EVAL_SEED = 88019
SITE, RANK, STRENGTH = 0, 4, 1.25
SITE_NAMES = ("block1", "block2", "block3", "final_norm")
DIRECTIONS = ((1, "Good→Bad"), (0, "Bad→Good"))


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


def jsonable(fitted):
    return dict(rows=fitted["rows"], r2=fitted["r2"], diagnostics=fitted["diagnostics"],
                encoder_norms={k: np.linalg.norm(v, axis=-1).tolist()
                               for k, v in fitted["encoder"].items()})
