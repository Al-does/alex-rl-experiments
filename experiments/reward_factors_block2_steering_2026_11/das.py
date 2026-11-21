"""Real-cache DAS fitting, with independent site-specific rank selection."""

import jax
import jax.numpy as jnp
import numpy as np

from experiments.belief_quotient_block2_steering_2026_11 import continuation as c
from experiments.belief_quotient_causal_decoding_2026_11 import decoders as d
from experiments.belief_quotient_replay_edit_2026_11 import replay as lr
from experiments.reward_factors_replay_edit_2026_11 import replay as rr


def fit(spec, params, process, ladder, variant, episodes, length, non_target, site, beta=1., frames=(2,)):
    targeted = (lambda p: p[:, jnp.asarray(rr.INVERSE)].sum(-2)) if not ladder and frames == (2,) else lambda p: p

    def pairs(seed):
        if ladder:
            original = lr.records(spec, process, params, seed, episodes, length)
            obs = np.asarray(original["obs"])
            alt, _ = lr.edit(obs, "swap", np.random.default_rng(seed), full=variant == 3)
        else:
            original = rr.collect(spec, params, process, seed, episodes, length)
            obs = np.asarray(original["obs"])
            alt = np.concatenate([rr.mirror_inputs(obs, frame) for frame in frames])
            obs = np.concatenate([obs] * len(frames))
        _, sites = lr.forward(spec, params, jnp.asarray(obs))
        ref_logits, donors = lr.forward(spec, params, jnp.asarray(alt))
        warmup = 8 if ladder else 64
        indices = np.linspace(0, len(obs) * (length-warmup)-1,
                              min(512, len(obs) * (length-warmup)), dtype=int)
        ep, times = np.divmod(indices, length-warmup)
        times += warmup
        caches = c.contexts(spec, params, sites, ep, times, site)
        h, donor = sites[ep, times, site], donors[ep, times, site]
        ref = jax.nn.softmax(ref_logits[ep, times] / 1.5)
        policy = lambda a: jax.nn.softmax(c.batch_logits(spec, params, a, caches, jnp.asarray(times), site) / 1.5)
        return h, donor, ref, policy

    train, selection, held = (pairs(seed) for seed in (42, 43, 44))

    def score(data, u):
        h, donor, ref, policy = data
        p, full = policy(h), policy(donor)
        written = policy(h + (donor-h) @ u @ u.T)
        effect = float((.5 * jnp.abs(p-ref).sum(-1)).sum())
        fidelity = lambda x: float(1 - (.5*jnp.abs(x-ref).sum(-1)).sum()/effect) if effect > 1e-6 else None
        return dict(kl=float(d.kl(ref, written).mean()), fidelity=fidelity(written),
                    reference_patch_fidelity=fidelity(full),
                    non_target_tv=float((.5*jnp.abs(non_target(written)-non_target(p)).sum(-1)).mean()))

    best, rows = None, {}
    h, donor, ref, policy = train
    for rank in (1, 2, 4, 8):
        u, loss = d.das(h, donor, policy, ref, non_target, beta=beta, rank=rank, targeted=targeted)
        selected = score(selection, u)
        rows[str(rank)] = dict(train_loss=float(loss), selection=selected, held_out=score(held, u))
        if best is None or selected["kl"] < best[0]:
            best = selected["kl"], rank, u
    return best[2], dict(site=site, rows=len(h), ranks=rows, selected_rank=best[1], real_cache=True)
