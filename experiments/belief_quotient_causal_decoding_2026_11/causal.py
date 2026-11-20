"""Action-fixed symmetry pairs and held-out latest-position interchange tests."""

import jax
import jax.numpy as jnp
import numpy as np

from experiments.belief_quotient_replay_edit_2026_11 import replay as lr
from experiments.reward_factors_replay_edit_2026_11 import replay as rr
from . import decoders as d


def fit(spec, params, process, ladder, variant, episodes, length, non_target, beta=1., frames=(2,)):
    targeted = (lambda p: p[:, jnp.asarray(rr.INVERSE)].sum(-2)) if not ladder and frames == (2,) else lambda p: p
    def pairs(seed, site):
        if ladder:
            original = lr.records(spec, process, params, seed, episodes, length)
            obs = original["obs"]
            alt, _ = lr.edit(obs, "swap", np.random.default_rng(seed), full=variant == 3)
            truth, counterfactual = (lr.filtered(process, x, variant)[..., :3] for x in (obs, alt))
            np.testing.assert_allclose(counterfactual, truth[..., lr.SWAP_TOKEN], atol=1e-6)
        else:
            original = rr.collect(spec, params, process, seed, episodes, length)
            obs = original["obs"]
            alt = np.concatenate([rr.mirror_inputs(obs, frame) for frame in frames])
            for frame, chunk in zip(frames, np.split(alt, len(frames))):
                np.testing.assert_array_equal(obs[..., 9:], chunk[..., 9:][..., rr.PERM[frame]])
            obs = np.concatenate([obs] * len(frames))
        z, h = lr.forward(spec, params, jnp.asarray(obs))
        zz, hh = lr.forward(spec, params, jnp.asarray(alt))
        warmup = 8 if ladder else 64
        if site == 3:
            orig = h[:, warmup:, site].reshape(-1, spec.width)
            donor = hh[:, warmup:, site].reshape(-1, spec.width)
            reference = jax.nn.softmax(zz[:, warmup:].reshape(-1, spec.n_actions) / 1.5)
            def policy(x):
                return jax.nn.softmax(lr.patch(spec, params, x, x, jnp.zeros(x.shape[:-1], bool), site) / 1.5)
        else:
            queries = np.arange(255, length, 256)
            windows = jnp.stack([h[:, q-128:q+1, site] for q in queries], 1).reshape(-1, 129, spec.width)
            orig = windows[:, -1]
            donor = hh[:, queries, site].reshape(-1, spec.width)
            reference = jax.nn.softmax(zz[:, queries].reshape(-1, spec.n_actions) / 1.5)
            def policy(x):
                changed = windows.at[:, -1].set(x)
                logits = lr.patch(spec, params, changed, changed, jnp.zeros(changed.shape[:-1], bool), site)
                return jax.nn.softmax(logits[:, -1] / 1.5)
        return orig, donor, reference, policy

    def score(data, u):
        orig, donor, ref, policy = data
        p = policy(orig)
        patch = policy(donor)
        written = policy(orig + (donor - orig) @ u @ u.T)
        effect = .5 * jnp.abs(p - ref).sum(-1).sum()
        def fidelity(q):
            return float(1 - .5 * jnp.abs(q - ref).sum(-1).sum() / effect) if float(effect) > 1e-6 else None
        return dict(kl=float(d.kl(ref, written).mean()), fidelity=fidelity(written),
                    reference_patch_fidelity=fidelity(patch), replay_tv=float(effect / len(orig)),
                    non_target_tv=float(.5 * jnp.abs(non_target(written) - non_target(p)).sum(-1).mean()))

    results, best = {}, None
    for site in (3, 1):
        train, selection, test = (pairs(seed, site) for seed in (42, 43, 44))
        orig, donor, reference, policy = train
        rows = {}
        for rank in (1, 2, 4, 8):
            u, loss = d.das(orig, donor, policy, reference, non_target, beta=beta, rank=rank, targeted=targeted)
            selected, held = score(selection, u), score(test, u)
            rows[str(rank)] = dict(train_loss=float(loss), selection=selected, held_out=held)
            if site == 3 and (best is None or selected["kl"] < best[0]):
                best = selected["kl"], rank, u
        results[str(site)] = rows
    results["selected_rank_block4"] = best[1]
    return best[2], results
