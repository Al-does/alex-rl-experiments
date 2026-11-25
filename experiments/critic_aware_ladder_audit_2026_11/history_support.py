"""Locate the Trivial Quotient seed-46 transfer error on rare-action histories."""

import os

import numpy as np

from .audit import ROOT, archive, fit_affine_probe, jax, load, targets, window


def main():
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    seed = 46
    spec, process, _, length, _, _, trunks, _ = load(1, seed)
    _, rollout, replay = archive.runtime()
    keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed), 2))
    result = {"variant": 1, "seed": seed, "trunk": "reinforce", "window": "64:600",
              "action_order": ["Noop", "Tilt X", "Tilt Y"]}
    lookback = spec.n_layers * spec.context_len
    with jax.default_matmul_precision("highest"):
        for donor in ("reinforce", "sac_autoalpha"):
            streams = [rollout(spec, process, trunks[donor], key, n, length)
                       for key, n in zip(keys, (32, 40))]
            beliefs = [targets(window(r["belief"], 64, 600, None)) for r in streams]
            features = [window(replay(spec, trunks["reinforce"], r["obs"], "embedding"),
                               64, 600, None) for r in streams]
            weight, bias = fit_affine_probe(features[0], beliefs[0], ridge=1e-6)
            error = np.square(features[1] @ weight + bias - beliefs[1])
            recorded = [window(r["action"], 64, 600, None) for r in streams]
            counts = [np.bincount(actions, minlength=spec.n_actions) for actions in recorded]
            # A preceding action is part of each input, including the final position.
            observations = np.asarray(streams[1]["obs"])
            action_inputs = observations[..., -spec.n_actions:]
            non_tilt_x = action_inputs[..., 0].astype(bool) | action_inputs[..., 2].astype(bool)
            recent = non_tilt_x.copy()
            for lag in range(1, lookback + 1):
                recent[:, lag:] |= non_tilt_x[:, :-lag]
            selected = window(recent, 64, 600, None)
            result[donor] = {
                "fit_action_counts": counts[0].tolist(),
                "test_action_counts": counts[1].tolist(),
                "test_rows_with_recent_non_tilt_x_input": int(selected.sum()),
                "test_rows": len(selected),
                "sse_share_in_recent_non_tilt_x_rows": (
                    error[selected].sum(0) / error.sum(0)
                ).tolist(),
            }
    archive.write(ROOT / "results" / "trivial_seed46_history_support.json", result)


if __name__ == "__main__":
    main()
