"""Descriptive rare-action history diagnostic; does not change primary scores."""

import os

from .analyze import FINAL, REPO, ROOT, discover, flatten, read, refit, restore, runtime, score, write

import numpy as np


def main():
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    from experiments.belief_quotient_history_jax_2026_11.models import ModelSpec
    from experiments.belief_quotient_history_jax_2026_11.env import make_process

    run = next(r for r in discover() if r['algorithm'] == 'sac_autoalpha' and r['variant'] == 1 and r['seed'] == 44)
    jax, rollout, replay = runtime()
    recipe = read(REPO / run['results_path'] / 'resolved_recipe.json')
    spec, process = ModelSpec(**recipe['model']), make_process(**recipe['process'])
    params, _ = restore(run, FINAL, spec)
    init, _ = restore(run, 0, spec)
    counts = (60000, 80000)
    with jax.default_matmul_precision('highest'):
        keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(44), 2))
        raw = [rollout(spec, process, params, k, -(-n // 960), 1024) for k, n in zip(keys, counts)]
        h = [flatten(replay(spec, init, r['obs'], 'embedding'), 64, n) for r, n in zip(raw, counts)]
        b = [flatten(r['belief'], 64, n) for r, n in zip(raw, counts)]
        classes = [flatten(r['action'], 64, n) for r, n in zip(raw, counts)]
        pred = refit(h[0], b[0], h[1], *classes)
    valid = np.isfinite(pred).all(1)
    masks = []
    for r, n in zip(raw, counts):
        obs = np.asarray(r['obs'])
        has_tilt_y = obs[:, :, 5] > 0
        mask = has_tilt_y.copy()
        for lag in range(1, 129):
            mask[:, lag:] |= has_tilt_y[:, :-lag]
        masks.append(flatten(mask, 64, n))
    error = ((pred-b[1])**2).sum(1)
    result = dict(algorithm='sac_autoalpha', variant=1, seed=44,
        fit_counts=np.bincount(classes[0], minlength=3).tolist(),
        test_counts=np.bincount(classes[1], minlength=3).tolist(),
        excluded_rows=int((~valid).sum()),
        rows_with_tilt_y_previous_action_in_current_and_128_preceding_positions=[int(m.sum()) for m in masks],
        fraction_matched_init_b_sse_in_these_valid_test_rows=float(error[valid & masks[1]].sum()/error[valid].sum()),
        matched_init_within_b_r2_all_valid=score(b[1][valid], pred[valid], classes[1][valid])['r2'],
        matched_init_within_b_r2_excluding_these_history_rows_diagnostic_only=score(
            b[1][valid & ~masks[1]], pred[valid & ~masks[1]], classes[1][valid & ~masks[1]])['r2'],
        diagnostic_only='No primary rows or scores modified; describes association with rare Tilt Y histories, not a causal intervention.')
    write(ROOT / 'results/trivial_seed44_diagnostic.json', result)
    print(result)


if __name__ == '__main__':
    main()
