"""Is there an intention between belief and action in the r7 RockSample policies?

Three measurements on one agent-checkpoint, following the probe / plan / intervene
outline of Bush et al. (2025):

``probes``   linear readout of the next rock the episode will sample, at every site,
             against linear readouts from observables and the exact belief;
``decode``   the same readout applied to belief-steered rollouts;
``onset``    steering that starts mid-episode. Rollouts share every random draw with
             the intact rollout, so they are identical to it until the onset step.

Onset modes: ``steer`` is the frozen twin-free protocol; ``current_token`` replaces
the block-1 output of every token from the onset on with the twin's; ``all_tokens``
replaces the final activation, i.e. acts as if every cached token had been rewritten.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from experiments.rocksample_jax_2026_10 import belief_probe as bp
from experiments.rocksample_jax_2026_10 import belief_steering as bs
from experiments.rocksample_jax_2026_10 import belief_steering_cf as cf
from experiments.rocksample_jax_2026_10 import belief_steering_cf_confirm as confirm
from experiments.rocksample_jax_2026_10 import belief_steering_continuation as continuation
from experiments.rocksample_jax_2026_10.belief_steering_cf_explore import DIRECTIONS

TEST_SEED = 79031
EVAL_SEED = 90217
RELATIVE = (0, 1, 2, 3, 4, 6, 8)
FIXED = (2, 4, 8, 12, 16, 24)
TRACE_LEAD = 4
MODES = dict(steer=confirm.steer(),
             current_token=dict(patch="full", patch_sites=(confirm.SITE,)),
             all_tokens=dict(patch="full", patch_sites=(cf.SITES - 1,)))


def next_sample(picked):
    """Next rock sampled at or after each step, and how many steps ahead; ``k`` if none."""
    episodes, horizon, k = picked.shape
    rock, lead = np.full((episodes, horizon), k), np.zeros((episodes, horizon), int)
    current, ahead = np.full(episodes, k), np.zeros(episodes, int)
    for t in range(horizon - 1, -1, -1):
        hit = picked[:, t].any(1)
        current = np.where(hit, picked[:, t].argmax(1), current)
        ahead = np.where(hit, 0, ahead + 1)
        rock[:, t], lead[:, t] = current, ahead
    return rock, lead


def first(mask):
    """First true step along axis 1, or the axis length when there is none."""
    return np.where(mask.any(1), mask.argmax(1), mask.shape[1])


def classifier(x, labels, classes):
    return bs.affine(x, np.eye(classes)[labels])


def predict(model, x):
    return (x @ model[0] + model[1]).argmax(1)


def score(pred, labels, classes):
    f1 = []
    for c in range(classes):
        tp = ((pred == c) & (labels == c)).sum()
        if (labels == c).any():
            f1.append(2 * tp / max((pred == c).sum() + (labels == c).sum(), 1))
    return dict(accuracy=float((pred == labels).mean()), macro_f1=float(np.mean(f1)),
                rows=int(len(labels)))


def dataset(spec, env, params, episodes, seed):
    rows = bp.collect(spec, env, params, params, episodes=episodes, key_seed=seed)
    data = bp.targets(rows, env)
    valid = rows["valid"]
    position = rows["position"]
    picked = ((rows["action"] == 4)[:, :, None] & valid[:, :, None]
              & np.all(position[:, :, None] == np.asarray(env.rocks), axis=-1))
    rock, lead = next_sample(picked)
    sampled = cf.sampled_history(rows, env)[valid]
    cell = np.eye(env.n**2)[(position[:, :, 0] + env.n * position[:, :, 1])[valid]]
    state = np.column_stack((data.marginals * ~sampled, sampled, np.ones(len(sampled))))
    observables = bs.nuisance_features(rows, env)
    features = dict(observables=observables,
                    observables_belief=np.column_stack((observables, data.marginals)),
                    position_x_belief=(cell[:, :, None] * state[:, None]).reshape(len(cell), -1))
    h = np.asarray(bs.replay(spec, params, rows["obs"]))[valid].astype(np.float64)
    for s in range(cf.SITES):
        features[confirm.SITE_NAMES[s]] = h[:, s]
    return dict(features=features, label=rock[valid], lead=lead[valid], rows=rows, data=data)


def resolution(train, env):
    """Per-rock belief statistics of the intact policy, for the steerability predictors."""
    rows, data = train["rows"], train["data"]
    quality = rows["qualities"][:, 0]
    out = []
    for rock in range(env.k):
        rel = data.relevant[:, rock]
        last = np.full(len(quality), np.nan)
        last[data.groups[rel]] = data.marginals[rel, rock]  # later steps overwrite earlier
        seen = ~np.isnan(last)
        m = data.marginals[rel, rock]
        check = (data.actions == 5 + rock) & rel
        distance = np.linalg.norm(data.positions[check] - np.asarray(env.rocks[rock]), axis=1)
        out.append(dict(
            rock=rock, checks=float(data.checks[:, rock].mean()),
            check_distance=float(distance.mean()) if len(distance) else None,
            check_accuracy=float((1 + 2 ** (-distance / env.half_efficiency_distance)).mean() / 2)
            if len(distance) else None,
            certainty=float(np.abs(2 * m - 1).mean()),
            last_good=float(last[seen & quality[:, rock]].mean()),
            last_bad=float(last[seen & ~quality[:, rock]].mean()),
            last_certainty=float(np.abs(2 * last[seen] - 1).mean()),
            sampled_good=float(data.good_samples[:, rock].sum() / max(quality[:, rock].sum(), 1)),
            relevant_steps=float(rel.sum() / len(quality))))
    return out


def probes(train, test, classes):
    out, models = {}, {}
    ahead = test["lead"] >= 2
    for name, x in train["features"].items():
        models[name] = classifier(x, train["label"], classes)
        pred = predict(models[name], test["features"][name])
        out[name] = dict(all=score(pred, test["label"], classes),
                         ahead=score(pred[ahead], test["label"][ahead], classes))
    return out, models


def run(spec, env, params, iv, *, rock, good, episodes, onset=None, trace=False):
    onset = None if onset is None else jnp.asarray(onset, jnp.int32)
    values = cf.rollout(spec, env, params, iv, episodes=episodes, seed=EVAL_SEED, rock=rock,
                        good=good, onset=onset, trace=trace)
    keys = ("actions", "valid_steps", "picked_steps", "rover_steps") + (("site_steps",) if trace else ())
    return {k: np.moveaxis(np.asarray(values[k]), 0, 1) for k in keys}


def outcome(base, steered, eligible, onset, rock, env):
    """Target sampling and timing among episodes that reach the onset with the rock unsampled."""
    horizon = base["actions"].shape[1]
    steps = np.arange(horizon)
    hit = first(steered["picked_steps"][:, :, rock])
    differ = ((steered["actions"] != base["actions"]) & base["valid_steps"]
              & steered["valid_steps"] & (steps >= onset[:, None]))
    diverge = first(differ)
    at_onset = steered["rover_steps"][np.arange(len(onset)), np.minimum(onset, horizon - 1)]
    distance = np.abs(at_onset - np.asarray(env.rocks[rock])).sum(1)
    got = eligible & (hit < horizon)
    moved = eligible & (diverge < horizon)
    n = int(eligible.sum())
    mean = lambda x, mask: float(x[mask].mean()) if mask.any() else None
    return dict(eligible=n, sampled=int(got.sum()),
                intact_sampled=int((eligible & (first(base["picked_steps"][:, :, rock]) < horizon)).sum()),
                sampled_ci95=cf.wilson(int(got.sum()), n) if n else None,
                diverged=int(moved.sum()), diverge_delay=mean(diverge - onset, moved),
                diverge_at_onset=mean(diverge == onset, eligible),
                sample_delay=mean(hit - onset, got), distance=mean(distance, got))


def decode(models, run_, rock, k):
    """Plan readout on a traced rollout, over steps where the target is still unsampled."""
    label, _ = next_sample(run_["picked_steps"])
    live = run_["valid_steps"] & ~(np.cumsum(run_["picked_steps"][:, :, rock], axis=1)
                                   - run_["picked_steps"][:, :, rock]).astype(bool)
    out, pred_sites = {}, {}
    for s in range(cf.SITES):
        name = confirm.SITE_NAMES[s]
        pred = predict(models[name], run_["site_steps"][:, :, s].reshape(-1, run_["site_steps"].shape[-1]))
        pred = pred_sites[name] = pred.reshape(live.shape)
        out[name] = dict(accuracy=float((pred == label)[live].mean()),
                         says_target=float((pred == rock)[live].mean()))
    out["is_target"] = float((label == rock)[live].mean())
    out["steps"] = int(live.sum())
    return out, pred_sites


def cell(spec, env, params, fitted, models, rock, good, episodes, trace_episodes):
    horizon = env.episode_length
    zero = cf.intervention(fitted, rock, good)
    base = run(spec, env, params, zero, rock=rock, good=good, episodes=episodes)
    target_step = first(base["picked_steps"][:, :, rock])
    out = dict(intact_sampled=float((target_step < horizon).mean()), relative={}, fixed={})
    ivs = {name: cf.intervention(fitted, rock, good, **options) for name, options in MODES.items()}
    for name, iv in ivs.items():
        out["fixed"][name] = {}
        for t0 in FIXED:
            onset = np.full(episodes, t0)
            eligible = base["valid_steps"][:, t0] & (target_step >= t0)
            steered = run(spec, env, params, iv, rock=rock, good=good, episodes=episodes, onset=onset)
            out["fixed"][name][t0] = outcome(base, steered, eligible, onset, rock, env)
        if not good:
            continue
        out["relative"][name] = {}
        for lead in RELATIVE:
            eligible = (target_step < horizon) & (target_step >= lead)
            onset = np.where(eligible, target_step - lead, horizon)
            steered = run(spec, env, params, iv, rock=rock, good=good, episodes=episodes, onset=onset)
            out["relative"][name][lead] = outcome(base, steered, eligible, onset, rock, env)
    # Plan readout: intact, steered from the first step, and (Good target) steered mid-route.
    traced = run(spec, env, params, zero, rock=rock, good=good, episodes=trace_episodes, trace=True)
    out["decode_intact"], intact_pred = decode(models, traced, rock, env.k)
    full = run(spec, env, params, ivs["steer"], rock=rock, good=good, episodes=trace_episodes,
               trace=True)
    out["decode_steer"], _ = decode(models, full, rock, env.k)
    if good:
        step = first(traced["picked_steps"][:, :, rock])
        eligible = (step < horizon) & (step >= TRACE_LEAD)
        onset = np.where(eligible, step - TRACE_LEAD, horizon)
        late = run(spec, env, params, ivs["steer"], rock=rock, good=good, episodes=trace_episodes,
                   onset=onset, trace=True)
        _, late_pred = decode(models, late, rock, env.k)
        steps = np.arange(horizon)
        window = (steps >= onset[:, None]) & late["valid_steps"]
        act = first((late["actions"] != traced["actions"]) & window & traced["valid_steps"])
        acted = eligible & (act < horizon)
        timing = dict(eligible=int(eligible.sum()), acted=int(acted.sum()),
                      action_delay=float((act - onset)[acted].mean()))
        before = (steps >= onset[:, None]) & (steps < step[:, None]) & eligible[:, None]
        after = window & eligible[:, None]
        for name in confirm.SITE_NAMES:
            # First step at which the readout departs from the intact rollout's readout.
            plan = first((late_pred[name] != intact_pred[name]) & window)
            both = acted & (plan < horizon)
            timing[name] = dict(
                changed=int((eligible & (plan < horizon)).sum()),
                plan_delay=float((plan - onset)[eligible & (plan < horizon)].mean()),
                plan_first=float((plan < act)[both].mean()), same_step=float((plan == act)[both].mean()),
                action_first=float((plan > act)[both].mean()),
                intact_says_target=float((intact_pred[name] == rock)[before].mean()),
                steered_says_target=float((late_pred[name] == rock)[after].mean()))
        out["timing"] = timing
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=int, required=True)
    parser.add_argument("--update", type=int, required=True, choices=(29, 316))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-episodes", type=int, default=512)
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--trace-episodes", type=int, default=256)
    args = parser.parse_args()
    mapping = continuation.mappings()[args.agent]
    params, env, spec, provenance = continuation.load(mapping, args.update)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"agent{args.agent}_update{args.update}.json"
    start = time.time()
    train = dataset(spec, env, params, args.fit_episodes, confirm.FIT_SEED)
    test = dataset(spec, env, params, args.fit_episodes, TEST_SEED)
    probe_scores, models = probes(train, test, env.k + 1)
    state = dict(agent=args.agent, update=args.update, run=mapping["source_run"],
                 seed_index=mapping["seed"], checkpoint_sha256=provenance["sha256"],
                 protocol=dict(site=confirm.SITE_NAMES[confirm.SITE], rank=confirm.RANK,
                               strength=confirm.STRENGTH),
                 fit_seed=confirm.FIT_SEED, test_seed=TEST_SEED, eval_seed=EVAL_SEED,
                 fit_episodes=args.fit_episodes, episodes=args.episodes,
                 trace_episodes=args.trace_episodes, revision=bp.revision(bp.ROOT),
                 probes=probe_scores, resolution=resolution(train, env),
                 label_share=(np.bincount(test["label"], minlength=env.k + 1) / len(test["label"])).tolist(),
                 cells={})
    print(f"probes done {time.time() - start:.0f}s", flush=True)
    fitted = cf.fit(spec, env, params, args.fit_episodes, confirm.FIT_SEED, ranks=(1, 2, 4))
    for rock in range(env.k):
        for good, direction in DIRECTIONS:
            state["cells"][f"{rock}|{direction}"] = cell(
                spec, env, params, fitted, models, rock, good, args.episodes, args.trace_episodes)
            path.write_text(json.dumps(state, allow_nan=False) + "\n")
            print(f"rock {rock} {direction} done {time.time() - start:.0f}s", flush=True)
    print(f"complete {path}")


if __name__ == "__main__":
    main()
