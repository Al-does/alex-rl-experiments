"""One-step residual pulses on intact open-loop token/executed-action histories."""

from __future__ import annotations

import time

import numpy as np
import torch

from envs.hmm import HMMEnv
from experiments.mess3_belief_geometry_2026_07.probe import _initial_state
from experiments.mess3_reward_state_action_symmetry_cycle_7 import belief_steering_followup as steering
from experiments.mess3_reward_state_action_symmetry_cycle_7.belief_steering_prediction import (
    EMISSION, target, transitions_for,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.coarse_probe_analysis import coarse_filter_spec


def clone_state(state):
    return {k: v.clone() for k, v in state.items()}


def read_coordinates(activation, mapping):
    decoded = activation.cpu().numpy().astype(float) @ mapping.decoder + mapping.bias
    return np.column_stack((1-decoded.sum(1), decoded[:, 0]-decoded[:, 1]))


def filter_step(b, previous_actions, tokens, transitions, emission):
    prior = np.einsum("bs,bsj->bj", b, transitions[previous_actions])
    posterior = prior*emission[:, tokens].T
    return posterior/posterior.sum(1, keepdims=True)


@torch.inference_mode()
def branch_trace(module, maps, observations, state, *, site, mode, strength=1.):
    """observations[0] is the pulse input; state is the intact pre-input cache."""
    if site not in (0, 1, 2) or len(observations) < 1:
        raise ValueError("pulse requires an early block and nonempty observations")
    current = clone_state(state)
    readings, probabilities = [], []
    with steering.ResidualSites(module) as sites:
        sites.mapping, sites.mode, sites.strength = maps[site], mode, strength
        for k, obs in enumerate(observations):
            sites.site = site if k == 0 else None
            h, current = module.encode_step(torch.as_tensor(obs), current)
            readings.append(read_coordinates(sites.activations[3], maps[3]))
            probabilities.append(module.action_distribution_inputs(h).softmax(-1).numpy())
    return np.stack(readings), np.stack(probabilities)


@torch.inference_mode()
def intact_history(module, mapping, variant, episodes, horizon, pulse_times, seed=3042):
    envs = [HMMEnv(steering.config(horizon, variant)) for _ in range(episodes)]
    try:
        seeds = np.random.default_rng(seed).integers(2**31, size=episodes)
        reset = [env.reset(seed=int(s)) for env, s in zip(envs, seeds)]
        obs, infos = map(list, zip(*reset))
        state = _initial_state(module, episodes, torch.device("cpu"))
        rows, beliefs, readings, probabilities, coarse_rows = [], [], [], [], []
        snapshots = {}
        if variant == 2:
            initial, coarse_emission, coarse_transitions = coarse_filter_spec()
            coarse_transition_array = np.stack([coarse_transitions[a] for a in range(3)])
            coarse = np.repeat(initial[None], episodes, axis=0)
        with steering.ResidualSites(module) as sites:
            for t in range(horizon):
                observation = np.stack(obs)
                if t in pulse_times:
                    snapshots[t] = clone_state(state)
                rows.append(observation)
                beliefs.append(np.stack([info["belief_current"] for info in infos]))
                if variant == 2:
                    token = (observation[:, :3].argmax(1) == 2).astype(int)
                    if t == 0:
                        coarse *= coarse_emission[:, token].T
                        coarse /= coarse.sum(1, keepdims=True)
                    else:
                        coarse = filter_step(coarse, observation[:, 3:].argmax(1), token,
                                             coarse_transition_array, coarse_emission)
                    coarse_rows.append(coarse.copy())
                h, state = module.encode_step(torch.as_tensor(observation), state)
                readings.append(read_coordinates(sites.activations[3], mapping))
                p = module.action_distribution_inputs(h).softmax(-1).numpy()
                probabilities.append(p)
                actions = p.argmax(1)
                outcomes = [env.step(int(a)) for env, a in zip(envs, actions)]
                if any((v[2] or v[3]) != (t == horizon-1) for v in outcomes):
                    raise ValueError("intact pulse rollout horizon mismatch")
                obs, infos = [v[0] for v in outcomes], [v[4] for v in outcomes]
        return {"observations": np.stack(rows), "beliefs": np.stack(beliefs),
                "readings": np.stack(readings), "probabilities": np.stack(probabilities),
                "snapshots": snapshots,
                "coarse": np.stack(coarse_rows) if coarse_rows else None,
                "episode_seeds": seeds.tolist()}
    finally:
        for env in envs:
            env.close()


def exact_trace(history, variant, t0, lag, mode):
    """Restart from the posterior at t0; do not re-emit its already-seen token."""
    observations = history["observations"][t0:t0+lag+1]
    base = history["beliefs"][t0].copy()
    branch = target(base, mode)
    transitions = transitions_for(variant)
    full = []
    if variant == 2:
        _, emission, matrices = coarse_filter_spec()
        matrices = np.stack([matrices[a] for a in range(3)])
        coarse_base = history["coarse"][t0].copy()
        b = np.column_stack((coarse_base[:, 0]/2, coarse_base[:, 0]/2, coarse_base[:, 1]))
        perturbed = target(b, mode)
        coarse_branch = np.column_stack((perturbed[:, :2].sum(1), perturbed[:, 2]))
        coarse = []
    for k, obs in enumerate(observations):
        if k:
            actions, tokens = obs[:, 3:].argmax(1), obs[:, :3].argmax(1)
            base = filter_step(base, actions, tokens, transitions, EMISSION)
            branch = filter_step(branch, actions, tokens, transitions, EMISSION)
            if variant == 2:
                coarse_base = filter_step(coarse_base, actions, (tokens == 2).astype(int), matrices, emission)
                coarse_branch = filter_step(coarse_branch, actions, (tokens == 2).astype(int), matrices, emission)
        difference = branch-base
        full.append(np.column_stack((difference[:, 2], difference[:, 0]-difference[:, 1])))
        if variant == 2:
            coarse.append(coarse_branch[:, 1]-coarse_base[:, 1])
    return np.stack(full), np.stack(coarse) if variant == 2 else None


def decay_rate(curve):
    """Fit log(mean absolute effect) = intercept + slope*k over lags 1..48.

    Values below max(1e-12, peak*1e-8) are excluded. Rate is exp(slope):
    0 means immediate loss, 1 no exponential decay, >1 growth. Zero signals
    have no fitted rate. Nonexponential curves need not have a good R².
    """
    values = np.asarray(curve, dtype=float)
    threshold = max(1e-12, float(values.max())*1e-8)
    valid = (values > threshold) & (np.arange(len(values)) > 0)
    if valid.sum() < 3:
        return {"per_step_multiplier": None, "log_slope": None, "r2": None,
                "n_lags": int(valid.sum())}
    x, y = np.arange(len(values))[valid], np.log(values[valid])
    slope, intercept = np.polyfit(x, y, 1)
    residual = np.sum((y-(intercept+slope*x))**2)
    total = np.sum((y-y.mean())**2)
    return {"per_step_multiplier": float(np.exp(slope)), "log_slope": float(slope),
            "r2": float(1-residual/total) if total else None, "n_lags": int(valid.sum()),
            "first_lag": int(x[0]), "last_lag": int(x[-1])}


def summarize_trace(values):
    # Input: pulse x lag x episode. Average repeated pulses within an episode.
    signed = values.mean(0)
    absolute = np.abs(values).mean(0)
    n = signed.shape[1]
    return {"mean": signed.mean(1).tolist(),
            "episode_se": (signed.std(1, ddof=1)/np.sqrt(n)).tolist(),
            "mean_absolute": absolute.mean(1).tolist(),
            "absolute_episode_se": (absolute.std(1, ddof=1)/np.sqrt(n)).tolist(),
            "decay": decay_rate(absolute.mean(1))}


def run_pulses(module, maps, variant, *, smoke=False):
    start = time.monotonic()
    episodes, horizon, times, lag = (2, 64, (16,), 8) if smoke else (32, 1024, (256, 512, 768), 48)
    history = intact_history(module, maps[3], variant, episodes, horizon, times)
    results = {}
    for mode in ("mirror", "rho_flip"):
        predictions = [exact_trace(history, variant, t0, lag, mode) for t0 in times]
        full = np.stack([p[0] for p in predictions])
        for site in range(3):
            measured, changes, tv = [], [], []
            for t0 in times:
                obs = history["observations"][t0:t0+lag+1]
                readings, probabilities = branch_trace(module, maps, obs, history["snapshots"][t0], site=site, mode=mode)
                base_p = history["probabilities"][t0:t0+lag+1]
                measured.append(readings-history["readings"][t0:t0+lag+1])
                changes.append((probabilities.argmax(2) != base_p.argmax(2)).astype(float))
                tv.append(np.abs(probabilities-base_p).sum(2)/2)
            measured = np.stack(measured)
            result = {"measured": {c: summarize_trace(measured[..., i]) for i, c in enumerate(("rho", "delta"))},
                      "full_filter": {c: summarize_trace(full[..., i]) for i, c in enumerate(("rho", "delta"))},
                      "action_change": summarize_trace(np.stack(changes)), "action_tv": summarize_trace(np.stack(tv))}
            if variant == 2:
                result["quotient_filter"] = {"rho": summarize_trace(np.stack([p[1] for p in predictions]))}
            results[f"{mode}/block_{site+1}"] = result
    return {"protocol": {"eval_seed": 3042, "episodes": episodes, "horizon": horizon,
                         "pulse_times": times, "max_lag": lag, "strength": 1,
                         "input": "intact tokens and intact executed preceding actions; no branch actions executed",
                         "readout": "block-4 affine decoder; delta=b1-b2; rho=b3",
                         "prediction": "posterior restart at t0; later filters consume actual preceding actions",
                         "rate_fit": "log mean absolute effect, lags 1..K; exclude below max(1e-12, peak*1e-8); multiplier=exp(slope)"},
            "episode_seeds": history["episode_seeds"], "results": results,
            "elapsed_seconds": time.monotonic()-start}


def aggregate_pulses(reports):
    results = {}
    for condition, first in reports[0]["results"].items():
        result = {}
        for source, curves in first.items():
            if source in ("action_change", "action_tv"):
                curves = {"value": curves}
            combined = {}
            for coordinate in curves:
                individual = [r["results"][condition][source] if source in ("action_change", "action_tv")
                              else r["results"][condition][source][coordinate] for r in reports]
                item = {}
                for key in ("mean", "mean_absolute"):
                    a = np.asarray([v[key] for v in individual])
                    item[key] = a.mean(0).tolist()
                    item[key+"_training_seed_se"] = (a.std(0, ddof=1)/np.sqrt(len(a))).tolist()
                item["decay"] = decay_rate(item["mean_absolute"])
                item["per_seed_decay"] = {str(r["training_seed"]): v["decay"] for r, v in zip(reports, individual)}
                combined[coordinate] = item
            result[source] = combined["value"] if source in ("action_change", "action_tv") else combined
        results[condition] = result
    return {"agent": reports[0]["agent"], "training_seeds": [r["training_seed"] for r in reports],
            "protocol": reports[0]["protocol"], "uncertainty": "SE across five training-seed means",
            "per_seed_provenance": {str(r["training_seed"]): r["provenance"] for r in reports},
            "results": results}


def pulse_md(report):
    lines = [f"# {report['agent'].upper()} single-step pulse and decay", "",
             "Open-loop intact histories; no branch action enters the environment or its action features.",
             "Coordinate differences are decoded at block 4. Signed and absolute lag curves are in JSON.",
             "Rate is exp(slope) from log mean absolute effect; <1 decay, >1 growth; null means insufficient nonzero lags.",
             "Fits exclude values below max(1e-12, peak*1e-8). This descriptive exponential fit is not a dynamical model.", "",
             "| Pulse/site | Coordinate | Measured rate | Full filter rate | Quotient filter rate |",
             "|---|---|---:|---:|---:|"]
    for condition, result in report["results"].items():
        for coord in ("rho", "delta"):
            sources = [result["measured"][coord], result["full_filter"][coord], result.get("quotient_filter", {}).get(coord)]
            rates = [v["decay"]["per_step_multiplier"] if v else None for v in sources]
            lines.append(f"| {condition} | {coord} | "+" | ".join(f"{v:.6g}" if v is not None else "—" for v in rates)+" |")
    return "\n".join(lines)
