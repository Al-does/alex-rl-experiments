"""Fresh pulse reproduction and controlled audit of the Reward-State GRU128 rebound."""
import argparse
from functools import partial
import json
from pathlib import Path

from .analyze import HERE, STUDY, restore, sha
from .reference import core, pulse
from . import recurrent as r

import jax
import jax.numpy as jnp
import numpy as np

from experiments.belief_quotient_history_jax_2026_11 import models
from experiments.belief_quotient_history_jax_2026_11.continuation import load_params
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment

TIMES = (256, 512, 768)
CONTROLS = ("normal", "all_layers", "same_top_with_lower_write", "constant_lower",
            "zero_lower", "drop_top_carry", "clip_source")


def rho_target(b, clip=False):
    rho = jnp.clip(b[..., 2], 0, 1) if clip else b[..., 2]
    mass = b[..., :2].sum(-1, keepdims=True)
    ratio = jnp.where(mass > 0, b[..., :2] / jnp.where(mass > 0, mass, 1), .5)
    return jnp.concatenate((ratio * rho[..., None], (1-rho)[..., None]), -1)


def cell_terms(block, x, h):
    ir, iz, inn = jnp.split(x @ block["w_i"] + block["b_i"], 3, -1)
    hr, hz, hn = jnp.split(h @ block["w_h"] + block["b_h"], 3, -1)
    reset = jax.nn.sigmoid(ir + hr)
    update = jax.nn.sigmoid(iz + hz)
    candidate = jnp.tanh(inn + reset * hn)
    return (1-update) * candidate + update * h, update, candidate


@partial(jax.jit, static_argnums=(0, 5))
def controlled_trace(spec, params, maps, initial, observations, control):
    """Independent cell-by-cell replay, with named diagnostic changes only."""
    q = -maps["decoder"][-1].sum(-1)
    fixed_lower = models._gru_cell(params["blocks"][0], models._dense(params["input"], observations[0]), initial[0])
    natural_top = models._gru_cell(params["blocks"][-1], fixed_lower, initial[-1])
    natural_b = core.decode(natural_top, maps, spec.n_layers-1)
    initial_written_top = natural_top + jnp.matmul(
        rho_target(natural_b)[..., :2]-natural_b[..., :2], maps["encoder"][-1],
        precision=jax.lax.Precision.HIGHEST)

    def body(carry, item):
        intact, written = carry
        lag, obs = item
        xi = models._dense(params["input"], obs)
        xb = xi
        hi, hb = [], []
        components = jnp.zeros((obs.shape[0], 3))
        for layer, block in enumerate(params["blocks"]):
            previous_i, previous_b = intact[layer], written[layer]
            if layer == spec.n_layers-1:
                if control == "constant_lower":
                    xi = jnp.where(lag > 0, fixed_lower, xi)
                    xb = jnp.where(lag > 0, fixed_lower, xb)
                if control == "zero_lower":
                    xi = jnp.where(lag > 0, 0., xi)
                    xb = jnp.where(lag > 0, 0., xb)
                if control == "drop_top_carry":
                    previous_b = jnp.where(lag > 0, previous_i, previous_b)
            out_i, zi, ni = cell_terms(block, xi, previous_i)
            out_b, zb, nb = cell_terms(block, xb, previous_b)
            if layer == spec.n_layers-1:
                parts = (zb * (previous_b-previous_i),
                         (zb-zi) * (previous_i-ni), (1-zb) * (nb-ni))
                components = jnp.stack([part @ q for part in parts], -1)
            if layer == spec.n_layers-1 or control in ("all_layers", "same_top_with_lower_write"):
                desired = rho_target(core.decode(out_i, maps, layer), control == "clip_source")
                displacement = desired[..., :2] - core.decode(out_b, maps, layer)[..., :2]
                shift = jnp.matmul(displacement, maps["encoder"][layer], precision=jax.lax.Precision.HIGHEST)
                out_b = out_b + jnp.where(lag == 0, shift, 0.)
            if layer == spec.n_layers-1 and control == "same_top_with_lower_write":
                out_b = jnp.where(lag == 0, initial_written_top, out_b)
            xi, xb = out_i, out_b
            hi.append(out_i)
            hb.append(out_b)
        hi, hb = jnp.stack(hi), jnp.stack(hb)
        return (hi, hb), {"intact": hi, "written": hb,
                "rho_diff": core.decode(hb[-1], maps, spec.n_layers-1)[:, 2] - core.decode(hi[-1], maps, spec.n_layers-1)[:, 2],
                "components": components}

    return jax.lax.scan(body, (initial, initial), (jnp.arange(len(observations)), observations))[1]


def describe(differences):
    # Input axes: pulse time, lag, episode.
    pooled = differences.transpose(1, 0, 2).reshape(differences.shape[1], -1)
    absolute = np.abs(pooled)
    per_episode = np.abs(differences).mean(0).T
    sign = np.sign(pooled[0])
    rows = []
    for lag in range(9):
        values = absolute[lag]
        rows.append({"lag": lag, "mean_absolute": float(values.mean()),
                     "signed_mean": float(pooled[lag].mean()),
                     "pulse_sign_aligned_mean": float((pooled[lag]*sign).mean()),
                     "absolute_quantiles_0_10_25_50_75_90_100": np.quantile(values, [0, .1, .25, .5, .75, .9, 1]).tolist(),
                     "top5_of96_share": float(np.sort(values)[-5:].sum()/values.sum()) if values.sum() else 0.,
                     "top5_of32_episodes_share": float(np.sort(per_episode[:, lag])[-5:].sum()/per_episode[:, lag].sum()) if values.sum() else 0.,
                     "opposite_to_pulse_sign_fraction": float((pooled[lag]*sign < 0).mean())})
    return {"lag_distribution": rows,
            "per_time_absolute_0_8": np.abs(differences[:, :9]).mean(-1).tolist(),
            "per_time_signed_0_8": differences[:, :9].mean(-1).tolist(),
            "per_time_absolute_quantiles_10_50_90_0_8": np.quantile(np.abs(differences[:, :9]), [.1, .5, .9], axis=-1).transpose(1, 2, 0).tolist(),
            "per_episode_absolute_mean_over_times_0_8": per_episode[:, :9].tolist(),
            "mean_absolute_0_128": absolute.mean(-1).tolist(),
            "mean_signed_0_128": pooled.mean(-1).tolist(),
            "per_time_fraction_lag4_above_lag1": (np.abs(differences[:, 4]) > np.abs(differences[:, 1])).mean(-1).tolist(),
            "leave_one_pulse_episode_pair_out_lag4_minus_lag1_range": [float(x) for x in (
                ((absolute[4].sum()-absolute[1].sum()) - (absolute[4]-absolute[1])).min()/95,
                ((absolute[4].sum()-absolute[1].sum()) - (absolute[4]-absolute[1])).max()/95)],
            "leave_one_of32_episodes_out_lag4_minus_lag1_range": [float(x) for x in (
                (per_episode[:, 4].sum()-per_episode[:, 1].sum() - (per_episode[:, 4]-per_episode[:, 1])).min()/31,
                (per_episode[:, 4].sum()-per_episode[:, 1].sum() - (per_episode[:, 4]-per_episode[:, 1])).max()/31)]}


def load_case(architecture, variant):
    recipe, checkpoint, provenance = restore(architecture, variant)
    spec = models.ModelSpec(**recipe["model"])
    params = load_params(checkpoint, spec)
    process = environment.ladder_process(variant)
    fit = r.rollout(spec, process, params, r.empty_maps(spec), jax.random.PRNGKey(42), 14, 1024)
    maps = r.fit_maps(fit, 12000)
    test = r.rollout(spec, process, params, maps, jax.random.PRNGKey(43), 14, 1024)
    probes = r.probe_scores(test, maps, 12000)
    records = r.rollout(spec, process, params, maps, jax.random.PRNGKey(3042), 32, 1024)
    return spec, params, maps, records, provenance, probes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--rerun-output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.raw_output, args.rerun_output):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    protected = list((HERE / "results").glob("variant_*_seed42.json"))
    protected += [HERE / "results" / name for name in ("summary.json", "summary.md")]
    paper = STUDY.parents[2] / "deep-beliefs" / "figures"
    protected += [paper / "fig_arch_pulse_decay.pdf", paper / "plot_arch_pulse_decay.py"]
    before = {str(p): sha(p) for p in protected}
    report = {"times": TIMES, "cases": {}, "source_sha256": sha(__file__), "protected_before": before}
    raw = {}
    for architecture, variant in (("gru128", 2), ("gru92", 2), ("gru128", 3)):
        spec, params, maps, records, provenance, probes = load_case(architecture, variant)
        old_path = HERE / "results" / f"variant_{variant}_{architecture}_seed42.json"
        old = json.loads(old_path.read_text())
        case = {"checkpoint": provenance, "probes_exactly_equal": probes == old["probes"], "sites": {}}
        saved = r.snapshots(spec, params, maps, records["obs"], jnp.asarray(TIMES), len(TIMES))
        for site in (0, spec.n_layers-1):
            differences = []
            max_intact_alignment = 0.
            for i, t in enumerate(TIMES):
                obs = records["obs"][t:t+129]
                _, intact = r.branch(spec, params, maps, saved[i], obs, t, site, 2, 0.)
                _, written = r.branch(spec, params, maps, saved[i], obs, t, site, 2, 1.)
                expected = core.decode(records["h"][t:t+129, :, -1], maps, spec.n_layers-1)
                max_intact_alignment = max(max_intact_alignment, float(np.max(np.abs(np.asarray(intact-expected)))))
                differences.append(np.asarray(written[:, :, 2]-intact[:, :, 2]))
            differences = np.stack(differences)
            raw[f"{architecture}_v{variant}_layer{site+1}"] = differences
            statistics = describe(differences)
            old_condition = old["pulse"]["conditions"][f"rho_flip_layer{site+1}"]["measured"]
            for label, values in (("absolute", np.abs(differences).mean((0, 2))), ("signed", differences.mean((0, 2)))):
                reference = np.asarray(old_condition[label])[:, 0]
                statistics[f"saved_{label}_max_difference"] = float(np.abs(values-reference).max())
            statistics["intact_replay_vs_recorded_readout_max_difference"] = max_intact_alignment
            case["sites"][f"layer{site+1}"] = statistics
        if architecture == "gru128" and variant == 2:
            fresh_pulse = r.run_pulse(spec, params, maps, records, variant, environment)
            fresh = {"checkpoint": provenance, "probes": probes, "pulse": fresh_pulse,
                     "protocol": old["protocol"], "source_sha256": {n: sha(HERE / n) for n in ("analyze.py", "reference.py", "recurrent.py")}}
            args.rerun_output.write_text(json.dumps(fresh, indent=2, allow_nan=False)+"\n")
            errors = [np.max(np.abs(np.asarray(item[kind][metric])-np.asarray(old["pulse"]["conditions"][name][kind][metric])))
                      for name, item in fresh_pulse["conditions"].items() for kind in ("measured", "full_filter") for metric in ("absolute", "signed")]
            case["full_rerun_max_difference"] = float(max(errors))
            controls = {}
            normal_top_at_pulse = None
            for control in CONTROLS:
                traces = [controlled_trace(spec, params, maps, saved[i], records["obs"][t:t+129], control) for i, t in enumerate(TIMES)]
                d = np.stack([np.asarray(x["rho_diff"]) for x in traces])
                controls[control] = describe(d)
                raw[f"control_{control}"] = d
                if control == "normal":
                    normal_top_at_pulse = np.stack([np.asarray(x["written"][0, -1]) for x in traces])
                    reference = raw["gru128_v2_layer2"]
                    controls[control]["independent_cell_replay_max_difference"] = float(np.abs(d-reference).max())
                    controls[control]["layer1_written_vs_intact_max_difference"] = float(max(np.max(np.abs(np.asarray(x["written"][:, 0]-x["intact"][:, 0]))) for x in traces))
                    components = np.stack([np.asarray(x["components"]) for x in traces])
                    raw["normal_components"] = components
                    controls[control]["component_reconstruction_max_difference_lags1_128"] = float(np.max(np.abs(components[:, 1:].sum(-1)-d[:, 1:])))
                    aligned = components*np.sign(reference[:, 0])[:, None, :, None]
                    controls[control]["pulse_sign_aligned_components_1_8"] = aligned[:, 1:9].mean((0, 2)).tolist()
                    h0 = np.concatenate([np.asarray(x["written"][0, -1]) for x in traces])
                    controls[control]["post_write_hidden_outside_unit_interval_fraction"] = float((np.abs(h0) > 1).mean())
                if control == "same_top_with_lower_write":
                    controls[control]["pulse_top_state_vs_normal_max_difference"] = float(np.max(np.abs(
                        np.stack([np.asarray(x["written"][0, -1]) for x in traces])-normal_top_at_pulse)))
            case["controls"] = controls
            b = np.concatenate([np.asarray(core.decode(records["h"][t, :, -1], maps, 1)) for t in TIMES])
            desired = np.asarray(rho_target(jnp.asarray(b)))
            case["target_audit"] = {"n": len(b), "decoded_rho_range": [float(b[:, 2].min()), float(b[:, 2].max())],
                    "decoded_rho_outside_0_1_count": int(((b[:, 2]<0)|(b[:, 2]>1)).sum()),
                    "nonpositive_nonreward_mass_count": int((b[:, :2].sum(-1)<=0).sum()),
                    "independent_target_vs_driver_max_difference": float(np.max(np.abs(desired-np.asarray(core.target(jnp.asarray(b), 2))))),
                    "rho_target_vs_one_minus_rho_max_difference": float(np.max(np.abs(desired[:, 2]-(1-b[:, 2]))))}
            raw["pulse_input_beliefs"] = b
        report["cases"][f"{architecture}_v{variant}"] = case
        print(architecture, variant, "top-site absolute saved difference", case["sites"][f"layer{spec.n_layers}"]["saved_absolute_max_difference"], flush=True)
    after = {str(p): sha(p) for p in protected}
    assert before == after, "An existing result or figure changed"
    report["protected_outputs_unchanged"] = True
    report["raw_output"] = str(args.raw_output)
    np.savez_compressed(args.raw_output, **raw)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print("Saved diagnostic:", args.output)
    print("Saved fresh rerun:", args.rerun_output)
    print("Saved per-episode arrays:", args.raw_output)
    print("Existing six reports, summaries and figure/script hashes unchanged.")


if __name__ == "__main__":
    main()
