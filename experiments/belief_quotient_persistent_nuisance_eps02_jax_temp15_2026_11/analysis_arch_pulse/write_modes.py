"""Ordered all-layer pulse writes, compared with verified existing first/top writes."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path

from .analyze import HERE, STUDY, sha
from .diagnose_rebound import load_case
from .reference import core, pulse
from . import recurrent as r

import jax
import jax.numpy as jnp
import numpy as np

from experiments.belief_quotient_history_jax_2026_11 import models
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment

TIMES = (256, 512, 768)
LAGS = (0, 1, 2, 4, 8, 16, 32, 64)


def all_layer_step(spec, params, carry, obs, maps, reference, mode, strength):
    """Targets come from intact per-layer activations; writes use current steered outputs."""
    x = models._dense(params["input"], obs)
    hidden = []
    for layer, block in enumerate(params["blocks"]):
        x = models._gru_cell(block, x, carry[layer])
        desired = core.target(core.decode(reference[:, layer], maps, layer), mode)
        displacement = desired[..., :2] - core.decode(x, maps, layer)[..., :2]
        shift = jnp.matmul(displacement, maps["encoder"][layer], precision=jax.lax.Precision.HIGHEST)
        x = x + strength * shift
        hidden.append(x)
    hidden = jnp.stack(hidden)
    return hidden, jnp.swapaxes(hidden, 0, 1)


@partial(jax.jit, static_argnums=(0,))
def all_layer_branch(spec, params, maps, initial, observations, mode):
    def body(carry, item):
        intact, written = carry
        lag, obs = item
        _, intact, reference = r.step(spec, params, intact, obs, lag, maps)
        written, sites = all_layer_step(spec, params, written, obs, maps, reference,
                                        mode, jnp.where(lag == 0, 1., 0.))
        difference = core.decode(sites[:, -1], maps, spec.n_layers-1) - core.decode(reference[:, -1], maps, spec.n_layers-1)
        return (intact, written), difference
    return jax.lax.scan(body, (initial, initial), (jnp.arange(len(observations)), observations))[1]


def run_all_layers(spec, params, maps, records, times=TIMES, horizon=128):
    observations = records["obs"]
    if not times or min(times) < 0 or horizon < 0 or max(times)+horizon >= len(observations):
        raise ValueError("pulse times and horizon must fit the complete recorded episode")
    saved = r.snapshots(spec, params, maps, observations, jnp.asarray(times), len(times))
    result = {}
    for name, mode in (("mirror", 1), ("rho_flip", 2)):
        difference = [np.asarray(all_layer_branch(spec, params, maps, saved[i],
                      observations[t:t+horizon+1], mode)) for i, t in enumerate(times)]
        result[name] = pulse.summarize(pulse.coordinates(np.concatenate(difference, 1)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("output directory must be new or empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paper = STUDY.parents[2] / "deep-beliefs" / "figures"
    protected = list((HERE / "results").glob("*.json")) + list((HERE / "results").glob("*.md"))
    protected += [paper / name for name in ("fig_arch_pulse_decay.pdf", "plot_arch_pulse_decay.py", "fig_arch_pulse_decay_method.md")]
    protected_hashes = {str(p): sha(p) for p in protected}
    rows, report_index = [], []
    for architecture in ("gru128", "gru92"):
        for variant in (3, 2):
            source = HERE / "results" / f"variant_{variant}_{architecture}_seed42.json"
            old = json.loads(source.read_text())
            spec, params, maps, records, provenance, probes = load_case(architecture, variant)
            assert provenance["sha256"] == old["checkpoint"]["sha256"]
            assert probes == old["probes"], "refitted probe scores differ"
            baseline = r.run_pulse(spec, params, maps, records, variant, environment)
            baseline_errors = [np.max(np.abs(np.asarray(item[kind][metric])-np.asarray(old["pulse"]["conditions"][name][kind][metric])))
                               for name, item in baseline["conditions"].items()
                               for kind in ("measured", "full_filter") for metric in ("absolute", "signed")]
            baseline_difference = float(max(baseline_errors))
            assert baseline_difference == 0., "original first/top results did not reproduce exactly"
            all_layers = run_all_layers(spec, params, maps, records)
            condition = "Full-State Quotient" if variant == 3 else "Reward-State Quotient"
            conditions = {}
            for mode, coordinate in (("mirror", 1), ("rho_flip", 0)):
                for write_mode, site in (("first", 1), ("top", spec.n_layers), ("all", None)):
                    measured = all_layers[mode] if site is None else old["pulse"]["conditions"][f"{mode}_layer{site}"]["measured"]
                    full_filter = old["pulse"]["conditions"][f"{mode}_layer1"]["full_filter"]
                    conditions[f"{mode}_{write_mode}"] = {"measured": measured, "full_filter": full_filter}
                    rows.append({"architecture": architecture, "condition": condition, "pulse": mode,
                                 "write_mode": write_mode, "lags": {str(k): measured["absolute"][k][coordinate] for k in LAGS}})
                rows.append({"architecture": architecture, "condition": condition, "pulse": mode,
                             "write_mode": "filter", "lags": {str(k): full_filter["absolute"][k][coordinate] for k in LAGS}})
            report = {"architecture": architecture, "variant": variant, "condition": condition, "seed": 42,
                      "checkpoint": provenance, "probes": probes, "probe_scores_identical": True,
                      "source_report": str(source), "source_report_sha256": sha(source),
                      "reused_first_top_original_path_max_difference": baseline_difference,
                      "first_layer_semantics": "site=0 only; modified output feeds the next layer and is carried; other layers update normally",
                      "all_layer_semantics": "At t0 each cell consumes steered output below, then shifts toward the flipped belief decoded from that layer's intact shadow output. Each modified output is carried. No writes after t0.",
                      "protocol": old["protocol"], "reference_driver_revision": old["reference_driver_revision"],
                      "source_sha256": {name: sha(HERE / name) for name in ("write_modes.py", "recurrent.py", "reference.py", "analyze.py", "diagnose_rebound.py")},
                      "probe_map_sha256": {name: hashlib.sha256(np.asarray(value).tobytes()).hexdigest() for name, value in maps.items()},
                      "pulse": {"seed": 3042, "times": list(TIMES), "horizon": 128, "conditions": conditions}}
            path = args.output_dir / f"variant_{variant}_{architecture}_seed42_write_modes.json"
            path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            report_index.append({"path": path.name, "sha256": sha(path)})
            print(f"Saved {path.name}; first/top baseline max difference: {baseline_difference}", flush=True)
    assert protected_hashes == {str(p): sha(p) for p in protected}
    summary = {"lags": list(LAGS), "rows": rows, "reports": report_index,
               "protected_outputs_sha256": protected_hashes, "protected_outputs_unchanged": True}
    (args.output_dir / "write_site_comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    lines = ["# GRU pulse write-mode comparison", "", "Seed 42; final step 2012217344; mean absolute change in the pulsed coordinate.",
             "First/top curves reuse exactly reproduced original results. All-layer targets use intact per-layer shadows, with sequential steered-input writes and carry replacement.", "",
             "| GRU | Condition | Pulse | Write | " + " | ".join(str(k) for k in LAGS) + " |",
             "|---|---|---|---|" + "---:|"*len(LAGS)]
    for row in rows:
        lines.append(f"| {row['architecture']} | {row['condition']} | {row['pulse']} | {row['write_mode']} | " + " | ".join(f"{v:.6f}" for v in row["lags"].values()) + " |")
    (args.output_dir / "write_site_comparison.md").write_text("\n".join(lines)+"\n")
    print(f"Saved {len(rows)} table rows; original reports, summaries and decay figure/script unchanged.")


if __name__ == "__main__":
    main()
