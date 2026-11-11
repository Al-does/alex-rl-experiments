"""Matched CPU steering comparison of r7 updates 29 and 316."""

import argparse
import json
from pathlib import Path

import jax
from experiments.rocksample_jax_2026_10 import belief_steering as bs
from experiments.rocksample_jax_2026_10 import belief_steering_paper as paper
from experiments.rocksample_jax_2026_10 import belief_steering_site0 as frozen

ROOT = bs.bp.ROOT
LEAF = ROOT / "ppo_bmax_r7_cont330m/results"
V4 = ROOT / "results/belief_steering_r7_20261110_v4_site0/results.json"
V5 = ROOT / "results/belief_steering_r7_20261110_v5_paper_protocol/results.json"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def mappings(leaf=LEAF, historical=None):
    historical = read(V4) if historical is None else historical
    result = []
    sources = [(p.parent, read(p)) for p in sorted(leaf.glob("*/continuation_source.json"))]
    for agent in historical["agents"]:
        matches = [(p, s) for p, s in sources if
                   (s["run_id"], s["seed_index"]) == (agent["run_id"], agent["seed"])]
        if len(matches) != 1:
            raise ValueError(f"expected one continuation for {agent['run_id']} seed {agent['seed']}")
        directory, source = matches[0]
        if any(source[k] != agent["checkpoint"][k] for k in ("sha256", "size_bytes", "uri")):
            raise ValueError("continuation source differs from historical checkpoint")
        if source["env_steps"] != 30_408_704:
            raise ValueError("unexpected source lifetime steps")
        result.append(dict(source_run=agent["run_id"], seed=agent["seed"],
                           continuation_run=directory.name, source=source,
                           source_record_sha256=bs.bp.sha256(directory/"continuation_source.json")))
    return result


def load(mapping, update):
    run = mapping["continuation_run"] if update == 316 else mapping["source_run"]
    leaf = "ppo_bmax_r7_cont330m" if update == 316 else "ppo_bmax_r7"
    results = ROOT/leaf/"results"/run
    remote = read(results/"run_manifest.json")["remote_artifacts"]
    prefix = f"experiments/rocksample_jax_2026_10/{leaf}/{run}"
    if (remote["prefix"] != prefix or
            remote["canonical_manifest_key"] != prefix+"/metadata/durability_manifest.json"):
        raise ValueError("canonical durability manifest layout differs from verified loader")
    directory = ROOT/"artifacts/belief_steering_continuation"/run
    directory.mkdir(parents=True, exist_ok=True)
    params, steps, arm, provenance = bs.bp.checkpoint(directory, mapping["seed"], True,
                                                     run_results=results, update=update)
    expected_steps = 331_350_016 if update == 316 else 30_408_704
    expected = dict(d_model=128, n_layers=3, n_heads=4, context_len=32)
    if steps != expected_steps or any(arm[k] != v for k, v in expected.items()):
        raise ValueError("checkpoint lifetime steps or recipe architecture differs")
    env = bs.rs.RockSampleParams.from_instance()
    spec = bs.tm.TransformerSpec(env.obs_dim, env.num_actions, **expected)
    template = bs.tm.init_params(spec, jax.random.key(0))
    if jax.tree.structure(params) != jax.tree.structure(template) or any(
            a.shape != b.shape for a, b in zip(jax.tree.leaves(params), jax.tree.leaves(template))):
        raise ValueError("checkpoint parameter architecture differs from recipe")
    provenance.update(lifetime_steps=steps, architecture=expected,
                      canonical_manifest_key=remote["canonical_manifest_key"],
                      continuation=update == 316, source=mapping["source"],
                      source_num_keys=4)
    print(f"Verified {run} seed {mapping['seed']} update {update}: {steps} steps; "
          f"{provenance['size_bytes']} bytes; SHA256 {provenance['sha256']}; architecture matches", flush=True)
    return params, env, spec, provenance


def strengths(tuning, representation):
    return {d: tuning["representations"][representation][d]["strength"] for d in paper.DIRECTIONS}


def same_strengths(a, b, representation):
    return strengths(a, representation) == strengths(b, representation)


def run_checkpoint(mapping, update, output, smoke=False, resume=False):
    params, env, spec, provenance = load(mapping, update)
    path = output/f"{mapping['source_run']}_seed{mapping['seed']}_update{update}.json"
    state = read(path) if resume and path.exists() else dict(mapping=mapping, checkpoint=provenance)
    if state["checkpoint"]["sha256"] != provenance["sha256"]:
        raise ValueError("resume checkpoint changed")
    episodes = 8 if smoke else 1024
    fit = 8 if smoke else 256
    old4 = next(a for a in read(V4)["agents"] if
                (a["run_id"], a["seed"]) == (mapping["source_run"], mapping["seed"]))
    old5 = next(a for a in read(V5)["agents"] if
                (a["run_id"], a["seed"]) == (mapping["source_run"], mapping["seed"]))
    if "v4" not in state or len(state["v4"]["sites"]) < 4:
        fitted, r2, controlled, nuisance = bs.maps(spec, env, params, fit)
        variants, calibration = bs.calibrated_maps((fitted[0], fitted[1], controlled, fitted[3]))
        state.setdefault("v4", dict(r2=r2, nuisance=nuisance, calibration=calibration, sites={}))
        baseline = {str(i): row for i, row in enumerate(old4["rows"])} if update == 29 and not smoke else None
        for site in range(4):
            if str(site) in state["v4"]["sites"]:
                continue
            print(f"v4 update {update} seed {mapping['seed']} site {site}", flush=True)
            if update == 29 and site == 0 and not smoke:
                rows = old4["rows"]
            else:
                intact = list(baseline.values()) if baseline is not None else None
                rows = bs.evaluate(spec, env, params, variants["calibrated_encoder"], episodes,
                                   frozen.CONFIRM_SEED, site, 1., intact_rows=intact)
                if baseline is None:
                    baseline = {str(i): row for i, row in enumerate(rows)}
            state["v4"]["sites"][str(site)] = rows
            write(path, state)
    if "v5" not in state or any(len(s) < 4 for s in state["v5"]["representations"].values()):
        fitted, r2, diagnostics = paper.fit_maps(spec, env, params, fit)
        if "v5" not in state:
            tuning = (read(V5)["tuning"] if update == 29 and mapping["seed"] == 3 and not smoke else
                      paper.tune(spec, env, params, fitted, 8 if smoke else 128))
            state["v5"] = dict(r2=r2, diagnostics=diagnostics, tuning=tuning,
                               representations={rep: {} for rep in paper.REPRESENTATIONS})
            write(path, state)
        baseline = None
        for representation in paper.REPRESENTATIONS:
            for site in range(4):
                if str(site) in state["v5"]["representations"][representation]:
                    continue
                selected = strengths(state["v5"]["tuning"], representation)
                print(f"v5 update {update} seed {mapping['seed']} {representation} site {site} λ={selected}", flush=True)
                if update == 29 and not smoke and same_strengths(state["v5"]["tuning"], read(V5)["tuning"], representation):
                    rows = old5["representations"][representation][str(site)]
                else:
                    if baseline is None:
                        baseline = (old5["representations"]["marginal"]["0"] if update == 29 and not smoke else
                            paper.evaluate(spec, env, params, fitted["marginal"], "marginal", episodes,
                                paper.CONFIRM_SEED, 2, dict.fromkeys(paper.DIRECTIONS, 0.),
                                conditions=("intact", "random")))
                    rows = paper.evaluate(spec, env, params, fitted[representation], representation, episodes,
                        paper.CONFIRM_SEED, site, selected, baselines=baseline)
                state["v5"]["representations"][representation][str(site)] = rows
                write(path, state)
    jax.clear_caches()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    agents = mappings()
    if args.verify_only:
        for mapping in agents:
            load(mapping, 316)
        return
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema=1, smoke=args.smoke, mappings=agents,
        fit_episodes=8 if args.smoke else 256, test_episodes=8 if args.smoke else 256,
        tune_episodes=8 if args.smoke else 128, confirmation_episodes=8 if args.smoke else 1024,
        v4_seeds={**bs.SEEDS, "confirm": frozen.CONFIRM_SEED},
        v5_seeds={**bs.SEEDS, "tune": paper.TUNE_SEED, "confirm": paper.CONFIRM_SEED},
        sources={p.name: bs.bp.sha256(p) for p in (Path(__file__), Path(bs.__file__), Path(paper.__file__),
            Path(frozen.__file__), ROOT/"belief_probe.py", ROOT/"model.py", ROOT/"env.py")},
        historical={str(p.relative_to(ROOT)): bs.bp.sha256(p) for p in (V4, V5)},
        experiment_revision=bs.bp.revision(ROOT), harness_revision=bs.bp.revision(ROOT.parents[2]/"rl-harness"),
        jax_version=jax.__version__)
    path = args.output/"manifest.json"
    if args.resume and path.exists() and read(path) != manifest:
        raise ValueError("resume configuration or execution sources changed")
    write(path, manifest)
    for mapping in agents[:1] if args.smoke else agents:
        for update in (29, 316):
            run_checkpoint(mapping, update, args.output, smoke=args.smoke, resume=args.resume)


if __name__ == "__main__":
    main()
