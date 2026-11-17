"""Which target does a frozen agent track: the approximate quotient, the full belief, or neither?

Offline analysis of one completed JAX history run: an emission-asymmetry
REINFORCE or PPO leaf, or a PR #234 ``variant_2_transformer`` (exact quotient)
or ``variant_3_transformer`` (Full-State control) run. The final policy plays greedy fit
and test episodes; the run's own step-0 encoder is replayed on the same
observations. Targets are refiltered in float64 from tokens and executed
actions, after the current token and before the current action:

- ``full_belief``: exact three-state posterior; ``reward_marginal`` and
  ``identity`` are its reward-state entry and state-1 minus state-2 contrast.
- ``quotient``: reward-class probability of the equal-weight two-class filter
  on merged tokens (``environment.quotient_model``); the
  ``quotient_occupancy_weighted`` variant averages states 1 and 2 by their
  fit-episode occupancy instead.
- ``quotient_gap``: ``reward_marginal - quotient``.

Probes measure affine accessibility, not causal use.
"""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

import analysis
from analysis.probes import controls
from analysis.probes.resampling import cluster_bootstrap_statistics, percentile_interval
from experiments.belief_quotient_history_jax_2026_11 import env, models
from experiments.belief_quotient_history_jax_2026_11.continuation import load_params
from experiments.belief_quotient_history_jax_2026_11.reinforce import rollout
from experiments.belief_quotient_emission_asymmetry_eps02_jax_temp15_2026_11 import (
    design, environment,
)
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_ppo_2026_11 import ppo

WARMUP = 128  # the four-layer, 32-position banded encoder reads 128 observations
BUDGET = {"fit_episodes": 64, "test_episodes": 128, "behavior_episodes": 2048,
          "reference_episodes": design.EPISODES, "n_resamples": 200}
SMOKE_BUDGET = {"fit_episodes": 8, "test_episodes": 8, "behavior_episodes": 16,
                "reference_episodes": design.SMOKE_EPISODES, "n_resamples": 20}
FIT_SEED, TEST_SEED, BEHAVIOR_SEED, REFERENCE_SEED = 20261201, 20261202, 20261203, 20261204
BEHAVIOR_CHUNK = 256
FILTER_TOLERANCE = 2e-5
FEATURES = ("trained", "initialization", "current_token", "token_previous_action")
RULES = {"quotient": ("quotient",), "reward_marginal": ("reward_marginal",),
         "quotient_and_identity": ("quotient", "identity_share"),
         "full_belief": ("reward_marginal", "identity_share")}
RULE_BINS = (64, 8)

_rollout = partial(jax.jit, static_argnums=(0, 4, 5),
                   static_argnames=("greedy", "track_belief"))(rollout)


@partial(jax.jit, static_argnums=(0,))
def replay(spec, params, observations):
    """Post-final-LayerNorm embeddings of ``params`` on recorded observations."""

    def body(carry, item):
        t, obs = item
        _, _, embedding, carry = models.step(spec, params, carry, obs, t)
        return carry, embedding

    _, embeddings = jax.lax.scan(
        body, models.init_carry(spec, observations.shape[0]),
        (jnp.arange(observations.shape[1]), jnp.swapaxes(observations, 0, 1)))
    return jnp.swapaxes(embeddings, 0, 1)


@partial(jax.jit, static_argnums=(0, 4, 5))
def _behavior_rollout(spec, process, params, key, episodes, horizon):
    records = rollout(spec, process, params, key, episodes, horizon, greedy=True,
                      track_belief=True)
    return {"rho": records["belief"][..., 2], "reward": records["reward"],
            "action": records["action"]}


def load_run(run_dir):
    run = {name: json.loads((Path(run_dir) / f"{name}.json").read_text())
           for name in ("resolved_recipe", "run_manifest", "condition_summary")}
    recipe = run["resolved_recipe"]
    if recipe.get("framework") != "jax" or recipe.get("continued_from") is not None:
        raise ValueError("expected a JAX history run trained from its own initialization")
    if run["run_manifest"].get("status") != "completed":
        raise ValueError("the run did not complete")
    return run


def full_model(recipe):
    full = {name: np.asarray(recipe["process"][name], dtype=np.float64)
            for name in ("initial", "transitions", "emission")}
    if full["emission"].shape != (3, 3):
        raise ValueError("expected a three-state, three-token process")
    return {name: value / value.sum(-1, keepdims=True) for name, value in full.items()}


def _b2_client():
    from harness.storage.b2 import B2StorageConfig

    storage = B2StorageConfig.from_env()
    if storage is None:
        raise RuntimeError("B2 credentials are required to download checkpoints")
    return storage.s3_client()


def verified_checkpoints(run, directory, *, allow_unverified=False):
    """Step-0 and final parameter files in ``directory``, downloading missing
    ones and checking size and SHA-256 against the run's durability manifest."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    remote = run["run_manifest"].get("remote_artifacts") or {}
    manifest_path, client, durable = directory / "durability_manifest.json", None, None
    if not manifest_path.is_file() and remote.get("canonical_manifest_key"):
        client = _b2_client()
        manifest_path.write_bytes(client.get_object(
            Bucket=remote["bucket"], Key=remote["canonical_manifest_key"])["Body"].read())
    if manifest_path.is_file():
        durable = json.loads(manifest_path.read_text())
        if remote and durable["prefix"] != remote["prefix"]:
            raise ValueError(f"{manifest_path} belongs to a different run")
    elif not allow_unverified:
        raise ValueError("no durability manifest; pass --allow-unverified for local-only runs")
    steps = {"initialization": 0, "final": int(run["condition_summary"]["training"]["env_steps"])}
    checkpoints = {}
    for name, step in steps.items():
        relative = f"checkpoints/step_{step:010d}.npz"
        path = directory / Path(relative).name
        record = durable and next(
            (item for item in durable["files"] if item["relative_path"] == relative), None)
        if durable and not record:
            raise ValueError(f"{relative} is missing from the durability manifest")
        if not path.is_file():
            if not record:
                raise FileNotFoundError(path)
            client = client or _b2_client()
            client.download_file(durable["bucket"], record["key"], str(path))
        digest, size = hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size
        if record and (digest, size) != (record["sha256"], record["size_bytes"]):
            raise ValueError(f"{path} does not match the durability manifest")
        checkpoints[name] = {"step": step, "path": str(path), "sha256": digest,
                             "size_bytes": size, "verified_against_durability_manifest": bool(record),
                             "key": record["key"] if record else None}
    return checkpoints


def collect(spec, process, full, params, *, episodes, seed, horizon, warmup=WARMUP):
    """Post-warmup rows of final-policy greedy episodes, with both encoders'
    embeddings and float64 targets filtered over complete episode prefixes."""

    records = jax.tree_util.tree_map(np.asarray, _rollout(
        spec, process, params["final"], jax.random.PRNGKey(seed), episodes, horizon,
        greedy=True, track_belief=True))
    tokens, actions = records["obs"][..., :3].argmax(-1), records["action"]
    belief = environment.filter_histories(full, tokens, actions)
    error = float(np.abs(belief - records["belief"]).max())
    if error > FILTER_TOLERANCE:
        raise ValueError(f"refiltered beliefs differ from the rollout filter by {error}")

    def flat(values):
        values = np.asarray(values)[:, warmup:]
        return values.reshape(-1, *values.shape[2:])

    return {
        "trained": flat(records["embedding"]),
        "initialization": flat(replay(spec, params["initialization"], jnp.asarray(records["obs"]))),
        "current_token": flat(records["obs"][..., :3]),
        "token_previous_action": flat(records["obs"]),
        "belief": flat(belief), "action": flat(actions), "state": flat(records["state"]),
        "groups": np.repeat(np.arange(episodes), horizon - warmup),
        "filter_max_error": error,
        "quotient": lambda weights=(0.5, 0.5): flat(
            environment.quotient_filter(full, tokens, actions, weights)),
    }


def probe_targets(split, occupancy_weights):
    belief = split["belief"]
    rho, quotient = belief[:, 2], split["quotient"]()
    return {
        "full_belief": belief,
        "reward_marginal": rho[:, None],
        "identity": (belief[:, 0] - belief[:, 1])[:, None],
        "quotient": quotient[:, None],
        "quotient_occupancy_weighted": split["quotient"](occupancy_weights)[:, None],
        "quotient_gap": (rho - quotient)[:, None],
    }


def _comparison(result):
    return {name: result[name] for name in (
        "delta_r_squared", "delta_r_squared_ci", "mse_improvement", "mse_improvement_ci")}


def probe_report(fit, test, fit_targets, test_targets, *, seed, n_resamples):
    """Held-out affine probes of every target from every feature set, with
    episode-bootstrap comparisons of the trained encoder against its controls."""

    report, predictions = {}, {}
    for name, target in fit_targets.items():
        scores = {}
        for feature in FEATURES:
            weight, bias, fitted = controls.fit_grouped_affine(
                fit[feature], target, fit["groups"], seed=seed)
            predictions[name, feature] = test[feature] @ weight + bias
            scores[feature] = {
                **controls.score_prediction(predictions[name, feature], test_targets[name]),
                "selected_rcond": fitted["selected_rcond"], "rank": fitted["rank"],
            }
        report[name] = {"scores": scores, **{
            f"trained_minus_{baseline}": _comparison(controls.paired_comparison(
                predictions[name, "trained"], predictions[name, baseline], test_targets[name],
                test["groups"], seed=seed, n_resamples=n_resamples))
            for baseline in ("initialization", "token_previous_action")
        }}
    return report, predictions


def which_target(predictions, targets, groups, *, seed, n_resamples):
    """Quotient-minus-reward-marginal held-out R² on identical rows. Positive
    values mean the features expose the approximate quotient more accurately;
    ``training_change`` subtracts the same contrast for the initialized encoder."""

    rho, report = targets["reward_marginal"][:, 0], {}
    for quotient in ("quotient", "quotient_occupancy_weighted"):
        target = targets[quotient][:, 0]
        errors = {feature: (np.square(predictions[quotient, feature][:, 0] - target),
                            np.square(predictions["reward_marginal", feature][:, 0] - rho))
                  for feature in ("trained", "initialization")}

        def advantage(feature, index=slice(None)):
            quotient_error, rho_error = errors[feature]
            return float(rho_error[index].mean() / rho[index].var()
                         - quotient_error[index].mean() / target[index].var())

        def change(index=slice(None)):
            return advantage("trained", index) - advantage("initialization", index)

        report[quotient] = {
            name: {"quotient_minus_reward_marginal_r_squared": statistic(),
                   "ci": list(percentile_interval(cluster_bootstrap_statistics(
                       groups, statistic, n_resamples=n_resamples, seed=seed)))}
            for name, statistic in (("trained", partial(advantage, "trained")),
                                    ("initialization", partial(advantage, "initialization")),
                                    ("training_change", change))
        }
    return report


def rule_predictions(fit_columns, fit_action, test_columns, n_actions, bins=RULE_BINS):
    """Majority fit-episode action on quantile cells of the given statistics."""

    fit_cell, test_cell, n_cells = 0, 0, 1
    for fit_column, test_column, count in zip(fit_columns, test_columns, bins):
        edges = np.unique(np.quantile(fit_column, np.linspace(0, 1, count + 1)[1:-1]))
        fit_cell = fit_cell * (len(edges) + 1) + np.searchsorted(edges, fit_column)
        test_cell = test_cell * (len(edges) + 1) + np.searchsorted(edges, test_column)
        n_cells *= len(edges) + 1
    counts = np.zeros((n_cells, n_actions))
    np.add.at(counts, (fit_cell, fit_action), 1)
    majority = counts.argmax(1)
    majority[counts.sum(1) == 0] = np.bincount(fit_action, minlength=n_actions).argmax()
    return majority[test_cell]


def action_rules(fit, test, fit_targets, test_targets, *, n_actions, seed, n_resamples):
    """Which statistic explains the greedy actions: held-out agreement of
    majority-action lookups on the quotient, the reward marginal, and each
    joined with the state-1/state-2 share of the non-reward belief."""

    def statistics(split, targets):
        belief = split["belief"]
        mass = belief[:, 0] + belief[:, 1]
        return {"quotient": targets["quotient"][:, 0], "reward_marginal": belief[:, 2],
                "identity_share": np.divide(belief[:, 0] - belief[:, 1], mass,
                                            out=np.zeros_like(mass), where=mass > 0)}

    fit_values, test_values = statistics(fit, fit_targets), statistics(test, test_targets)
    predicted = {
        name: rule_predictions([fit_values[c] for c in columns], fit["action"],
                               [test_values[c] for c in columns], n_actions)
        for name, columns in RULES.items()
    }
    correct = {name: (value == test["action"]).astype(float) for name, value in predicted.items()}

    def difference(first, second):
        rows = correct[first] - correct[second]
        estimates = cluster_bootstrap_statistics(
            test["groups"], lambda index: rows[index].mean(), n_resamples=n_resamples, seed=seed)
        return {"difference": float(rows.mean()), "ci": list(percentile_interval(estimates))}

    differ = predicted["quotient"] != predicted["reward_marginal"]
    return {
        "bins": list(RULE_BINS),
        "agreement": {name: float(value.mean()) for name, value in correct.items()},
        "reward_marginal_minus_quotient": difference("reward_marginal", "quotient"),
        "identity_beyond_quotient": difference("quotient_and_identity", "quotient"),
        "rule_disagreement_fraction": float(differ.mean()),
        "agent_follows_reward_marginal_when_rules_disagree": (
            float(correct["reward_marginal"][differ].mean()) if differ.any() else None),
    }


def behavior(spec, process, params, full, *, episodes, reference_episodes, horizon,
             warmup=WARMUP):
    """Greedy reward-state occupancy against the two simulated threshold
    controllers on the same process, horizon and reset distribution. The
    controllers assume states 1 and 2 share action effects, so they are
    omitted when the transition lumpability defect is nonzero."""

    chunk = min(BEHAVIOR_CHUNK, episodes)
    keys = jax.random.split(jax.random.PRNGKey(BEHAVIOR_SEED), -(-episodes // chunk))
    records = [jax.tree_util.tree_map(np.asarray, _behavior_rollout(
        spec, process, params, key, chunk, horizon)) for key in keys]
    rho, reward, action = (np.concatenate([record[name] for record in records])
                           for name in ("rho", "reward", "action"))
    agent = {
        "episodes": len(rho), "seed": BEHAVIOR_SEED, "policy_mode": "greedy",
        "expected_reward": design.episode_estimate(rho.mean(1)),
        "post_warmup_expected_reward": design.episode_estimate(rho[:, warmup:].mean(1)),
        "realized_reward": design.episode_estimate(reward.mean(1)),
        "action_fractions": (np.bincount(action.ravel(), minlength=spec.n_actions)
                             / action.size).tolist(),
    }
    if environment.lumpability_defect(full)["transition"] > 1e-9:
        return {"agent": agent, "references": None}
    references = design.reference_controllers(
        full, episodes=reference_episodes, seed=REFERENCE_SEED, horizon=horizon, warmup=warmup)

    def gap(statistic):
        reference = references[statistic]["expected_reward"]
        return {"difference": agent["expected_reward"]["mean"] - reference["mean"],
                "se": float(np.hypot(agent["expected_reward"]["episode_mc_se"],
                                     reference["episode_mc_se"]))}

    return {"agent": agent, "references": references,
            "agent_minus_quotient_controller": gap("quotient"),
            "agent_minus_reward_marginal_controller": gap("reward_marginal")}


def _revision(root):
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def analyze(run_dir, checkpoint_dir, *, smoke=False, allow_unverified=False):
    run = load_run(run_dir)
    recipe = run["resolved_recipe"]
    budget = SMOKE_BUDGET if smoke else BUDGET
    horizon = int(recipe["training"]["episode_length"])
    spec, full = models.ModelSpec(**recipe["model"]), full_model(recipe)
    reward = (np.arange(3) == 2).astype(float)
    if not np.array_equal(recipe["process"]["reward"], reward):
        raise ValueError("expected state 3 to be the only rewarded state")
    process = env.make_process(full["initial"], full["transitions"], full["emission"],
                               reward, reward[None])
    checkpoints = verified_checkpoints(run, checkpoint_dir,
                                       allow_unverified=allow_unverified or smoke)
    init_fn = (ppo.init_params if recipe["training"].get("algorithm") == "PPO"
               else models.init_params)
    params = {name: load_params(Path(record["path"]), spec, init_fn=init_fn)
              for name, record in checkpoints.items()}
    fit, test = (
        collect(spec, process, full, params, episodes=budget[f"{name}_episodes"], seed=seed,
                horizon=horizon)
        for name, seed in (("fit", FIT_SEED), ("test", TEST_SEED))
    )
    occupancy = np.bincount(fit["state"], minlength=3)[:2] + 1
    fit_targets, test_targets = probe_targets(fit, occupancy), probe_targets(test, occupancy)
    probes, predictions = probe_report(fit, test, fit_targets, test_targets,
                                       seed=FIT_SEED, n_resamples=budget["n_resamples"])
    resampling = {"seed": FIT_SEED, "n_resamples": budget["n_resamples"]}
    return {
        "study": recipe["study"], "condition": recipe["condition"], "seed": recipe["seed"],
        "run_id": run["run_manifest"]["run_id"], "smoke": smoke,
        "emission_asymmetry_kappa": recipe.get("emission_asymmetry_kappa"),
        "lumpability_defect": environment.lumpability_defect(full),
        "process": {name: value.tolist() for name, value in full.items()},
        "quotient_model": {name: value.tolist()
                           for name, value in environment.quotient_model(full).items()},
        "quotient_occupancy_weights": (occupancy / occupancy.sum()).tolist(),
        "checkpoints": checkpoints,
        "training_revisions": run["run_manifest"].get("git"),
        "analysis_revisions": {
            "experiments": _revision(Path(__file__).resolve().parents[2]),
            "rl_harness": _revision(Path(analysis.__file__).resolve().parents[1]),
            "jax": jax.__version__,
        },
        "protocol": {
            **budget, "horizon": horizon, "warmup": WARMUP, "policy_mode": "greedy",
            "fit_seed": FIT_SEED, "test_seed": TEST_SEED, "site": "post_final_layer_norm",
            "probe": "grouped SVD-cutoff affine fit on fit episodes, scored on test episodes",
            "intervals": "95% episode bootstrap with fixed fitted probes; evaluation "
                         "uncertainty for one training seed, not training-seed variation",
            "n_fit_rows": len(fit["groups"]), "n_test_rows": len(test["groups"]),
            "filter_max_error": max(fit["filter_max_error"], test["filter_max_error"]),
        },
        "target_separation": design.target_separation({
            "rho": test_targets["reward_marginal"][:, 0], "quotient": test_targets["quotient"][:, 0],
            "identity": test_targets["identity"][:, 0]}),
        "probes": probes,
        "which_target": which_target(predictions, test_targets, test["groups"], **resampling),
        "action_rules": action_rules(fit, test, fit_targets, test_targets,
                                     n_actions=spec.n_actions, **resampling),
        "behavior": behavior(spec, process, params["final"], full,
                             episodes=budget["behavior_episodes"],
                             reference_episodes=budget["reference_episodes"], horizon=horizon),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="results/<run-id> directory holding the run's compact JSON files")
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="directory holding (or receiving) step_<N>.npz and durability_manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-unverified", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} exists; outputs are never overwritten")
    report = analyze(args.run_dir, args.checkpoint_dir, smoke=args.smoke,
                     allow_unverified=args.allow_unverified)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "x") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    print("Wrote", args.output)
