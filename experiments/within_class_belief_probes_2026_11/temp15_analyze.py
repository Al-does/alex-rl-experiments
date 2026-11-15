"""CPU conditional probes and complete-episode frozen-input replays for PR202."""

import argparse
from functools import partial
import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np

from experiments.belief_quotient_history_jax_2026_11 import models
from experiments.belief_quotient_history_jax_2026_11.probe import _probe_rollout
from .temp15 import ROOT, RAW, runs, restore, histories, rows, gate_passes
from .metrics import refit, score, null_basis, bootstrap

TARGETS = {"belief": slice(0, 3), "delta": slice(3, 4), "rho": slice(4, 5)}


@partial(jax.jit, static_argnums=0)
def replay(spec, params, obs):
    def body(carry, item):
        t, observation = item
        _, _, embedding, carry = models.step(spec, params, carry, observation, t)
        return carry, embedding
    _, embeddings = jax.lax.scan(body, models.init_carry(spec, obs.shape[0]),
                                (jnp.arange(obs.shape[1]), jnp.swapaxes(obs, 0, 1)))
    return jnp.swapaxes(embeddings, 0, 1)


def perturb(obs, mode, seed):
    rng = np.random.default_rng(seed)
    changed = obs.copy()
    if mode == "history_shuffle":
        for i in range(len(obs)):
            changed[i, 1:] = rng.permutation(obs[i, 1:])
    elif mode == "random_input":
        reset, later = obs[:, 0], obs[:, 1:].reshape(-1, obs.shape[-1])
        changed[:, 0] = reset[rng.integers(len(reset), size=len(obs))]
        changed[:, 1:] = later[rng.integers(len(later), size=len(obs)*(obs.shape[1]-1))].reshape(changed[:, 1:].shape)
    else:
        raise ValueError("unknown history control")
    return changed


def summarize(y, prediction, classes):
    valid = np.isfinite(prediction).all(1)
    result = score(y[valid], prediction[valid], classes[valid])
    result["excluded_rows"] = int((~valid).sum())
    result["per_class"] = {str(k): score(y[(classes == k) & valid], prediction[(classes == k) & valid]) for k in range(3)}
    return result


def analyze_encoder(spec, params, data, embeddings, classes):
    counts = (60000, 80000)
    h = [rows({"h": value}, "h", n) for value, n in zip(embeddings, counts)]
    belief = [rows(d, "belief", n) for d, n in zip(data, counts)]
    targets = [np.concatenate([b, b[:, 0:1]-b[:, 1:2], b[:, 2:3]], axis=1).astype(float) for b in belief]
    weight = np.asarray(params["policy"]["w"], dtype=float)
    bias = np.asarray(params["policy"]["b"], dtype=float)
    basis = null_basis(weight.T)
    features = {"full_h": h, "head_blind": [x @ basis for x in h], "logits": [x @ weight+bias for x in h]}
    result = {"fit_counts": np.bincount(classes[0], minlength=3).tolist(),
              "test_counts": np.bincount(classes[1], minlength=3).tolist(),
              "head_null_dimension": basis.shape[1], "metrics": {}, "controls": {}}
    result["one_class"] = int(np.count_nonzero(np.sum([result["fit_counts"], result["test_counts"]], axis=0))) == 1
    predictions = {}
    for name, (fit, test) in features.items():
        predictions[name] = refit(fit, targets[0], test, *classes)
    for index, mode in enumerate(("history_shuffle", "random_input")):
        changed = [perturb(d["obs"], mode, 20261106+10*index+i) for i, d in enumerate(data)]
        values = [rows({"h": np.asarray(replay(spec, params, obs))}, "h", n) for obs, n in zip(changed, counts)]
        predictions[mode] = refit(values[0], targets[0], values[1], *classes)
    predictions["label_permutation"] = refit(h[0], targets[0], h[1], *classes, permute_seed=853)
    for name, predicted in predictions.items():
        group = "metrics" if name in features else "controls"
        result[group][name] = {target: summarize(targets[1][:, sl], predicted[:, sl], classes[1]) for target, sl in TARGETS.items()}
    return result, predictions, targets[1]


def gru92_diagnosis():
    result = {"training_seed": 42, "arms": []}
    fingerprints = []
    for run in [r for r in runs() if r.parent.parent.name.endswith("gru92")]:
        spec, process, params, provenance = restore(run, 0)
        keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(42), 2))
        records = _probe_rollout(spec, process, params, keys[1], 84, 1024)
        h = rows(records, "embedding", 80000)
        logits = np.asarray(jax.jit(lambda x, w, b: x @ w+b)(h, params["policy"]["w"], params["policy"]["b"]))
        ordered = np.sort(logits, axis=1)
        gaps = ordered[:, -1]-ordered[:, -2]
        archived = json.loads((run/"condition_summary.json").read_text())["checkpoint_reports"][0]
        digest = hashlib.sha256()
        for key, value in jax.tree_util.tree_leaves_with_path(params):
            array = np.asarray(value)
            digest.update(jax.tree_util.keystr(key).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        fingerprints.append(digest.hexdigest())
        result["arms"].append({"arm": run.parent.parent.name, "checkpoint_sha256": provenance["checkpoint"]["sha256"],
            "parameter_sha256": fingerprints[-1], "archived_policy": archived["policy"],
            "cpu_action_fractions": (np.bincount(rows(records, "action", 80000), minlength=3)/80000).tolist(),
            "cpu_mean_reward": float(rows(records, "reward", 80000).mean()),
            "top1_minus_top2_median": float(np.median(gaps)), "top1_minus_top2_min": float(gaps.min()),
            "head_argmax_agreement": float(np.mean(logits.argmax(1) == rows(records, "action", 80000)))})
    result["identical_parameters_across_variants"] = len(set(fingerprints)) == 1
    result["identical_npz_bytes_across_variants"] = len({r["checkpoint_sha256"] for r in result["arms"]}) == 1
    (ROOT/"results/ladder_temp15_gru92_diagnosis.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)


def main():
    gate = json.loads((ROOT / "results/ladder_temp15_gate.json").read_text())
    assert len(gate["checks"]) == 24
    annotations = {(c["arm"], c["step"]): None if gate_passes(c) else "archived global R² not reproduced on CPU" for c in gate["checks"]}
    for run in runs():
        output = RAW / run.parent.parent.name / "analysis.json"
        if output.exists():
            continue
        archived = json.loads((run / "condition_summary.json").read_text())["checkpoint_reports"]
        result = {"arm": run.parent.parent.name, "seed": 42, "stages": {}}
        init = None
        for stage, point in (("init", archived[0]), ("final", archived[-1])):
            step = point["checkpoint_step"]
            spec, process, params, provenance = restore(run, step)
            if init is None:
                init = params
            data = histories(run, step, spec, process, params)
            classes = [rows(d, "action", n) for d, n in zip(data, (60000, 80000))]
            metrics, predictions, y = analyze_encoder(spec, params, data, [d["embedding"] for d in data], classes)
            result["stages"][stage] = {"step": step, "provenance": provenance, "annotation": annotations[run.parent.parent.name, step], **metrics}
            if stage == "final":
                own_replay = np.asarray(replay(spec, params, data[1]["obs"]))
                error = float(np.max(np.abs(own_replay-data[1]["embedding"])))
                result["final_self_replay_max_abs"] = error
                matched_embeddings = [np.asarray(replay(spec, init, d["obs"])) for d in data]
                matched, matched_predictions, _ = analyze_encoder(spec, init, data, matched_embeddings, classes)
                result["matched_init"] = matched
                result["paired_differences"] = {}
                raw_predictions = {}
                for feature in predictions:
                    result["paired_differences"][feature] = {}
                    for target, sl in TARGETS.items():
                        p, q = predictions[feature][:, sl], matched_predictions[feature][:, sl]
                        valid = np.isfinite(p).all(1) & np.isfinite(q).all(1)
                        c, episodes = classes[1][valid], (np.arange(80000)//960)[valid]
                        difference = score(y[valid, sl], p[valid], c)["r2"]-score(y[valid, sl], q[valid], c)["r2"]
                        interval = bootstrap(y[valid, sl], p[valid], c, episodes, n=10000, comparison=q[valid])
                        result["paired_differences"][feature][target] = {"difference": difference, **interval}
                    raw_predictions[feature] = predictions[feature]
                    raw_predictions[feature+"_matched"] = matched_predictions[feature]
                np.savez_compressed(output.with_name("final_predictions.npz"), **raw_predictions)
                np.savez_compressed(output.with_name("matched_embeddings.npz"), fit=matched_embeddings[0], test=matched_embeddings[1])
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        print("DONE", result["arm"], "self-replay max abs", result["final_self_replay_max_abs"], flush=True)
        jax.clear_caches()


def intermediate():
    for run in runs():
        output = RAW / run.parent.parent.name / "intermediate.json"
        if output.exists():
            continue
        points = json.loads((run/"condition_summary.json").read_text())["checkpoint_reports"]
        point = min(points[1:-1], key=lambda p: abs(p["checkpoint_step"]-2_000_000_000))
        step = point["checkpoint_step"]
        spec, process, params, provenance = restore(run, step)
        data = histories(run, step, spec, process, params)
        classes = [rows(d, "action", n) for d, n in zip(data, (60000, 80000))]
        metrics, _, _ = analyze_encoder(spec, params, data, [d["embedding"] for d in data], classes)
        result = {"step": step, "provenance": provenance, "annotation": None,
                  "archive_check": "not part of required init/final protocol checks", **metrics}
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        print("MIDPOINT", run.parent.parent.name, step, flush=True)
        jax.clear_caches()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["analyze", "gru92_diagnosis", "intermediate"])
    args = parser.parse_args()
    {"gru92_diagnosis": gru92_diagnosis, "analyze": main, "intermediate": intermediate}[args.mode]()
