"""Verified paper checkpoints and complete, stochastic-policy episodes."""
from functools import partial
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.belief_quotient_history_jax_2026_11 import models
from experiments.belief_quotient_history_jax_2026_11.continuation import load_params, locate_parent_run
from experiments.belief_quotient_history_jax_2026_11.env import make_process
from experiments.belief_quotient_history_jax_2026_11.reinforce import rollout
from harness.storage.b2 import B2StorageConfig

LADDER = "belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11"
RR = "reward_factors_persistent_nuisance_eps02_jax_temp15_ctx32_2026_11"
ROOT = Path(__file__).resolve().parents[2]


def restore(study, condition, seed, cache):
    leaf = ROOT / study / (condition + "_transformer")
    parent = locate_parent_run(leaf, study=study, seed=seed)
    directory = leaf / "results" / parent["run_id"]
    recipe = json.loads((directory / "resolved_recipe.json").read_text())
    manifest = json.loads((directory / "run_manifest.json").read_text())
    remote = manifest["remote_artifacts"]
    config = B2StorageConfig.from_env()
    if config is None:
        raise RuntimeError("B2 credentials required for checkpoint integrity verification")
    client = config.s3_client()
    raw = client.get_object(Bucket=remote["bucket"], Key=remote["canonical_manifest_key"])["Body"].read()
    row, = [r for r in json.loads(raw)["files"] if r["relative_path"] == parent["checkpoint_file"]]
    path = Path(cache) / parent["run_id"] / Path(parent["checkpoint_file"]).name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        client.download_file(remote["bucket"], row["key"], str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != row["sha256"] or path.stat().st_size != row.get("size_bytes", row.get("size")):
        raise ValueError("checkpoint durability verification failed")
    spec = models.ModelSpec(**recipe["model"])
    provenance = {**parent, "sha256": digest, "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                  "results_path": str(directory.relative_to(ROOT.parent)), "training_git": manifest["git"]}
    return spec, make_process(**recipe["process"]), load_params(path, spec), recipe, provenance


@partial(jax.jit, static_argnums=(0, 4))
def collect_batch(spec, process, params, key, count):
    records = rollout(spec, process, params, key, count, 1024,
                      temperature=1.5, epsilon=0., track_belief=True)
    embedding = records["embedding"]
    logits = embedding @ params["policy"]["w"] + params["policy"]["b"]
    obs, b, action = records["obs"], records["belief"], records["action"]
    token = obs[..., :process.n_tokens].argmax(-1)
    previous = obs[..., process.n_tokens:].argmax(-1)
    predictive = jnp.einsum("bti,btij,jy->bty", b, process.transitions[action], process.emission)
    all_predictive = jnp.einsum("bti,aij,jy->btay", b, process.transitions, process.emission)
    reset = jnp.broadcast_to(jnp.arange(1024)[None, :] == 0, token.shape)
    return {"x": previous*process.n_tokens+token, "token": token, "reset": reset,
            "previous_action": jnp.where(reset, -1, previous), "action": action, "belief": b,
            "policy": jax.nn.softmax(logits/1.5, -1), "predictive": predictive,
            "all_predictive": all_predictive}


def exact_targets(records, process):
    """Re-filter complete histories in float64, independently of GPU matmuls."""
    source = process if isinstance(process, dict) else process._asdict()
    initial, T, E = (np.asarray(source[key], dtype=np.float64) for key in ("initial", "transitions", "emission"))
    initial = initial/initial.sum()
    T, E = T/T.sum(-1, keepdims=True), E/E.sum(-1, keepdims=True)
    tokens, previous = records["token"], records["previous_action"]
    b = np.broadcast_to(initial, (len(tokens), len(initial))).copy()
    beliefs = np.empty((*tokens.shape, len(initial)), dtype=np.float64)
    predictions = np.empty((*tokens.shape, len(T), E.shape[-1]), dtype=np.float64)
    for t in range(tokens.shape[1]):
        if t:
            b = np.einsum("bi,bij->bj", b, T[previous[:, t]])
        b *= E[:, tokens[:, t]].T
        b /= b.sum(-1, keepdims=True)
        beliefs[:, t] = b
        predictive = np.einsum("bi,aij,jy->bay", b, T, E)
        predictions[:, t] = predictive/predictive.sum(-1, keepdims=True)
    selected = np.take_along_axis(predictions, records["action"][..., None, None], axis=2)[..., 0, :]
    return {"belief": beliefs, "all_predictive": predictions, "predictive": selected}


def collect(spec, process, params, *, episodes, seed, batch=32, truth_process=None):
    if episodes < 1 or batch < 1:
        raise ValueError("positive episode count and rollout batch required")
    chunks = []
    for i in range(0, episodes, batch):
        key = jax.random.fold_in(jax.random.PRNGKey(seed), i)
        records = jax.tree.map(np.asarray, collect_batch(spec, process, params, key, min(batch, episodes-i)))
        records.update(exact_targets(records, process if truth_process is None else truth_process))
        chunks.append(records)
    return {name: np.concatenate([c[name] for c in chunks]) for name in chunks[0]}


def quotient_model(process):
    """Reward-State Quotient matrices for merged A/B observations."""
    T, E = np.asarray(process.transitions), np.asarray(process.emission)
    T = np.stack((T[:, [0, 2], :2].sum(-1), T[:, [0, 2], 2]), -1)
    E = np.stack((E[[0, 2], :2].sum(-1), E[[0, 2], 2]), -1)
    return np.array([2/3, 1/3]), T, E


def truth_targets(data, process, ladder):
    b = data["belief"]
    if not ladder:
        joint = b.reshape(*b.shape[:-1], 3, 3)
        return {"belief": b, "factor_1": joint.sum(-1), "factor_2": joint.sum(-2)}
    # The same exact Reward-State model is the comparison target in every arm.
    from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.environment import ladder_process
    prior, T, E = quotient_model(ladder_process(2))
    c = np.broadcast_to(prior, (len(b), 2)).copy()
    cs = []
    for t in range(b.shape[1]):
        if t:
            c = np.einsum("bi,bij->bj", c, T[data["previous_action"][:, t]])
        c *= E[:, (data["token"][:, t] == 2).astype(int)].T
        c /= c.sum(-1, keepdims=True)
        cs.append(c.copy())
    return {"belief": b, "quotient": np.stack(cs, 1)}
