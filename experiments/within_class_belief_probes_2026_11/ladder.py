"""Frozen Belief-Quotient Ladder probes and matched-history initialization."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
import torch

from analysis.checkpoints import load_module_only
from harness.seeding import child_seed_sequence, named_seed_sequences, seed_sequence_to_int
from experiments.mess3_reward_state_action_symmetry_cycle_7.component_probe_campaign import all_checkpoint_records
from experiments.mess3_reward_state_action_symmetry_cycle_7.control_campaign import _download_module, _sha256, WorkItem, STUDY, CONDITIONS, SEEDS
from experiments.mess3_reward_state_action_symmetry_cycle_7.control_analysis import (
    HistoryData, collect_history_data, replay_beliefs, replay_activations,
    shuffled_histories, empirical_random_histories, _CONTROL_STREAMS,
)
from .gate import ROOT
from .metrics import affine, refit, score, null_basis

SCRATCH = ROOT / "artifacts" / "ladder"
REPO = ROOT.parents[1]
STREAMS = {"probe_train": (300,), "probe_test": (301,)}
NAMES = {1: "Trivial Quotient", 2: "Reward-State Quotient", 3: "Full-State Quotient"}


def items():
    result = []
    for variant, condition in enumerate(CONDITIONS, 1):
        for seed in SEEDS:
            run = f"mess3_reward_state_action_symmetry_cycle_6-{condition}-seed{seed}-10m"
            directory = REPO / STUDY / condition / "results" / run
            curve = directory / "checkpoint_probe_curve.json"
            for index, (stage, record) in enumerate(all_checkpoint_records(curve)):
                result.append(WorkItem(condition, variant, seed, stage, index,
                                       record.get("training_iteration"), int(record["agent_steps"]), run,
                                       record.get("checkpoint_name"), curve, directory / "run_manifest.json",
                                       SCRATCH / "modules" / condition / f"seed_{seed}" / f"checkpoint_{index}",
                                       SCRATCH / "reports" / condition / f"seed_{seed}" / f"checkpoint_{index}.json"))
    return result


def checked_module(item):
    download = _download_module(item)
    for name, metadata in download["module_files"].items():
        path = item.module_path / name
        assert path.stat().st_size == metadata["size_bytes"]
        assert _sha256(path) == metadata["sha256"]
    module = load_module_only(item.module_path).to("cpu").eval()
    download["source_run_git"] = json.loads(item.run_manifest_path.read_text())["git"]
    download["source_curve_sha256"] = _sha256(item.curve_path)
    return module, download


def histories(item, module):
    streams = named_seed_sequences(item.seed, STREAMS)
    result = []
    for split, n, stream in (("fit", 60000, "probe_train"), ("test", 80000, "probe_test")):
        cache = item.module_path.parent / f"history_{item.stage_index}_{split}.npz"
        if cache.exists():
            with np.load(cache) as arrays:
                metadata = json.loads(str(arrays["metadata"]))
                data = HistoryData(**{k: arrays[k] for k in HistoryData.__dataclass_fields__ if k != "metadata"}, metadata=metadata)
        else:
            data = collect_history_data(module, variant=item.variant, n_steps=n,
                                        seed=streams[stream], device=torch.device("cpu"), warmup=64)
            np.savez_compressed(cache, **{k: getattr(data, k) for k in HistoryData.__dataclass_fields__ if k != "metadata"}, metadata=json.dumps(dict(data.metadata)))
        assert int(data.mask.sum()) == n
        error = np.max(np.abs(replay_beliefs(data, variant=item.variant)-data.beliefs))
        assert error < 1e-10, error
        result.append(data)
    return result


@torch.inference_mode()
def policy_logits(module, h):
    values = []
    for start in range(0, len(h), 4096):
        values.append(module.action_distribution_inputs(torch.as_tensor(h[start:start+4096], dtype=torch.float32)).numpy())
    return np.concatenate(values).astype(float)


def gate_item(item):
    torch.set_num_threads(1)
    module, download = checked_module(item)
    train, test = histories(item, module)
    pred = affine(train.activations[train.mask], train.beliefs[train.mask], test.activations[test.mask])
    actual = 1-score(test.beliefs[test.mask], pred)["r2"]
    records = json.loads(item.curve_path.read_text())["checkpoints"]
    record = next(r for r in records if int(r["agent_steps"]) == item.agent_steps)
    expected = record["global_mse_ratio"]
    result = dict(condition=NAMES[item.variant], variant=item.variant, seed=item.seed,
                  stage=item.stage, agent_steps=item.agent_steps, archived=expected,
                  reproduced=actual, delta=actual-expected, checkpoint=download)
    print(f"GATE {NAMES[item.variant]} seed={item.seed} {item.stage} archived={expected:.12f} reproduced={actual:.12f} delta={actual-expected:+.3e}", flush=True)
    if abs(actual-expected) > 1e-6:
        raise ValueError(f"Full-belief reproduction gate failed: {result}")
    return result


def summary(y, pred, classes):
    valid = np.isfinite(pred).all(1)
    result = score(y[valid], pred[valid], classes[valid])
    result["n_undefined_predictions"] = int((~valid).sum())
    result["per_class"] = {str(k): score(y[(classes == k) & valid], pred[(classes == k) & valid]) for k in range(3)}
    return result


def metrics_for(module, train, test, fit_h, test_h, fit_classes, test_classes, seed):
    weight = module.heads.policy.weight.detach().numpy().astype(float)
    basis = null_basis(weight)
    logits = (policy_logits(module, fit_h), policy_logits(module, test_h))
    features = {"full_h": (fit_h, test_h), "head_blind": (fit_h @ basis, test_h @ basis), "logits": logits}
    targets = {"belief": (train.beliefs[train.mask], test.beliefs[test.mask])}
    targets["rho"] = tuple(b[:, 2:3] for b in targets["belief"])
    targets["delta"] = tuple(b[:, 0:1]-b[:, 1:2] for b in targets["belief"])
    output = {"fit_counts": np.bincount(fit_classes, minlength=3).tolist(),
              "test_counts": np.bincount(test_classes, minlength=3).tolist(), "head_null_dimension": basis.shape[1], "metrics": {}}
    for name, (fit, held) in features.items():
        output["metrics"][name] = {}
        for target, (yfit, ytest) in targets.items():
            pred = refit(fit, yfit, held, fit_classes, test_classes)
            output["metrics"][name][target] = summary(ytest, pred, test_classes)
    output["controls"] = {}
    streams = named_seed_sequences(seed, _CONTROL_STREAMS)
    for name, transform, stream in (("history_shuffle", shuffled_histories, "shuffle"), ("random_input", empirical_random_histories, "empirical_random")):
        changed = [transform(data, seed=seed_sequence_to_int(streams[stream] if i == 0 else child_seed_sequence(streams[stream], 1))) for i, data in enumerate((train, test))]
        values = [replay_activations(module, data, device=torch.device("cpu"))[data.mask] for data in changed]
        output["controls"][name] = {}
        for target, (yfit, ytest) in targets.items():
            pred = refit(values[0], yfit, values[1], fit_classes, test_classes)
            output["controls"][name][target] = summary(ytest, pred, test_classes)
    output["controls"]["label_permutation"] = {}
    for target, (yfit, ytest) in targets.items():
        pred = refit(fit_h, yfit, test_h, fit_classes, test_classes, permute_seed=seed+811)
        output["controls"]["label_permutation"][target] = summary(ytest, pred, test_classes)
    return output


def analyze_item(item):
    torch.set_num_threads(1)
    output = item.module_path.parent / f"within_{item.stage_index}.json"
    if output.exists():
        return str(output)
    module, download = checked_module(item)
    train, test = histories(item, module)
    fit_h, test_h = train.activations[train.mask], test.activations[test.mask]
    fit_classes, test_classes = policy_logits(module, fit_h).argmax(1), policy_logits(module, test_h).argmax(1)
    report = dict(condition=NAMES[item.variant], variant=item.variant, seed=item.seed, stage=item.stage,
                  agent_steps=item.agent_steps, run_id=item.run_id, checkpoint=download,
                  trained_or_own_init=metrics_for(module, train, test, fit_h, test_h, fit_classes, test_classes, item.seed))
    np.savez(item.module_path.parent / f"head_{item.stage_index}.npz", weight=module.heads.policy.weight.detach().numpy(), bias=module.heads.policy.bias.detach().numpy())
    if item.stage == "final":
        init_item = next(i for i in items() if i.variant == item.variant and i.seed == item.seed and i.stage == "init")
        initial, init_download = checked_module(init_item)
        same_fit, same_test = [replay_activations(initial, data, device=torch.device("cpu"))[data.mask] for data in (train, test)]
        np.savez_compressed(item.module_path.parent / "init_on_final_histories.npz", fit=same_fit, test=same_test,
                            fit_classes=fit_classes, test_classes=test_classes)
        report["init_encoder_same_rows"] = metrics_for(initial, train, test, same_fit, same_test, fit_classes, test_classes, item.seed)
        report["init_checkpoint"] = init_download
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(f"DONE {NAMES[item.variant]} seed={item.seed} {item.stage}", flush=True)
    return str(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("gate", "analyze"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--stages", nargs="+", default=["init", "final"])
    args = parser.parse_args()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    selected = [i for i in items() if i.stage in args.stages]
    if args.mode == "gate":
        selected = [i for i in items() if i.seed == 42 and i.stage in ("init", "iteration_4", "final")]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            checks = list(pool.map(gate_item, selected))
        (ROOT / "results" / "ladder_reproduction_gate.json").write_text(json.dumps({"streams": STREAMS, "checks": checks}, indent=2)+"\n")
    else:
        gate = json.loads((ROOT / "results" / "ladder_reproduction_gate.json").read_text())
        assert len(gate["checks"]) == 9 and all(abs(c["delta"]) <= 1e-6 for c in gate["checks"])
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(analyze_item, selected))


if __name__ == "__main__":
    main()
