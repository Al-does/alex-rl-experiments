"""Hash-verified, module-only reproduction of the archived passive probes."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import boto3
import numpy as np
import torch

from analysis.checkpoints import load_module_only
from analysis.probes import fit_affine_probe, probe_predict, r2_score
from envs.hmm import HMMEnv
from experiments.mess3_belief_geometry_2026_07.probe import collect_probe_data, make_transducer_target
from experiments.mess3_token_guess_cycle_2.analysis import _STREAM_KEYS
from experiments.mess3_token_guess_cycle_2.shared import ENV_CONFIG
from harness.seeding import named_seed_sequences

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "mess3_token_guess_cycle_2"
RUNS = {
    "ppo": "mess3_token_guess_cycle_2-ppo-seed45-2p5m",
    "supervised_ce": "mess3_token_guess_cycle_2-supervised_ce-seed45-2p5m-ppomatch",
}


def collect(module, n, seed):
    original = module.encode_step
    windows, logits, clusters = [], [], []
    steps = np.zeros(16, dtype=int)
    episodes = np.full(16, -1, dtype=int)

    def recording(obs, state):
        reset = state["len"].numpy().reshape(-1) == 0
        steps[reset] = 0
        episodes[reset] += 1
        h, next_state = original(obs, state)
        keep = steps >= 64
        window = torch.cat([state["ctx"], obs[:, None]], dim=1).numpy()
        windows.extend(window[keep])
        logits.extend(module.action_distribution_inputs(h).detach().numpy()[keep])
        clusters.extend((episodes * 16 + np.arange(16))[keep])
        steps[:] += 1
        return h, next_state

    config = {**ENV_CONFIG, "diagnostics": {"state": True, "belief": True, "tokens": True, "transitions": True}}
    env = HMMEnv(config)
    initial, operator, initial_operator = make_transducer_target(env)
    env.close()
    module.encode_step = recording
    try:
        data = collect_probe_data(module, lambda: HMMEnv(config), n_steps=n, seed=seed,
                                  policy_mode="greedy", n_envs=16, device="cpu", warmup=64,
                                  initial_belief=initial, action_outcome_operator=operator,
                                  initial_outcome_operator=initial_operator)
    finally:
        module.encode_step = original
    assert np.max(np.abs(data.beliefs - data.diagnostic_beliefs)) < 1e-10
    return dict(h=data.activations, b=data.beliefs, logits=np.asarray(logits)[:n],
                action=data.actions, episode=np.asarray(clusters)[:n],
                window=np.asarray(windows)[:n], episode_step=data.episode_steps,
                env=data.env_indices)


def main():
    torch.set_num_threads(2)
    client = boto3.client("s3", endpoint_url=os.environ["B2_ENDPOINT"],
                          aws_access_key_id=os.environ["B2_APPLICATION_KEY_ID"],
                          aws_secret_access_key=os.environ["B2_APPLICATION_KEY"])
    report = {"protocol": "archived greedy 16-env 60000/80000, warmup64, ridge1e-6, seed45 streams200/201",
              "experiment_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "harness_commit": subprocess.check_output(["git", "-C", "/home/ubuntu/repos/rl-harness", "rev-parse", "HEAD"], text=True).strip(),
              "checks": []}
    for arm, run in RUNS.items():
        source = SOURCE / arm / "results" / run
        remote = json.loads((source / "run_manifest.json").read_text())["remote_artifacts"]
        raw = client.get_object(Bucket=remote["bucket"], Key=remote["canonical_manifest_key"])["Body"].read()
        manifest = json.loads(raw)
        base = ROOT / "artifacts" / run
        base.mkdir(parents=True, exist_ok=True)
        (base / "durability_manifest.json").write_bytes(raw)
        points = json.loads((source / "checkpoint_probe_curve.json").read_text())["checkpoints"]
        for point in points:
            name = point.get("checkpoint_name", "initial_checkpoint")
            entries = [f for f in manifest["files"] if "/learner_group/learner/rl_module/default_policy/" in f["relative_path"] and name in Path(f["relative_path"]).parts]
            assert len(entries) == 3, (name, len(entries))
            hashes = []
            for entry in entries:
                path = base / entry["relative_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    client.download_file(remote["bucket"], entry["key"], str(path))
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                assert digest == entry["sha256"] and path.stat().st_size == entry["size_bytes"]
                hashes.append({"path": entry["relative_path"], "sha256": digest})
            module = load_module_only(path.parent).to("cpu").eval()
            streams = named_seed_sequences(45, _STREAM_KEYS)
            rows = []
            for split, n, stream in [("fit", 60000, "probe_train"), ("test", 80000, "probe_test")]:
                cache = base / f"{point['agent_steps']}_{split}.npz"
                if cache.exists():
                    data = dict(np.load(cache))
                else:
                    data = collect(module, n, streams[stream])
                    np.savez_compressed(cache, **data)
                rows.append(data)
            train, test = rows
            w, bias = fit_affine_probe(train["h"], train["b"], ridge=1e-6)
            actual = float(r2_score(probe_predict(w, bias, test["h"]), test["b"]))
            expected = point["r_squared"]
            check = dict(arm=arm, steps=point["agent_steps"], archived=expected,
                         reproduced=actual, delta=actual-expected, hashes=hashes,
                         n_fit=len(train["b"]), n_test=len(test["b"]))
            report["checks"].append(check)
            (ROOT / "results" / "reproduction_gate.json").write_text(json.dumps(report, indent=2)+"\n")
            print(f"{arm} {point['agent_steps']} archived={expected:.12f} reproduced={actual:.12f} delta={actual-expected:+.3e}", flush=True)
            if abs(actual-expected) > 1e-6:
                raise RuntimeError("Reproduction gate failed; no downstream analyses permitted")


if __name__ == "__main__":
    main()
