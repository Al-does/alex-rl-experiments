"""Verified encoder-only warm starts for the guess-or-pass study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from devops.serverless.retrieve import retrieve_manifest_artifacts
from harness.cli import execute_experiment, load_experiment, make_run_context
from harness.storage.b2 import B2StorageConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.rl_module.rl_module import RLModule

import experiments.pusher_b.learning  # noqa: F401
from experiments.pusher_b10_discrete_SAC_2026_09.model import (
    SharedTrunkSACTorchLearner,
    SingleBackwardSACTorchLearner,
)

ROOT = Path(__file__).resolve().parents[3]
OVERRIDES_ENV = "PUSHER_B_CONTINUE_OVERRIDES"
MODULE_SUBDIR = Path("learner_group/learner/rl_module/default_policy")
SOURCE_CATALOG: dict[str, dict[str, Any]] = {}

for iteration, steps, probe in (
    (4, 1_081_024, 0.555),
    (8, 2_162_048, 0.444),
    (16, 4_324_096, 0.188),
    (32, 8_648_192, 0.084),
    (56, 15_134_336, 0.059),
):
    SOURCE_CATALOG[f"ppo_iter{iteration}"] = {
        "algorithm": "ppo",
        "architecture": None,
        "leaf": "experiments/pusher_b/rl_b10",
        "run_id": "20260913T221640Z-e0010176",
        "env_steps": steps,
        "probe_1_minus_r2": probe,
        "training_iteration": iteration,
        "checkpoint_index": "log_spaced_checkpoints/index.json",
        "pretraining_objective": "ppo",
    }

SOURCE_CATALOG["ppo_iter143"] = {
    "algorithm": "ppo",
    "architecture": None,
    "leaf": "experiments/pusher_b/rl_b10_continue",
    "run_id": "20260914T050618Z-88d0fb36",
    "env_steps": 115_901_216,
    "probe_1_minus_r2": 0.038,
    "checkpoint_index": None,
    "pretraining_objective": "ppo",
}

for label, iteration, steps, probe, checkpoint_index in (
    (
        "ppo_ce_iter4",
        4,
        1_137_920,
        0.509,
        "log_spaced_checkpoints/index.json",
    ),
    (
        "ppo_ce_iter8",
        8,
        2_275_840,
        0.1098,
        "log_spaced_checkpoints/index.json",
    ),
    (
        "ppo_ce_iter211",
        211,
        60_025_280,
        0.0396,
        "interval_checkpoints/index.json",
    ),
):
    SOURCE_CATALOG[label] = {
        "algorithm": "ppo",
        "architecture": None,
        "leaf": "experiments/pusher_b/rl_b10_aux_ce",
        "run_id": "20260915T033547Z-913cef7f",
        "env_steps": steps,
        "probe_1_minus_r2": probe,
        "greedy_accuracy": 0.6373 if iteration == 211 else 0.6375,
        "training_iteration": iteration,
        "checkpoint_index": checkpoint_index,
        "pretraining_objective": "ppo+ntp_ce",
    }

for architecture, run_id, points in (
    (
        "split_transformers",
        "pusher_b10_discrete_SAC-split_transformers-seed42-300m",
        ((32, 455_168, 0.396), (64, 910_336, 0.124), (128, 1_820_672, 0.062), (512, 7_282_688, 0.058)),
    ),
    (
        "shared_trunk",
        "pusher_b10_discrete_SAC-shared_trunk-seed42-300m",
        ((32, 455_168, 0.522), (64, 910_336, 0.167), (512, 7_282_688, 0.055)),
    ),
):
    for iteration, steps, probe in points:
        SOURCE_CATALOG[f"sac_{'split' if architecture == 'split_transformers' else 'shared'}_iter{iteration}"] = {
            "algorithm": "sac",
            "architecture": architecture,
            "leaf": f"experiments/pusher_b10_discrete_SAC_2026_09/{architecture}",
            "run_id": run_id,
            "env_steps": steps,
            "probe_1_minus_r2": probe,
            "training_iteration": iteration,
            "checkpoint_index": "scheduled_checkpoints/index.json",
            "pretraining_objective": "sac",
        }


class FrozenEncoderPPOTorchLearner(PPOTorchLearner):
    """Exclude the shared PPO encoder from the optimizer parameter set."""

    def configure_optimizers_for_module(self, module_id, config=None) -> None:
        module = self.module[module_id]
        freeze_module_encoders(module, "ppo")
        parameters = [
            parameter
            for parameter in self.get_parameters(module)
            if parameter.requires_grad
        ]
        if not parameters:
            raise ValueError("frozen PPO encoder left no trainable parameters")
        optimizer = torch.optim.Adam(parameters)
        self.register_optimizer(
            module_id=module_id,
            optimizer=optimizer,
            params=parameters,
            lr_or_lr_schedule=config.lr,
        )


class _FrozenSACEncoderMixin:
    def get_parameters(self, module) -> list[torch.Tensor]:
        return [
            parameter
            for parameter in super().get_parameters(module)
            if parameter.requires_grad
        ]

    def configure_optimizers_for_module(self, module_id, config=None) -> None:
        module = self.module[module_id]
        freeze_module_encoders(module, "sac")
        super().configure_optimizers_for_module(module_id, config)


class FrozenEncoderSplitSACTorchLearner(
    _FrozenSACEncoderMixin,
    SingleBackwardSACTorchLearner,
):
    """Use the source split-SAC optimizers without encoder parameters."""


class FrozenEncoderSharedTrunkSACTorchLearner(
    _FrozenSACEncoderMixin,
    SharedTrunkSACTorchLearner,
):
    """Optimize fresh SAC heads while excluding the shared encoder."""

    def configure_optimizers_for_module(self, module_id, config=None) -> None:
        if config is None:
            raise ValueError("shared-trunk SAC requires an algorithm config")
        module = self.module[module_id]
        freeze_module_encoders(module, "sac")
        optimizer_specs = (
            ("policy", self.get_parameters(module.pi), config.actor_lr),
            ("qf", self.get_parameters(module.qf), config.critic_lr),
            ("qf_twin", self.get_parameters(module.qf_twin), config.critic_lr),
        )
        for name, parameters, learning_rate in optimizer_specs:
            optimizer = torch.optim.Adam(parameters, eps=1e-7)
            self.register_optimizer(
                module_id=module_id,
                optimizer_name=name,
                optimizer=optimizer,
                params=parameters,
                lr_or_lr_schedule=learning_rate,
            )
        temperature = self.curr_log_alpha[module_id]
        optimizer = torch.optim.Adam([temperature], eps=1e-7)
        self.register_optimizer(
            module_id=module_id,
            optimizer_name="alpha",
            optimizer=optimizer,
            params=[temperature],
            lr_or_lr_schedule=config.alpha_lr,
        )


def transfer_overrides() -> dict[str, object] | None:
    value = os.environ.get(OVERRIDES_ENV)
    if not value:
        return None
    metadata = json.loads(value)
    if not isinstance(metadata, dict):
        raise TypeError(f"{OVERRIDES_ENV} must contain a JSON object")
    required = {
        "source",
        "run_id",
        "checkpoint_path",
        "source_env_steps",
        "source_probe_1_minus_r2",
        "trunk_mode",
        "architecture",
        "pretraining_objective",
    }
    missing = required.difference(metadata)
    if missing:
        raise ValueError(f"transfer metadata is missing {sorted(missing)}")
    return metadata


def _encoder_prefixes(algorithm_name: str) -> tuple[str, ...]:
    if algorithm_name == "ppo":
        return ("encoder.",)
    if algorithm_name == "sac":
        return ("pi_encoder.", "qf_encoder.", "qf_twin_encoder.")
    raise ValueError(f"unsupported transfer algorithm: {algorithm_name}")


def freeze_module_encoders(module, algorithm_name: str) -> None:
    names = (
        ("encoder",)
        if algorithm_name == "ppo"
        else ("pi_encoder", "qf_encoder", "qf_twin_encoder")
        if algorithm_name == "sac"
        else ()
    )
    if not names:
        raise ValueError(f"unsupported transfer algorithm: {algorithm_name}")
    for name in names:
        encoder = getattr(module, name, None)
        if encoder is None:
            raise ValueError(f"module is missing the {name} encoder")
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)


def _reset_sac_targets(learner) -> None:
    module = learner.module["default_policy"]
    module.make_target_networks()


def initialize_algorithm_transfer(
    algorithm,
    *,
    module_checkpoint: str | Path,
    algorithm_name: str,
    trunk_mode: str,
    metadata: dict[str, object],
    recipe_path: str | Path,
) -> dict[str, object]:
    if trunk_mode not in {"full", "frozen"}:
        raise ValueError("trunk mode must be 'full' or 'frozen'")
    source_module = RLModule.from_checkpoint(str(module_checkpoint))
    source_state = source_module.get_state()
    weights = algorithm.learner_group.get_weights()
    if "default_policy" not in weights:
        raise RuntimeError("learner does not contain the default policy module")
    target_state = weights["default_policy"]
    if algorithm_name == "ppo" and any(
        "next_token" in key.lower() or "aux" in key.lower()
        for key in target_state
    ):
        raise RuntimeError("target PPO module unexpectedly contains an auxiliary head")
    prefixes = _encoder_prefixes(algorithm_name)
    encoder_keys = [
        key for key in target_state if key.startswith(prefixes)
    ]
    if not encoder_keys:
        raise RuntimeError(f"no {algorithm_name} encoder parameters were found")
    for key in encoder_keys:
        if key not in source_state:
            raise ValueError(f"source module is missing encoder parameter {key}")
        source_value = np.asarray(source_state[key])
        target_value = np.asarray(target_state[key])
        if source_value.shape != target_value.shape:
            raise ValueError(
                f"encoder shape mismatch for {key}: "
                f"{source_value.shape} != {target_value.shape}"
            )
        target_state[key] = source_value.copy()
    weights["default_policy"] = target_state
    algorithm.learner_group.set_weights(weights)
    if algorithm_name == "sac":
        algorithm.learner_group.foreach_learner(_reset_sac_targets)
    algorithm.env_runner_group.sync_weights(
        from_worker_or_learner_group=algorithm.learner_group,
        inference_only=True,
    )

    expected = {
        key: np.asarray(source_state[key])
        for key in encoder_keys
    }
    learner_state = algorithm.learner_group.get_weights()["default_policy"]
    runner_state = algorithm.get_module().get_state()
    for key, value in expected.items():
        if not np.array_equal(np.asarray(learner_state[key]), value):
            raise RuntimeError(f"learner encoder verification failed for {key}")
    runner_encoder_keys = [key for key in expected if key in runner_state]
    if not runner_encoder_keys:
        raise RuntimeError("env-runner module has no encoder parameters")
    for key in runner_encoder_keys:
        if not np.array_equal(np.asarray(runner_state[key]), expected[key]):
            raise RuntimeError(f"env-runner encoder verification failed for {key}")

    report = {
        **metadata,
        "encoder_weights_copied": True,
        "encoder_weight_keys": encoder_keys,
        "env_runner_encoder_weight_keys": runner_encoder_keys,
        "encoder_weights_verified_on_learner_and_env_runner": True,
    }
    path = Path(recipe_path)
    if path.is_file():
        recipe = json.loads(path.read_text())
        recipe["transfer"] = report
        path.write_text(json.dumps(recipe, indent=2) + "\n")
    if trunk_mode == "frozen":
        head_keys = [key for key in target_state if key not in encoder_keys]
        state_path = path.with_name("frozen_transfer_state.json")
        state_path.write_text(
            json.dumps(
                {
                    "encoder_digest": _state_digest(target_state, encoder_keys),
                    "head_digest": _state_digest(target_state, head_keys),
                    "encoder_keys": encoder_keys,
                    "head_keys": head_keys,
                    "verified": False,
                },
                indent=2,
            )
            + "\n"
        )
    print(
        f"TRANSFER ENCODER CHECK source={metadata['source']} "
        f"trunk={trunk_mode} keys={len(encoder_keys)} exact=True",
        flush=True,
    )
    return report


def _state_digest(state: dict[str, Any], keys: list[str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        value = np.asarray(state[key])
        digest.update(key.encode())
        digest.update(value.dtype.str.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def verify_frozen_update(*, algorithm, recipe_path: str, **kwargs) -> None:
    del kwargs
    recipe_file = Path(recipe_path)
    state_path = recipe_file.with_name("frozen_transfer_state.json")
    if not state_path.is_file():
        return
    before = json.loads(state_path.read_text())
    if before.get("verified"):
        return
    state = algorithm.learner_group.get_weights()["default_policy"]
    encoder_keys = list(before["encoder_keys"])
    head_keys = list(before["head_keys"])
    encoder_digest = _state_digest(state, encoder_keys)
    head_digest = _state_digest(state, head_keys)
    if encoder_digest != before["encoder_digest"]:
        raise RuntimeError("frozen transfer encoder changed during its first update")
    if head_digest == before["head_digest"]:
        raise RuntimeError("fresh transfer heads did not change during their first update")
    before.update(
        {
            "verified": True,
            "encoder_digest_after_update": encoder_digest,
            "head_digest_after_update": head_digest,
        }
    )
    state_path.write_text(json.dumps(before, indent=2) + "\n")
    if recipe_file.is_file():
        recipe = json.loads(recipe_file.read_text())
        recipe["transfer"]["frozen_encoder_update_verified"] = True
        recipe["transfer"]["fresh_heads_changed_after_update"] = True
        recipe_file.write_text(json.dumps(recipe, indent=2) + "\n")
    print(
        "TRANSFER FROZEN CHECK encoder_bit_identical=True "
        "fresh_heads_changed=True",
        flush=True,
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_verified_json(client, manifest: dict[str, Any], relative_path: str) -> dict[str, Any]:
    rows = [
        row
        for row in manifest.get("files", [])
        if row.get("relative_path") == relative_path
    ]
    if not rows:
        raise FileNotFoundError(f"missing {relative_path} in B2 durability manifest")
    row = rows[-1]
    payload = client.get_object(
        Bucket=manifest["bucket"],
        Key=row["key"],
    )["Body"].read()
    if _sha256(payload) != str(row["sha256"]).lower():
        raise ValueError(f"SHA-256 mismatch for {relative_path}")
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise TypeError(f"{relative_path} must contain a JSON object")
    return decoded


def _checkpoint_relative_path(
    source: dict[str, Any],
    *,
    spec: dict[str, Any],
    manifest: dict[str, Any],
    client,
) -> str:
    index_path = spec["checkpoint_index"]
    if index_path is None:
        summary_path = ROOT / spec["leaf"] / "results" / spec["run_id"] / "tune_summary.json"
        trials = json.loads(summary_path.read_text()).get("trials") or []
        if len(trials) != 1:
            raise RuntimeError(f"expected one final PPO checkpoint in {summary_path}")
        checkpoint = str(trials[0].get("checkpoint") or "")
        marker = f"/artifacts/{spec['run_id']}/"
        if marker not in checkpoint:
            raise ValueError("final PPO checkpoint path does not match its run id")
        return checkpoint.split(marker, 1)[1].rstrip("/")

    index = _read_verified_json(client, manifest, index_path)
    records = index.get("checkpoints")
    if not isinstance(records, list):
        raise TypeError(f"invalid checkpoint index: {index_path}")
    selected = []
    for record in records:
        if not isinstance(record, dict):
            continue
        iteration = record.get("training_iteration")
        steps = record.get("env_steps", record.get("agent_steps"))
        if (
            iteration is not None
            and steps is not None
            and int(iteration) == int(spec["training_iteration"])
            and int(float(steps)) == int(spec["env_steps"])
        ):
            selected.append(record)
    if len(selected) != 1:
        raise RuntimeError(
            f"expected one checkpoint at {spec['env_steps']} steps, "
            f"found {len(selected)} in {index_path}"
        )
    record = selected[0]
    run_id = spec["run_id"]
    checkpoint = str(record.get("path") or "")
    marker = f"/artifacts/{run_id}/"
    if marker in checkpoint:
        relative = checkpoint.split(marker, 1)[1].rstrip("/")
    else:
        relative = str(record.get("checkpoint_name") or "")
        if not relative:
            raise ValueError(f"checkpoint index entry is missing a path: {record}")
        if "/" not in relative:
            relative = f"{Path(index_path).parent.as_posix()}/{relative}"
    return relative


def _module_directory(root: Path) -> Path:
    if root.name == "default_policy" and (root / "metadata.json").is_file():
        return root
    candidate = root / MODULE_SUBDIR
    if candidate.is_dir():
        return candidate
    matches = list(root.rglob("default_policy"))
    matches = [path for path in matches if (path / "metadata.json").is_file()]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"could not resolve RLModule directory under {root}")


def fetch_source_module(
    *,
    source_label: str,
    destination: Path,
) -> tuple[Path, str]:
    spec = SOURCE_CATALOG[source_label]
    run_manifest_path = (
        ROOT / spec["leaf"] / "results" / spec["run_id"] / "run_manifest.json"
    )
    run_manifest = json.loads(run_manifest_path.read_text())
    remote = run_manifest.get("remote_artifacts") or {}
    if remote.get("status") != "completed":
        raise RuntimeError(f"B2 upload is incomplete for {source_label}")
    config = B2StorageConfig.from_env()
    if config is None:
        raise RuntimeError("B2 credentials are not configured")
    client = config.s3_client()
    manifest = json.loads(
        client.get_object(
            Bucket=remote["bucket"],
            Key=remote["canonical_manifest_key"],
        )["Body"].read()
    )
    if manifest.get("status") != "completed":
        raise RuntimeError("source durability manifest is not completed")
    relative = _checkpoint_relative_path(
        SOURCE_CATALOG[source_label],
        spec=spec,
        manifest=manifest,
        client=client,
    )
    subtree = (Path(relative) / MODULE_SUBDIR).as_posix()
    module_rows = [
        row
        for row in manifest.get("files", [])
        if str(row.get("relative_path", "")).startswith(subtree + "/")
    ]
    if not module_rows:
        raise FileNotFoundError(
            f"no RLModule files under {subtree!r} in the B2 manifest"
        )
    retrieve_manifest_artifacts(
        {**manifest, "files": module_rows},
        destination,
        client=client,
        config=config,
    )
    module_dir = _module_directory(destination / relative)
    return module_dir, relative


def _local_source_module(source_dir: Path) -> Path:
    source_dir = source_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source_dir}")
    return _module_directory(source_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_module")
    parser.add_argument("--source", choices=sorted(SOURCE_CATALOG), required=True)
    parser.add_argument("--trunk", choices=("full", "frozen"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--hardware-profile", "--hardware", default="auto")
    parser.add_argument(
        "--upload-artifacts",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument(
        "--architecture",
        choices=("split_transformers", "shared_trunk"),
        help="SAC architecture; defaults to the source checkpoint architecture",
    )
    args = parser.parse_args(argv)
    if "." not in args.experiment_module:
        args.experiment_module = (
            "experiments.pusher_b10_checkpoint_transfer_2026_10.guess_or_pass_rllib."
            f"{args.experiment_module}.experiment"
        )

    spec = SOURCE_CATALOG[args.source]
    parts = args.experiment_module.split(".")
    leaf = parts[-2] if parts[-1] == "experiment" else parts[-1]
    algorithm_name = "ppo" if leaf.startswith("ppo_") else "sac" if leaf.startswith("sac_") else None
    if algorithm_name != spec["algorithm"]:
        raise ValueError(
            f"source {args.source} uses {spec['algorithm']}, "
            f"but {args.experiment_module} is not a matching leaf"
        )
    architecture = spec["architecture"]
    if algorithm_name == "sac":
        if args.architecture is not None and args.architecture != architecture:
            raise ValueError(
                f"source {args.source} uses {architecture}; "
                f"requested architecture {args.architecture} is incompatible"
            )
        architecture = args.architecture or architecture

    experiment = load_experiment(args.experiment_module)
    if args.source_dir is not None:
        module_dir = _local_source_module(args.source_dir)
        checkpoint_path = str(args.source_dir.expanduser().resolve())
    else:
        module_dir, relative = fetch_source_module(
            source_label=args.source,
            destination=experiment.directory / "artifacts" / "sources" / args.source,
        )
        checkpoint_path = relative
    metadata: dict[str, object] = {
        "source": args.source,
        "run_id": spec["run_id"],
        "checkpoint_path": checkpoint_path,
        "source_module_directory": str(module_dir),
        "source_env_steps": spec["env_steps"],
        "source_probe_1_minus_r2": spec["probe_1_minus_r2"],
        "pretraining_objective": spec["pretraining_objective"],
        "trunk_mode": args.trunk,
        "architecture": architecture,
    }
    os.environ[OVERRIDES_ENV] = json.dumps(metadata)
    context = make_run_context(
        experiment,
        seed=args.seed,
        run_id=args.run_id,
        smoke=args.smoke,
        resume_from=module_dir,
        hardware_profile=args.hardware_profile,
    )
    execute_experiment(
        experiment,
        context,
        command=[
            "python",
            "-m",
            "experiments.pusher_b10_checkpoint_transfer_2026_10.guess_or_pass_rllib.transfer",
            args.experiment_module,
            "--source",
            args.source,
            "--trunk",
            args.trunk,
            "--seed",
            str(args.seed),
            *(["--run-id", args.run_id] if args.run_id is not None else []),
        ],
        runtime_overrides={
            **metadata,
            "run_id": context.run_id,
            "source_dir": str(args.source_dir) if args.source_dir else None,
            "resume_from": str(module_dir),
            OVERRIDES_ENV: os.environ[OVERRIDES_ENV],
        },
        upload_artifacts=args.upload_artifacts,
    )


if __name__ == "__main__":
    main()
