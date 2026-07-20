"""Compare SGD and Muon parameter and representation movement from initialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from io import BytesIO
from pathlib import Path
from typing import Any

import torch

from harness.storage.b2 import B2StorageConfig

from .paper_supervised_replication.mess3 import enumerate_paths
from .paper_supervised_replication.model import PaperModelConfig, PaperTransformer


_STEPS = (
    0,
    5_000,
    10_000,
    15_000,
    20_000,
    25_000,
    30_000,
    35_000,
    40_000,
    45_000,
    50_000,
    55_000,
    60_000,
    61_446,
    62_500,
)
_DEFAULT_SGD_PREFIX = (
    "alex/experiments/mess3_belief_geometry_2026_07/"
    "large_batch_replication/20260719-large-batch-sgd-trajectory-seed0/"
    "checkpoints"
)
_DEFAULT_MUON_PREFIX = (
    "alex/experiments/mess3_belief_geometry_2026_07/"
    "muon_large_batch_replication/20260719-large-batch-muon-seed0-v2/"
    "checkpoints"
)
_R2_PATH = (
    Path(__file__).parent
    / "muon_large_batch_replication"
    / "results"
    / "20260719-sgd-muon-r2-comparison-v2"
    / "checkpoint_r2_comparison.json"
)


def _load_state(
    *,
    client: Any,
    bucket: str,
    prefix: str,
    step: int,
) -> dict[str, torch.Tensor]:
    key = f"{prefix}/step_{step:07d}.pt"
    payload = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    checkpoint = torch.load(
        BytesIO(payload),
        map_location="cpu",
        weights_only=False,
    )
    return {
        name: tensor.detach().float()
        for name, tensor in checkpoint["model_state"].items()
    }


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def _block3_features(
    state: dict[str, torch.Tensor],
    contexts: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    model = PaperTransformer(PaperModelConfig()).eval()
    model.load_state_dict(state)
    chunks = []
    for start in range(0, len(contexts), batch_size):
        _, activations = model(
            contexts[start : start + batch_size],
            return_activations=True,
        )
        chunks.append(activations["block_3"].reshape(-1, 64))
    return torch.cat(chunks).float()


@torch.no_grad()
def _linear_cka(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left - left.mean(dim=0, keepdim=True)
    right = right - right.mean(dim=0, keepdim=True)
    cross = left.T @ right
    left_covariance = left.T @ left
    right_covariance = right.T @ right
    denominator = (
        left_covariance.square().sum().sqrt()
        * right_covariance.square().sum().sqrt()
    )
    return float(cross.square().sum() / denominator)


@torch.no_grad()
def _centered_relative_feature_drift(
    initial: torch.Tensor,
    current: torch.Tensor,
) -> float:
    initial = initial - initial.mean(dim=0, keepdim=True)
    current = current - current.mean(dim=0, keepdim=True)
    return float(
        torch.linalg.vector_norm(current - initial)
        / torch.linalg.vector_norm(initial)
    )


def _parameter_group(name: str) -> str:
    if "token_embedding" in name or "position_embedding" in name:
        return "embeddings"
    if ".attention." in name:
        return "attention"
    if ".mlp." in name:
        return "mlp"
    if "norm" in name:
        return "layer_norm"
    if "unembedding" in name:
        return "unembedding"
    return "other"


def _aggregate_parameter_metrics(
    initial: dict[str, torch.Tensor],
    current: dict[str, torch.Tensor],
    names: list[str],
) -> dict[str, float | None]:
    initial_squared_norm = sum(
        float(initial[name].double().square().sum()) for name in names
    )
    current_squared_norm = sum(
        float(current[name].double().square().sum()) for name in names
    )
    delta_squared_norm = sum(
        float((current[name] - initial[name]).double().square().sum())
        for name in names
    )
    dot_product = sum(
        float((current[name].double() * initial[name].double()).sum())
        for name in names
    )
    denominator = math.sqrt(initial_squared_norm * current_squared_norm)
    return {
        "delta_l2": math.sqrt(delta_squared_norm),
        "relative_displacement": (
            math.sqrt(delta_squared_norm / initial_squared_norm)
            if initial_squared_norm
            else None
        ),
        "norm_ratio": (
            math.sqrt(current_squared_norm / initial_squared_norm)
            if initial_squared_norm
            else None
        ),
        "cosine_to_initialization": (
            dot_product / denominator if denominator else None
        ),
    }


def _parameter_metrics(
    initial: dict[str, torch.Tensor],
    current: dict[str, torch.Tensor],
) -> dict[str, Any]:
    names = list(initial)
    matrix_names = [name for name in names if initial[name].ndim == 2]
    auxiliary_names = [name for name in names if initial[name].ndim != 2]
    groups = sorted({_parameter_group(name) for name in names})
    return {
        "all": _aggregate_parameter_metrics(initial, current, names),
        "matrix_2d": _aggregate_parameter_metrics(
            initial,
            current,
            matrix_names,
        ),
        "auxiliary_non_2d": _aggregate_parameter_metrics(
            initial,
            current,
            auxiliary_names,
        ),
        "groups": {
            group: _aggregate_parameter_metrics(
                initial,
                current,
                [name for name in names if _parameter_group(name) == group],
            )
            for group in groups
        },
    }


def analyze_feature_movement(
    *,
    sgd_prefix: str,
    muon_prefix: str,
    context_count: int,
    feature_batch_size: int,
) -> dict[str, Any]:
    storage = B2StorageConfig.from_env()
    if storage is None:
        raise RuntimeError("B2 artifact storage is not configured")
    client = storage.s3_client()
    prefixes = {"Large-batch SGD": sgd_prefix, "Large-batch Muon": muon_prefix}

    initial_states = {
        label: _load_state(
            client=client,
            bucket=storage.bucket,
            prefix=prefix,
            step=0,
        )
        for label, prefix in prefixes.items()
    }
    initial = initial_states["Large-batch SGD"]
    same_initialization = initial.keys() == initial_states["Large-batch Muon"].keys()
    if same_initialization:
        same_initialization = all(
            torch.equal(initial[name], initial_states["Large-batch Muon"][name])
            for name in initial
        )
    if not same_initialization:
        raise ValueError("SGD and Muon step-0 model states differ")

    all_contexts = enumerate_paths(10)
    indices = torch.linspace(0, len(all_contexts) - 1, context_count).long()
    contexts = all_contexts.index_select(0, indices)
    initial_features = _block3_features(
        initial,
        contexts,
        batch_size=feature_batch_size,
    )

    r2_curves = json.loads(_R2_PATH.read_text())["curves"]
    r2_lookup = {
        label: {point["step"]: point["r2"] for point in curve}
        for label, curve in r2_curves.items()
    }
    curves: dict[str, list[dict[str, Any]]] = {}
    analyzed_features: dict[str, torch.Tensor] = {}
    for label, prefix in prefixes.items():
        curve = []
        for step in _STEPS:
            state = (
                initial
                if step == 0
                else _load_state(
                    client=client,
                    bucket=storage.bucket,
                    prefix=prefix,
                    step=step,
                )
            )
            features = (
                initial_features
                if step == 0
                else _block3_features(
                    state,
                    contexts,
                    batch_size=feature_batch_size,
                )
            )
            if step == 61_446:
                analyzed_features[label] = features
            curve.append(
                {
                    "step": step,
                    "belief_probe_r2": r2_lookup[label][step],
                    "block3_linear_cka_to_initialization": _linear_cka(
                        initial_features,
                        features,
                    ),
                    "block3_centered_relative_drift": (
                        _centered_relative_feature_drift(
                            initial_features,
                            features,
                        )
                    ),
                    "parameters": _parameter_metrics(initial, state),
                }
            )
        curves[label] = curve

    sgd_analyzed = curves["Large-batch SGD"][-2]["parameters"]["matrix_2d"]
    muon_analyzed = curves["Large-batch Muon"][-2]["parameters"]["matrix_2d"]
    displacement_ratio = (
        muon_analyzed["relative_displacement"]
        / sgd_analyzed["relative_displacement"]
    )
    return {
        "seed": 0,
        "same_initialization_exact": same_initialization,
        "initialization_sha256": _state_digest(initial),
        "checkpoint_prefixes": prefixes,
        "feature_metric": {
            "layer": "block_3 residual stream before final LayerNorm",
            "context_count": context_count,
            "context_selection": (
                "evenly spaced indices from all length-10 MESS3 contexts"
            ),
            "metric": "centered linear CKA",
        },
        "analyzed_step": 61_446,
        "headline": {
            "muon_to_sgd_2d_relative_displacement_ratio": displacement_ratio,
            "sgd_2d_relative_displacement": sgd_analyzed[
                "relative_displacement"
            ],
            "muon_2d_relative_displacement": muon_analyzed[
                "relative_displacement"
            ],
            "sgd_2d_norm_ratio": sgd_analyzed["norm_ratio"],
            "muon_2d_norm_ratio": muon_analyzed["norm_ratio"],
            "sgd_2d_cosine_to_initialization": sgd_analyzed[
                "cosine_to_initialization"
            ],
            "muon_2d_cosine_to_initialization": muon_analyzed[
                "cosine_to_initialization"
            ],
            "sgd_block3_cka_to_initialization": _linear_cka(
                initial_features,
                analyzed_features["Large-batch SGD"],
            ),
            "muon_block3_cka_to_initialization": _linear_cka(
                initial_features,
                analyzed_features["Large-batch Muon"],
            ),
            "sgd_muon_block3_cka": _linear_cka(
                analyzed_features["Large-batch SGD"],
                analyzed_features["Large-batch Muon"],
            ),
        },
        "curves": curves,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sgd-prefix", default=_DEFAULT_SGD_PREFIX)
    parser.add_argument("--muon-prefix", default=_DEFAULT_MUON_PREFIX)
    parser.add_argument("--context-count", type=int, default=4_096)
    parser.add_argument("--feature-batch-size", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_feature_movement(
        sgd_prefix=args.sgd_prefix,
        muon_prefix=args.muon_prefix,
        context_count=args.context_count,
        feature_batch_size=args.feature_batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
