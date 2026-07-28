"""Plot held-out belief-probe R² across SGD and Muon checkpoints."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
import torch

from harness.seeding import named_seed_sequences, seed_sequence_to_int

from .paper_supervised_replication.analysis import run_layer_probes
from .paper_supervised_replication.mess3 import enumerate_paths
from .paper_supervised_replication.model import (
    PaperModelConfig,
    PaperTransformer,
)
from .paper_supervised_replication.training import load_checkpoint

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


_CHECKPOINT_PATTERN = re.compile(r"step_(\d+)\.pt")


def _checkpoint_paths(directory: Path) -> list[tuple[int, Path]]:
    checkpoints = []
    for path in directory.glob("step_*.pt"):
        match = _CHECKPOINT_PATTERN.fullmatch(path.name)
        if match is not None:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort()
    if not checkpoints:
        raise ValueError(f"no named checkpoints found under {directory}")
    return checkpoints


def _probe_curve(
    checkpoint_directory: Path,
    *,
    contexts: torch.Tensor,
    probe_seed: int,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, float | int]]:
    curve = []
    for step, checkpoint in _checkpoint_paths(checkpoint_directory):
        model = PaperTransformer(PaperModelConfig()).to(device)
        load_checkpoint(
            checkpoint,
            model=model,
            optimizers=None,
            generator=None,
            device=device,
        )
        result, _, _ = run_layer_probes(
            model,
            contexts,
            seed=probe_seed,
            batch_size=batch_size,
        )
        headline = result["layers"][result["headline_layer"]]
        curve.append(
            {
                "step": step,
                "r2": headline["r2"],
                "mse": headline["mse"],
            }
        )
        del model
    return curve


def plot_checkpoint_r2(
    *,
    sgd_checkpoints: Path,
    muon_checkpoints: Path,
    output_directory: Path,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    probe_seed = seed_sequence_to_int(
        named_seed_sequences(seed, {"probe_split": (2,)})["probe_split"]
    )
    contexts = enumerate_paths(10)
    curves = {
        "Large-batch SGD": _probe_curve(
            sgd_checkpoints,
            contexts=contexts,
            probe_seed=probe_seed,
            batch_size=batch_size,
            device=device,
        ),
        "Large-batch Muon": _probe_curve(
            muon_checkpoints,
            contexts=contexts,
            probe_seed=probe_seed,
            batch_size=batch_size,
            device=device,
        ),
    }

    data_path = output_directory / "checkpoint_r2_comparison.json"
    data_path.write_text(json.dumps({"seed": seed, "curves": curves}, indent=2) + "\n")

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for label, curve in curves.items():
        axis.plot(
            [point["step"] for point in curve],
            [point["r2"] for point in curve],
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            label=label,
        )
    axis.set_xlabel("Optimizer updates")
    axis.set_ylabel("Held-out affine belief-probe R²")
    axis.set_title("MESS3 belief geometry across training")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure_path = output_directory / "checkpoint_r2_comparison.png"
    figure.savefig(figure_path, dpi=220)
    plt.close(figure)
    return data_path, figure_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sgd-checkpoints", type=Path, required=True)
    parser.add_argument("--muon-checkpoints", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4_096)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    device_name = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    outputs = plot_checkpoint_r2(
        sgd_checkpoints=args.sgd_checkpoints,
        muon_checkpoints=args.muon_checkpoints,
        output_directory=args.output_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        device=torch.device(device_name),
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
