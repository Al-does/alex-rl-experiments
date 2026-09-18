"""Longitudinal weighted-belief probes and paper-style visualizations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from analysis.belief_geometry import (
    evaluate_belief_geometry,
    prediction_null_basis,
)
from analysis.probes.controls import score_prediction
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.nonergodic_mess3_token_guess_cycle_1.process import (
    nonergodic_mess3_model,
)

from .data import (
    BOS_TOKEN,
    COMPONENT_COUNT,
    NonergodicSequenceSampler,
    component_posteriors,
    next_token_distributions,
    weighted_beliefs,
)
from .model import ModelConfig, NonergodicMess3Transformer
from .training import load_analysis_checkpoint


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    fit_sequences: int = 128
    test_sequences: int = 128
    inference_batch_size: int = 32
    n_null_repeats: int = 2
    n_bootstrap_resamples: int = 100
    max_plot_points: int = 20_000

    @classmethod
    def smoke(cls) -> AnalysisConfig:
        return cls(
            fit_sequences=6,
            test_sequences=6,
            inference_batch_size=2,
            n_null_repeats=1,
            n_bootstrap_resamples=10,
            max_plot_points=1_000,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProbeDataset:
    tokens: np.ndarray
    beliefs: np.ndarray
    next_token_probabilities: np.ndarray
    component_probabilities: np.ndarray
    generating_components: np.ndarray
    groups: np.ndarray
    positions: np.ndarray

    @property
    def flat_beliefs(self) -> np.ndarray:
        return self.beliefs.reshape(-1, self.beliefs.shape[-1])

    @property
    def flat_next_token_probabilities(self) -> np.ndarray:
        return self.next_token_probabilities.reshape(
            -1,
            self.next_token_probabilities.shape[-1],
        )

    @property
    def flat_tokens(self) -> np.ndarray:
        return self.tokens.reshape(-1)


def make_probe_dataset(seed: int, sequence_count: int) -> ProbeDataset:
    sampled = NonergodicSequenceSampler(seed).sample(sequence_count)
    beliefs = weighted_beliefs(sampled.tokens)
    batch_size, sequence_length = sampled.tokens.shape
    return ProbeDataset(
        tokens=sampled.tokens,
        beliefs=beliefs,
        next_token_probabilities=next_token_distributions(beliefs),
        component_probabilities=component_posteriors(beliefs),
        generating_components=np.repeat(
            sampled.components,
            sequence_length,
        ),
        groups=np.repeat(np.arange(batch_size), sequence_length),
        positions=np.tile(np.arange(sequence_length), batch_size),
    )


@torch.no_grad()
def extract_residuals(
    model: NonergodicMess3Transformer,
    tokens: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    model.eval()
    chunks = {name: [] for name in model.residual_names}
    for start in range(0, len(tokens), batch_size):
        token_batch = torch.as_tensor(
            tokens[start : start + batch_size],
            dtype=torch.long,
            device=device,
        )
        _, residuals = model(token_batch, return_residuals=True)
        for name, values in residuals.items():
            chunks[name].append(
                values.reshape(-1, model.config.d_model)
                .float()
                .cpu()
                .numpy()
            )
    return {
        name: np.concatenate(values, axis=0)
        for name, values in chunks.items()
    }


def _fit_affine(
    features: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    design = np.concatenate(
        [
            np.asarray(features, dtype=np.float64),
            np.ones((len(features), 1), dtype=np.float64),
        ],
        axis=1,
    )
    solution, _, _, _ = np.linalg.lstsq(
        design,
        np.asarray(targets, dtype=np.float64),
        rcond=1e-8,
    )
    return solution[:-1], solution[-1]


def _probe_checkpoint(
    train_residuals: dict[str, np.ndarray],
    test_residuals: dict[str, np.ndarray],
    train_beliefs: np.ndarray,
    test_beliefs: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    report = {}
    predictions = {}
    for name in train_residuals:
        weight, bias = _fit_affine(
            train_residuals[name],
            train_beliefs,
        )
        prediction = test_residuals[name] @ weight + bias
        report[name] = score_prediction(prediction, test_beliefs)
        predictions[name] = prediction
    return report, predictions


def _one_hot_tokens(dataset: ProbeDataset) -> np.ndarray:
    return np.eye(BOS_TOKEN + 1, dtype=np.float64)[dataset.flat_tokens]


def _contrasts() -> dict[str, np.ndarray]:
    emission = np.asarray(
        nonergodic_mess3_model().emission_matrix,
        dtype=np.float64,
    )
    null_basis = prediction_null_basis(emission)
    result = {
        "component_A_minus_B": np.asarray(
            [0.5, 0.5, 0.5, -0.5, -0.5, -0.5],
            dtype=np.float64,
        )
    }
    result.update(
        {
            f"next_token_null_{index + 1}": null_basis[:, index]
            for index in range(null_basis.shape[1])
        }
    )
    return result


def _plot_training_and_probe_curves(
    history: list[dict[str, Any]],
    checkpoint_report: list[dict[str, Any]],
    *,
    path: Path,
) -> None:
    training = [
        record for record in history if record["kind"] == "training"
    ]
    validation = {
        int(record["step"]): record
        for record in history
        if record["kind"] == "validation"
    }
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    if training:
        axes[0].plot(
            [record["step"] for record in training],
            [record["training_loss_nats"] for record in training],
            label="training CE",
            linewidth=1.0,
        )
    ordered_validation = [
        validation[step] for step in sorted(validation)
    ]
    axes[0].plot(
        [max(1, record["step"]) for record in ordered_validation],
        [record["validation_loss_nats"] for record in ordered_validation],
        "o-",
        label="held-out CE",
        markersize=3,
    )
    axes[0].plot(
        [max(1, record["step"]) for record in ordered_validation],
        [record["bayesian_floor_nats"] for record in ordered_validation],
        "--",
        color="black",
        label="Bayesian floor",
        linewidth=0.9,
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("cross-entropy (nats)")
    axes[0].legend()

    for layer in checkpoint_report[-1]["layers"]:
        points = [
            (
                validation[record["step"]]["excess_loss_nats"],
                max(1e-8, 1.0 - record["layers"][layer]["r_squared"]),
            )
            for record in checkpoint_report
            if record["step"] in validation
            and validation[record["step"]]["excess_loss_nats"] > 0.0
            and record["layers"][layer]["r_squared"] is not None
        ]
        if points:
            axes[1].plot(
                [point[0] for point in points],
                [point[1] for point in points],
                "o-",
                label=layer,
                markersize=3,
            )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("excess held-out loss (nats)")
    axes[1].set_ylabel(r"$1-R^2$ weighted belief probe")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=190)
    plt.close(figure)


def _posterior_entropy(probabilities: np.ndarray) -> np.ndarray:
    safe = np.clip(probabilities, 1e-12, 1.0)
    return -(safe * np.log(safe)).sum(axis=-1)


def _plot_geometry(
    target: np.ndarray,
    prediction: np.ndarray,
    component_probabilities: np.ndarray,
    *,
    path: Path,
    seed: int,
    max_points: int,
) -> None:
    if len(target) > max_points:
        indices = np.random.default_rng(seed).choice(
            len(target),
            size=max_points,
            replace=False,
        )
        target = target[indices]
        prediction = prediction[indices]
        component_probabilities = component_probabilities[indices]
    entropy = _posterior_entropy(component_probabilities)
    colors = plt.get_cmap("viridis")(
        entropy / max(np.log(COMPONENT_COUNT), 1e-12)
    )
    figure = plt.figure(figsize=(10.5, 9.0))
    for row, (label, values) in enumerate(
        (("Exact Bayesian belief", target), ("Linear readout", prediction))
    ):
        for component in range(COMPONENT_COUNT):
            axis = figure.add_subplot(
                2,
                COMPONENT_COUNT,
                row * COMPONENT_COUNT + component + 1,
                projection="3d",
            )
            block = values[:, component * 3 : (component + 1) * 3]
            axis.scatter(
                block[:, 0],
                block[:, 1],
                block[:, 2],
                c=colors,
                s=1.0,
                alpha=0.35,
                linewidths=0,
                rasterized=True,
            )
            axis.set(
                xlim=(0.0, 1.0),
                ylim=(0.0, 1.0),
                zlim=(0.0, 1.0),
                xlabel="S1",
                ylabel="S2",
                zlabel="S3",
                title=f"{label}: component {'AB'[component]}",
            )
    figure.tight_layout()
    figure.savefig(path, dpi=190)
    plt.close(figure)


def _cumulative_explained_variance(features: np.ndarray) -> np.ndarray:
    centered = np.asarray(features, dtype=np.float64)
    centered -= centered.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(
        centered,
        compute_uv=False,
        full_matrices=False,
    )
    variance = np.square(singular_values)
    return np.cumsum(variance) / variance.sum()


def _plot_activation_variance(
    features: np.ndarray,
    dataset: ProbeDataset,
    *,
    path: Path,
) -> None:
    entropy = _posterior_entropy(
        dataset.component_probabilities.reshape(-1, COMPONENT_COUNT)
    )
    figure, axes = plt.subplots(
        1,
        COMPONENT_COUNT,
        figsize=(10.5, 4.2),
        sharey=True,
    )
    for component, axis in enumerate(axes):
        component_mask = dataset.generating_components == component
        for label, mask in (
            ("all contexts", component_mask),
            (
                "posterior entropy ≤ 0.15",
                component_mask & (entropy <= 0.15),
            ),
        ):
            if mask.sum() < 2:
                continue
            cumulative = _cumulative_explained_variance(features[mask])
            axis.plot(
                np.arange(1, min(32, len(cumulative)) + 1),
                cumulative[:32],
                label=label,
            )
        axis.set(
            xlabel="principal components",
            title=f"Generating component {'AB'[component]}",
            ylim=(0.0, 1.01),
        )
        axis.legend(fontsize=8)
    axes[0].set_ylabel("cumulative explained variance")
    figure.tight_layout()
    figure.savefig(path, dpi=190)
    plt.close(figure)


def run_checkpoint_analysis(
    *,
    checkpoint_paths: list[Path],
    history: list[dict[str, Any]],
    model_config: ModelConfig,
    device: torch.device,
    fit_seed: int,
    test_seed: int,
    analysis_seed: int,
    config: AnalysisConfig,
    results_dir: Path,
) -> dict[str, Any]:
    fit_dataset = make_probe_dataset(fit_seed, config.fit_sequences)
    test_dataset = make_probe_dataset(test_seed, config.test_sequences)
    fit_beliefs = fit_dataset.flat_beliefs
    test_beliefs = test_dataset.flat_beliefs
    model = NonergodicMess3Transformer(model_config).to(device)

    checkpoint_report = []
    initialization_train = None
    initialization_test = None
    analyzed_train = None
    analyzed_test = None
    analyzed_predictions = None
    for path in checkpoint_paths:
        step = load_analysis_checkpoint(path, model=model, device=device)
        train_residuals = extract_residuals(
            model,
            fit_dataset.tokens,
            device=device,
            batch_size=config.inference_batch_size,
        )
        test_residuals = extract_residuals(
            model,
            test_dataset.tokens,
            device=device,
            batch_size=config.inference_batch_size,
        )
        layers, predictions = _probe_checkpoint(
            train_residuals,
            test_residuals,
            fit_beliefs,
            test_beliefs,
        )
        checkpoint_report.append({"step": step, "layers": layers})
        if step == 0:
            initialization_train = train_residuals
            initialization_test = test_residuals
        if path == checkpoint_paths[-1]:
            analyzed_train = train_residuals
            analyzed_test = test_residuals
            analyzed_predictions = predictions

    if any(
        value is None
        for value in (
            initialization_train,
            initialization_test,
            analyzed_train,
            analyzed_test,
            analyzed_predictions,
        )
    ):
        raise RuntimeError("analysis checkpoints did not include initialization")

    fit_next = fit_dataset.flat_next_token_probabilities
    test_next = test_dataset.flat_next_token_probabilities
    fit_tokens = _one_hot_tokens(fit_dataset)
    test_tokens = _one_hot_tokens(test_dataset)
    battery = evaluate_belief_geometry(
        analyzed_train,
        analyzed_test,
        fit_beliefs,
        test_beliefs,
        train_groups=fit_dataset.groups,
        test_groups=test_dataset.groups,
        nuisance_features={
            "next_token_probabilities": (fit_next, test_next),
            "log_next_token_probabilities": (
                np.log(np.clip(fit_next, 1e-12, 1.0)),
                np.log(np.clip(test_next, 1e-12, 1.0)),
            ),
            "current_token": (fit_tokens, test_tokens),
        },
        contrasts=_contrasts(),
        initialization_features={
            name: (initialization_train[name], initialization_test[name])
            for name in analyzed_train
        },
        matched_keys=(
            np.column_stack((fit_dataset.flat_tokens, fit_dataset.positions)),
            np.column_stack((test_dataset.flat_tokens, test_dataset.positions)),
        ),
        seed=analysis_seed,
        n_null_repeats=config.n_null_repeats,
        n_resamples=config.n_bootstrap_resamples,
    )

    _plot_training_and_probe_curves(
        history,
        checkpoint_report,
        path=results_dir / "training_and_probe_curves.png",
    )
    _plot_geometry(
        test_beliefs,
        analyzed_predictions["block_3"],
        test_dataset.component_probabilities.reshape(-1, COMPONENT_COUNT),
        path=results_dir / "block_3_geometry.png",
        seed=analysis_seed,
        max_points=config.max_plot_points,
    )
    _plot_activation_variance(
        analyzed_test["block_3"],
        test_dataset,
        path=results_dir / "block_3_activation_variance.png",
    )

    return {
        "config": config.to_dict(),
        "sampling": {
            "fit_seed": fit_seed,
            "test_seed": test_seed,
            "independent_sequence_groups": True,
            "positions_per_sequence": fit_dataset.tokens.shape[1],
        },
        "checkpoint_curve": checkpoint_report,
        "battery": battery.report,
        "headline": {
            "block_3": checkpoint_report[-1]["layers"]["block_3"],
            "block_4": checkpoint_report[-1]["layers"]["block_4"],
            "initialization_block_3": checkpoint_report[0]["layers"][
                "block_3"
            ],
        },
    }
