"""Matched next-token CE controls for factored MESS3 representations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import matplotlib
import torch
import torch.nn.functional as F
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.mess3_token_guess_cycle_2.model import (
    PaperActorCriticConfig,
    PaperResidualEncoder,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES


FULL_NEXT_TOKEN_EXAMPLES = 50_000_000
SMOKE_NEXT_TOKEN_EXAMPLES = 4_096
CONTEXT_LENGTH = 11
FACTOR_CARDINALITY = 3
LEARNING_RATE = 3e-4
FULL_BATCH_SEQUENCES = 512
SMOKE_BATCH_SEQUENCES = 16

_TRANSITION = (
    (0.90, 0.05, 0.05),
    (0.05, 0.90, 0.05),
    (0.05, 0.05, 0.90),
)
_EMISSION = (
    (0.85, 0.075, 0.075),
    (0.075, 0.85, 0.075),
    (0.075, 0.075, 0.85),
)


@dataclass(frozen=True, slots=True)
class StudyCondition:
    """Validated scientific parameters for one matched control leaf."""

    n_factors: int
    d_model: int

    def __post_init__(self) -> None:
        if self.n_factors not in {2, 3, 5}:
            raise ValueError("the matched study supports 2, 3, or 5 factors")
        if self.d_model not in {64, 120}:
            raise ValueError("the matched study supports 64d or 120d models")
        if self.n_factors != 5 and self.d_model != 64:
            raise ValueError("only the five-factor condition has a 120d arm")

    @property
    def vocabulary_size(self) -> int:
        return FACTOR_CARDINALITY**self.n_factors

    @property
    def name(self) -> str:
        return f"{self.n_factors}f_{self.d_model}d"


def build_model_config(condition: StudyCondition) -> dict[str, Any]:
    """Return a fresh paper-residual encoder configuration."""

    if condition.d_model == 64:
        config = PaperActorCriticConfig(
            d_model=64,
            n_layers=4,
            n_heads=1,
            d_head=8,
            d_mlp=256,
            context_length=CONTEXT_LENGTH,
            max_seq_len=CONTEXT_LENGTH,
        )
    else:
        config = PaperActorCriticConfig(
            d_model=120,
            n_layers=4,
            n_heads=3,
            d_head=40,
            d_mlp=480,
            context_length=CONTEXT_LENGTH,
            max_seq_len=CONTEXT_LENGTH,
        )
    return config.to_dict()


class SupervisedNextTokenModel(nn.Module):
    """Paper residual encoder plus one next-token head, with no critic."""

    def __init__(self, condition: StudyCondition) -> None:
        super().__init__()
        self.condition = condition
        self.model_config = PaperActorCriticConfig.from_dict(
            build_model_config(condition)
        )
        self.encoder = PaperResidualEncoder(
            condition.vocabulary_size,
            self.model_config,
        )
        self.next_token_head = nn.Linear(
            condition.d_model,
            condition.vocabulary_size,
        )
        nn.init.normal_(self.next_token_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.next_token_head.bias)

    def encode_tokens(
        self,
        tokens: torch.Tensor,
        *,
        apply_final_norm: bool,
    ) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch, sequence)")
        if tokens.shape[1] > CONTEXT_LENGTH:
            raise ValueError("token sequence exceeds the context length")
        observations = F.one_hot(
            tokens,
            num_classes=self.condition.vocabulary_size,
        ).to(dtype=self.encoder.input_embedding.weight.dtype)
        batch_size = tokens.shape[0]
        context = torch.zeros(
            batch_size,
            self.model_config.context_length - 1,
            self.condition.vocabulary_size,
            dtype=observations.dtype,
            device=tokens.device,
        )
        context_lengths = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=tokens.device,
        )
        return self.encoder(
            context,
            context_lengths,
            observations,
            apply_final_norm=apply_final_norm,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        return_activations: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        pre_final_norm = self.encode_tokens(tokens, apply_final_norm=False)
        logits = self.next_token_head(self.encoder.final_norm(pre_final_norm))
        if return_activations:
            return logits, pre_final_norm
        return logits

    def token_embedding_table(self) -> torch.Tensor:
        """Return one embedding vector for every Cartesian joint token."""

        return self.encoder.input_embedding.weight.transpose(0, 1)


def _device(context: RunContext) -> torch.device:
    profile = context.hardware or PROFILES["cpu"]
    if profile.learner_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA profile selected but CUDA is unavailable")
        return torch.device("cuda")
    if (
        profile.learner_device == "mps"
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def _generator(device: torch.device, seed: int) -> torch.Generator:
    try:
        generator = torch.Generator(device=device)
    except RuntimeError:
        generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def sample_factor_paths(
    count: int,
    *,
    n_factors: int,
    length: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample independent stationary MESS3 subtoken paths on-device."""

    if count <= 0 or n_factors <= 0 or length <= 0:
        raise ValueError("count, n_factors, and length must be positive")
    transition = torch.tensor(
        _TRANSITION,
        dtype=torch.float32,
        device=device,
    )
    emission = torch.tensor(
        _EMISSION,
        dtype=torch.float32,
        device=device,
    )
    states = torch.randint(
        FACTOR_CARDINALITY,
        (count, n_factors),
        device=device,
        generator=generator,
    )
    paths: list[torch.Tensor] = []
    for _ in range(length):
        flat_states = states.reshape(-1)
        token = torch.multinomial(
            emission.index_select(0, flat_states),
            1,
            generator=generator,
        ).reshape(count, n_factors)
        paths.append(token)
        states = torch.multinomial(
            transition.index_select(0, flat_states),
            1,
            generator=generator,
        ).reshape(count, n_factors)
    return torch.stack(paths, dim=1)


def combine_factor_tokens(factor_paths: torch.Tensor) -> torch.Tensor:
    """Encode (..., factor) base-three subtokens as one joint token."""

    if factor_paths.ndim < 2:
        raise ValueError("factor_paths must include a factor dimension")
    n_factors = factor_paths.shape[-1]
    powers = FACTOR_CARDINALITY ** torch.arange(
        n_factors - 1,
        -1,
        -1,
        device=factor_paths.device,
    )
    return (factor_paths * powers).sum(dim=-1).to(torch.long)


def make_next_token_batch(
    count: int,
    *,
    condition: StudyCondition,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return aligned length-11 inputs and their immediate next joint tokens."""

    factors = sample_factor_paths(
        count,
        n_factors=condition.n_factors,
        length=CONTEXT_LENGTH + 1,
        device=device,
        generator=generator,
    )
    joint = combine_factor_tokens(factors)
    return joint[:, :-1], joint[:, 1:], factors[:, :-1]


def _checkpoint_payload(
    *,
    model: SupervisedNextTokenModel,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    examples_seen: int,
    condition: StudyCondition,
    curve: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "format": "factored-mess3-supervised-ce-v1",
        "condition": asdict(condition),
        "model_state": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "optimizer_state": optimizer.state_dict(),
        "generator_state": generator.get_state().cpu(),
        "examples_seen": examples_seen,
        "training_curve": curve,
    }


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_supervised_checkpoint(
    path: Path,
    *,
    model: SupervisedNextTokenModel,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Restore the experiment-local portable checkpoint contract."""

    checkpoint_path = path / "final.pt" if path.is_dir() else path
    payload = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if payload.get("format") != "factored-mess3-supervised-ce-v1":
        raise ValueError("resume checkpoint is not a supervised control checkpoint")
    if payload.get("condition") != asdict(model.condition):
        raise ValueError("resume checkpoint condition does not match this leaf")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if generator is not None:
        generator.set_state(payload["generator_state"].cpu())
    return payload


@torch.no_grad()
def _validation_metrics(
    model: SupervisedNextTokenModel,
    *,
    condition: StudyCondition,
    device: torch.device,
    seed: int,
    smoke: bool,
) -> dict[str, float | int]:
    model.eval()
    count = 128 if smoke else 2_048
    inputs, targets, _ = make_next_token_batch(
        count,
        condition=condition,
        device=device,
        generator=_generator(device, seed),
    )
    logits = model(inputs)
    return {
        "validation_examples": int(targets.numel()),
        "cross_entropy_nats": float(
            F.cross_entropy(
                logits.reshape(-1, condition.vocabulary_size),
                targets.reshape(-1),
            ).detach().cpu()
        ),
        "accuracy": float(
            (
                logits.argmax(dim=-1) == targets
            ).float().mean().detach().cpu()
        ),
    }


def train_supervised_control(
    context: RunContext,
    *,
    condition: StudyCondition,
) -> tuple[SupervisedNextTokenModel, dict[str, Any], list[dict[str, Any]], Path]:
    """Train pure next-token CE for an exact example budget."""

    if context.seed is None:
        raise ValueError("the supervised controls require a resolved seed")
    device = _device(context)
    torch.manual_seed(context.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(context.seed)
    model = SupervisedNextTokenModel(condition).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    sampling_generator = _generator(device, context.seed + 1_001)
    target_examples = (
        SMOKE_NEXT_TOKEN_EXAMPLES
        if context.smoke
        else FULL_NEXT_TOKEN_EXAMPLES
    )
    batch_sequences = (
        SMOKE_BATCH_SEQUENCES
        if context.smoke
        else FULL_BATCH_SEQUENCES
    )
    examples_seen = 0
    curve: list[dict[str, Any]] = []
    if context.resume_from is not None:
        payload = load_supervised_checkpoint(
            context.resume_from,
            model=model,
            device=device,
            optimizer=optimizer,
            generator=sampling_generator,
        )
        examples_seen = int(payload["examples_seen"])
        curve = list(payload.get("training_curve", []))
        if examples_seen > target_examples:
            raise ValueError("resume checkpoint exceeds this leaf's budget")

    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    checkpoint_dir = outputs.checkpoints_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if examples_seen == 0:
        _save_checkpoint(
            checkpoint_dir / "initial.pt",
            _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                generator=sampling_generator,
                examples_seen=0,
                condition=condition,
                curve=curve,
            ),
        )

    started_at = time.monotonic()
    update = 0
    log_every = 4 if context.smoke else 100
    model.train()
    while examples_seen < target_examples:
        remaining = target_examples - examples_seen
        count = min(
            batch_sequences,
            math.ceil(remaining / CONTEXT_LENGTH),
        )
        inputs, targets, _ = make_next_token_batch(
            count,
            condition=condition,
            device=device,
            generator=sampling_generator,
        )
        logits = model(inputs)
        take = min(remaining, targets.numel())
        flat_logits = logits.reshape(-1, condition.vocabulary_size)[:take]
        flat_targets = targets.reshape(-1)[:take]
        loss = F.cross_entropy(flat_logits, flat_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        examples_seen += take
        update += 1
        if update % log_every == 0 or examples_seen == target_examples:
            with torch.no_grad():
                accuracy = (
                    flat_logits.argmax(dim=-1) == flat_targets
                ).float().mean()
            record = {
                "optimizer_update": update,
                "next_token_examples": examples_seen,
                "cross_entropy_nats": float(loss.detach().cpu()),
                "accuracy": float(accuracy.detach().cpu()),
                "wall_seconds": time.monotonic() - started_at,
            }
            curve.append(record)
            outputs.append_result(record)

    final_checkpoint = checkpoint_dir / "final.pt"
    _save_checkpoint(
        final_checkpoint,
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            generator=sampling_generator,
            examples_seen=examples_seen,
            condition=condition,
            curve=curve,
        ),
    )
    validation = _validation_metrics(
        model,
        condition=condition,
        device=device,
        seed=context.seed + 2_003,
        smoke=context.smoke,
    )
    summary = {
        "objective": "next_joint_token_cross_entropy_only",
        "has_ppo_objective": False,
        "has_value_or_critic_head": False,
        "seed": context.seed,
        "smoke": context.smoke,
        "condition": asdict(condition),
        "target_next_token_examples": target_examples,
        "completed_next_token_examples": examples_seen,
        "optimizer_updates_this_invocation": update,
        "learning_rate": LEARNING_RATE,
        "validation": validation,
        "checkpoint": str(final_checkpoint),
        "wall_seconds": time.monotonic() - started_at,
    }
    outputs.write_json("training_summary.json", summary)
    outputs.write_json("training_curve.json", curve)
    _plot_training_curve(
        curve,
        context.results_dir / "training_curve.png",
    )
    return model, summary, curve, final_checkpoint


def _plot_training_curve(curve: list[dict[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    if curve:
        axis.plot(
            [row["next_token_examples"] for row in curve],
            [row["cross_entropy_nats"] for row in curve],
            marker=".",
        )
    axis.set_xlabel("Next-token examples")
    axis.set_ylabel("Cross-entropy (nats)")
    axis.set_title("Supervised next-token training")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_recipe(
    context: RunContext,
    condition: StudyCondition,
) -> None:
    outputs = RunArtifacts.from_context(context)
    target_examples = (
        SMOKE_NEXT_TOKEN_EXAMPLES
        if context.smoke
        else FULL_NEXT_TOKEN_EXAMPLES
    )
    outputs.write_json(
        "resolved_recipe.json",
        {
            "study": "factored_mess3_representation_controls_2026_08",
            "hypothesis": (
                "If approximately 2N-dimensional geometry is induced by "
                "next-token prediction rather than PPO, a matched pure-CE "
                "transformer should show the same factorwise collapse."
            ),
            "primary_comparison": (
                "matched_supervised_next_token_ce_vs_completed_ppo_baselines"
            ),
            "condition": asdict(condition),
            "seed": context.seed,
            "independent_mess3_factors": condition.n_factors,
            "factor_alpha": 0.85,
            "factor_transition_stay_probability": 0.90,
            "joint_vocabulary_size": condition.vocabulary_size,
            "context_length": CONTEXT_LENGTH,
            "objective": "shifted joint-token next-token cross_entropy",
            "ppo_objective": None,
            "critic": None,
            "next_token_example_budget": target_examples,
            "full_next_token_example_budget": FULL_NEXT_TOKEN_EXAMPLES,
            "smoke_next_token_example_budget": SMOKE_NEXT_TOKEN_EXAMPLES,
            "model": build_model_config(condition),
            "analysis_suite": [
                "held-out additive-factor vs full-joint activation encoding",
                "held-out target R2 by activation rank",
                "controlled vary-one factor geometry",
                "joint-token embedding additive decomposition",
            ],
            "resume_contract": (
                "RunContext.resume_from must point to final.pt (or its "
                "containing checkpoint directory) produced by this same leaf."
            ),
        },
    )


def run_condition(
    context: RunContext,
    *,
    n_factors: int,
    d_model: int,
) -> dict[str, Any]:
    """Train one control and run every representation analysis."""

    from .analysis import run_analysis_suite

    condition = StudyCondition(n_factors=n_factors, d_model=d_model)
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    _write_recipe(context, condition)
    model, training, _, checkpoint = train_supervised_control(
        context,
        condition=condition,
    )
    analysis = run_analysis_suite(
        context,
        model=model,
        condition=condition,
    )
    required = [
        "resolved_recipe.json",
        "training_summary.json",
        "training_curve.json",
        "training_curve.png",
        "reverse_encoding_metrics.json",
        "reduced_rank_curves.json",
        "reduced_rank_curves.png",
        "vary_one_metrics.json",
        "embedding_additivity_metrics.json",
        "analysis_summary.json",
    ]
    missing = [
        name
        for name in required
        if not (context.results_dir / name).is_file()
    ]
    if missing:
        raise RuntimeError("missing compact outputs: " + ", ".join(missing))
    validation = {
        "status": "completed",
        "condition": condition.name,
        "checkpoint": str(checkpoint),
        "required_files": required,
        "full_run_triggered": False if context.smoke else True,
    }
    outputs.write_json("output_validation.json", validation)
    return {
        "training": training,
        "analysis": analysis,
        "output_validation": validation,
    }


__all__ = [
    "CONTEXT_LENGTH",
    "FACTOR_CARDINALITY",
    "FULL_NEXT_TOKEN_EXAMPLES",
    "SMOKE_NEXT_TOKEN_EXAMPLES",
    "StudyCondition",
    "SupervisedNextTokenModel",
    "build_model_config",
    "combine_factor_tokens",
    "load_supervised_checkpoint",
    "make_next_token_batch",
    "run_condition",
    "sample_factor_paths",
]
