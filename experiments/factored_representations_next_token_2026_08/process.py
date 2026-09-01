"""Device-native independent MESS3 sequence generation and exact beliefs."""

from __future__ import annotations

from dataclasses import dataclass

import torch


MESS3_ALPHA = 0.6
MESS3_X = 0.15
FACTOR_CARDINALITY = 3
FACTOR_STATE_DIMENSION = 3
FACTOR_COUNTS = (2, 3)
SEQUENCE_LENGTH = 8


@dataclass(frozen=True, slots=True)
class SequenceBatch:
    """Joint tokens with exact local predictive-state trajectories."""

    tokens: torch.Tensor
    factor_beliefs: torch.Tensor
    target_probabilities: torch.Tensor


def joint_token_count(factor_count: int) -> int:
    if factor_count not in FACTOR_COUNTS:
        raise ValueError(f"factor_count must be one of {FACTOR_COUNTS}")
    return FACTOR_CARDINALITY**factor_count


def encode_joint_tokens(subtokens: torch.Tensor) -> torch.Tensor:
    """Encode final-axis factor subtokens in Cartesian mixed-radix order."""

    if subtokens.ndim < 1 or subtokens.shape[-1] not in FACTOR_COUNTS:
        raise ValueError("subtokens must end with a supported factor count")
    if subtokens.dtype != torch.long:
        raise TypeError("subtokens must use torch.long")
    powers = FACTOR_CARDINALITY ** torch.arange(
        subtokens.shape[-1] - 1,
        -1,
        -1,
        device=subtokens.device,
        dtype=torch.long,
    )
    return (subtokens * powers).sum(dim=-1)


def decode_joint_tokens(tokens: torch.Tensor, factor_count: int) -> torch.Tensor:
    """Decode joint tokens to one base-three subtoken per factor."""

    count = joint_token_count(factor_count)
    if tokens.dtype != torch.long:
        raise TypeError("tokens must use torch.long")
    powers = FACTOR_CARDINALITY ** torch.arange(
        factor_count - 1,
        -1,
        -1,
        device=tokens.device,
        dtype=torch.long,
    )
    return torch.div(
        tokens.unsqueeze(-1),
        powers,
        rounding_mode="floor",
    ).remainder(FACTOR_CARDINALITY)


def mess3_labeled_operators(
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return Appendix C.1.1 operators as ``(token, state, next_state)``."""

    alpha = MESS3_ALPHA
    beta = (1.0 - alpha) / 2.0
    x = MESS3_X
    y = 1.0 - 2.0 * x
    return torch.tensor(
        [
            [[alpha * y, beta * x, beta * x],
             [alpha * x, beta * y, beta * x],
             [alpha * x, beta * x, beta * y]],
            [[beta * y, alpha * x, beta * x],
             [beta * x, alpha * y, beta * x],
             [beta * x, alpha * x, beta * y]],
            [[beta * y, beta * x, alpha * x],
             [beta * x, beta * y, alpha * x],
             [beta * x, beta * x, alpha * y]],
        ],
        device=device,
        dtype=dtype,
    )


def sample_sequences(
    *,
    batch_size: int,
    factor_count: int,
    sequence_length: int = SEQUENCE_LENGTH,
    device: torch.device | str,
    generator: torch.Generator | None,
    dtype: torch.dtype = torch.float32,
) -> SequenceBatch:
    """Sample independent-factor sequences without leaving the training device.

    Belief index zero is the stationary prior. Index ``position + 1`` is the
    posterior predictive vector after observing the token at ``position``.
    ``target_probabilities`` contains the Bayes probability of each sampled
    joint token given its preceding context.
    """

    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("batch_size and sequence_length must be positive")
    joint_token_count(factor_count)
    device = torch.device(device)
    operators = mess3_labeled_operators(device=device, dtype=dtype)
    belief = torch.full(
        (batch_size, factor_count, FACTOR_STATE_DIMENSION),
        1.0 / FACTOR_STATE_DIMENSION,
        device=device,
        dtype=dtype,
    )
    beliefs = [belief]
    joint_tokens = []
    joint_probabilities = []

    for _ in range(sequence_length):
        local_probabilities = torch.einsum(
            "bfs,tsq->bft",
            belief,
            operators,
        )
        subtokens = torch.multinomial(
            local_probabilities.reshape(-1, FACTOR_CARDINALITY),
            num_samples=1,
            replacement=True,
            generator=generator,
        ).reshape(batch_size, factor_count)
        selected_probabilities = local_probabilities.gather(
            -1,
            subtokens.unsqueeze(-1),
        ).squeeze(-1)
        selected_operators = operators[subtokens]
        unnormalized = torch.einsum(
            "bfs,bfsq->bfq",
            belief,
            selected_operators,
        )
        belief = unnormalized / selected_probabilities.unsqueeze(-1)
        beliefs.append(belief)
        joint_tokens.append(encode_joint_tokens(subtokens))
        joint_probabilities.append(selected_probabilities.prod(dim=-1))

    return SequenceBatch(
        tokens=torch.stack(joint_tokens, dim=1),
        factor_beliefs=torch.stack(beliefs, dim=1),
        target_probabilities=torch.stack(joint_probabilities, dim=1),
    )


def product_beliefs(factor_beliefs: torch.Tensor) -> torch.Tensor:
    """Form aligned joint beliefs from independent local beliefs."""

    if factor_beliefs.ndim != 4:
        raise ValueError("factor_beliefs must have shape (B, L, F, 3)")
    joint = factor_beliefs[:, :, 0, :]
    for factor in range(1, factor_beliefs.shape[2]):
        local = factor_beliefs[:, :, factor, :]
        joint = (
            joint.unsqueeze(-1) * local.unsqueeze(-2)
        ).flatten(start_dim=-2)
    return joint
