"""Exact MESS3 path distribution and Bayesian belief updates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


N_STATES = 3
N_TOKENS = 3
PAPER_X = 0.05
PAPER_ALPHA = 0.85


def labeled_operators(
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return the paper's ``T[token, state, next_state]`` matrices."""
    x = PAPER_X
    alpha = PAPER_ALPHA
    beta = (1.0 - alpha) / 2.0
    stay = 1.0 - 2.0 * x
    ay, ax = alpha * stay, alpha * x
    by, bx = beta * stay, beta * x
    return torch.tensor(
        [
            [[ay, bx, bx], [ax, by, bx], [ax, bx, by]],
            [[by, ax, bx], [bx, ay, bx], [bx, ax, by]],
            [[by, bx, ax], [bx, by, ax], [bx, bx, ay]],
        ],
        dtype=dtype,
        device=device,
    )


def stationary_prior(
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """MESS3 is cyclically symmetric, so its stationary prior is uniform."""
    return torch.full(
        (N_STATES,),
        1.0 / N_STATES,
        dtype=dtype,
        device=device,
    )


def enumerate_paths(
    length: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Enumerate all base-three token paths in lexicographic order."""
    if length <= 0:
        raise ValueError("path length must be positive")
    values = torch.arange(N_TOKENS**length, device=device)
    powers = N_TOKENS ** torch.arange(
        length - 1,
        -1,
        -1,
        device=device,
    )
    return torch.div(
        values[:, None],
        powers[None, :],
        rounding_mode="floor",
    ).remainder(N_TOKENS).to(torch.long)


def bayesian_beliefs(
    paths: torch.Tensor,
    *,
    operators: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return posterior hidden-state beliefs after each observed token."""
    if paths.ndim != 2:
        raise ValueError("paths must have shape (batch, length)")
    operators = (
        labeled_operators(device=paths.device)
        if operators is None
        else operators
    )
    dtype = operators.dtype
    belief = stationary_prior(device=paths.device, dtype=dtype).expand(
        paths.shape[0],
        -1,
    )
    trajectory = []
    for position in range(paths.shape[1]):
        selected = operators.index_select(0, paths[:, position])
        unnormalized = torch.bmm(
            belief.unsqueeze(1),
            selected,
        ).squeeze(1)
        normalizer = unnormalized.sum(dim=-1, keepdim=True)
        if torch.any(normalizer <= 0):
            raise ValueError("encountered a zero-probability MESS3 path")
        belief = unnormalized / normalizer
        trajectory.append(belief)
    return torch.stack(trajectory, dim=1)


def path_probabilities(
    paths: torch.Tensor,
    *,
    operators: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute exact stationary probabilities for token paths."""
    if paths.ndim != 2:
        raise ValueError("paths must have shape (batch, length)")
    operators = (
        labeled_operators(device=paths.device)
        if operators is None
        else operators
    )
    dtype = operators.dtype
    belief = stationary_prior(device=paths.device, dtype=dtype).expand(
        paths.shape[0],
        -1,
    )
    probabilities = torch.ones(
        paths.shape[0],
        dtype=dtype,
        device=paths.device,
    )
    for position in range(paths.shape[1]):
        selected = operators.index_select(0, paths[:, position])
        unnormalized = torch.bmm(
            belief.unsqueeze(1),
            selected,
        ).squeeze(1)
        token_probability = unnormalized.sum(dim=-1)
        probabilities = probabilities * token_probability
        belief = unnormalized / token_probability.unsqueeze(-1)
    return probabilities


def next_token_probabilities(
    beliefs: torch.Tensor,
    *,
    operators: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map hidden-state beliefs to the next-token distribution."""
    operators = (
        labeled_operators(device=beliefs.device, dtype=beliefs.dtype)
        if operators is None
        else operators
    )
    emission_by_state = operators.sum(dim=-1).transpose(0, 1)
    return beliefs @ emission_by_state


def exact_bayesian_loss(paths: torch.Tensor, probabilities: torch.Tensor) -> float:
    """Expected ten-position next-token loss for length-eleven paths."""
    if paths.shape[1] < 2:
        raise ValueError("at least two tokens are needed for next-token loss")
    operators = labeled_operators(
        device=paths.device,
        dtype=probabilities.dtype,
    )
    beliefs = bayesian_beliefs(paths[:, :-1], operators=operators)
    predictions = next_token_probabilities(beliefs, operators=operators)
    targets = paths[:, 1:].unsqueeze(-1)
    target_probabilities = predictions.gather(-1, targets).squeeze(-1)
    per_path = -target_probabilities.log().mean(dim=-1)
    normalized = probabilities / probabilities.sum()
    return float((normalized * per_path).sum().cpu())


@dataclass(frozen=True, slots=True)
class AliasTable:
    """Walker alias table for constant-time GPU path sampling."""

    threshold: torch.Tensor
    alias: torch.Tensor

    @classmethod
    def from_probabilities(
        cls,
        probabilities: torch.Tensor,
        *,
        device: torch.device | str,
    ) -> "AliasTable":
        values = probabilities.detach().cpu().double().numpy()
        values = values / values.sum()
        count = len(values)
        scaled = values * count
        threshold = np.empty(count, dtype=np.float64)
        alias = np.empty(count, dtype=np.int64)
        small = [index for index, value in enumerate(scaled) if value < 1.0]
        large = [index for index, value in enumerate(scaled) if value >= 1.0]
        while small and large:
            low = small.pop()
            high = large.pop()
            threshold[low] = scaled[low]
            alias[low] = high
            scaled[high] -= 1.0 - scaled[low]
            (small if scaled[high] < 1.0 else large).append(high)
        for index in (*small, *large):
            threshold[index] = 1.0
            alias[index] = index
        return cls(
            threshold=torch.as_tensor(
                threshold,
                dtype=probabilities.dtype,
                device=device,
            ),
            alias=torch.as_tensor(alias, dtype=torch.long, device=device),
        )

    def sample(
        self,
        count: int,
        *,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Draw indices using device-native random operations."""
        buckets = torch.randint(
            len(self.alias),
            (count,),
            device=self.alias.device,
            generator=generator,
        )
        uniforms = torch.rand(
            count,
            dtype=self.threshold.dtype,
            device=self.threshold.device,
            generator=generator,
        )
        return torch.where(
            uniforms < self.threshold.index_select(0, buckets),
            buckets,
            self.alias.index_select(0, buckets),
        )
