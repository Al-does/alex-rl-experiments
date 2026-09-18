"""Exact path distribution and Bayesian filtering for RRXOR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from experiments.rrxor_token_guess_cycle_1.process import (
    RRXOR_STATIONARY,
    RRXOR_TENSOR,
)


N_STATES = 5
N_TOKENS = 2


def labeled_operators(
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return ``T[token, state, next_state]`` for the paper's RRXOR process."""

    return torch.as_tensor(RRXOR_TENSOR, dtype=dtype, device=device)


def stationary_prior(
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return the exact stationary source-state distribution."""

    return torch.as_tensor(RRXOR_STATIONARY, dtype=dtype, device=device)


def enumerate_paths(
    length: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Enumerate all binary token paths in lexicographic order."""

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
    """Return source-state posteriors after each observed token."""

    if paths.ndim != 2:
        raise ValueError("paths must have shape (batch, length)")
    operators = (
        labeled_operators(device=paths.device)
        if operators is None
        else operators
    )
    belief = stationary_prior(
        device=paths.device,
        dtype=operators.dtype,
    ).expand(paths.shape[0], -1)
    trajectory = []
    for position in range(paths.shape[1]):
        selected = operators.index_select(0, paths[:, position])
        unnormalized = torch.bmm(
            belief.unsqueeze(1),
            selected,
        ).squeeze(1)
        normalizer = unnormalized.sum(dim=-1, keepdim=True)
        if torch.any(normalizer <= 0):
            raise ValueError("encountered a zero-probability RRXOR path")
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
    joint_state = stationary_prior(
        device=paths.device,
        dtype=operators.dtype,
    ).expand(paths.shape[0], -1)
    for position in range(paths.shape[1]):
        selected = operators.index_select(0, paths[:, position])
        joint_state = torch.bmm(
            joint_state.unsqueeze(1),
            selected,
        ).squeeze(1)
    return joint_state.sum(dim=-1)


def possible_paths(
    length: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return all positive-probability paths and their exact probabilities."""

    paths = enumerate_paths(length, device=device)
    operators = labeled_operators(device=device, dtype=dtype)
    probabilities = path_probabilities(paths, operators=operators)
    possible = probabilities > 0
    return paths[possible], probabilities[possible]


def next_token_probabilities(
    beliefs: torch.Tensor,
    *,
    operators: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map source-state beliefs to exact next-token distributions."""

    operators = (
        labeled_operators(device=beliefs.device, dtype=beliefs.dtype)
        if operators is None
        else operators
    )
    emission_by_state = operators.sum(dim=-1).transpose(0, 1)
    return beliefs @ emission_by_state


def exact_bayesian_loss(
    paths: torch.Tensor,
    probabilities: torch.Tensor,
) -> float:
    """Expected shifted next-token CE over the supplied complete paths."""

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
    """Walker alias table for device-native exact path sampling."""

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
        """Draw path indices from the represented categorical distribution."""

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
