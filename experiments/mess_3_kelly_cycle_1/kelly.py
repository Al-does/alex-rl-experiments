"""Pure Kelly-betting math for the three-token experiment."""

from __future__ import annotations

import numpy as np
import torch


N_OUTCOMES = 3
NET_WIN_ODDS = float(N_OUTCOMES - 1)
MAX_WAGER = 1.0 - 1e-4
COLLAPSE_THRESHOLD = 0.01


def kelly_fraction(
    probability,
    *,
    n_outcomes: int = N_OUTCOMES,
    max_wager: float = MAX_WAGER,
):
    """Return the fair-odds Kelly fraction for one selected outcome."""

    if n_outcomes < 2:
        raise ValueError("n_outcomes must be at least two")
    if not 0.0 < max_wager < 1.0:
        raise ValueError("max_wager must lie strictly between zero and one")
    fraction = (n_outcomes * probability - 1.0) / (n_outcomes - 1.0)
    if isinstance(probability, torch.Tensor):
        return fraction.clamp(min=0.0, max=max_wager)
    return np.clip(fraction, 0.0, max_wager)


def realized_log_growth(correct, wager, *, net_win_odds: float = NET_WIN_ODDS):
    """Return log bankroll growth for observed outcomes and wager fractions."""

    if net_win_odds <= 0.0:
        raise ValueError("net_win_odds must be positive")
    if isinstance(wager, torch.Tensor):
        won = torch.log1p(wager * net_win_odds)
        lost = torch.log1p(-wager)
        return torch.where(correct.to(dtype=torch.bool), won, lost)
    wager_array = np.asarray(wager)
    return np.where(
        np.asarray(correct, dtype=bool),
        np.log1p(wager_array * net_win_odds),
        np.log1p(-wager_array),
    )


def expected_log_growth(
    probability,
    wager,
    *,
    net_win_odds: float = NET_WIN_ODDS,
):
    """Return expected log growth under a calibrated correctness probability."""

    if isinstance(wager, torch.Tensor):
        return probability * torch.log1p(wager * net_win_odds) + (
            1.0 - probability
        ) * torch.log1p(-wager)
    probability_array = np.asarray(probability)
    wager_array = np.asarray(wager)
    return probability_array * np.log1p(wager_array * net_win_odds) + (
        1.0 - probability_array
    ) * np.log1p(-wager_array)
