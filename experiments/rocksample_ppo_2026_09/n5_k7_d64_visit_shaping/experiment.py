"""d64 PPO on RockSample[5,7] with annealed potential-based visit shaping.

Motivation: the baseline d64 policy exits after ~1.3 sampled rocks even
though the ideal tour (visit every rock, check on-cell, sample if good)
is worth ~+45. This subclass adds potential-based reward shaping toward
unvisited rocks so coverage pays during learning:

    r' = r + w * (gamma * Phi(s') - Phi(s))
    Phi(s) = -POTENTIAL_WEIGHT * min_manhattan(rover, unvisited rocks)

A rock counts as visited once the rover has occupied its cell. Phi is 0
when every rock is visited, and it is also evaluated on terminal
transitions (no artificial zeroing) so early exits while rocks remain
unvisited lose a small amount of shaped return.

The weight w anneals on this env instance's lifetime step counter: held
at 1.0 for HOLD_STEPS, then linear to 0 by ANNEAL_END_STEPS, after which
the env returns unshaped rewards for the rest of training. At 4 runners
x 24 envs, 7.5k env-local steps is ~720k global steps (~72% of a 1M
budget): the shaped phase bootstraps the tour habit and the final phase
trains on the clean task reward.
"""

from __future__ import annotations

from envs.rocksample.env import RockSampleEnv
from experiments.rocksample_ppo_2026_09 import shared


ENV_CONFIG = {}
D_MODEL = 64


class VisitShapedRockSampleEnv(RockSampleEnv):
    """Potential-based shaping toward the nearest unvisited rock."""

    GAMMA = 0.99
    POTENTIAL_WEIGHT = 0.2
    HOLD_STEPS = 2_500
    ANNEAL_END_STEPS = 7_500

    def __init__(self, config=None):
        super().__init__(config)
        self._t = 0
        self._visited: set[tuple[int, int]] = set()
        self._last_potential = 0.0

    def _potential(self) -> float:
        rover = tuple(self._rover)
        unvisited = [
            cell for cell in self._rock_by_position if cell not in self._visited
        ]
        if not unvisited:
            return 0.0
        d = min(abs(rover[0] - x) + abs(rover[1] - y) for x, y in unvisited)
        return -self.POTENTIAL_WEIGHT * d

    def _weight(self) -> float:
        if self._t <= self.HOLD_STEPS:
            return 1.0
        if self._t >= self.ANNEAL_END_STEPS:
            return 0.0
        return 1.0 - (self._t - self.HOLD_STEPS) / (
            self.ANNEAL_END_STEPS - self.HOLD_STEPS
        )

    def reset(self, *, seed=None, options=None):
        self._visited = set()
        out = super().reset(seed=seed, options=options)
        self._last_potential = self._potential()
        return out

    def step(self, action):
        phi_before = self._last_potential
        obs, reward, terminated, truncated, info = super().step(action)
        self._t += 1
        rover = tuple(self._rover)
        if rover in self._rock_by_position:
            self._visited.add(rover)
        phi_after = self._potential()
        self._last_potential = phi_after
        reward = float(reward) + self._weight() * (
            self.GAMMA * phi_after - phi_before
        )
        return obs, reward, terminated, truncated, info


def build_config(context):
    return shared.build_config(
        context,
        env_config=ENV_CONFIG,
        d_model=D_MODEL,
        env_class=VisitShapedRockSampleEnv,
    )


def run(context):
    return shared.run_condition(
        context,
        env_config=ENV_CONFIG,
        d_model=D_MODEL,
        env_class=VisitShapedRockSampleEnv,
    )
