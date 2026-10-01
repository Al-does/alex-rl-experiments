"""d64 PPO on RockSample[5,7] 10M with trapezoidal visit shaping.

Same potential as n5_k7_d64_visit_shaping (Phi = -0.2 * manhattan to the
nearest unvisited rock, added as gamma*Phi(s') - Phi(s)), but scheduled
as a trapezoid over training instead of front-loaded: shaping is OFF
for the first phase so the policy learns core task competence on clean
rewards, ramps in around the point earlier runs stalled (~1M steps, the
+10 plateau region), holds, ramps back out, and leaves a long clean tail.

    weight(t): 0  for t < RAMP_UP_START
               0 -> 1  linear over [RAMP_UP_START, HOLD_START]
               1  for t in [HOLD_START, RAMP_DOWN_START]
               1 -> 0  linear over [RAMP_DOWN_START, ANNEAL_END]
               0  after ANNEAL_END

t is this env instance's lifetime step counter; at 4 runners x 24 envs
the env-local values below map to roughly 1.0M / 1.5M / 4.2M / 5.0M
global steps, leaving ~5M of unshaped training inside the 10M budget.
"""

from __future__ import annotations

from envs.rocksample.env import RockSampleEnv
from experiments.rocksample_ppo_2026_09 import shared


ENV_CONFIG = {}
D_MODEL = 64
TOTAL_ENV_STEPS = 10_000_000


class VisitShapedRockSampleEnv(RockSampleEnv):
    """Potential-based shaping toward the nearest unvisited rock."""

    GAMMA = 0.99
    POTENTIAL_WEIGHT = 0.2
    RAMP_UP_START = 10_000
    HOLD_START = 15_000
    RAMP_DOWN_START = 43_000
    ANNEAL_END = 52_000

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
        if self._t < self.RAMP_UP_START or self._t >= self.ANNEAL_END:
            return 0.0
        if self._t < self.HOLD_START:
            return (self._t - self.RAMP_UP_START) / (
                self.HOLD_START - self.RAMP_UP_START
            )
        if self._t < self.RAMP_DOWN_START:
            return 1.0
        return 1.0 - (self._t - self.RAMP_DOWN_START) / (
            self.ANNEAL_END - self.RAMP_DOWN_START
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
        total_env_steps=TOTAL_ENV_STEPS,
        env_class=VisitShapedRockSampleEnv,
    )


def run(context):
    return shared.run_condition(
        context,
        env_config=ENV_CONFIG,
        d_model=D_MODEL,
        total_env_steps=TOTAL_ENV_STEPS,
        env_class=VisitShapedRockSampleEnv,
    )
