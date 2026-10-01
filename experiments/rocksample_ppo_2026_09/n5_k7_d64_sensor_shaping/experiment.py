"""d64 PPO on RockSample[5,7] with reward shaping on sensor (check) actions.

The environment subclass adds a decreasing bonus to the first
``BONUS_USES`` check actions per episode: the i-th check earns
``BONUS_BASE * (BONUS_USES - i + 1) / BONUS_USES`` — 0.5, 0.4, 0.3, 0.2,
0.1 for five uses, then nothing. Total shaping potential is +1.5 per
episode against +10/-10 rock and +10 exit rewards, so the bonus nudges
early exploration of the sensor without dominating task reward.
"""

from __future__ import annotations

from envs.rocksample.env import RockSampleEnv
from envs.rocksample.model import Action
from experiments.rocksample_ppo_2026_09 import shared


ENV_CONFIG = {}
D_MODEL = 64


class SensorShapedRockSampleEnv(RockSampleEnv):
    """Adds a decreasing bonus to the first BONUS_USES check actions."""

    BONUS_USES = 5
    BONUS_BASE = 0.5

    def __init__(self, config=None):
        super().__init__(config)
        self._sensor_uses = 0

    def reset(self, *, seed=None, options=None):
        self._sensor_uses = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if int(action) >= len(Action) and self._sensor_uses < self.BONUS_USES:
            bonus = (
                self.BONUS_BASE
                * (self.BONUS_USES - self._sensor_uses)
                / self.BONUS_USES
            )
            self._sensor_uses += 1
            reward = float(reward) + bonus
        return obs, reward, terminated, truncated, info


def build_config(context):
    return shared.build_config(
        context,
        env_config=ENV_CONFIG,
        d_model=D_MODEL,
        env_class=SensorShapedRockSampleEnv,
    )


def run(context):
    return shared.run_condition(
        context,
        env_config=ENV_CONFIG,
        d_model=D_MODEL,
        env_class=SensorShapedRockSampleEnv,
    )
