"""Named MESS3 operating points for the token-guess task.

The shipped point was inherited rather than chosen. Measured with
``process_design``, it leaves almost no room on either reported axis: the best
policy that looks only at the last token is within 0.017 of Bayes, and an
8-token window already explains 0.964 of the belief variance a probe could
ever explain. Both numbers are properties of the process alone.

``PROPOSED`` was selected by scanning the symmetric family. Three floors have
to be cleared before a result means anything, and they do not move together.

* A short token window. Falls as the channel gets noisier.
* The Bayes argmax cell, which any task-solving agent must encode anyway.
  Rises as the chain gets stickier and the belief concentrates on the decision.
* A randomly initialised transformer, measured rather than derived. It stays
  near 0.82-0.88 across the whole family, and *rises* as ``alpha`` falls,
  because a smoother belief is easier for a random causal filter to track.

``stay`` is set where the first two cross, and ``alpha`` where the third
crosses them. At ``stay=0.96`` the binding floor by ``alpha`` runs
0.859, 0.831, 0.852, 0.882 for 0.60, 0.55, 0.50, 0.45, so 0.55 leaves the most
room. Chain memory must also outlast the channel: below ``stay > alpha`` the
Bayes-optimal guess is the last token at every step and the accuracy axis is
exactly zero wide.

The token alphabet is deliberately left at three. A policy objective can only
reward what the one-step predictive distribution distinguishes, and that
distribution determines the belief only when the emission matrix has full rank.
Shrinking the alphabet to two lowers every window floor but caps the probe at
0.60, which is a worse instrument despite the lower floors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


TASK_PATH = "experiments.mess3_token_guess_cycle_1.task:NextTokenGuessTask"
EPISODE_LENGTH = 512


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """One MESS3 parameterisation with the delay-one token-guess task."""

    name: str
    stay: float
    alpha: float
    note: str

    def env_config(self, **diagnostics: bool) -> dict[str, Any]:
        config: dict[str, Any] = {
            "model": {
                "factory": "envs.mess3.model:symmetric_model",
                "kwargs": {"stay": self.stay, "alpha": self.alpha},
            },
            "task": {"class": TASK_PATH},
            "observation": {"action": None},
            "delay": 1,
            "episode_length": EPISODE_LENGTH,
            "randomize_first_episode_length": True,
        }
        if diagnostics:
            config["diagnostics"] = dict(diagnostics)
        return config


SHIPPED = OperatingPoint(
    name="shipped",
    stay=0.90,
    alpha=0.85,
    note="inherited from mess3_belief_geometry; barely clears stay > alpha",
)
PROPOSED = OperatingPoint(
    name="proposed",
    stay=0.96,
    alpha=0.55,
    note="where the window, argmax-cell, and untrained-network floors cross",
)
POINTS = (SHIPPED, PROPOSED)


def point_by_name(name: str) -> OperatingPoint:
    try:
        return next(point for point in POINTS if point.name == name)
    except StopIteration as error:
        raise ValueError(f"unknown operating point {name!r}") from error
