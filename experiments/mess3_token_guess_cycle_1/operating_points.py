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

Widening the global-R2 band this way costs the Cantor picture, and that cost is
not incidental. The mixed-state set is the attractor of one contractive map per
token, so disjoint first-level images need a contraction ratio below
``1/sqrt(3)``. Between-branch variance is then at least ``2/3``, and
between-branch variance is exactly what a probe on the last token reads. Visible
gaps and a low last-token floor are the same quantity with opposite signs: box
dimension runs 0.93, 1.11, 1.22, 1.38, 1.68 as ``alpha`` falls 0.85, 0.80, 0.75,
0.70, 0.55, while the last-token floor runs 0.80, 0.72, 0.54, 0.46, 0.26.

``stay`` carries no such cost. Raising it from 0.90 to 0.95 at fixed ``alpha``
leaves the box dimension at 0.93 to 0.85 while accuracy resolution goes from 6
to 21 sigma, so ``CANTOR_SHARP`` and ``CANTOR`` keep the picture and take the
accuracy axis anyway.

The deeper answer is that global R2 is the wrong readout for a Cantor-structured
belief: nearly all of its variance is which cluster you are in, which the last
token already gives away. Within-branch residual R2 discards exactly that term,
so any depth-two token model scores zero by construction at every operating
point.
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
    note="floors cross here, but the mixed-state set is dense, not Cantor",
)
CANTOR_SHARP = OperatingPoint(
    name="cantor_sharp",
    stay=0.95,
    alpha=0.85,
    note="alpha unchanged, so the gaps are as crisp as the shipped point",
)
CANTOR = OperatingPoint(
    name="cantor",
    stay=0.95,
    alpha=0.75,
    note="widest global-R2 band that still shows visible level-one gaps",
)
POINTS = (SHIPPED, PROPOSED)
FRACTAL_POINTS = (CANTOR_SHARP, CANTOR)
ALL_POINTS = POINTS + FRACTAL_POINTS


def point_by_name(name: str) -> OperatingPoint:
    try:
        return next(point for point in ALL_POINTS if point.name == name)
    except StopIteration as error:
        raise ValueError(f"unknown operating point {name!r}") from error
