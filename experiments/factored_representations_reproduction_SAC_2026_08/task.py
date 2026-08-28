"""The unchanged rewarded next-joint-token task for the SAC comparison."""

from experiments.factored_representations_reproduction_PPO_2026_08.task import (
    NextJointTokenGuessTask as _PPONextJointTokenGuessTask,
)


class NextJointTokenGuessTask(_PPONextJointTokenGuessTask):
    """Keep SAC's serialized task path local while preserving task semantics."""
