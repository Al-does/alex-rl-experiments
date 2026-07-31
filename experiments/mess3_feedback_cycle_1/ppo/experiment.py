"""Two-million-step plain PPO action-feedback study."""

from harness.context import RunContext

from experiments.mess3_feedback_cycle_1.shared import build_config, run_condition


def run(context: RunContext):
    return run_condition(context)


__all__ = ["build_config", "run"]
