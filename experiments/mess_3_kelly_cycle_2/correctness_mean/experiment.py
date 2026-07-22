"""Straight gamma-zero correctness PPO with a scalar critic."""

from experiments.mess_3_kelly_cycle_2.shared import build_config as _build, run_condition

ARM = "correctness_mean"


def build_config(context):
    return _build(context, ARM)


def run(context):
    return run_condition(context, ARM)
