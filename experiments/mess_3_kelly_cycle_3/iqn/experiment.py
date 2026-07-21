"""Gamma-.99 correctness PPO with an IQN critic."""

from experiments.mess_3_kelly_cycle_3.shared import build_config as _build, run_condition

ARM = "iqn"


def build_config(context):
    return _build(context, ARM)


def run(context):
    return run_condition(context, ARM)
