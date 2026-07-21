"""Correctness PPO plus scalar Kelly shaping with an IQN critic."""

from experiments.mess_3_kelly_cycle_2.shared import build_config as _build, run_condition

ARM = "decoupled_kelly_iqn"


def build_config(context):
    return _build(context, ARM)


def run(context):
    return run_condition(context, ARM)
