"""Gamma-.99 correctness PPO, conditional Kelly shaping, and IQN."""

from experiments.mess_3_kelly_cycle_3.shared import build_config as _build, run_condition

ARM = "conditional_decoupled_kelly_iqn"


def build_config(context):
    return _build(context, ARM)


def run(context):
    return run_condition(context, ARM)
