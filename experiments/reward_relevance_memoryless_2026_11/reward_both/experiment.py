"""reward_relevance_memoryless_2026_11: reward_both."""

from experiments.policy_baselines_2026_11 import (
    build_config as _build_config,
    run_condition,
)

STUDY = "reward_relevance_memoryless_2026_11"
CONDITION = "reward_both"
ARCHITECTURE = "memoryless"


def build_config(context):
    return _build_config(context, CONDITION, architecture=ARCHITECTURE, ladder=False)


def run(context):
    return run_condition(context, CONDITION, study=STUDY,
                         architecture=ARCHITECTURE, ladder=False)
