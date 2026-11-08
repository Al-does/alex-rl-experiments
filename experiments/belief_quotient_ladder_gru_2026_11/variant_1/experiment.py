"""belief_quotient_ladder_gru_2026_11: variant_1."""

from experiments.policy_baselines_2026_11 import (
    build_config as _build_config,
    run_condition,
)

STUDY = "belief_quotient_ladder_gru_2026_11"
CONDITION = 1
ARCHITECTURE = "gru"


def build_config(context):
    return _build_config(context, CONDITION, architecture=ARCHITECTURE, ladder=True)


def run(context):
    return run_condition(context, CONDITION, study=STUDY,
                         architecture=ARCHITECTURE, ladder=True)
