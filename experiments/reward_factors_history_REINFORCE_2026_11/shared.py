from experiments.belief_quotient_history_REINFORCE_2026_11.design import (
    analytic_design_summary,
)
from experiments.belief_quotient_history_REINFORCE_2026_11.models import model_spec
from experiments.belief_quotient_history_REINFORCE_2026_11.training import run_study
from experiments.reward_factors_history_REINFORCE_2026_11.analysis import (
    analyze_checkpoint,
)
from experiments.reward_factors_history_REINFORCE_2026_11.process import (
    CONDITION_NAMES,
    environment_config,
)
from experiments.two_factor_reward_state_REINFORCE_cycle_4.shared import (
    build_config as build_source_config,
)
from experiments.two_factor_reward_state_SAC_cycle_1.analysis import (
    FULL_PROBE_TEST_STEPS,
    FULL_PROBE_TRAIN_STEPS,
    SMOKE_PROBE_STEPS,
    WARMUP,
)
from experiments.two_factor_reward_state_SAC_cycle_2.task import (
    ACTION_LABELS,
    ACTION_PAIRS,
)


STUDY = "reward_factors_history_REINFORCE_2026_11"
TOTAL_ENV_STEPS = 30_000_000


def build_config(context, condition, *, architecture):
    return (
        build_source_config(context, condition)
        .environment(env_config=environment_config(condition))
        .rl_module(rl_module_spec=model_spec(architecture, ladder=False))
    )


def condition_recipe(condition, *, architecture):
    return {
        "study": STUDY,
        "condition": condition,
        "condition_name": CONDITION_NAMES[condition],
        "architecture": architecture,
        "total_env_steps": TOTAL_ENV_STEPS,
        "baseline_recipe": "two_factor_reward_state_REINFORCE_cycle_4",
        "hypothesis": (
            "reward relevance selectively changes factor-belief accessibility "
            "when temporal integration can improve control"
        ),
        "factor_count": 2,
        "factor_design": analytic_design_summary(),
        "joint_reference_caveat": (
            "single-factor memoryless bounds do not establish a joint-token "
            "memoryless bound; joint reactive policies may couple factor actions"
        ),
        "action_pairs_by_flat_index": [list(pair) for pair in ACTION_PAIRS],
        "action_labels_by_flat_index": list(ACTION_LABELS),
        "probe_representation": "pre_final_layer_norm",
        "probe_fit_rows": FULL_PROBE_TRAIN_STEPS,
        "probe_test_rows": FULL_PROBE_TEST_STEPS,
        "probe_warmup": WARMUP,
        "smoke_probe_fit_rows": SMOKE_PROBE_STEPS,
        "smoke_probe_test_rows": SMOKE_PROBE_STEPS,
    }


def run_condition(context, condition, *, architecture):
    return run_study(
        context,
        build_config(context, condition, architecture=architecture),
        recipe=condition_recipe(condition, architecture=architecture),
        analyze_checkpoint=analyze_checkpoint,
    )
