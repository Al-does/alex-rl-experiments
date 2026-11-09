from experiments.belief_quotient_history_REINFORCE_2026_11.analysis import (
    analyze_checkpoint,
)
from experiments.belief_quotient_history_REINFORCE_2026_11.design import (
    CONDITION_NAMES,
    EFFECT_SIZE,
    analytic_design_summary,
    model_kwargs,
)
from experiments.belief_quotient_history_REINFORCE_2026_11.models import model_spec
from experiments.belief_quotient_history_REINFORCE_2026_11.training import (
    SMOKE_ENV_STEPS,
    run_study,
)
from experiments.mess3_reward_state_action_symmetry_cycle_7.shared import (
    build_config as build_source_config,
)


STUDY = "belief_quotient_history_REINFORCE_2026_11"
TOTAL_ENV_STEPS = 10_000_000
ENTROPY_COEFF = 0.03
ENTROPY_ANNEAL_REMAINING_STEPS = 1_000_000


def environment_config(variant):
    if variant not in CONDITION_NAMES:
        raise ValueError("variant must be 1, 2, or 3")
    return {
        "model": {"factory": "envs.mess3.model:control_model", "kwargs": model_kwargs()},
        "task": {
            "class": (
                "experiments.mess3_reward_state_action_symmetry_cycle_7.task:"
                "ActionSymmetryTask"
            ),
            "kwargs": {"variant": variant, "effect_size": EFFECT_SIZE},
        },
        "delay": 0,
        "episode_length": 1024,
        "randomize_first_episode_length": True,
    }


def entropy_schedule(context):
    steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    return [[0, ENTROPY_COEFF], [max(1, steps - ENTROPY_ANNEAL_REMAINING_STEPS), 0.0]]


def build_config(context, variant, *, architecture):
    spec = model_spec(architecture, ladder=True)
    config = build_source_config(context, variant, model_config=spec.model_config)
    return (
        config.environment(env_config=environment_config(variant))
        .rl_module(rl_module_spec=spec)
        .training(entropy_coeff=entropy_schedule(context))
    )


def run_condition(context, variant, *, architecture):
    return run_study(
        context,
        build_config(context, variant, architecture=architecture),
        recipe={
            "study": STUDY,
            "condition": f"variant_{variant}",
            "condition_name": CONDITION_NAMES[variant],
            "architecture": architecture,
            "total_env_steps": TOTAL_ENV_STEPS,
            "baseline_recipe": "mess3_reward_state_action_symmetry_cycle_7/variant_*_ctx32_ent",
            "hypothesis": (
                "history improves control in the two nontrivial quotient conditions; "
                "compare initialization-to-final affine belief accessibility"
            ),
            "analytic_design": analytic_design_summary(),
            "entropy_anneal_remaining_steps": ENTROPY_ANNEAL_REMAINING_STEPS,
            "probe_representation": "post_final_layer_norm",
            "probe_fit_rows": 60_000,
            "probe_test_rows": 80_000,
            "probe_warmup": 64,
            "smoke_probe_fit_rows": 4_096,
            "smoke_probe_test_rows": 4_096,
            "smoke_probe_warmup": 4,
        },
        analyze_checkpoint=analyze_checkpoint,
    )
