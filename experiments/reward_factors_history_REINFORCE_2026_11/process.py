from experiments.belief_quotient_history_REINFORCE_2026_11.design import model_kwargs
from experiments.two_factor_reward_state_SAC_cycle_2.task import CONDITIONS


CONDITION_NAMES = {
    "reward_both": "Joint Reward",
    "reward_factor_1": "Unrewarded Factor 2",
}


def environment_config(condition):
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    return {
        "model": {
            "factory": "envs.hmm:factored_model",
            "kwargs": {
                "factors": [
                    {
                        "factory": "envs.mess3.model:control_model",
                        "kwargs": model_kwargs(),
                    }
                    for _ in range(2)
                ]
            },
        },
        "task": {
            "class": (
                "experiments.reward_factors_history_REINFORCE_2026_11.task:"
                "HistoryRewardFactorsTask"
            ),
            "kwargs": {"condition": condition},
        },
        "observation": {"token": {"depth": 1}, "action": {"depth": 1}},
        "delay": 0,
        "episode_length": 1024,
        "randomize_first_episode_length": True,
    }
