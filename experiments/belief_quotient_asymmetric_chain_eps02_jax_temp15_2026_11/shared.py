"""Asymmetric-future uncontrolled chain (1 -> 2 -> 3 -> 1) separating D_V from D_R."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from experiments.belief_quotient_history_jax_2026_11.env import make_process
from experiments.belief_quotient_history_jax_2026_11.shared import model_spec, train_config
from experiments.belief_quotient_history_jax_2026_11.training import run_study
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11.environment import (
    EMISSION_MATRIX,
    INITIAL_STATE_DISTRIBUTION,
    REWARD_STATE,
)
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_ppo_2026_11 import ppo
from experiments.belief_quotient_persistent_nuisance_jax_temp15_2026_11.shared import (
    ENTROPY_SCHEDULE,
    TEMPERATURE,
    TOTAL_ENV_STEPS,
)

STUDY = "belief_quotient_asymmetric_chain_eps02_jax_temp15_2026_11"
TRANSITION_MATRIX = np.array([
    [0.99, 0.01, 0.00],
    [0.00, 0.98, 0.02],
    [0.05, 0.00, 0.95],
], dtype=np.float64)
N_ACTIONS = 3
GAMMA_DISCOUNTED = 0.99

_REWARD = (np.arange(3) == REWARD_STATE).astype(np.float64)
_PI = np.eye(3) - 1.0 / 3.0
ALPHA_GAMMA99 = np.linalg.solve(np.eye(3) - GAMMA_DISCOUNTED * TRANSITION_MATRIX, _REWARD)
_U_V_RAW = ALPHA_GAMMA99 @ _PI
U_V = _U_V_RAW / np.linalg.norm(_U_V_RAW)
U_PERP = np.cross(U_V, np.ones(3) / np.sqrt(3.0))
U_PERP = U_PERP / np.linalg.norm(U_PERP)
RHO_DIR = np.array([-1.0, -1.0, 2.0]) / np.sqrt(6.0)
DELTA_DIR = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)

ARMS = {
    "ppo_gamma99": {"algorithm": "PPO", "gamma": 0.99},
    "ppo_gamma0": {"algorithm": "PPO", "gamma": 0.0},
    "reinforce_gamma99": {"algorithm": "REINFORCE", "gamma": 0.99},
}


def chain_process():
    transitions = np.broadcast_to(TRANSITION_MATRIX[None], (N_ACTIONS, 3, 3)).copy()
    return make_process(
        INITIAL_STATE_DISTRIBUTION.copy(),
        transitions,
        EMISSION_MATRIX.copy(),
        _REWARD.copy(),
        _REWARD[None].copy(),
    )


def chain_targets(belief):
    return {
        "belief": belief,
        "rho": belief[:, 2:3],
        "delta": belief[:, 0:1] - belief[:, 1:2],
        "y_v": (belief @ U_V)[:, None],
        "y_perp": (belief @ U_PERP)[:, None],
    }


def _ppo_backend(gamma):
    def configure(config):
        return replace(ppo.configure(config), gamma=gamma)
    return SimpleNamespace(configure=configure, train=ppo.train)


def environment_metadata():
    return {
        "environment_revision": "asymmetric_chain_eps02",
        "transition_matrix": TRANSITION_MATRIX.tolist(),
        "emission_matrix": EMISSION_MATRIX.tolist(),
        "initial_state_distribution": INITIAL_STATE_DISTRIBUTION.tolist(),
        "reward_state": REWARD_STATE,
        "alpha_gamma99": ALPHA_GAMMA99.tolist(),
        "u_v": U_V.tolist(),
        "u_perp": U_PERP.tolist(),
    }


def run_condition(context, *, arm):
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {tuple(ARMS)}")
    spec_arm = ARMS[arm]
    process = chain_process()
    config = train_config(context.smoke, TOTAL_ENV_STEPS)
    config = replace(
        config,
        entropy_schedule=tuple(ENTROPY_SCHEDULE),
        temperature=TEMPERATURE,
        exploration_epsilon=0.0,
        gamma=spec_arm["gamma"],
    )
    trainer_kwargs = {}
    if spec_arm["algorithm"] == "PPO":
        backend = _ppo_backend(spec_arm["gamma"])
        config = backend.configure(config)
        trainer_kwargs["trainer"] = backend.train
    return run_study(
        context,
        study=STUDY,
        condition=arm,
        architecture="transformer",
        process=process,
        spec=model_spec("transformer", process),
        config=config,
        targets=chain_targets,
        representation="embedding",
        recipe_extra={
            **environment_metadata(),
            "arm": arm,
            "reward": "pre-transition state-2 occupancy",
            "representation": "post_final_layer_norm",
        },
        **trainer_kwargs,
    )
