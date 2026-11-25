"""PPO with policy and value readouts on independent transformer trunks."""
from dataclasses import replace
from functools import partial

import jax

from experiments.belief_quotient_history_jax_2026_11 import models, reinforce
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_ppo_2026_11 import ppo


def init_params(spec, key):
    params = ppo.init_params(spec, key)
    critic = models.init_params(replace(spec, n_actions=1), jax.random.fold_in(key, 18))
    critic["policy"] = params.pop("value")
    params["critic"] = critic
    return params


def forward(spec, params, obs):
    logits, _, _ = models.sequence(spec, params, obs)
    values, _, _ = models.sequence(spec, params["critic"], obs)
    return logits, values[..., 0]


train = partial(
    reinforce.train, init_fn=init_params,
    iteration_fn=partial(ppo.make_iteration, forward_fn=forward),
)
