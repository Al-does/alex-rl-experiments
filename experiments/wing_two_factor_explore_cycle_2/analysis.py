from __future__ import annotations

from pathlib import Path
from typing import Any

from envs.wing.model import wing_model
from experiments.wing_two_factor_explore_cycle_1 import analysis as probe
from experiments.wing_two_factor_explore_cycle_2.process import (
    REWARD_STATE,
    WING_ALPHA,
    WING_X,
    environment_config,
)
from harness.context import RunContext
from harness.seeding import named_seed_sequences


def analyze_checkpoint(
    context: RunContext,
    *,
    checkpoint: Path,
    condition: str,
    checkpoint_label: str,
    agent_steps: int,
    training_iteration: int,
) -> dict[str, Any]:
    if context.seed is None:
        raise ValueError("belief probing requires a resolved seed")
    streams = named_seed_sequences(context.seed, probe._STREAM_KEYS)
    train_steps = probe.SMOKE_PROBE_STEPS if context.smoke else probe.FULL_PROBE_TRAIN_STEPS
    test_steps = probe.SMOKE_PROBE_STEPS if context.smoke else probe.FULL_PROBE_TEST_STEPS
    config = environment_config(condition)
    with probe.load_algorithm(checkpoint) as algorithm:
        module = algorithm.get_module()
        if module is None:
            raise KeyError("checkpoint has no default RLModule")
        train = probe.collect_probe_data(
            module, condition=condition, reward_state=REWARD_STATE,
            n_steps=train_steps, seed=streams["probe_train"], device=probe._device(context),
            env_config=config,
        )
        test = probe.collect_probe_data(
            module, condition=condition, reward_state=REWARD_STATE,
            n_steps=test_steps, seed=streams["probe_test"], device=probe._device(context),
            env_config=config,
        )
    return probe._analyze_samples(
        context, condition=condition, reward_state=REWARD_STATE,
        checkpoint_label=checkpoint_label, agent_steps=agent_steps,
        training_iteration=training_iteration, train=train, test=test,
        streams=streams,
        emission_matrix=wing_model(alpha=WING_ALPHA, x=WING_X).emission_matrix,
    )
