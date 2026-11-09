import copy

import torch

from analysis.checkpoints import load_algorithm
from analysis.rollouts import collect_batched_rollout_data
from envs.hmm import HMMEnv
from experiments.factored_representations_reproduction_PPO_2026_08.probe import (
    _initial_state,
)
from experiments.reward_factors_history_REINFORCE_2026_11.process import (
    environment_config,
)
from experiments.two_factor_reward_state_SAC_cycle_1.analysis import (
    FULL_PROBE_TEST_STEPS,
    FULL_PROBE_TRAIN_STEPS,
    N_ENVS,
    SMOKE_PROBE_STEPS,
    WARMUP,
)
from experiments.two_factor_reward_state_SAC_cycle_2.analysis import (
    _STREAM_KEYS,
    _analyze_samples,
    _device,
    _probe_data,
    _target_adapter,
)
from harness.seeding import named_seed_sequences


@torch.inference_mode()
def collect_probe_data(module, *, config, n_steps, seed, device):
    config = copy.deepcopy(config)
    config["diagnostics"] = {"belief": True, "state": True}
    module = module.to(device).eval()

    def initial_state(batch_size):
        return _initial_state(module, batch_size, device)

    def reset_state(state, indices):
        fresh = initial_state(len(indices))
        index = torch.as_tensor(indices, dtype=torch.long, device=device)
        for key, value in state.items():
            value.index_copy_(0, index, fresh[key])
        return state

    def step_adapter(observations, state, randomness, action_spaces):
        tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
        residual, state_out = module.encode_step_pre_final_norm(tensor, state)
        logits = module.action_distribution_inputs(module.encoder.final_norm(residual))
        return logits.argmax(dim=-1).cpu().numpy(), state_out, residual.cpu().numpy()

    collected = collect_batched_rollout_data(
        lambda: HMMEnv(config),
        step_adapter,
        _target_adapter,
        n_steps=n_steps,
        seed=seed,
        n_envs=N_ENVS,
        initial_state=initial_state,
        reset_state=reset_state,
        warmup=WARMUP,
    )
    data = _probe_data(collected)
    if data.product_consistency_max_abs > 1e-10:
        raise AssertionError("independent-factor filter targets must remain factorized")
    return data


def analyze_checkpoint(
    context,
    *,
    checkpoint,
    condition,
    checkpoint_label,
    agent_steps,
    training_iteration,
):
    if context.seed is None:
        raise ValueError("belief probing requires a resolved seed")
    streams = named_seed_sequences(context.seed, _STREAM_KEYS)
    train_steps = SMOKE_PROBE_STEPS if context.smoke else FULL_PROBE_TRAIN_STEPS
    test_steps = SMOKE_PROBE_STEPS if context.smoke else FULL_PROBE_TEST_STEPS
    with load_algorithm(checkpoint) as algorithm:
        module = algorithm.get_module()
        config = dict(algorithm.config.env_config)
        if config != environment_config(condition):
            raise ValueError("checkpoint environment does not match the new factor study")
        common = {"config": config, "device": _device(context)}
        train = collect_probe_data(
            module, n_steps=train_steps, seed=streams["probe_train"], **common
        )
        test = collect_probe_data(
            module, n_steps=test_steps, seed=streams["probe_test"], **common
        )
    return _analyze_samples(
        context,
        condition=condition,
        checkpoint_label=checkpoint_label,
        agent_steps=agent_steps,
        training_iteration=training_iteration,
        train=train,
        test=test,
        streams=streams,
        representation="final_block_residual_before_final_layer_norm",
    )
