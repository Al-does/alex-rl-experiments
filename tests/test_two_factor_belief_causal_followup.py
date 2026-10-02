import numpy as np
import torch

from experiments.two_factor_reward_state_REINFORCE_cycle_4.belief_causal_followup import (
    BeliefMap,
    ResidualSites,
    opposite_delta,
)


def test_joint_opposite_shift_preserves_first_marginal_before_embedding():
    joint = np.array([
        [0.15, 0.05, 0.10, 0.05, 0.20, 0.05, 0.05, 0.10, 0.25],
        [0.10, 0.10, 0.10, 0.20, 0.05, 0.05, 0.10, 0.20, 0.10],
    ])
    activation = joint[:, :8]
    mapping = BeliefMap(
        decoder=np.eye(8), bias=np.zeros(8), embedding=np.eye(8),
        mean=joint[:, :8].mean(axis=0), target="joint",
    )
    altered = activation + opposite_delta(activation, mapping)
    updated = np.column_stack((altered, 1 - altered.sum(axis=1))).reshape(-1, 3, 3)
    np.testing.assert_allclose(updated.sum(axis=2), joint.reshape(-1, 3, 3).sum(axis=2))
    np.testing.assert_allclose(updated[:, :, 2], 0, atol=1e-12)
    np.testing.assert_allclose(updated.sum(axis=(1, 2)), 1)


class TinyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln2 = torch.nn.LayerNorm(4)
        self.mlp = torch.nn.Linear(4, 4, bias=False)

    def forward_step(self, tensor):
        return tensor + self.mlp(self.ln2(tensor))


class TinyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.blocks = torch.nn.ModuleList([TinyBlock(), TinyBlock()])

    def encode_step(self, tensor):
        x = tensor[:, None, :]
        for block in self.encoder.blocks:
            x = block.forward_step(x)
        return x[:, 0, :]


def test_cached_step_hooks_capture_residual_and_inject_before_downstream_block():
    torch.manual_seed(3)
    module = TinyModule()
    observation = torch.randn(2, 4)
    unmodified = module.encode_step(observation)
    with ResidualSites(module) as sites:
        assert torch.allclose(module.encode_step(observation), unmodified)
        first = sites.activations[0].clone()
        assert torch.allclose(sites.activations[1], unmodified)
        sites.steer_layer = 0
        sites.mapping = BeliefMap(
            decoder=torch.eye(4).numpy(), bias=np.zeros(4),
            embedding=np.eye(4), mean=np.zeros(4), target="marginal",
        )
        sites.strength = 0.5
        modified = module.encode_step(observation)
        expected_first = first + 0.5 * torch.as_tensor(
            opposite_delta(first.detach().numpy(), sites.mapping),
            dtype=first.dtype,
        )
        expected_final = module.encoder.blocks[1].forward_step(expected_first[:, None, :])[:, 0, :]
        assert torch.allclose(modified, expected_final, atol=1e-6)
        assert not torch.allclose(modified, unmodified)
    assert torch.allclose(module.encode_step(observation), unmodified)
