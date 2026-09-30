"""Learned Policy Optimisation (LPO) drift objective for RLlib PPO.

Mirror-learning view of PPO (Kuba et al. 2022; Lu et al. 2022): the clipped
surrogate is ``r * A - F(r, A)`` with PPO's drift
``F_ppo = relu((r - clip(r, 1 - eps, 1 + eps)) * A)``. LPO replaces ``F`` with
a small network whose parameters are meta-learned in an outer loop. Here the
drift is

    F(r, A) = relu(w_ppo * F_ppo(r, A) + g(x(r, A)) - g(x(1, A)))

where ``g`` is a tanh MLP over ratio/advantage features. ``g``'s output layer
starts at zero and ``w_ppo`` at one, so the initial meta-parameters recover
PPO exactly. Subtracting ``g`` at ``r = 1`` keeps ``F(1, A) = 0``; the outer
relu keeps ``F >= 0``.

The flat meta-parameter vector is
``[g.W1, g.b1, g.W2, g.b2, w_ppo, log_entropy_coeff]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ray.rllib.algorithms.ppo.ppo import (
    LEARNER_RESULTS_KL_KEY,
    LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY,
    LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY,
)
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.learner import ENTROPY_KEY, POLICY_LOSS_KEY, VF_LOSS_KEY
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.utils.torch_utils import explained_variance

NAMESPACE = "meta_rl"
CANDIDATE_KEY = f"{NAMESPACE}/candidate"
DRIFT_METRIC_KEY = f"{NAMESPACE}/drift"
PPO_DRIFT_METRIC_KEY = f"{NAMESPACE}/ppo_drift"
ENTROPY_COEFF_METRIC_KEY = f"{NAMESPACE}/entropy_coeff"

NUM_FEATURES = 8
LOG_RATIO_BOUND = 5.0


@dataclass(frozen=True, slots=True)
class DriftSpec:
    """Shape and PPO reference constants of the learned drift."""

    hidden: int = 32
    clip_param: float = 0.2
    initial_entropy_coeff: float = 0.05

    @property
    def num_params(self) -> int:
        return NUM_FEATURES * self.hidden + self.hidden + self.hidden + 1 + 2

    def initial_params(self, seed: int) -> np.ndarray:
        """PPO-equivalent meta-parameters with a seeded hidden layer."""
        rng = np.random.default_rng(seed)
        w1 = rng.normal(0.0, 1.0 / math.sqrt(NUM_FEATURES), (self.hidden, NUM_FEATURES))
        b1 = np.zeros(self.hidden)
        w2 = np.zeros(self.hidden)
        b2 = np.zeros(1)
        extra = np.array([1.0, math.log(self.initial_entropy_coeff)])
        return np.concatenate([w1.ravel(), b1, w2, b2, extra]).astype(np.float64)


def drift_features(ratio: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    """LPO-style inputs: ratio deviations and their products with ``A``."""
    log_ratio = torch.log(ratio.clamp_min(1e-8)).clamp(
        -LOG_RATIO_BOUND, LOG_RATIO_BOUND
    )
    base = torch.stack(
        [ratio - 1.0, log_ratio, (ratio - 1.0) ** 2, log_ratio**2], dim=-1
    )
    return torch.cat([base, base * advantages.unsqueeze(-1)], dim=-1)


class LearnedDrift(torch.nn.Module):
    """Frozen drift function built from one flat meta-parameter vector."""

    def __init__(self, params, spec: DriftSpec):
        super().__init__()
        vector = torch.as_tensor(np.asarray(params, dtype=np.float32))
        if vector.numel() != spec.num_params:
            raise ValueError(
                f"expected {spec.num_params} meta-parameters, got {vector.numel()}"
            )
        h = spec.hidden
        offset = 0

        def take(count: int) -> torch.Tensor:
            nonlocal offset
            out = vector[offset : offset + count]
            offset += count
            return out

        self.spec = spec
        self.register_buffer("w1", take(NUM_FEATURES * h).reshape(h, NUM_FEATURES))
        self.register_buffer("b1", take(h))
        self.register_buffer("w2", take(h))
        self.register_buffer("b2", take(1))
        self.register_buffer("w_ppo", take(1))
        self.register_buffer("log_entropy_coeff", take(1))

    @property
    def entropy_coeff(self) -> torch.Tensor:
        return self.log_entropy_coeff.exp().squeeze(0)

    def _g(self, features: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(features @ self.w1.T + self.b1)
        return hidden @ self.w2 + self.b2

    def ppo_drift(self, ratio: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        eps = self.spec.clip_param
        return F.relu((ratio - ratio.clamp(1.0 - eps, 1.0 + eps)) * advantages)

    def forward(self, ratio: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        learned = self._g(drift_features(ratio, advantages)) - self._g(
            drift_features(torch.ones_like(ratio), advantages)
        )
        return F.relu(self.w_ppo * self.ppo_drift(ratio, advantages) + learned)


class LearnedDriftPPOLearner(PPOTorchLearner):
    """PPO Learner whose surrogate is ``r * A - F_theta(r, A)``.

    Reads ``meta_rl/candidate`` from ``learner_config_dict``: a mapping with
    ``params`` (flat meta-parameter list) and ``hidden``/``clip_param``. The
    meta-learned entropy coefficient replaces RLlib's scheduler value. Critic
    loss, adaptive KL penalty, advantage estimation and masking are PPO's.
    """

    def build(self):
        super().build()
        self._drifts = {}

    def _drift(self, module_id, config) -> LearnedDrift:
        drift = self._drifts.get(module_id)
        if drift is None:
            candidate = config.learner_config_dict[CANDIDATE_KEY]
            spec = DriftSpec(
                hidden=int(candidate["hidden"]),
                clip_param=float(candidate["clip_param"]),
            )
            drift = LearnedDrift(candidate["params"], spec).to(self._device)
            self._drifts[module_id] = drift
        return drift

    def compute_loss_for_module(self, *, module_id, config, batch, fwd_out):
        module = self.module[module_id].unwrapped()
        drift_fn = self._drift(module_id, config)

        if Columns.LOSS_MASK in batch:
            mask = batch[Columns.LOSS_MASK]
            num_valid = torch.sum(mask)

            def masked_mean(data):
                return torch.sum(data[mask]) / num_valid

        else:
            masked_mean = torch.mean

        dist = module.get_train_action_dist_cls().from_logits(
            fwd_out[Columns.ACTION_DIST_INPUTS]
        )
        prev_dist = module.get_exploration_action_dist_cls().from_logits(
            batch[Columns.ACTION_DIST_INPUTS]
        )
        ratio = torch.exp(
            dist.logp(batch[Columns.ACTIONS]) - batch[Columns.ACTION_LOGP]
        )
        advantages = batch[Postprocessing.ADVANTAGES]

        drift = drift_fn(ratio, advantages)
        surrogate = ratio * advantages - drift
        entropy = dist.entropy()
        entropy_coeff = drift_fn.entropy_coeff

        values = module.compute_values(
            batch, embeddings=fwd_out.get(Columns.EMBEDDINGS)
        )
        vf_loss = torch.pow(values - batch[Postprocessing.VALUE_TARGETS], 2.0)
        vf_loss_clipped = torch.clamp(vf_loss, 0, config.vf_clip_param)

        total = masked_mean(
            -surrogate
            + config.vf_loss_coeff * vf_loss_clipped
            - entropy_coeff * entropy
        )

        if config.use_kl_loss:
            action_kl = masked_mean(prev_dist.kl(dist))
            total = total + self.curr_kl_coeffs_per_module[module_id] * action_kl
        else:
            action_kl = torch.zeros((), device=ratio.device)
        self.metrics.log_dict(
            {
                POLICY_LOSS_KEY: -masked_mean(surrogate),
                VF_LOSS_KEY: masked_mean(vf_loss_clipped),
                LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY: masked_mean(vf_loss),
                LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY: explained_variance(
                    batch[Postprocessing.VALUE_TARGETS], values
                ),
                ENTROPY_KEY: masked_mean(entropy),
                LEARNER_RESULTS_KL_KEY: action_kl,
                DRIFT_METRIC_KEY: masked_mean(drift),
                PPO_DRIFT_METRIC_KEY: masked_mean(
                    drift_fn.ppo_drift(ratio, advantages)
                ),
                ENTROPY_COEFF_METRIC_KEY: entropy_coeff,
            },
            key=module_id,
            window=1,
        )
        return total
