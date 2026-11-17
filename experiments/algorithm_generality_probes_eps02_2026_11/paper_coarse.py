"""Unmodified function extracts: deep-beliefs@177ba8d9081933b83a3834b7a1f92ac31bb3f02b experiments/persistent_nuisance_eps02/compare_coarse.py."""

import numpy as np


def merged_filter(observations, process):
    """Three-state posterior given executed actions and A/B-merged tokens."""
    obs = np.asarray(observations)
    emission = np.asarray(process["emission"], dtype=np.float64)
    transitions = np.asarray(process["transitions"], dtype=np.float64)
    if obs.ndim != 3 or not obs.shape[0] or not obs.shape[1] or obs.shape[2] != 6:
        raise ValueError("expected nonempty episode histories with three tokens and actions")
    emission = np.stack((emission[:, :2].sum(1), emission[:, 2]), axis=1)
    tokens = (obs[..., :3].argmax(-1) == 2).astype(int)
    actions = obs[..., 3:].argmax(-1)
    belief = np.broadcast_to(np.asarray(process["initial"], dtype=np.float64), (len(obs), 3)).copy()
    result = np.empty(obs.shape[:2] + (3,), dtype=np.float64)
    for t in range(obs.shape[1]):
        if t:
            belief = np.einsum("bi,bij->bj", belief, transitions[actions[:, t]])
        belief *= emission[:, tokens[:, t]].T
        belief /= belief.sum(1, keepdims=True)
        result[:, t] = belief
    return result
