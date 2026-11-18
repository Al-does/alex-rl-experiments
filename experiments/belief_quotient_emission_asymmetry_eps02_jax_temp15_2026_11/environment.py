"""Emission-asymmetry perturbation of the PR #234 Reward-State Quotient process.

States 1 and 2 (zero-based 0 and 1) keep the source transitions, reward and
within-row A:B ratio, but emit C with probability ``0.85 - kappa`` and
``0.85 + kappa``. ``kappa = 0`` is the source process, whose two classes
``{1, 2}`` / ``{3}`` with tokens merged to not-C / C form an exact two-state
POMDP. For ``kappa > 0`` the classes stay transition-lumpable but emit the
merged token at different rates, so that two-state filter is only approximate.
Its equal-weight version is the source quotient filter at every ``kappa``.
"""
import numpy as np

from experiments.belief_quotient_history_jax_2026_11.env import make_process
from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import (
    environment as source,
)

SOURCE_STUDY = "belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11"
VARIANT = 2
LEVELS = {"025": 0.025, "050": 0.05, "075": 0.075, "100": 0.1}


def emission_matrix(kappa):
    emission = source.emission_matrix()
    off = emission[:2, :2].sum(1)
    if not 0 <= kappa < off.min() - emission[2, :2].sum():
        raise ValueError("kappa must keep states 1 and 2 likelier than state 3 to emit A or B")
    shift = np.array([kappa, -kappa])
    emission[:2, :2] *= ((off + shift) / off)[:, None]
    emission[:2, 2] -= shift
    return emission


def model(kappa):
    """Float64 ``initial`` / ``transitions`` (action, source, destination) / ``emission``."""

    return {"initial": source.INITIAL_STATE_DISTRIBUTION.copy(),
            "transitions": source.action_transitions(VARIANT),
            "emission": emission_matrix(kappa)}


def ladder_process(kappa):
    arrays = model(kappa)
    reward = (np.arange(3) == source.REWARD_STATE).astype(float)
    return make_process(arrays["initial"], arrays["transitions"], arrays["emission"],
                        reward, reward[None])


def _merge(values):
    return np.stack([values[..., :2].sum(-1), values[..., 2]], -1)


def quotient_model(full, weights=(0.5, 0.5)):
    """Two-class model over (non-reward, reward) and (not-C, C).

    States 1 and 2 are averaged with ``weights``; this is an exact POMDP only
    when both lumpability defects vanish.
    """

    weights = np.asarray(weights, dtype=np.float64) / np.sum(weights)

    def classes(values):
        return np.stack([np.tensordot(weights, values[..., :2, :], (0, -2)),
                         values[..., 2, :]], -2)

    return {"initial": _merge(full["initial"]),
            "transitions": _merge(classes(full["transitions"])),
            "emission": _merge(classes(full["emission"]))}


def lumpability_defect(full):
    """Largest state-1/state-2 gap in reward-class transition and merged-token emission mass."""

    return {"transition": float(np.abs(np.diff(full["transitions"][:, :2, 2], axis=1)).max()),
            "emission": float(abs(full["emission"][0, 2] - full["emission"][1, 2]))}


def filter_histories(model, tokens, actions):
    """Posterior over the current state after ``tokens[:, t]``, given the earlier
    tokens and the executed ``actions[:, :t]``; rows are complete episodes."""

    tokens, actions = np.asarray(tokens), np.asarray(actions)
    belief = np.empty((*tokens.shape, len(model["initial"])))
    current = np.broadcast_to(model["initial"], belief[:, 0].shape)
    for t in range(tokens.shape[1]):
        if t:
            current = np.einsum("bi,bij->bj", current, model["transitions"][actions[:, t - 1]])
        current = current * model["emission"][:, tokens[:, t]].T
        current = current / current.sum(1, keepdims=True)
        belief[:, t] = current
    return belief


def quotient_filter(full, tokens, actions, weights=(0.5, 0.5)):
    """Reward-class probability of the two-class filter on merged tokens."""

    merged = (np.asarray(tokens) == 2).astype(int)
    return filter_histories(quotient_model(full, weights), merged, actions)[..., 1]


def environment_metadata(kappa):
    full = model(kappa)
    return {
        **source.environment_metadata(),
        "environment_revision": "persistent_nuisance_eps02_emission_asymmetry",
        "source_environment_study": SOURCE_STUDY,
        "variant": VARIANT,
        "emission_asymmetry_kappa": kappa,
        "emission_matrix": full["emission"].tolist(),
        "lumpability_defect": lumpability_defect(full),
        "condition_name": f"Reward-State Quotient with emission asymmetry {kappa:g}",
        "reward": "pre-transition state-2 occupancy",
        "representation": "post_final_layer_norm",
    }
