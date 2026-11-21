"""Edge-emitting MRRXOR, with decision-time beliefs before each token."""

import numpy as np


def transition_tensor(eps=0.05):
    if not 0 <= eps <= 1:
        raise ValueError("eps must be in [0, 1]")
    emit = np.zeros((2, 5, 5), dtype=np.float64)
    emit[0, 0, 1] = emit[1, 0, 2] = 0.5
    emit[0, 1, 4] = emit[1, 1, 3] = 0.5
    emit[0, 2, 3] = emit[1, 2, 4] = 0.5
    emit[1, 3, 0] = emit[0, 4, 0] = 1
    jump = np.full((5, 5), eps / 4)
    np.fill_diagonal(jump, 1 - eps)
    return jump @ emit


def stationary_distribution(edges):
    values, vectors = np.linalg.eig(edges.sum(0).T)
    pi = vectors[:, np.argmin(abs(values - 1))].real
    return pi / pi.sum()


def sample(rng, batch, length, edges=None):
    edges = transition_tensor() if edges is None else edges
    state = rng.choice(5, size=batch, p=stationary_distribution(edges))
    cdf = edges.transpose(1, 0, 2).reshape(5, 10).cumsum(-1)
    tokens = np.empty((batch, length), dtype=np.int32)
    for n in range(length):
        outcome = (rng.random((batch, 1)) > cdf[state, :-1]).sum(-1)
        tokens[:, n], state = outcome // 5, outcome % 5
    return tokens


def beliefs(tokens, edges=None):
    edges = transition_tensor() if edges is None else edges
    tokens = np.asarray(tokens)
    if tokens.ndim != 2 or not np.isin(tokens, [0, 1]).all():
        raise ValueError("tokens must be a batch of binary sequences")
    out = np.empty((*tokens.shape[:1], tokens.shape[1] + 1, 5))
    out[:, 0] = stationary_distribution(edges)
    for n in range(tokens.shape[1]):
        posterior = np.einsum("bs,bst->bt", out[:, n], edges[tokens[:, n]])
        mass = posterior.sum(-1, keepdims=True)
        if np.any(mass <= 0):
            raise ValueError("impossible history")
        out[:, n + 1] = posterior / mass
    return out


def predictive(belief, edges=None):
    edges = transition_tensor() if edges is None else edges
    return belief @ edges.sum(-1).T


def policy_inputs(tokens):
    return np.concatenate([np.full((len(tokens), 1), 2, dtype=np.int32), tokens[:, :-1]], axis=1)


def reward(action, token):
    return (action == token).astype("float32")


def fixed_pools(seed, train_sequences=1024, heldout_sequences=4096, length=32):
    if min(train_sequences, heldout_sequences, length) <= 0:
        raise ValueError("pool sizes and sequence length must be positive")
    rng = np.random.default_rng(seed)
    train = sample(rng, train_sequences, length)
    keys = {row.tobytes() for row in train}
    heldout = []
    while len(heldout) < heldout_sequences:
        for row in sample(rng, heldout_sequences - len(heldout), length):
            if row.tobytes() not in keys:
                heldout.append(row)
    return {"train": train, "heldout": np.asarray(heldout)}
