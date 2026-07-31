# MESS3 feedback cycle 1

## Locked study

- One condition: plain clipped PPO, with no predictive or Kelly auxiliary.
- Five seeds: `42, 43, 44, 45, 46`.
- Budget: `2,000,000` environment steps per seed.
- MESS3 emissions: `alpha=0.85`.
- Timing: `delay=1`.
- Agent observation: current visible token and previous executed action.
- Transformer: cycle-2 paper architecture, context length 10.
- Return objective: `gamma=0`, `lambda=0`.

The categorical token guess is also the intervention. For passive transition
matrix `T`, action `a` executes

```text
U_a = (1 - eta) T + eta R_a
eta = 0.10
R_a[i, j] = 1[j = a]
```

Thus every action is nontrivial and the three token labels remain permutation
symmetric. Reward still scores the pending pre-transition token, so the current
reward is not changed by the intervention selected on that step.

## Evaluation

The standard held-out affine belief probe runs at initialization and every
retained checkpoint. The final checkpoint additionally receives:

1. Closed-loop previous-action mask and cross-environment shuffle evaluations,
   reporting token-accuracy degradation and probe-MSE inflation.
2. A local counterfactual evaluation which holds the preceding token/action
   context fixed, replaces the previous action in the current observation with
   each alternative, and compares the affine-decoded belief shift with the
   exact delay-one Bayesian shift.

Absolute MSE, target-normalized MSE, episode-clustered bootstrap intervals, and
the held-out permutation null remain part of the standard probe report.
