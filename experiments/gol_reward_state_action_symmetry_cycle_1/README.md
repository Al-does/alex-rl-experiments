# Gol reward-state action-symmetry cycle 1

Two action-symmetry variants, each at half and quarter speed:

| Experiment leaf | Speed | Preferred actions in `(M1, M2, E, S)` under full information |
|---|---|---|
| `variant_2` | Half | `(a1, a1, aE, aS)` |
| `variant_3` | Half | `(a1, a2, aE, aS)` |
| `variant_2_quarter` | Quarter | `(a1, a1, aE, aS)` |
| `variant_3_quarter` | Quarter | `(a1, a2, aE, aS)` |

The preferred actions above describe fully observed control, not an action rule for every uncertain posterior. Both variants retain all four actions and all four fine hidden states in the experiments.

## Simulated Bayesian-controller reward occupancy

These estimates were supplied by the experiment owner on 2026-09-09. They used an **exact Bayesian belief filter with an approximate planner**, with **discount gamma = 0.99**.

| Setting | Variant 2 | Variant 3 |
|---|---:|---:|
| Half speed | approximately 37.62% | approximately 36.02% |
| Quarter speed | approximately 31.51% | approximately 29.62% |

These are **achieved reward-occupancy estimates, not certified Bayes-optimal values and not upper bounds**. A better planner could improve them. They are reference-controller simulations, not measured PPO results. Reward is `1[destination_state=E]`, so the occupancy fraction equals mean reward; 37.62% corresponds to approximately 0.3762 reward per step.

The planner implementation, simulation budgets/seeds, and uncertainty intervals were not supplied with these estimates. The simulations have not been independently rerun here. The archived verifier does not reproduce or certify this table.

### Separate full-information upper bounds

| Speed | Full-information long-run maximum E occupancy, either variant |
|---|---:|
| Half | 42.481203% |
| Quarter | 35.646688% |

These verified bounds assume the true hidden state is available at every decision. They must not be confused with the simulated action/token-history controller estimates above or with a certified partially observed Bayes optimum.

## Experiment contract

- Full training targets 2,500,000 environment steps per leaf; smoke targets 2,048. Runs stop after crossing the target at a complete training iteration.
- Reset samples the known prior without emitting a token or reward.
- The actor and shared critic receive only latest-token and previous-action features; rewards, hidden states, and exact beliefs are excluded from their inputs.
- The exact filter conditions on actions and tokens, not rewards.
- Continuing trajectories use the shared `ContinuingSingleAgentEnvRunner` to release metrics-only episode references without resetting the environment, filter, or model context.

For example, from the experiment repository root:

```bash
uv run rl-harness experiments.gol_reward_state_action_symmetry_cycle_1.variant_2.experiment --smoke --hardware cpu --no-upload-artifacts
```

Substitute any other leaf name from the first table. Omit `--smoke` for the full budget.

## Supplied frozen specification

All six originally supplied reference files are archived under [`specification/`](specification/):

- [`README.md`](specification/README.md): frozen v1.0 specification, supplemented with the simulated occupancy estimates above.
- [`four_action_hmm.py`](specification/four_action_hmm.py): standalone NumPy model generator, simulator, and exact filter.
- [`hmm_values.json`](specification/hmm_values.json): numerical kernels, priors, emissions, and reward predictions for all four fine models and both variant-2 quotients.
- [`requirements.txt`](specification/requirements.txt): dependencies for the standalone reference implementation.
- [`verification.json`](specification/verification.json): frozen model checks, full-information oracle bounds, and memory witnesses.
- [`verify_spec.py`](specification/verify_spec.py): reference checks and JSON regeneration.

The Python and JSON reference files are preserved as supplied; the README is the updated source document. These files document the specification. The experiments use the reusable `envs.gol` implementation in the sibling `rl-harness` repository, not the archived simulator.

To run the standalone reference tests without regenerating the JSON files, run from `specification/` using Python with NumPy installed:

```bash
python -m unittest -v verify_spec.ReferenceChecks
```

Running `python verify_spec.py` instead also regenerates `hmm_values.json` and `verification.json` in the current directory. Those generated numerical checks do not contain the supplied approximate-planner occupancy estimates.
