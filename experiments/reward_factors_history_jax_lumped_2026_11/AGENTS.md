# Reward-factor history study (pure JAX)

This study supersedes the environment of PR #195’s `reward_factors_history_jax_2026_11`; that folder remains unchanged as the historical record.

Derived from `reward_factors_history_jax_2026_11`: 8 arms
(`{reward_both,reward_factor_1}_{transformer,memoryless,gru128,gru92}`). It
reuses the JAX environment, models, critic-free PPO trainer and probes from
`belief_quotient_history_jax_2026_11`; `shared.py` owns the baseline, controlled kernels and task construction.

## Science (deliberately revised baseline)

- Two independent controlled MESS3 factors with the ladder's emission 0.75 and
  revised uniform-nonreward-destination baseline, applied independently before
  tilting each factor. Joint state/token use `np.kron` ordering (`3*f1 + f2`); the
  nine joint actions apply variant-3 tilts independently to each factor in the
  source `ACTION_PAIRS` order (parity with the source `HMMEnv` is tested).
- `reward_both`: occupancy of state 2 in factor 1 plus factor 2;
  `reward_factor_1`: factor 1 only. Both components are logged.
- Same recipe as the ladder (64 minibatches, constant entropy 0.015; the earlier
  RLlib study used entropy 0.0) and a 2B env-step budget (PR #195: 200M; same 19,456-env
  batch, 101 rollouts, 6,464 Adam steps). Transformer window 10
  (lookback 40); memoryless reads only the current joint token.
- Probes use the pre-final-LayerNorm representation and report the joint
  nine-state belief plus each factor marginal.

See `../belief_quotient_history_jax_lumped_2026_11/AGENTS.md` for commands
and execution differences from RLlib.
Resolved provenance identifies `uniform_nonreward_destination_v2` and the
factor baseline. Historical saved results retain their original kernels and
training recipes and are not results for this revised environment.

Generic env/models/rollout/loss/training/probe code is reused by import from
PR #195. Each new shared.py owns its baseline and kernels. The new process is
passed explicitly throughout rollout, exact filtering, evaluation and probes.
