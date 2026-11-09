# Reward-factor history study (pure JAX)

JAX rewrite of `reward_factors_history_REINFORCE_2026_11`: 8 arms
(`{reward_both,reward_factor_1}_{transformer,memoryless,gru128,gru92}`). It
reuses the JAX environment, models, REINFORCE trainer and probes from
`belief_quotient_history_jax_2026_11`; `process.py` is the only task adapter.

## Science (unchanged from the source study)

- Two independent controlled MESS3 factors with the ladder's emission 0.75 and
  sticky baseline. Joint state/token use `np.kron` ordering (`3*f1 + f2`); the
  nine joint actions apply variant-3 tilts independently to each factor in the
  source `ACTION_PAIRS` order (parity with the source `HMMEnv` is tested).
- `reward_both`: occupancy of state 2 in factor 1 plus factor 2;
  `reward_factor_1`: factor 1 only. Both components are logged.
- Same recipe as the ladder (64 minibatches, constant entropy 0.015; the source
  study used entropy 0.0) and a 200M env-step budget (source: 30M; same 19,456-env batch, 11 rollouts). Transformer window 10
  (lookback 40); memoryless reads only the current joint token.
- Probes use the pre-final-LayerNorm representation and report the joint
  nine-state belief plus each factor marginal.

See the ladder `AGENTS.md` for execution differences from RLlib and commands.
