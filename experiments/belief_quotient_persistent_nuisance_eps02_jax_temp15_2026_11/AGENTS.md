# belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11

Copy of the PR #231 recipe with emission epsilon 0.02 (not exploration).
T0, effect size 2.5, uniform initial distribution, reward state 2, architecture,
training budgets, temperature 1.5, entropy 0, and probes are unchanged.
No epsilon-greedy exploration. Smoke retains 4,096 steps.
The effect size scales the existing destination-state-2 log-odds tilt directions.
The reward-factor study uses two independent variant-3 factors with Kronecker
transitions/emissions and uniform joint initialization.
No historical analytic optima are asserted for this emission matrix.
