# eps02 reward references

**Only variant 1 is solved exactly.** Other full-history values have a numerical
upper bound and a Monte Carlo feasible-policy estimate; reactive values are
best-known stochastic-search lower bounds, NOT certified global optima.

All primary values below are expected reward per step over a complete 1,024-step
episode, including reset, with a uniform initial distribution. reward_both sums
factor rewards. This matches the training metric's units and horizon, not the
gamma=.99 gradient surrogate or a stationary-distribution approximation.

| Task | Full-history feasible policy, mean [95% CI] | Bayes upper bound | Memoryless best known | Transformer final train mean (15 seeds) | Train range |
|---|---:|---:|---:|---:|---:|
| variant 1 | **0.941335636 exact optimal** | 0.941335636 | **0.941335636 exact optimal** | 0.941158 | 0.940435–0.942020 |
| variant 2 | 0.455902 [0.454391, 0.457413] | 0.456254 | 0.307072 | 0.429778 | 0.382419–0.451810 |
| variant 3 | 0.420969 [0.419385, 0.422552] | 0.421413 | 0.271768 | 0.408487 | 0.394594–0.413005 |
| reward_both | 0.841937 [0.838771, 0.845104]* | 0.842825 | 0.543535 | 0.780216 | 0.732088–0.793027 |
| reward_factor_1 | 0.420969 [0.419385, 0.422552] | 0.421413 | 0.271768 | 0.406712 | 0.400337–0.411503 |

*The joint row scales the per-factor value/CI by two, using additivity, rather
than claiming a separately simulated joint-policy CI. Each full-history estimate
uses 16,384 independent episodes, seed 208, the 320-division belief mesh and exact
filtered beliefs. The feasible-policy CI is not a CI on the optimum. Numerical
upper bounds are valid up to floating-point error.

The CLI also ran an 80-division mesh: eps02 upper bounds were
0.456710832 (variant 2) and 0.422360609 (variant 3). Refinement reduced these to
0.456254193 and 0.421412617. This is refinement evidence, not exact convergence.

Memoryless inputs are current tokens only. All 27 deterministic single-factor
maps plus 64 random stochastic starts were optimized (91/91 SLSQP successes
per single-factor task); joint tasks used unrestricted 9-token/9-action maps,
64 random starts and a product-policy start (65/65 successes each). No searched
stochastic policy beat [1,1,0] for variant 2 or [1,2,0] for variant 3. This does
not prove a better stochastic or joint reactive controller cannot exist.
No memoryless/GRU achieved results were present in the checked-out eps02 result
folders. Empty result columns would mean absent measurements, not zero reward.

## Separate greedy and continuation metrics

Final transformer greedy probe means (15 seeds), which **exclude 64 warmup
steps**, were 0.960445 / 0.436448 / 0.419082 for ladder variants 1/2/3,
0.800582 for reward_both, and 0.415319 for reward_factor_1. These should not be
compared directly with the complete-episode Bayes bounds above. Full-history
feasible policies' corresponding post-warmup estimates are recorded in JSON.

The continuation study has one seed (42), evaluated separately at 4,024,434,688
steps:

| Variant | Final training reward/step | Final greedy post-warmup reward/step |
|---|---:|---:|
| 1 | 0.941982448 | 0.965162516 |
| 2 | 0.438793778 | 0.436549991 |
| 3 | 0.410647839 | 0.404325008 |

report.json contains all **78 per-run records**, including seed, leaf, final
checkpoint step, final training and greedy metrics, source paths/hashes, and
training gaps to the numerical Bayes upper and best-known memoryless reference.
Finite-sample training estimates can exceed a population bound (as variant 1
does for some seeds); no episode-level CIs can be reconstructed from saved means.

## v2 and historical sanity checks

PR #208 is v1 and explicitly supplies feasible stationary controllers, not Bayes
optima. Its existing design.py was reused to reproduce the checked references:
memoryless 0.228861871655982 and three-token 0.251424641014780. The stationary
solver gives v2 variant 2's deterministic [1,1,0] value 0.345326852727761.
The exact 1,024-step evaluation is instead 0.344560072066844.

| v2 task | Full-history feasible mean [95% CI] | Bayes upper | Memoryless best known |
|---|---:|---:|---:|
| variant 1 | 0.941335636 exact | 0.941335636 | 0.941335636 exact |
| variant 2 | 0.525442 [0.523719, 0.527165] | 0.525746 | 0.344560 |
| variant 3 | 0.481332 [0.479506, 0.483158] | 0.481532 | 0.306868 |

The searched files contained no recorded v2 Bayes optimum to certify against.
Do not relabel PR #208's v1 numbers or PR #231's learned scores as such.

See ../../optimal_reward.md for algorithms, objective choices, limitations and
the reproducible CLI command. Historical code, results, and continuation files
were left unchanged.
