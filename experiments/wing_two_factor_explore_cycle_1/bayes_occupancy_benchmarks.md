# Bayesian reward-occupancy benchmarks

## Reference targets for the six arms

For the current Wing process, use the following approximate Bayesian planning
ceilings for regular **1,024-action episodes**, including the initial inference
transient after each reset:

| Rewarding state | `reward_factor_1` mean reward | `reward_both` mean reward | Expected episode return |
|---|---:|---:|---:|
| 0 | **46.67%** | **46.67%** | **477.91** |
| 1 | **51.64%** | **51.64%** | **528.75** |
| 2 | **49.55%** | **49.55%** | **507.35** |

These are finely discretized, information-relaxed planning upper bounds, not
closed-form exact optima or forecasts of what PPO will achieve. Simulated
implementable belief policies achieve very close to them. The full-history
planning bound is also an upper benchmark for the 32-frame transformer; it is
not an exact solution of the optimal 32-frame policy problem.

The reward conditions have the same normalized ceiling because their factors
are independent, have separately selectable actions, and use the **mean**
reward over selected factors. In `reward_factor_1`, reward is the first factor's
arrival-state indicator. In `reward_both`, it is the average of the two
indicators. The latter is **not** the indicator that both factors are rewarded
simultaneously, and its expected reward is not the square of the table entry.
The unselected factor's occupancy is not constrained by the single-factor
reward objective.

The 30M-step single-rewarded-factor and 50M-step both-rewarded-factor training
budgets affect learning, not these ceilings. Entropy regularization, imperfect
inference, and optimization can leave PPO below the reference policies.

## Process and information assumptions

- Wing parameters: `alpha = 0.94`, `x = 0.4`, `b = 0.03`.
- Control strength: `0.15`; action 0 holds, action 1 rotates forward with this
  probability, and action 2 rotates backward with this probability.
- Edge kernels: `K[a, x] = K[x] @ C[a]`, with token emission and destination
  jointly distributed according to the controlled kernel.
- Reward: indicator of the **arrival state**, averaged over rewarded factors.
- The Bayesian reference knows the process parameters and its past actions.
  Hidden states and rewards are **not** observations used for filtering.
- Exact filter: `belief_next = normalize(belief @ K[action, token])`.
- Reset starts from the uniform stationary source prior and executes a neutral
  edge. Its token is observed and its arrival-state posterior is the initial
  decision belief. Reset itself produces no reward.
- Episode calculations use a fixed length of 1,024 actions. The recipe's
  randomized first-episode length is not included in these calculations; it is
  negligible as a fraction of a 30M/50M-step training run.

Relevant implementation: [process.py](process.py),
[design.py](design.py), and the sibling harness's `envs/wing/model.py`,
`envs/wing/tasks/reward_state.py`, and `envs/hmm/env.py`.

## Numerical planning method

The two-factor optimization separates into single-factor problems. Each
single-factor belief lies on the three-state probability simplex. The
calculation used triangular grids with coordinates
`(i/n, j/n, 1 - (i+j)/n)`, where `i >= 0`, `j >= 0`, and `i+j <= n`.

For each grid belief and action, the calculation obtained:

1. The exact expected arrival-state reward.
2. Each token's probability and its exact posterior belief.
3. Nonnegative barycentric weights representing that posterior using the
   surrounding triangle's three grid vertices.

Replacing each posterior with a random grid vertex using these weights defines
an information relaxation: an extra signal can reveal which component of the
posterior mixture applies. Its optimal value therefore upper-bounds that of the
original partially observed process. Equivalently, convexity of finite-horizon
POMDP values gives the interpolation upper bound by induction. Reported bounds
are numerical floating-point calculations, not interval-arithmetic certificates.

For the long-run average-reward objective, relative value iteration was run
until the Bellman-increment span across grid points was below `1e-10` for all
three reward placements. For the episodic objective, 1,024 undiscounted Bellman
backups from zero terminal value were evaluated at the reset posterior mixture.
The final grid had **321,201 points** (`n = 800`).

### Episodic planning bounds, before rounding

| Reward state | Mean reward upper bound | Expected 1,024-step return upper bound |
|---|---:|---:|
| 0 | 0.4667137416419691 | 477.91487144137636 |
| 1 | 0.5163595172701350 | 528.7521456846182 |
| 2 | 0.4954624187537571 | 507.35351680384724 |

### Long-run grid refinement

These percentages assume continuing operation without episode resets:

| Grid subdivisions `n` | Grid points | Reward state 0 | Reward state 1 | Reward state 2 |
|---|---:|---:|---:|---:|
| 50 | 1,326 | 46.845480% | 51.798454% | 49.707891% |
| 100 | 5,151 | 46.757740% | 51.751889% | 49.676373% |
| 200 | 20,301 | 46.743903% | 51.740165% | 49.671023% |
| 400 | 80,601 | 46.739061% | 51.738112% | 49.669991% |
| 800 | 321,201 | **46.738033%** | **51.737823%** | **49.669788%** |

For the episodic bounds, refining `n = 400` to `n = 800` decreased the
percentages by only **0.00101344**, **0.00029951**, and **0.00020105 percentage
points**, respectively. Resolution stability is evidence of numerical accuracy,
not by itself a proof of the remaining gap to the continuous-belief optimum.

## Simulation validation

Policies selected actions greedily from interpolated grid Q-values, without
receiving the artificial grid-revelation signal. Token trajectories were
sampled from the exact controlled process's predictive distribution. Rewards
were evaluated as conditional expected arrival-state indicators, rather than
sampled hidden-state counts, to reduce Monte Carlo variance. Filtering never
used reward observations.

Each validation used **4,096 independent chains per reward placement**, each
with **8,192 actions**. This is **33,554,432 simulated actions per placement**
for each validation condition. Episodic validations contained eight complete
1,024-step episodes per chain and discarded no warmup samples.

For the 32-frame policies, the decision belief was recomputed using the last
32 generating-action/token frames, starting from the uniform stationary prior
before the oldest frame. Exact full-history beliefs were retained separately
only to evaluate expected rewards and sample the true predictive process.

### Practical 32-frame reference matching PPO's discount

The policy was planned with `gamma = 0.99`, but the reported metric remains
**undiscounted mean reward / rewarding-state occupancy**, as in the experiment.

| Reward state | Mean occupancy | Chainwise standard error, probability units | Approximate 95% half-width, percentage points |
|---|---:|---:|---:|
| 0 | **46.6538716%** | 0.00003921706093948536 | **0.00768654** |
| 1 | **51.6458758%** | 0.0001476770153405272 | **0.02894470** |
| 2 | **49.5456743%** | 0.00009746404738289444 | **0.01910295** |

Standard errors are the sample standard deviation of independent chain means
divided by `sqrt(4096)`; the approximate 95% half-width is `1.96 * SE`.
An empirical mean slightly exceeding a numerical upper bound is compatible
with Monte Carlo uncertainty. These intervals do not quantify grid error or
PPO training uncertainty.

### Other validation conditions

| Policy / evaluation | Seed | Reward state 0 | Reward state 1 | Reward state 2 |
|---|---:|---:|---:|---:|
| Average-reward planner, full belief, continuing | 20260907 | 46.7344175% | 51.7440769% | 49.6629740% |
| Average-reward planner, full belief, 1,024-step episodes | 20260908 | 46.6696754% | 51.6538546% | 49.5249529% |
| Average-reward planner, 32-frame belief, 1,024-step episodes | 20260909 | 46.6548469% | 51.6474725% | 49.5510878% |
| Discount-0.99 planner, 32-frame belief, 1,024-step episodes | 20260910 | 46.6538716% | 51.6458758% | 49.5456743% |

The continuing validation discarded the first 1,024 actions per chain, leaving
29,360,128 measured actions per reward placement. All other rows used every
action. Within each validation, a single NumPy generator drove the three
reward-placement batches together; the seed reproduces that batching convention,
not three independent invocations with the same seed.

## Distinguish these references from other ceilings

- A constant action has **33.3333%** stationary occupancy in every state.
- The fully observed state-feedback oracle reaches **73.5099338%** continuing
  occupancy for every reward state. It sees the actual hidden state, so it is
  **not an attainable Bayesian-observation benchmark** for this task.
- The earlier full-belief QMDP/myopic audit in `design.py` is a feasible policy
  benchmark, not an optimal POMDP solver. Its roughly 46.16% / 51.42% / 49.35%
  continuing results must not be mislabeled as Bayes maxima.
- Linear belief decodability does not establish that PPO uses a Bayesian
  planning algorithm or that it will attain these rewards.

## Provenance and reproducibility limits

The computation used experiment commit
`1eacb6f0ecd9765c7a5caee271edb169db55a1e2` and harness commit
`305ed0f064b6f02a8a1d4d43183c0b1eebc3803b`, with NumPy and SciPy 1.18.0's sparse
matrix operations. It ran locally in the assistant session; no PPO training,
cloud jobs, or subagents were launched for this calculation.

This file records the session's numerical outputs and method. The temporary
in-memory grid solver was **not saved as a runnable script**, and there are no
saved policies or raw validation trajectories. `design.py` contains the existing
analytic and QMDP audits, but running it does **not** reproduce the grid-planning
bounds above. Reproducing those requires implementing the grid procedure and
validation conventions described here; seeds alone are not sufficient.
