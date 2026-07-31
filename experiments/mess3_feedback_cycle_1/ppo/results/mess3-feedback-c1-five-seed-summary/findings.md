# MESS3 feedback cycle 1 — five-seed findings

Plain PPO with `eta=0.10`, `delay=1`, previous executed action in the observation, and ~2M environment steps per seed (seeds 42–46). Each action is a token intervention; reward still scores the pre-transition token, so the policy must infer hidden state from emissions and its own feedback history.

## Summary metrics

| metric | mean ± sample std |
|---|---:|
| final probe MSE | 0.000290733 ± 9.33252e-05 |
| normalized MSE (MSE / target variance) | 0.00233692 ± 0.00075166 |
| probe R² | 0.997663 ± 0.00075166 |
| greedy token accuracy | 69.61% ± 0.34% |
| on-rollout Bayes accuracy | 69.97% ± 0.05% |
| masked prev-action probe MSE | 0.010355 ± 0.00214072 |
| shuffled prev-action probe MSE | 0.0140919 ± 0.000608161 |
| counterfactual belief-shift MSE | 0.00351934 ± 0.000566855 |
| counterfactual shift cosine | 0.951074 ± 0.0209115 |

See [`mse_over_training.png`](mse_over_training.png) for the training curve and final-checkpoint ablations on a shared log-MSE axis.

## Belief geometry over training

The held-out affine probe MSE falls sharply from initialization (0.0038) to the final checkpoint (0.0002907 mean across seeds), with normalized MSE reaching ~0.2–0.4% of target variance (0.00233692 ± 0.00075166). The network learns a representation that is nearly linearly decodable into the exact predictive Bayesian belief under the training distribution.

## Action choice distribution

Counts come from the counterfactual evaluation rollouts (greedy closed loop at the final checkpoint). Actions coincide with the three token interventions and remain close to uniform:

| seed | token 0 | token 1 | token 2 |
|---:|---:|---:|---:|
| 42 | 32.9% | 33.4% | 33.7% |
| 43 | 29.5% | 39.7% | 30.8% |
| 44 | 30.4% | 36.3% | 33.3% |
| 45 | 33.2% | 32.3% | 34.5% |
| 46 | 34.8% | 32.9% | 32.3% |
| **pooled** | **32.2%** | **34.9%** | **32.9%** |

Pooled over 780,750 evaluated steps. Symmetry is expected because the three interventions are permutation-equivalent at `eta=0.10`.

## Interpretation

### Token accuracy near the Bayes ceiling (hypothesis)

The result shows ~69.6% greedy token accuracy against ~70.0% on-rollout Bayesian accuracy. **Hypothesis:** the policy is not far from the best token guesser allowed by the partially observed process, given its architecture and training budget.

Mechanistically, each step the agent sees the current emission and the previous executed action. With `delay=1`, that action shifted the transition kernel from passive `T` toward a rank-one intervention `R_a`, so the observation stream is informative but stochastic (`alpha=0.85`). A Bayes-optimal filter maintains the predictive belief over hidden states; greedy argmax on that belief sets an accuracy ceiling under the evaluation rollouts. The tight gap between agent and Bayes accuracy suggests the learned representation is not merely decodable (probe R² ≈ 0.998) but is used in a way that tracks the filter’s token ranking reasonably well. It does **not** by itself prove optimality: the probe measures belief geometry, not the policy head, and remaining error could reflect approximation in the actor or mismatch between training and evaluation sampling.

### Previous-action sensitivity (hypothesis)

Masking or shuffling the previous-action block at the final checkpoint inflates probe MSE by ~40–50× and lowers token accuracy by ~0.6–1.1 percentage points. **Hypothesis:** the policy and representation genuinely condition on executed feedback, not only on emissions.

In this task the previous action is the only channel through which the agent’s own interventions enter the observation. Masking removes that channel while leaving transitions driven by executed actions intact; shuffling breaks the temporal pairing between the true intervention and the current belief state. The ablation therefore tests whether the network’s internal state tracks *this* history rather than a generic action marginal. The accuracy drop is modest because emissions already carry substantial information about hidden state, so a emission-only policy can still score well. The large MSE inflation under ablation is the stronger signal: the affine probe was fit under intact observations, and corrupting the action block moves representations off the training manifold even when token choice degrades only slightly.

### Counterfactual belief-shift alignment (hypothesis)

The counterfactual probe holds context fixed, swaps the previous action in the observation, and compares the decoded belief shift to the exact delay-one Bayesian update induced by that swap. Mean cosine alignment is 0.951 (sample std 0.021). **Hypothesis:** local representational sensitivity to action changes follows the direction of the true belief update, not just arbitrary feature noise.

Mechanistically, under `delay=1` the previous action enters the transition that produced the current token distribution. Changing the recorded action while holding earlier context fixed isolates the Bayesian effect of that counterfactual intervention. High cosine similarity means the post-intervention representation moves toward the Bayes-updated belief; `shift_mse` (~0.0035 mean) reports the remaining magnitude error in that subspace. This is distinct from the ablation MSEs (~0.013): ablations measure global decodability under corrupted inputs, while the counterfactual test measures whether infinitesimal action substitutions propagate through the representation in the same direction as the filter. Values below 1.0 leave room for residual non-Bayesian components in the actor trunk or probe mismatch.

See `five_seed_summary.json` for per-seed values and the full MSE-over-training aggregate.
