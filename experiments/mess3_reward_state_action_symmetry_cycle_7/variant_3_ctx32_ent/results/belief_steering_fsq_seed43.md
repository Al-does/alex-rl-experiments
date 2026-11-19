# Full-State Quotient belief steering

Occupancy is decision-time reward-state occupancy, including reset decisions.
Actions: Noop, Tilt X, Tilt Y. States: 1, 2, 3 (reward state).

Reference: N=60 nearest-grid, gamma=0.97, 800 iterations; near-optimal, not certified Bayes-optimal.
Raw affine decoded beliefs are not clipped. Point ties choose state 2.
Warmup is eight steps only for probe fitting/testing (baseline protocol); the banded four-layer context-32 encoder can depend on up to 128 observations.
Shadow forward uses the current executed cache and is discarded; earlier shifts affect subsequent-block caches only.
Selected strengths: {"point": 0.5, "mirror": 0.6}

Uncertainty: episode standard error within one frozen training seed

| Condition | Occupancy ± SE |
|---|---:|
| block_1/point | 0.118805 ± 0.002630 |
| block_2/point | 0.130402 ± 0.002480 |
| block_3/point | 0.135803 ± 0.002283 |
| block_4/point | 0.126038 ± 0.002387 |
| post_final_norm/point | 0.146210 ± 0.001785 |
| orthogonal/point | 0.313019 ± 0.002399 |
| block_1/mirror | 0.112701 ± 0.002305 |
| block_2/mirror | 0.122925 ± 0.002244 |
| block_3/mirror | 0.150635 ± 0.002546 |
| block_4/mirror | 0.160889 ± 0.002250 |
| post_final_norm/mirror | 0.190186 ± 0.002388 |
| orthogonal/mirror | 0.312927 ± 0.002392 |
| intact | 0.313019 ± 0.002399 |
| random | 0.162842 ± 0.001746 |

## Final pre-LayerNorm curves

| λ | Point ± SE | Mirror ± SE | Predicted point | Predicted mirror |
|---:|---:|---:|---:|---:|
| 0 | 0.313019 ± 0.002399 | 0.313019 ± 0.002399 | 0.311584 | 0.311584 |
| 0.25 | 0.313019 ± 0.002399 | 0.313019 ± 0.002399 | 0.311584 | 0.311584 |
| 0.4 | 0.306885 ± 0.002457 | 0.312927 ± 0.002382 | 0.289948 | 0.311584 |
| 0.45 | 0.240448 ± 0.002690 | 0.310364 ± 0.002369 | 0.272522 | 0.311584 |
| 0.5 | 0.126038 ± 0.002387 | 0.244751 ± 0.002077 | 0.071381 | 0.183350 |
| 0.55 | 0.086853 ± 0.002234 | 0.193115 ± 0.002113 | 0.070862 | 0.076202 |
| 0.6 | 0.074432 ± 0.001728 | 0.160889 ± 0.002250 | 0.069427 | 0.076202 |
| 0.75 | 0.073639 ± 0.001682 | 0.076874 ± 0.001788 | 0.069122 | 0.069427 |
| 0.9 | 0.073639 ± 0.001682 | 0.073700 ± 0.001693 | 0.068787 | 0.069336 |
| 1 | 0.073639 ± 0.001682 | 0.073639 ± 0.001682 | 0.068726 | 0.069122 |
