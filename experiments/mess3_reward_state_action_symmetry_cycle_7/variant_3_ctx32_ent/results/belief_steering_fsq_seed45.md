# Full-State Quotient belief steering

Occupancy is decision-time reward-state occupancy, including reset decisions.
Actions: Noop, Tilt X, Tilt Y. States: 1, 2, 3 (reward state).

Reference: N=60 nearest-grid, gamma=0.97, 800 iterations; near-optimal, not certified Bayes-optimal.
Raw affine decoded beliefs are not clipped. Point ties choose state 2.
Warmup is eight steps only for probe fitting/testing (baseline protocol); the banded four-layer context-32 encoder can depend on up to 128 observations.
Shadow forward uses the current executed cache and is discarded; earlier shifts affect subsequent-block caches only.
Selected strengths: {"point": 0.5, "mirror": 0.75}

Uncertainty: episode standard error within one frozen training seed

| Condition | Occupancy ± SE |
|---|---:|
| block_1/point | 0.111603 ± 0.002668 |
| block_2/point | 0.125671 ± 0.002681 |
| block_3/point | 0.148865 ± 0.002492 |
| block_4/point | 0.170074 ± 0.002346 |
| post_final_norm/point | 0.200226 ± 0.002416 |
| orthogonal/point | 0.313019 ± 0.002399 |
| block_1/mirror | 0.073700 ± 0.001682 |
| block_2/mirror | 0.078003 ± 0.001734 |
| block_3/mirror | 0.073944 ± 0.001659 |
| block_4/mirror | 0.077148 ± 0.001769 |
| post_final_norm/mirror | 0.113312 ± 0.002096 |
| orthogonal/mirror | 0.312653 ± 0.002405 |
| intact | 0.313019 ± 0.002399 |
| random | 0.162842 ± 0.001746 |

## Final pre-LayerNorm curves

| λ | Point ± SE | Mirror ± SE | Predicted point | Predicted mirror |
|---:|---:|---:|---:|---:|
| 0 | 0.313019 ± 0.002399 | 0.313019 ± 0.002399 | 0.311584 | 0.311584 |
| 0.25 | 0.313019 ± 0.002399 | 0.313019 ± 0.002399 | 0.311584 | 0.311584 |
| 0.4 | 0.312225 ± 0.002415 | 0.312714 ± 0.002408 | 0.289948 | 0.311584 |
| 0.45 | 0.265076 ± 0.002132 | 0.303894 ± 0.002457 | 0.272522 | 0.311584 |
| 0.5 | 0.170074 ± 0.002346 | 0.224548 ± 0.002238 | 0.071381 | 0.183350 |
| 0.55 | 0.074554 ± 0.001691 | 0.208435 ± 0.002414 | 0.070862 | 0.076202 |
| 0.6 | 0.073639 ± 0.001682 | 0.201050 ± 0.002703 | 0.069427 | 0.076202 |
| 0.75 | 0.073639 ± 0.001682 | 0.077148 ± 0.001769 | 0.069122 | 0.069427 |
| 0.9 | 0.073639 ± 0.001682 | 0.073639 ± 0.001682 | 0.068787 | 0.069336 |
| 1 | 0.073639 ± 0.001682 | 0.073639 ± 0.001682 | 0.068726 | 0.069122 |
