# Full-State Quotient belief steering

Occupancy is decision-time reward-state occupancy, including reset decisions.
Actions: Noop, Tilt X, Tilt Y. States: 1, 2, 3 (reward state).

Reference: N=60 nearest-grid, gamma=0.97, 800 iterations; near-optimal, not certified Bayes-optimal.
Raw affine decoded beliefs are not clipped. Point ties choose state 2.
Warmup is eight steps only for probe fitting/testing (baseline protocol); the banded four-layer context-32 encoder can depend on up to 128 observations.
Shadow forward uses the current executed cache and is discarded; earlier shifts affect subsequent-block caches only.
Selected strengths: {"point": 0.55, "mirror": 0.75}

Uncertainty: episode standard error within one frozen training seed

| Condition | Occupancy ± SE |
|---|---:|
| block_1/point | 0.119476 ± 0.002563 |
| block_2/point | 0.082642 ± 0.002140 |
| block_3/point | 0.083099 ± 0.002232 |
| block_4/point | 0.090973 ± 0.002207 |
| post_final_norm/point | 0.095215 ± 0.002556 |
| orthogonal/point | 0.313019 ± 0.002399 |
| block_1/mirror | 0.113068 ± 0.002517 |
| block_2/mirror | 0.077454 ± 0.001811 |
| block_3/mirror | 0.074493 ± 0.001725 |
| block_4/mirror | 0.097870 ± 0.002260 |
| post_final_norm/mirror | 0.092438 ± 0.002189 |
| orthogonal/mirror | 0.313019 ± 0.002399 |
| intact | 0.313019 ± 0.002399 |
| random | 0.162842 ± 0.001746 |

## Final pre-LayerNorm curves

| λ | Point ± SE | Mirror ± SE | Predicted point | Predicted mirror |
|---:|---:|---:|---:|---:|
| 0 | 0.313019 ± 0.002399 | 0.313019 ± 0.002399 | 0.311584 | 0.311584 |
| 0.25 | 0.313019 ± 0.002399 | 0.313019 ± 0.002399 | 0.311584 | 0.311584 |
| 0.4 | 0.307129 ± 0.002345 | 0.312592 ± 0.002395 | 0.289948 | 0.311584 |
| 0.45 | 0.262756 ± 0.002619 | 0.299469 ± 0.002580 | 0.272522 | 0.311584 |
| 0.5 | 0.193848 ± 0.002985 | 0.220612 ± 0.002197 | 0.071381 | 0.183350 |
| 0.55 | 0.090973 ± 0.002207 | 0.208832 ± 0.002571 | 0.070862 | 0.076202 |
| 0.6 | 0.075165 ± 0.001792 | 0.204742 ± 0.002500 | 0.069427 | 0.076202 |
| 0.75 | 0.073639 ± 0.001682 | 0.097870 ± 0.002260 | 0.069122 | 0.069427 |
| 0.9 | 0.073639 ± 0.001682 | 0.076843 ± 0.001775 | 0.068787 | 0.069336 |
| 1 | 0.073639 ± 0.001682 | 0.074249 ± 0.001677 | 0.068726 | 0.069122 |
