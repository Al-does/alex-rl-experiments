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
| block_1/point | 0.139862 ± 0.002351 |
| block_2/point | 0.170227 ± 0.002690 |
| block_3/point | 0.173340 ± 0.002188 |
| block_4/point | 0.142914 ± 0.002756 |
| post_final_norm/point | 0.180878 ± 0.002235 |
| orthogonal/point | 0.313019 ± 0.002399 |
| block_1/mirror | 0.076782 ± 0.001763 |
| block_2/mirror | 0.109711 ± 0.002258 |
| block_3/mirror | 0.087219 ± 0.001928 |
| block_4/mirror | 0.087006 ± 0.001804 |
| post_final_norm/mirror | 0.088104 ± 0.002192 |
| orthogonal/mirror | 0.313019 ± 0.002399 |
| intact | 0.313019 ± 0.002399 |
| random | 0.162842 ± 0.001746 |

## Final pre-LayerNorm curves

| λ | Point ± SE | Mirror ± SE | Predicted point | Predicted mirror |
|---:|---:|---:|---:|---:|
| 0 | 0.313019 ± 0.002399 | 0.313019 ± 0.002399 | 0.311584 | 0.311584 |
| 0.25 | 0.312958 ± 0.002414 | 0.312958 ± 0.002414 | 0.311584 | 0.311584 |
| 0.4 | 0.309906 ± 0.002400 | 0.312225 ± 0.002413 | 0.289948 | 0.311584 |
| 0.45 | 0.270660 ± 0.002324 | 0.302795 ± 0.002321 | 0.272522 | 0.311584 |
| 0.5 | 0.142914 ± 0.002756 | 0.245697 ± 0.002437 | 0.071381 | 0.183350 |
| 0.55 | 0.084167 ± 0.001809 | 0.208923 ± 0.002268 | 0.070862 | 0.076202 |
| 0.6 | 0.074890 ± 0.001760 | 0.188263 ± 0.003136 | 0.069427 | 0.076202 |
| 0.75 | 0.073639 ± 0.001682 | 0.087006 ± 0.001804 | 0.069122 | 0.069427 |
| 0.9 | 0.073639 ± 0.001682 | 0.075104 ± 0.001753 | 0.068787 | 0.069336 |
| 1 | 0.073639 ± 0.001682 | 0.073761 ± 0.001700 | 0.068726 | 0.069122 |
