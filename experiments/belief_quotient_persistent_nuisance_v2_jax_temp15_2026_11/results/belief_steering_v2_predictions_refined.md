# Refined FSQ numerical predictions

Barycentric interpolation of Bellman continuation values and policy action values; fixed deterministic action ties.
All γ/N solves have span Bellman residual below 1e-8. Prior nearest-grid references remain in the JSON for comparison.

| γ | Resolution comparison | Maximum occupancy difference | Below .01 |
|---:|---|---:|---|
| 0.995 | 150 → 200 | 0.00140381 | True |
| 0.995 | 200 → 300 | 0.00000000 | True |
| 0.99 | 150 → 200 | 0.00106812 | True |
| 0.99 | 200 → 300 | 0.00268555 | True |
| 0.97 | 150 → 200 | 0.00057983 | True |
| 0.97 | 200 → 300 | 0.00057983 | True |

## γ=.995, N=300 curves

| λ | Point | Mirror | ρ flip | ρ point |
|---:|---:|---:|---:|---:|
| 0 | 0.506104 | 0.506104 | 0.506104 | 0.506104 |
| 0.25 | 0.497864 | 0.491638 | 0.209869 | 0.209869 |
| 0.4 | 0.444946 | 0.508575 | 0.209869 | 0.209869 |
| 0.45 | 0.424744 | 0.503174 | 0.209869 | 0.209869 |
| 0.5 | 0.111420 | 0.226471 | 0.209869 | 0.209869 |
| 0.55 | 0.111420 | 0.117310 | 0.209869 | 0.209869 |
| 0.6 | 0.111420 | 0.117310 | 0.209869 | 0.209869 |
| 0.75 | 0.111420 | 0.117188 | 0.209869 | 0.181000 |
| 0.9 | 0.109253 | 0.115601 | 0.209869 | 0.063446 |
| 1 | 0.098358 | 0.115265 | 0.124969 | 0.063446 |

Matched 32 complete 1024-step reference episodes, seed 2042; resolution agreement is empirical, not an optimality certificate.

At mirror λ=.5, installed b1=b2 exactly: numerical X/Y ties must not become a history-dependent arbitrary policy. A fixed lowest-index tie rule removes that artifact. Discount sensitivity and finite-episode uncertainty remain distinct from grid sensitivity.
