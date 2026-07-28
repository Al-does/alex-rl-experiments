# PPO vs decoupled Kelly — paired t-test

Control: `ppo`. Candidate: `decoupled_kelly`.

Metric: held-out affine probe MSE at the **third checkpoint** (index 2, ~0.66M
environment steps). Later checkpoints excluded because PPO-family MSE degrades
after this point.

Design: three same-seed pairs (42, 43, 44). For each pair,
`diff_i = candidate_i − control_i` (negative ⇒ candidate lower MSE / better).

## Per-seed values

| seed | PPO (control) MSE | Kelly (candidate) MSE | diff (kelly − ppo) |
|---:|---:|---:|---:|
| 42 | 0.000484657 | 0.000356847 | −1.278e-04 |
| 43 | 0.000277524 | 0.000239974 | −3.755e-05 |
| 44 | 0.000383542 | 0.000320406 | −6.314e-05 |

Mean paired difference: **−7.617e-05** (SD = 4.652e-05).

## Normality check (Shapiro–Wilk on the 3 differences)

| statistic | value |
|---|---:|
| W | 0.941161 |
| p | 0.532062 |

Normality **passes** at α = 0.05 (p ≥ 0.05), so a paired t-test is treated as
applicable. With n = 3, Shapiro–Wilk has low power and is only a weak gate.

## Paired t-test

`scipy.stats.ttest_rel(candidate, control)`:

| statistic | value |
|---|---:|
| t | −2.835887 |
| p | 0.105103 |
| df | 2 |

Kelly has lower MSE on every seed, but the mean paired improvement is **not
significant** at α = 0.05 (p = 0.105).

Machine-readable copy: [`ppo_vs_kelly_paired_ttest.json`](ppo_vs_kelly_paired_ttest.json).
