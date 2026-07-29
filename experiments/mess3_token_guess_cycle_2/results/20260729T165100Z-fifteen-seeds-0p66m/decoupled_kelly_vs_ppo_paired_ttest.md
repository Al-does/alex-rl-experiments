# decoupled_kelly vs ppo — paired t-test

Control: `ppo`. Candidate: `decoupled_kelly`.

Metric: held-out affine probe MSE at checkpoint index **2** (~659,185 env steps).

Design: 15 same-seed pairs (45–59). For each pair, `diff_i = candidate_i − control_i` (negative ⇒ candidate lower MSE / better).

## Per-seed values

| seed | ppo (control) MSE | decoupled_kelly (candidate) MSE | diff (candidate − control) |
|---:|---:|---:|---:|
| 45 | 0.000448986 | 0.000294358 | -1.546279e-04 |
| 46 | 0.000324730 | 0.000281112 | -4.361806e-05 |
| 47 | 0.000399874 | 0.000279336 | -1.205386e-04 |
| 48 | 0.000337158 | 0.000314884 | -2.227454e-05 |
| 49 | 0.000377872 | 0.000356624 | -2.124865e-05 |
| 50 | 0.000428121 | 0.000315354 | -1.127674e-04 |
| 51 | 0.000354228 | 0.000319857 | -3.437111e-05 |
| 52 | 0.000372093 | 0.000287940 | -8.415282e-05 |
| 53 | 0.000384146 | 0.000289356 | -9.479025e-05 |
| 54 | 0.000553477 | 0.000278992 | -2.744848e-04 |
| 55 | 0.000364099 | 0.000262999 | -1.010996e-04 |
| 56 | 0.000442033 | 0.000245226 | -1.968067e-04 |
| 57 | 0.000335096 | 0.000232056 | -1.030403e-04 |
| 58 | 0.000377485 | 0.000289358 | -8.812627e-05 |
| 59 | 0.000355868 | 0.000202015 | -1.538538e-04 |

Mean paired difference: **-1.070534e-04** (SD = 6.850356e-05).

## Normality check (Shapiro–Wilk on the 15 differences)

| statistic | value |
|---|---:|
| W | 0.923173 |
| p | 0.215294 |

Normality **passes** at α = 0.05.

## Paired t-test

`scipy.stats.ttest_rel(candidate, control)`:

| statistic | value |
|---|---:|
| t | -6.052474 |
| p | 0.000030 |
| df | 14 |

decoupled_kelly MSE is lower than ppo on mean paired delta=-0.000107053; paired t=-6.05247, p=2.97456e-05.

Machine-readable copy: [`decoupled_kelly_vs_ppo_paired_ttest.json`](decoupled_kelly_vs_ppo_paired_ttest.json).
