# decoupled_kelly vs predictive_loss — paired t-test

Control: `predictive_loss`. Candidate: `decoupled_kelly`.

Metric: held-out affine probe MSE at checkpoint index **2** (~659,185 env steps).

Design: 15 same-seed pairs (45–59). For each pair, `diff_i = candidate_i − control_i` (negative ⇒ candidate lower MSE / better).

## Per-seed values

| seed | predictive_loss (control) MSE | decoupled_kelly (candidate) MSE | diff (candidate − control) |
|---:|---:|---:|---:|
| 45 | 0.000463035 | 0.000294358 | -1.686774e-04 |
| 46 | 0.000428907 | 0.000281112 | -1.477951e-04 |
| 47 | 0.000530100 | 0.000279336 | -2.507641e-04 |
| 48 | 0.000335098 | 0.000314884 | -2.021390e-05 |
| 49 | 0.000515539 | 0.000356624 | -1.589154e-04 |
| 50 | 0.000470413 | 0.000315354 | -1.550598e-04 |
| 51 | 0.000379831 | 0.000319857 | -5.997413e-05 |
| 52 | 0.000427771 | 0.000287940 | -1.398308e-04 |
| 53 | 0.000553380 | 0.000289356 | -2.640240e-04 |
| 54 | 0.000622278 | 0.000278992 | -3.432853e-04 |
| 55 | 0.000430882 | 0.000262999 | -1.678829e-04 |
| 56 | 0.000300203 | 0.000245226 | -5.497659e-05 |
| 57 | 0.000289862 | 0.000232056 | -5.780617e-05 |
| 58 | 0.000291554 | 0.000289358 | -2.195777e-06 |
| 59 | 0.000314693 | 0.000202015 | -1.126783e-04 |

Mean paired difference: **-1.402720e-04** (SD = 9.475299e-05).

## Normality check (Shapiro–Wilk on the 15 differences)

| statistic | value |
|---|---:|
| W | 0.948404 |
| p | 0.499680 |

Normality **passes** at α = 0.05.

## Paired t-test

`scipy.stats.ttest_rel(candidate, control)`:

| statistic | value |
|---|---:|
| t | -5.733550 |
| p | 0.000052 |
| df | 14 |

decoupled_kelly MSE is lower than predictive_loss on mean paired delta=-0.000140272; paired t=-5.73355, p=5.17146e-05.

Machine-readable copy: [`decoupled_kelly_vs_predictive_loss_paired_ttest.json`](decoupled_kelly_vs_predictive_loss_paired_ttest.json).
