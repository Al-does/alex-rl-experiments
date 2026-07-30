# Cycle 5 action-asymmetry — Vast multi-seed findings

Five seeded runs per variant (`seeds 42–46`) on separate Vast RTX 4090 boxes (`delay=0`, `700k` env steps, experiment SHA `1903c88`, harness `2d2c2ff`).

## Mean episode return (training)

| variant | mean return | std |
|---------|------------:|----:|
| variant_1 | 533.70 | 19.02 |
| variant_2 | 315.68 | 6.10 |
| variant_3 | 284.17 | 4.67 |

## Final greedy reward-state-2 occupancy (held-out probe rollout)

Fraction of greedy steps while the hidden state is reward state 2.

| seed | variant_1 | variant_2 | variant_3 |
|-----:|----------:|----------:|----------:|
| 42 | 57.7% | 33.3% | 31.6% |
| 43 | 57.0% | 33.0% | 31.3% |
| 44 | 56.5% | 32.7% | 31.0% |
| 45 | 57.1% | 33.0% | 31.4% |
| 46 | 57.1% | 33.3% | 31.7% |

- **variant_1**: 57.1% ± 0.4% (range 56.5–57.7%)
- **variant_2**: 33.1% ± 0.2% (range 32.7–33.3%)
- **variant_3**: 31.4% ± 0.2% (range 31.0–31.7%)

## Final greedy action mix

Percentages are `[noop, positive, negative]` from final checkpoint probes.

| variant | mean noop | mean pos | mean neg |
|---------|----------:|---------:|---------:|
| variant_1 | 0.0% | 100.0% | 0.0% |
| variant_2 | 33.2% | 66.8% | 0.0% |
| variant_3 | 31.9% | 34.0% | 34.1% |

## Final probe MSE

| seed | variant_1 | variant_2 | variant_3 |
|-----:|----------:|----------:|----------:|
| 42 | 0.005329 | 0.001183 | 0.000322 |
| 43 | 0.000575 | 0.002523 | 0.000358 |
| 44 | 0.004872 | 0.002582 | 0.000228 |
| 45 | 0.000345 | 0.002809 | 0.000422 |
| 46 | 0.001029 | 0.003053 | 0.000290 |

See `multi_seed_summary.json` for machine-readable aggregates.
