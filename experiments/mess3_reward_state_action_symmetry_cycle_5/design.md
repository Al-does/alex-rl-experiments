# Cycle 5 sticky-state design

Cycle 5 repeats cycle 4's sticky-state environment: the baseline transition
row from reward state 2 is `[0.30, 0.30, 0.40]` instead of `[0.45, 0.45,
0.10]`. Its sole recipe change is the transformer scale: 64 residual
dimensions, 4 layers, 1 attention head, and a context length of 10, matching
token-guess cycle 2. It retains delay 0, emission accuracy 0.85, action-odds
effect size 1.5, PPO hyperparameters, 700k environment steps, and the three
plain-PPO variants.

## Selection criteria

A useful baseline should:

1. give variant 2 a material return incentive to learn its state-dependent
   noop exception;
2. keep the noop boundary reachable but nontrivial under partial observation;
3. preserve the intended fully observed oracle policies;
4. avoid making variant 1 nearly absorbing in state 2; and
5. establish a plain-PPO baseline before auxiliary-loss follow-ups.

For a candidate baseline persistence `p = P(s₂' | s₂, noop)`, the other two
entries in that row are each `(1-p)/2`. Rows 0 and 1 remain unchanged.

## Analytic sweep

All values use action-odds effect size 1.5. `V2 gap` is the fully observed
oracle occupancy minus always-positive occupancy. The belief figures come from
simplex value iteration followed by stationary simulation at emission accuracy
0.85.

| p | V2 noop threshold | V1 oracle occupancy | V2 oracle occupancy | V2 gap | belief-optimal V2 occupancy / noop |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.7541 | 0.3324 | 0.2697 | 0.0156 | 0.2588 / 0.2317 |
| 0.20 | 0.6123 | 0.4135 | 0.2936 | 0.0338 | 0.2770 / 0.2897 |
| 0.30 | 0.5221 | 0.4926 | 0.3220 | 0.0550 | 0.3009 / 0.3082 |
| **0.40** | **0.4621** | **0.5700** | **0.3565** | **0.0802** | **0.3296 / 0.3305** |
| 0.50 | 0.4226 | 0.6457 | 0.3993 | 0.1103 | 0.3652 / 0.3580 |
| 0.60 | 0.3996 | 0.7197 | 0.4539 | 0.1465 | 0.4105 / 0.3931 |
| 0.70 | 0.3939 | 0.7920 | 0.5256 | 0.1899 | 0.4711 / 0.4401 |
| 0.80 | 0.4144 | 0.8629 | 0.6244 | 0.2382 | 0.5583 / 0.5076 |

`p=0.40` increases the V2 oracle gap 5.1-fold while leaving its partially
observed policy genuinely mixed. At `p>=0.60`, V1 becomes dominated by a highly
sticky reward state and V2's runner-up is no longer simply always-positive,
which weakens the intended comparison.

## Selected transition values

The action tilt transforms a baseline state-2 probability `q` in direction
`d ∈ {-1, 0, 1}` as

```text
q_d = q * exp(1.5 d) / (1 - q + q * exp(1.5 d)).
```

For the selected baseline:

| Source state | noop `P(s₂')` | upward tilt | downward tilt |
|---|---:|---:|---:|
| 0 or 1 | 0.100000 | 0.332428 | 0.024192 |
| 2 | 0.400000 | 0.749235 | 0.129491 |

Variant 2's positive action uses the upward value outside state 2 and the
downward value inside state 2. Its one-step noop boundary is therefore

```text
b₂* = (0.332428 - 0.1)
      / ((0.332428 - 0.1) + (0.4 - 0.129491))
    = 0.462141.
```

Exhaustive enumeration of all 27 fully observed deterministic policies gives:

| Variant | Oracle `[s0,s1,s2]` | Oracle occupancy | Runner-up | Gap |
|---|---|---:|---:|---:|
| 1 | `[positive, positive, positive]` | 0.570013 | 0.440409 | 0.129605 |
| 2 | `[positive, positive, noop]` | 0.356519 | 0.276347 | 0.080172 |
| 3 | `[positive, negative, noop]` | 0.356519 | 0.276347 | 0.080172 |

MSE separation is intentionally not treated as analytically predictable:
probe MSE depends on learned representation geometry. The battery holds the
training and probing protocols fixed so five-seed results can test whether the
stronger policy incentive changes the cycle-2 ordering.
