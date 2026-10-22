# Untrained-network reference

A randomly initialised transformer, never optimised, scored by the same rank-2 affine belief probe as every trained arm.

| init | belief R² | greedy accuracy | within-branch R² (depth 2) |
|---:|---:|---:|---:|
| 0 | 0.8877 | 0.3331 | -0.6477 |
| 1 | 0.8748 | 0.3190 | -0.8370 |
| 2 | 0.8844 | 0.5016 | -0.6956 |

Mean untrained belief R²: 0.8823. Any trained arm at or below this value has not been shown to represent the belief simplex; the probe is reading the recent token window that a random causal filter already exposes.
