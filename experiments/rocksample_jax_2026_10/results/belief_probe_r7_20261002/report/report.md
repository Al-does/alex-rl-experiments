# Matched r7 RockSample belief trajectories

88 measurements: exact initialization + 10 archived checkpoints for each of eight agents.

| Agent | Independent return ± SEM | M R² on policy | J R² on policy | Rocks sampled in <5% of initially-Good episodes |
|---|---:|---:|---:|---|
| run 1 · seed 0 | 39.319 ± 0.197 | 0.7665 | 0.3924 | [5] |
| run 1 · seed 1 | 39.221 ± 0.198 | 0.7735 | 0.3879 | [0] |
| run 1 · seed 2 | 43.972 ± 0.214 | 0.9074 | 0.5941 | [] |
| run 1 · seed 3 | 39.126 ± 0.198 | 0.8134 | 0.4280 | [0] |
| run 2 · seed 0 | 39.072 ± 0.198 | 0.7518 | 0.3925 | [5] |
| run 2 · seed 1 | 39.006 ± 0.199 | 0.7493 | 0.3951 | [0] |
| run 2 · seed 2 | 43.875 ± 0.214 | 0.9063 | 0.6286 | [] |
| run 2 · seed 3 | 44.023 ± 0.214 | 0.9215 | 0.6167 | [] |

## Measurement contract

512 complete held-out episodes sampled from each saved policy; 256 fit/256 test, split seed 831; CV seed 551 uses fit episodes only; raw affine predictions; 200 fixed-probe episode bootstrap resamples. Full joint null battery at final only. Policy histories vary with each checkpoint. Reset rows included; exit ends histories.

Representation: post-final LayerNorm; exact action/observation-only targets M and J; previous action in observations. All held-out policy timesteps are scored. Histories start at reset, and terminal/post-exit rows are excluded. Predictions are not clipped. Read BELIEF_PROBES.md for reproduction commands.

Each nominal seed has identical reconstructed initialization hashes across the two runs. Policy histories vary with checkpoint; their 1−R² curves change with visitation as well as representations. Joint scores use 128 configurations; marginal scores use seven coordinates. Bootstrap bands measure held-out episode uncertainty conditional on each fitted decoder, not training variability.

The <5% sampling rule is a descriptive behavioral flag, not a threshold for absence of a representation. Coordinate-level target variances and undefined scores are preserved in JSON.

## Run comparison

- Seed 0: first saved parameter divergence at update 1; matched initialization identical.
- Seed 1: first saved parameter divergence at update 1; matched initialization identical.
- Seed 2: first saved parameter divergence at update 1; matched initialization identical.
- Seed 3: first saved parameter divergence at update 1; matched initialization identical.
