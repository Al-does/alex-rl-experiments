# Fresh-episode env runner retrain — probe comparison

Retrained `variant_2_ctx32_ent` (Reward-State Quotient) and
`variant_3_ctx32_ent` (Full-State Quotient) on seeds 42–46 with
`harness.env_runners.FreshEpisodeSingleAgentEnvRunner`
(`batch_mode="complete_episodes"`, `shared.py`), replacing RLlib's default
runner which re-seeded every environment identically at each sample call.
Nothing else in the recipe changed (10M steps, 10 runs). New runs publish as
`mess3_reward_state_action_symmetry_cycle_7-*` beside the historical
`cycle_6-*` result dirs — no historical results were modified.

Metric: `checkpoint_probe_curve.json` → `global_mse_ratio` (normalized
held-out probe MSE; lower = belief more linearly accessible).
Reference spread: historical seeds 47–56 (n=10).

## Final-checkpoint error (10.19M steps)

### variant_2_ctx32_ent — Reward-State

| seed | cycle_6 | cycle_7 | Δ | |
|------|---------|---------|---|---|
| 42 | 0.014864 | 0.014877 | +0.000013 | |
| 43 | 0.019302 | 0.019348 | +0.000047 | |
| 44 | 0.018329 | 0.017919 | −0.000410 | |
| 45 | 0.019010 | 0.018927 | −0.000084 | |
| 46 | 0.016838 | 0.016640 | −0.000198 | |

Reference spread (47–56): mean 0.018995, sd 0.001196, range
0.01697–0.02078. All Δ ≤ 0.34 sd.

### variant_3_ctx32_ent — Full-State

| seed | cycle_6 | cycle_7 | Δ | |
|------|---------|---------|---|---|
| 42 | 0.002498 | 0.002529 | +0.000031 | |
| 43 | 0.002488 | 0.002580 | +0.000093 | |
| 44 | 0.002601 | 0.002551 | −0.000051 | |
| 45 | 0.002546 | 0.002573 | +0.000027 | |
| 46 | 0.002419 | 0.002386 | −0.000033 | |

Reference spread (47–56): mean 0.002551, sd 0.000108, range
0.00247–0.00284. All Δ ≤ 0.86 sd.

## Error over training (mean over seeds 42–46)

```
steps        v2 c6    v2 c7    Δ          v3 c6    v3 c7    Δ
0            0.04445  0.04445  +0.00000   0.05262  0.05262  +0.00000
362128       0.02638  0.02638  -0.00000   0.04675  0.04675  -0.00000
755344       0.02418  0.02407  -0.00011   0.04295  0.04195  -0.00101
1541776      0.02339  0.02331  -0.00007   0.01179  0.01346  +0.00167
3114640      0.01871  0.01917  +0.00046   0.00304  0.00400  +0.00096
6260368      0.01711  0.01684  -0.00027   0.00251  0.00252  +0.00001
10192528     0.01767  0.01754  -0.00013   0.00251  0.00252  +0.00001
```

Curves are bit-identical through ~360K steps (reseed-vs-continue diverges
only after episode boundaries, which arrive ~every 12 iterations), then
diverge modestly mid-training (worst single-seed deviation: v2 +0.0016,
v3 +0.0137 transient at 1.5M steps on seed 43) and re-converge to
near-identical finals — so the fix demonstrably engaged the RNG path
without moving the outcome.

## Reward-State vs Full-State gap

Final `global_mse_ratio` ratio (v2/v3) per seed:

| seed | cycle_6 | cycle_7 |
|------|---------|---------|
| 42 | 5.95 | 5.88 |
| 43 | 7.76 | 7.50 |
| 44 | 7.05 | 7.03 |
| 45 | 7.47 | 7.36 |
| 46 | 6.96 | 6.97 |

**Verdict: unchanged.** Full-State beliefs remain ~6–8× more linearly
accessible than Reward-State beliefs at the final checkpoint; the gap moved
by ≤0.26 (≈3%) per seed, well inside seed-to-seed spread. The reseed bug in
the default runner did not materially affect the published result.

## Provenance

- Branch `devin/1790311511-fresh-episode-runner`; run IDs
  `mess3_reward_state_action_symmetry_cycle_7-variant_{2,3}_ctx32_ent-seed{42..46}-10m`.
- Each box pushed compact results only; `artifacts/` trees uploaded to B2
  (`slop-bucket/experiments/.../<run_id>/`, durability manifests verified).
- Compute note: complete-episode runners buffer ~40–50 GB of episode
  objects in the object store; runs only succeed on hosts with ≥~50 GB
  `/dev/shm` (the ≥500 GB-RAM class on vast.ai — observed shm 53–62 GB).
  Smaller hosts died by plasma spill → disk-full or Ray OOM worker kills.
