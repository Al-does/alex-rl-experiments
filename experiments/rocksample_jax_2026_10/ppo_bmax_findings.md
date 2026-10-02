# Large-batch PPO campaign (ppo_bmax_r1–r6): every agent, by final return

Run ID = `<leaf>/<arm>`; each leaf has one harness run ID (`results/<timestamp>-<hash>/`), and arms share it, so arms are identified by name inside that run's `summary.json`. Curves: `results/ppo_bmax_r5.png`, `results/ppo_bmax_r6.png`. Reward = final-update episode return (baselines: tail-20% return), mean over seeds, per-seed in brackets. RLlib PPO plateau ≈ 25.

**Control recipe** (`r3/b1m_d128_loose_ep16`): batch 1,048,576 (8,192 envs × 128 steps), 64 minibatches of 16k, 16 epochs, lr 1e-3, clip 0.3, KL target 0.05 (adaptive KL loss), entropy 0.05, vf_coeff 0.5, vf_clip 1e6, transformer d128 / 3 layers / 4 heads / context 32. "Changed" is relative to this recipe; r6/r6b seeds are merged with their r5 seed-0 run (same keys).

## 30M-step agents (29 updates)

| Run ID | Final reward | Seeds | Changed vs control / notes |
|---|---:|---|---|
| `r5`+`r6`/`d128_kl0.1` | **42.5** [43.6, 39.1, 43.7, 43.7] | 4 | KL target 0.1. Ties control (3/4 seeds in the ~43.5 basin); better worst seed (39 vs 34) |
| `r5`+`r6`/`d128_mb8k` | **42.4** [43.7, 39.2, 43.4, 43.5] | 4 | 128 minibatches of 8k (2,048 grad steps/update). Ties control, 3/4 |
| `r6`+`r6b`/`d128_kl0.1_mb8k` | 42.2 [39.1, 43.9, 43.7] | 3 (s1–3) | both of the above; 2/3 in top basin, seed 0 not run |
| `r3/b1m_d128_loose_ep16` | **41.3** [43.5, 34.1, 43.4, 44.1] | 4 | **control**; r2's 10M run continued to 30M. Best recipe; 3/4 seeds at 43–44 |
| `r4/b1m_d256_loose_ep16` | 39.2 [39.2] | 1 (s0) | d_model 256; seed-matched vs control s0 43.5 |
| `r5/d128_l4` | 39.1 [39.1] | 1 | 4 layers |
| `r5/d128_ctx16` | 39.1 [39.1] | 1 | context 16 |
| `r5/d64` | 39.1 [39.1] | 1 | d_model 64 |
| `r3/b1m_d256_loose` | 39.1 [34.2, 43.7, 39.1, 39.2] | 4 | d_model 256, 8 epochs; r2 arm continued to 30M |
| `r5/d128_clip0.5` | 39.0 [39.0] | 1 | clip 0.5 |
| `r5/d128_ep32` | 38.9 [38.9] | 1 | 32 epochs |
| `r5/d128_l2` | 38.9 [38.9] | 1 | 2 layers |
| `r5/d128_ctx64` | 38.8 [38.8] | 1 | context 64 |
| `r3/b1m_d128_push` | 36.6 [34.2, 29.2, 43.8, 39.0] | 4 | clip 0.5, KL target 0.1, entropy 0.01; entropy collapses to ~0.4, freezes in worse optima |
| `r5/d192` | 34.1 [34.1] | 1 | d_model 192; worst single-seed arm |

All 30M runs end on one of three levels (~34, ~39, ~43.5); all plateau by ~20M.

## 10M-step agents (10 updates at batch 1M)

| Run ID | Final reward | Seeds | Changed vs control / notes |
|---|---:|---|---|
| `r2/b1m_d128_push` | 29.9 [30.4, 29.0, 30.5, 29.7] | 4 | clip 0.5, KL 0.1, entropy 0.01; best at 10M but see r3 (36.6 at 30M) |
| `r2/b1m_d128_loose_ep16` | 29.2 [30.7, 30.0, 28.9, 27.1] | 4 | control recipe at 10M; epochs 8→16 was the load-bearing change |
| `r2/b1m_d256_loose` | 28.2 [30.1, 29.3, 28.7, 24.7] | 4 | d_model 256, 8 epochs |
| `r1/b1m_d128_loose` | 26.0 [26.3, 28.0, 25.4, 24.3] | 4 | 8 epochs ("loose" base: lr 1e-3, clip 0.3, KL 0.05 vs defaults) |
| `r1/b1m_d64_loose` | 25.0 [31.0, 26.7, 22.3, 20.0] | 4 | d_model 64, 8 epochs |
| `r1/b1m_d64_nokl_clip0.5` | 24.8 [28.4, 27.0, 19.6, 24.1] | 4 | d64, clip 0.5, KL loss off, 32 minibatches of 32k, 8 epochs |
| `r2/b1m_d128_loose_ent0.01` | 24.3 [24.9, 26.7, 22.1, 23.6] | 4 | 8 epochs, entropy 0.01 |
| `r1/b1m_d128_loose_vfclip100` | 23.6 [24.9, 25.1, 23.2, 21.2] | 4 | 8 epochs, vf_clip 100 (never binds) |
| `r1/b1m_d128_loose_vf1` | 23.4 [24.4, 25.4, 24.0, 19.7] | 4 | 8 epochs, vf_coeff 1.0 |
| `r2/b1m_d128_loose_vfclip10` | 22.1 [24.9, 23.9, 20.7, 19.0] | 4 | 8 epochs, vf_clip 10 (binds; hurts) |
| `r1/b2m_d128_nokl_clip0.5` | 22.1 [22.9, 22.9, 19.5, 23.2] | 4 | batch 2M (16,384 envs, 5 updates), clip 0.5, KL loss off |
| `r2/b1m_d128_loose_lr3e-3` | 20.9 [23.8, 20.4, 19.5, 19.7] | 4 | 8 epochs, lr 3e-3 |
| `r1/b262k_d128_loose` | 20.7 [24.3, 19.6, 15.0, 24.1] | 4 | batch 262k (2,048 envs, 39 updates), 16 minibatches, 4 epochs |
| `r1/b1m_d64_ref` | 11.5 [11.6, 11.2, 12.8, 10.6] | 4 | naive scale-up: d64, PPO defaults (lr 3e-4, clip 0.2, KL 0.01, 4 epochs, 32 mb of 32k) |

## 5M-step baselines (16 vmapped seeds, original recipes)

| Run ID | Final reward | Seeds | Notes |
|---|---:|---|---|
| `ppo_n5_k7_d64_b65k` | 22.4 mean / 23.8 median [19.5–24.2] | 16 | batch 65,536 (8,192 envs × 8 steps), 16 mb of 4k, 4 epochs, d64, lr 3e-4, clip 0.2 |
| `ppo_n5_k7_d64` | 17.7 mean / 18.4 median [10.0–24.1] | 16 | batch 8,192 (1,024 envs × 8 steps), 8 mb of 1k, 4 epochs, d64 |
