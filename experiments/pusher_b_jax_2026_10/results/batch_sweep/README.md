# JAX PPO batch / minibatch / LR sweep (Pusher-B, RTX 4090)

10 arms × presets {b10, b90} × seeds {42, 43}, 10M env steps each (target),
6 epochs, 127-step episodes. Arm name `B<n>k_mb<m>_lr<s>` = `num_envs` =
n×1024 synchronised episodes per update, `minibatch_episodes` = m,
`--lr-scale` = s (multiplies the lr schedule). `B32k` arms used
`--rollout-chunks 4` (single-chunk rollout KV cache does not fit 32k envs;
no OOM/fail anywhere — all 40 runs completed). Baseline `B2k_mb64_lr1`
reproduces the legacy RLlib recipe (~262k-step batch, 64-episode
minibatches).

Raw per-iteration curves are in `runs/*.jsonl`; per-run and per-arm
aggregates are in `summary.json`; the table is in `sweep_table.md`;
`batch_sweep.png` plots the Bayes gap vs env steps and vs wall-clock.
`jax_ce{,_bf16}.jsonl` are the matched CE benchmarks (below).

## Headline result

**The baseline wins everywhere.** Bigger batches give no throughput gain
(the 4090 is already saturated at 2,048 envs: ~187k env steps/s for B2k
vs ~189k for B8k_mb64 and *worse* 164–174k for B32k), and learning is
strictly worse at matched env steps *and* matched wall-clock. The
"reduce gradient noise without losing gradient updates" hypothesis fails
on Pusher-B: holding the minibatch fixed preserves grad steps per env
step but not per-update data freshness, and staleness costs more than
the lower gradient noise buys.

Gap = `bayes_greedy_return_mean` − `return_mean` (exact Bayes-greedy
ceiling minus sampled-policy return), mean over 2 seeds.

| arm | env steps/s | wall s | grad/env | gap@2M | gap@5M | gap@10M | final gap |
|---|---|---|---|---|---|---|---|
| **B2k_mb64_lr1 (b10)** | 187k | 82 | 7.4e-4 | 5.36 | 0.47 | **0.00** | 0.09 |
| **B2k_mb64_lr1 (b90)** | 187k | 81 | 7.4e-4 | 2.24 | 0.53 | **0.32** | 0.29 |
| B8k_mb64_lr1 | 189k | 82 | 7.4e-4 | 14.6 / 20.7 | 10.0 / 11.0 | 2.89 / 1.40 | 2.61 / 1.28 |
| B8k_mb64_lr2 | 189k | 82 | 7.4e-4 | 14.4 / 20.9 | 8.9 / 11.0 | **1.94** / **1.25** | 1.76 / 1.13 |
| B8k_mb256_lr1 | 174k | 88 | 1.8e-4 | 14.6 / 21.7 | 9.5 / 11.5 | 3.26 / 1.52 | 2.97 / 1.41 |
| B8k_mb256_lr2 | 174k | 86 | 1.8e-4 | 14.3 / 21.7 | 8.5 / 10.9 | 2.88 / 1.35 | 2.64 / 1.27 |
| B8k_mb256_lr4 | 174k | 87 | 1.8e-4 | 14.3 / 22.0 | 9.0 / 10.8 | 2.94 / 1.34 | 2.70 / 1.25 |
| B32k_mb256_lr1 | 174k | 98 | 1.8e-4 | – / – | 16.5 / 26.7 | 13.5 / 19.0 | 12.7 / 16.5 |
| B32k_mb256_lr2 | 174k | 99 | 1.8e-4 | – / – | 16.5 / 26.6 | 13.8 / 18.6 | 12.9 / 16.4 |
| B32k_mb1024_lr2 | 164k | 103 | 4.6e-5 | – / – | 16.5 / 26.9 | 13.4 / 19.8 | 12.5 / 17.4 |
| B32k_mb1024_lr4 | 164k | 104 | 4.6e-5 | – / – | 16.5 / 26.8 | 13.6 / 19.7 | 12.7 / 17.5 |

(b10 / b90 gaps; B32k does only 3 updates of 4.16M steps, so no 2M point.)

## Findings

- **No throughput upside to bigger batches.** Peak is ~189k env steps/s
  (B8k_mb64), +1% over baseline; B32k is *slower* (164–174k) and burns
  25% more env steps (3 updates × 4.16M = 12.5M) for a 10M target.
  The rollout is a KV-cached sequential scan, so more envs do not help.
- **Staleness dominates gradient noise.** B8k_mb64 keeps the same
  grad-steps/env-step as the baseline (768 grad steps per update, same
  minibatch) yet reaches only gap ~1.9–2.9 at 10M vs ~0–0.3: 6 epochs
  over one stale 1M-step batch underperforms 39 passes over fresh 260k
  batches. All B8k arms lag the baseline from the first update and never
  catch up within 10M steps.
- **Bigger minibatches trade grad steps for noise ~1:1.** B8k_mb256
  (¼ the grad steps/env) ≈ B8k_mb64_lr1 — the noise reduction and the
  lost updates cancel. Raising LR (×2, ×4) does not recover it on
  minibatch-256 arms; on mb64, lr×2 helps modestly (2.89→1.94 on b10).
- **B32k is unusable at this budget.** Only 3 updates in 10M steps;
  gap stays ~13–20, worse throughput, and it needed `--rollout-chunks 4`
  to fit (KV cache). Memory was never the binding constraint — the 4090
  held every config; *on-policy freshness* is.
- **Seed agreement** is decent (gap@10M spread 0.03–4.2 across the 2
  seeds; ordering identical on both seeds and both presets).

## Recommendation

Keep **`B2k_mb64_lr1`** — the RLlib-matched recipe (`num_envs=2048`,
`minibatch_episodes=64`, 6 epochs, base lr schedule). It nearly closes
the Bayes gap on both presets by 10M steps (0.09 b10 / 0.29 b90) at the
best wall-clock. If a bigger batch is ever needed, the only arm worth
extending is `B8k_mb64_lr2`, and only because lr×2 partly compensates —
but it still trails by ~1.5–2 return at matched env steps and offers no
speed advantage. There is no reason to raise the batch on this GPU for
this env.

## bf16 CE side-benchmark

`jax_ce.jsonl` (fp32) vs `jax_ce_bf16.jsonl` (`--compute-dtype
bfloat16`), 3,000 updates, b10, seed 42:

| run | updates/s | end-to-end wall s | excess loss @3k (nats) |
|---|---|---|---|
| fp32 | 30.3 | 164 | 3.79e-4 |
| bf16 | **1.8** | 1715 | **1.38e-4** |

bf16 learns *slightly better* per update (lower excess loss at matched
update count) but is ~16× slower — `jax.nn.dot_product_attention` on
bf16 hits a much slower kernel path than fp32 on this GPU/SHAPE
(B=512, T=127, 4 heads). Not worth it for CE; not tried for PPO
(rollout uses the separate `encode_step` path, still fp32).
