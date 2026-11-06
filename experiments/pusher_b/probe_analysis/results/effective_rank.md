# Does the late probe-fit improvement in Pusher-B b10 pretraining look like grokking?

Scripts: `effective_rank.py` (metrics → `effective_rank.json`), `plot_effective_rank.py` (figures).
Figures: `pusher_b_b10_erank_ppo.png`, `pusher_b_b10_erank_ppo_ce.png` (twins of the task-1 probe
figure with the accuracy/reward axis replaced by effective rank) and
`pusher_b_b10_grokking_diagnostics.png` (per-stage effective rank, participation ratio, weight norms,
weight-matrix effective rank, activation scale, return and policy entropy).

Protocol: the same 2048 held-out b10 sequences × 127 decision positions as the belief probe
(`SEEDS["test"] = 7002`). Effective rank is Roy & Vetterli's exp(entropy of normalised singular
values) of the mean-centred activation matrix at a residual-stream stage (max 128). The probe
stream is the output of block 4 before the final RMSNorm. Weight metrics are per 2-D parameter of
the restored module; the distance from initialisation is the encoder-wide L2 distance from
`rl_b10/initial_checkpoint` (PPO and PPO+CE share that initialisation, seed 42).

## Numbers (probe stream = block_4, centred)

| run | steps | 1-R² | erank | PR | n99 | ‖θ_enc−θ₀‖ / ‖θ₀‖ | mean weight erank |
|---|---|---|---|---|---|---|---|
| PPO init | 0 | 0.565 | 11.2 | 3.24 | 8 | 0 | 108.0 |
| PPO iter 8 | 2.2M | 0.444 | 9.4 | 1.52 | 5 | 0.040 | 107.9 |
| PPO iter 16 | 4.3M | 0.188 | 10.8 | 1.28 | 6 | 0.065 | 107.6 |
| PPO iter 56 | 15.1M | 0.059 | 14.3 | 1.22 | 8 | 0.088 | 107.3 |
| PPO iter 91 | 55.7M | 0.044 | 13.5 | 1.17 | 7 | 0.091 | 107.2 |
| PPO iter 143 | 115.9M | 0.038 | 13.8 | 1.13 | 7 | 0.095 | 107.1 |
| PPO+CE iter 4 | 1.1M | 0.509 | 10.6 | 2.26 | 6 | 0.034 | 107.9 |
| PPO+CE iter 8 | 2.3M | 0.110 | 16.0 | 2.29 | 9 | 0.069 | 107.5 |
| PPO+CE iter 32 | 9.1M | 0.056 | 30.0 | 1.68 | 19 | 0.118 | 106.7 |
| PPO+CE iter 211 | 60.0M | 0.040 | 28.9 | 1.38 | 17 | 0.149 | 106.3 |
| PPO+CE iter 408 | 116.1M | 0.052 | 32.4 | 1.35 | 21 | 0.176 | 105.9 |

PPO+CE aux-head held-out excess NLL over the Bayes floor (nats/token): 0.067 (init) → 0.0176 (iter 1)
→ 0.0175 (iter 2) → 0.0168 (iter 4) → 0.0061 (iter 8) → 0.0033 (iter 16) → 0.0016–0.0037 afterwards.

## Reading

1. **Timing.** Return (≈81/127) and held-out greedy accuracy (0.638 = Bayes greedy) saturate by
   ~1–2M steps. Most of the probe improvement (1-R² 0.44 → 0.06) happens between 2M and 15M steps,
   i.e. after task performance has plateaued, and it keeps creeping down to 0.038 by 116M. That
   "performance flat, representation still changing" ordering is the one grokking-like ingredient.
2. **No compression.** Grokking in the technical sense (Power et al. 2022; Nanda et al. 2023;
   Liu et al. 2022/2023) is a delayed jump in *generalisation* that is accompanied by weight-norm
   decrease / a low-rank, structured circuit replacing a high-rank memorising solution. Here the
   probe stream is low-dimensional from the start (erank 11 at init, 9–14 throughout, participation
   ratio dropping from 3.2 to 1.1, i.e. one dominant direction plus a tail), the effective rank
   *rises* slightly (9.4 → 14.3) during the 2M–15M probe-fit phase and is then flat (13.2–14.5)
   over the entire 15M–116M window in which 1-R² falls from 0.059 to 0.038. Weight-matrix effective
   ranks stay at their near-initialisation values (mean 108 → 107, max 128) and the encoder moves
   less than 10% of its own norm from initialisation (‖θ‖ grows 39.6 → 39.9). There is no
   collapse, no norm drop, no reorganisation signature.
3. **No memorisation set.** Every PPO batch is freshly sampled from the HMM, so there is no
   train/test gap to close; the probe is held-out but so is every training batch. The late phase
   is a slow refinement inside a fixed ~7-dimensional subspace while the policy is already
   near-deterministic (entropy < 0.01 nats after ~8M steps, 0.001 at 116M): a small, continuous
   drift, not a phase transition.
4. **PPO+CE.** The CE head makes the representation much higher-dimensional (erank 30–37 vs
   13–14; 17–26 PCs for 99% of variance vs 7) and moves the weights further (relative distance 0.18
   vs 0.095). Its aux-head excess NLL has a short plateau at ≈0.017 nats (iters 1–4, 0.3–1.1M
   steps) and then drops to 0.006 at 2.3M, coincident with the probe-fit drop (0.51 → 0.11). That is
   the only step-like event in either run, and it is a plateau-then-drop on the loss being
   optimised with no accompanying rank or norm change, so it is a loss plateau rather than grokking.

Conclusion: the probe-fit improvement after accuracy saturates is real but is not grokking in
the technical sense. Effective rank, participation ratio, weight-matrix rank and weight norm are
all flat over the window where 1-R² keeps improving; nothing compresses or reorganises. Describe it
as delayed/continued refinement of an already low-dimensional belief representation.

Caveats: one seed per arm; checkpoints are log-spaced early so a transition between 2.2M and 4.3M
steps could be sharper than it looks; effective rank of activations is scale- and
normalisation-dependent (centred and uncentred values are both in the JSON; they agree after ~5M
steps once the mean activation shrinks).
