# Results — nonergodic MESS3 `ppo_h100_estimated` campaign (2026-09-12)

Three full runs of the two-component MESS3 token-guess PPO experiment on a
Vast H100 PCIe. All use the updated process parameters
`mess3_a: x=0.20, α=0.95` / `mess3_b: x=0.4667, α=0.95`, `FreshEpisodeSingleAgentEnvRunner`
(fresh episodes — fixes the earlier replay bug that invalidated the first run),
train batch 262,144, minibatch 8,192, d_model=128.

Bayes-optimal accuracy for these parameters: **49.94%** (chance 33.33%; the
probes' empirical Bayes-greedy estimate came out at 49.2%).

## Runs

| Run dir | Config | Total steps | Final held-out reward | Final belief R² | Notes |
|---|---|---|---|---|---|
| `20260912T005726Z-e3230013` | entropy 0.01 constant | 10.27M | 47.7% | 0.817 | First clean run; reward plateaus just under Bayes |
| `20260912T024736Z-fc126d96` | + entropy anneal 0.01→0 over 1.5M–2.5M | 6.22M | 48.4% | 0.845 | Faster to ceiling; slightly better final R² |
| `20260912T043214Z-1bbe2a96` | + LR anneal 1e-4→1e-5 over 0–10M | 15.13M | **48.6%** | **0.889** | **Best run** |

## Best run: `20260912T043214Z-1bbe2a96`

- Reward reached the Bayes neighborhood by ~4M steps and stayed there;
  held-out 48.6% vs empirical Bayes 49.2%.
- **Surprise**: weighted-belief R² peaked early at **0.958 @ 2.16M steps**,
  well *before* reward plateaued — the belief representation forms before the
  policy exploits it, then partially relaxes (final 0.889).
- CEV90 expanded 6 → 28 dims mid-run, then consolidated back to 6 at the final
  checkpoint once LR bottomed out — while R² stayed ~0.89.
- See `run3_r2_cev_reward.png` in this folder for the R²/CEV90/reward overlay.

## Provenance

- Code: branch `devin/nem3-h100-run` (commits `06bdf2c`, `ca37d97`, `8ece22e`
  for the three configs) on top of PR #119.
- Library: `rl-harness` at `bafed8e` (FreshEpisodeSingleAgentEnvRunner).
- Checkpoints and Tune trees are on Backblaze B2; see each run's
  `run_manifest.json` → `remote_artifacts` for the prefix.
