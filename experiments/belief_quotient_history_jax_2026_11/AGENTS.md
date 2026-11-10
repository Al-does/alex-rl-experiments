# Belief-quotient history ladder (pure JAX)

JAX rewrite of `belief_quotient_history_REINFORCE_2026_11`: the same 12 arms
(`variant_{1,2,3}_{transformer,memoryless,gru128,gru92}`) with no RLlib, Ray or
PyTorch in the training path. Environment, policy, rollout, returns, loss and
Adam update all run inside one jitted function per iteration (`reinforce.py`).

## Science (unchanged from the source study)

- Controlled MESS3, emission accuracy 0.75, sticky baseline
  `0.75 I + 0.25 T_old`, reward-state odds tilts with effect size 1.5
  (`shared.py`; parity with the source `HMMEnv` is tested).
- Reward 1 when the pre-transition state is 2; zero-delay token emission;
  observation `[one_hot(token), one_hot(previous action)]`, zero action at reset.
- Critic-free Monte Carlo policy gradient in the PPO shell: gamma 0.99, no
  critic, advantages = batch-standardized discounted reward-to-go, one epoch
  split into 64 minibatches of whole episodes (one Adam step each, lr 4.2e-4,
  PPO clip 0.2, no gradient clipping), constant additive entropy bonus 0.015
  (not annealed). Budget 200M env steps (source: 10M); fresh runs only, seed default 42.
  The source study used one full-batch minibatch (pure REINFORCE) and entropy
  0.03 annealed to zero at 9M steps.
- Architectures: transformer 64x4, 1 head, banded RoPE window 32 (lookback
  128); memoryless residual MLP 64x6 reading only the current token;
  GRU 128x2 and GRU 92x4 with 32-step truncated BPTT chunks seeded from the
  rollout hidden state. PyTorch default initialization; no value head.

## Deliberate execution differences

- Each iteration samples `num_envs=19,456` synchronized, complete 1,024-step
  episodes (19,922,944 steps), split into 64 minibatches of 304 episodes
  (311,296 steps); 200M steps = 11 rollouts = 704 Adam steps. 19,456 is the
  largest multiple of 64 keeping the transformer (the most memory-hungry arm,
  ~0.86 GB per 1,000 envs; OOM at 22,528) under ~70% of a 24 GB RTX 4090, and
  is shared by every arm. The RLlib source used ~344K-393K steps per update. RLlib's randomized first
  episode lengths only desynchronize its runners and are not reproduced.
- Probes: greedy JAX rollouts of fresh episodes, 64-step warmup per episode,
  60K fit / 80K test rows, ridge 1e-6 affine probe on the post-final-LayerNorm
  embedding against the exact filtered belief. The same probe streams are used
  at every checkpoint, so changes isolate parameter changes. Bootstrap CIs and
  conditional-branch metrics from the RLlib battery are not ported.
- Checkpoints are parameter `.npz` files under `artifacts/<run>/checkpoints/`
  at step 0, log-spaced iterations, every 2M steps, and the end.

## Run

```bash
uv sync --group jax            # CPU; use --group jax-cuda on GPU boxes
uv run rl-harness experiments.belief_quotient_history_jax_2026_11.variant_2_gru128.experiment --smoke --hardware-profile cpu --no-upload-artifacts
uv run pytest -q tests/test_history_studies_jax.py
python -m experiments.belief_quotient_history_jax_2026_11.benchmark --study both --iterations 4 --out bench.json
```

`env.py`, `models.py`, `reinforce.py`, `probe.py` and `training.py` are shared
with `reward_factors_history_jax_2026_11`.
