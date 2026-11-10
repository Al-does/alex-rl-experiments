# Belief-quotient history ladder (pure JAX)

This study supersedes the environment of PR #195’s `belief_quotient_history_jax_2026_11`; that folder remains unchanged as the historical record.

Derived from `belief_quotient_history_jax_2026_11`: the same 12 arms
(`variant_{1,2,3}_{transformer,memoryless,gru128,gru92}`) with no RLlib, Ray or
PyTorch in the training path. Environment, policy, rollout, returns, loss and
Adam update all run inside one jitted function per iteration (`reinforce.py`).

## Science (deliberately revised baseline)

- Controlled MESS3, emission accuracy 0.75, baseline
  `[[.4875,.4875,.025],[.4875,.4875,.025],[.075,.075,.85]]`, applied before
  reward-state odds tilts with effect size 1.5. Non-reward destination identity
  is resampled uniformly; reward-class dynamics remain sticky. Reward-State
  Quotient has an exact scalar C/not-C filter; A/B give identical reward-class
  evidence but still distinguish fine states. Full-State Quotient retains
  opposing source-state action preferences. Actor tokens are not merged.
  HMMEnv parity tests explicitly substitute these revised parameters.
- Reward 1 when the pre-transition state is 2; zero-delay token emission;
  observation `[one_hot(token), one_hot(previous action)]`, zero action at reset.
- Critic-free PPO with Monte Carlo returns: gamma 0.99, no
  critic, advantages = batch-standardized discounted reward-to-go, one epoch
  split into 64 minibatches of whole episodes (one Adam step each, lr 4.2e-4,
  PPO clip 0.2, no gradient clipping), constant additive entropy bonus 0.015
  (not annealed). Budget 2B env steps (PR #195: 200M); fresh runs only, seed default 42.
  The earlier RLlib study used one full-batch minibatch (pure REINFORCE) and entropy
  0.03 annealed to zero at 9M steps.
- Architectures: transformer 64x4, 1 head, banded RoPE window 32 (lookback
  128); memoryless residual MLP 64x6 reading only the current token;
  GRU 128x2 and GRU 92x4 with 32-step truncated BPTT chunks seeded from the
  rollout hidden state. PyTorch default initialization; no value head.

## Deliberate execution differences

- Each iteration samples `num_envs=19,456` synchronized, complete 1,024-step
  episodes (19,922,944 steps), split into 64 minibatches of 304 episodes
  (311,296 steps); 2B target = 101 rollouts = 6,464 Adam steps
  (2,012,217,344 actual steps). 19,456 is the
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
uv run rl-harness experiments.belief_quotient_history_jax_lumped_2026_11.variant_2_gru128.experiment --smoke --hardware-profile cpu --no-upload-artifacts
uv run --group jax pytest -q tests/test_history_studies_jax.py tests/test_history_jax_lumped_baseline.py tests/test_history_jax_lumped_parity.py
uv run --group jax python -m experiments.belief_quotient_history_jax_lumped_2026_11.design
```

The historical ladder provides `env.py`, `models.py`, `reinforce.py`,
`probe.py` and `training.py` by import.

Resolved provenance labels this environment `uniform_nonreward_destination_v2`.
Saved `results/` in the original PR #195 folders are historical: their recorded kernels, budgets and entropy
recipes remain authoritative, including the old persistent-identity baseline,
10M/30M runs and old 200M capacity projections; none have been rewritten.

Recompute stationary references with `uv run --group jax python -m
experiments.belief_quotient_history_jax_lumped_2026_11.design`: current-token maxima
are 0.2950888739 (Reward-State) and 0.2668669030 (Full-State); feasible
three-token controllers attain 0.3171048338 and 0.2896779494 (7.46% and 8.55%
gains). These are constructive lower bounds, not Bayes optima. The former uses
Noop for current C or two preceding Cs, otherwise Tilt X; the latter uses Noop
after C, otherwise current A/B/C maps to Tilt X/Tilt Y/Noop. Smoke scores do
not establish learned coarsening.

Generic env/models/rollout/loss/training/probe code is reused by import from
PR #195. Each new shared.py owns its baseline and kernels. The new process is
passed explicitly throughout rollout, exact filtering, evaluation and probes.
