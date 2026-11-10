# Persistent-nuisance history ladder (pure JAX)

Twelve leaves: variant_{1,2,3}_{transformer,memoryless,gru128,gru92}.
Source: belief_quotient_history_jax_2026_11, PR #195 merge
`0ca34b25ea14efe9e5668c79d258b90fb2cd7200`.
Development origin/main: `2fb56dce3d63b4e01927100958a431babf5a79ab`.
The shared trainer has since gained continuation and temperature arguments;
these fresh runs use the default temperature 1.0 and no restoration.
Historical studies and trainer are unchanged.

## Process: persistent_nuisance_v1

Uniform initial distribution; S1/S2/S3 are indices 0/1/2, tokens A/B/C.
Explicit matrices (E is not an emission-accuracy alpha):
```text
T0 = [[.9375,.0375,.025], [.0375,.9375,.025], [.075,.075,.85]]
E  = [[.1875,.0625,.75], [.0625,.1875,.75], [0,0,1]]
```
Actions 0/1/2 are Noop/Tilt X/Tilt Y. Tilt multiplies the S3 destination
weight by exp(1.5*d[source state]), then normalizes each row:
- v1 Trivial: X=(+1,+1,+1), Y=(-1,-1,-1).
- v2 Reward-State: X=(+1,+1,-1), Y=(-1,-1,-1).
- v3 Full-State: X=(+1,-1,-1), Y=(-1,+1,-1).

These are the source DIRECTIONS tables. Reward is pre-transition S3 occupancy.
Reset emits from the initial state; action transitions then emits the next token.
The actor sees only current-token and previous-action one-hots (zero action at
reset); memoryless reads only the current token. No hidden state, reward or
belief is exposed. Exact zero emission probabilities remain zero; categorical
sampling accepts -inf logits and reachable filter normalizers are positive.
Training, evaluation and exact probe targets share the same explicit process.

## Unchanged PR #195 recipe

Constants/config factories and models are imported from the original JAX study,
not the lumped studies: target 200M steps, 11 complete rollouts
(219,152,384 actual steps), 19,456 envs x 1,024 steps, 64 episode minibatches,
one epoch, Adam lr 4.2e-4, gamma .99, PPO clip .2, no critic or gradient
clipping, standardized Monte Carlo reward-to-go, constant entropy .015.
Seed defaults to 42. Transformer 64x4, one head, banded RoPE window 32
(lookback 128); memoryless residual MLP 64x6; GRU 128x2 / 92x4 with 32-step
truncated BPTT seeded from rollout hidden states. Initialization is unchanged.
Probes: greedy independent fit/test episodes, 64 warmup steps,
60K/80K rows, affine ridge 1e-6, post-final-LayerNorm embedding.
Checkpoints include actual step-0 initialization, log-spaced iterations,
every 2M threshold crossing, and final. All execution remains pure JAX.

## Commands and checks

```bash
uv sync --group jax  # use --group jax-cuda for GPU dependencies
uv run --group jax python -m experiments.belief_quotient_persistent_nuisance_jax_2026_11.design
uv run --group jax pytest -q tests/test_history_jax_persistent_nuisance.py tests/test_history_studies_jax.py
uv run --group jax rl-harness experiments.belief_quotient_persistent_nuisance_jax_2026_11.variant_2_gru128.experiment --smoke --hardware-profile cpu --no-upload-artifacts
```

Replace the leaf to launch any arm; omit --smoke and choose a GPU hardware
profile for full training. Smoke uses 2 envs/2 minibatches, two iterations
(4,096 steps), with step 0, 2,048 and 4,096 probes; ignored leaf-local
.smoke outputs are not scientific results.

## Analytical references and interpretation

For v2, rho=P(S3|history), r=u_a(1-rho)+v_a*rho.
A/B force rho'=0; C gives rho'=r/(r+.75*(1-r)).
P(A or B)=.25*(1-r), P(C)=r+.75*(1-r), independent of the S1/S2 split.
Tilt X between successive observations gives:
AAAC=[.792774050481,.074363078022,.132862871497],
BBAC=[.295147599840,.571989528662,.132862871497].
The fine coordinate persists even though the reward coordinate agrees.

Enumerating deterministic current-token policies gives [X,X,Noop],
stationary expected occupancy .228861871656. With
U_a=.25*u_X+.75*u_a, max U_a/(U_a+1-v_a) is attained at Noop;
randomizing the C action cannot improve this linear-fractional objective.
The feasible controller Noop iff the current token AND the preceding two
tokens are all C, otherwise X, attains .251424641015 (9.85868428% gain).
The design calculation augments the hidden state with two preceding tokens;
emission of the current token precedes action selection.

These are model-based stationary values, not hard caps on finite-episode noisy
evaluation. The feasible history controller is not Bayes-optimal.
Optimal Reward-State control can ignore the non-reward coordinate, but reward
optimization is not guaranteed to erase it from learned representations.
Probe R² measures linear accessibility, not causal use.

## Smoke verification

All 12 CPU smoke leaves passed: two training iterations and finite belief
probe r_squared at steps 0, 2,048 and 4,096. Outputs stay in ignored .smoke/.
