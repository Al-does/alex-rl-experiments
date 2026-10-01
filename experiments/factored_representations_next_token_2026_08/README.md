# Pure next-token factored-representation experiment

This study is the supervised counterpart to PR #59's
`factored_representations_reproduction_2026_08` experiment. It keeps that
study's two- versus three-independent-MESS3 comparison and 64-dimensional
transformer, but removes reinforcement learning completely. Training minimizes
only shifted next-joint-token cross entropy.

The design follows
[“Transformers learn factored representations”](https://arxiv.org/abs/2602.02385)
and the authors'
[released implementation](https://github.com/Astera-org/factored-reps).

## Scientific recipe

- Every factor is Appendix C.1.1 MESS3 with `alpha=0.6`, `x=0.15`, and
  `y=1-2x=0.7`.
- Factors evolve independently. Their hidden predictive states therefore stay
  on the product-state manifold.
- Factor subtokens are hidden from the model and encoded as one Cartesian
  joint token: 9 visible tokens for two factors and 27 for three.
- Each batch contains eight generated tokens. The language-model shift is
  `[BOS, x1, ..., x7] -> [x1, ..., x8]`, matching the paper's released
  generator.
- The only loss is cross entropy over the next joint token. There is no Gym
  environment, action, reward, policy, critic, value target, RLlib Algorithm,
  or PPO term.
- Full training uses the paper's 500,000 optimizer updates, batch size 25,000,
  Adam/AdamW learning rate `5e-4`, betas `(0.9, 0.999)`, and zero weight decay.
- Checkpoints retain initialization, power-of-two updates, and the final
  update. Large checkpoint files stay under ignored `artifacts/`.

The architecture keeps PR #59's controlled adaptation of the paper:

| Property | Paper | This study |
|---|---:|---:|
| Factors | 3 MESS3 + 2 Bloch Walk | 2 or 3 MESS3 |
| Residual width | 120 | 64 |
| Attention heads | 3 × 40D | 4 × 16D |
| Blocks | 4 | 4 |
| MLP width | 480 | 256 |
| Context capacity | 9 | 9 |
| Objective | next-token CE | next-token CE |

Four heads are necessary because the paper's three heads do not evenly divide
a 64-dimensional residual stream.

The geometric alternatives both fit in the residual stream:

| Factors | Factored dimensions | Joint dimensions | Residual capacity |
|---:|---:|---:|---:|
| 2 | 4 | 8 | 64 |
| 3 | 6 | 26 | 64 |

## Analysis

Every retained checkpoint is evaluated on fixed held-out process samples:

1. validation cross entropy and joint-token accuracy, with the exact sampled
   Bayesian loss for the same contexts;
2. held-out affine regression from the final block's pre-final-LayerNorm
   residual stream to concatenated local predictive vectors;
3. activation CEV compared with empirical factored and joint belief CEV;
4. controlled vary-one PCA, centered within frozen context and sequence
   position;
5. effective-dimension additivity and basis-invariant principal-angle overlap;
6. rank-two projected recovery of each factor belief simplex; and
7. visible-token embedding geometry, excluding BOS.

Geometry probes exclude the BOS position. Since the training input has
`BOS,x1,...,x7`, this yields seven posterior-aligned activation positions per
sequence.

## Run

Run both factor counts:

```bash
uv run rl-harness \
  experiments.factored_representations_next_token_2026_08.next_token.experiment \
  --smoke --hardware-profile cpu
```

Run one cell:

```bash
uv run rl-harness \
  experiments.factored_representations_next_token_2026_08.next_token_2_factors.experiment \
  --smoke --hardware-profile cpu

uv run rl-harness \
  experiments.factored_representations_next_token_2026_08.next_token_3_factors.experiment \
  --smoke --hardware-profile cpu
```

Smoke mode follows the authors' released smoke scale: 10 updates with batch
size 24. Full mode is intentionally paper-scale and should run on a GPU.
