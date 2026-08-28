# PPO reproduction of “Transformers learn factored representations”

This study adapts the independent-factor experiment from
[arXiv:2602.02385](https://arxiv.org/abs/2602.02385) with only the requested
scientific changes:

1. correctness-reward PPO replaces next-token-only supervised training;
2. the environments contain either two or three independent MESS3 factors;
3. the residual width is 64.

The second objective arm adds next-joint-token cross entropy with coefficient
`1.0` to the same PPO loss. Each arm runs both factor counts:

The completed seed-42 results, Bayes baselines, checkpoint trajectories, CEV
dimension counts, and interpretation are in [`FINDINGS.md`](FINDINGS.md).

```bash
uv run rl-harness \
  experiments.factored_representations_reproduction_PPO_2026_08.ppo.experiment \
  --smoke --hardware-profile cpu

uv run rl-harness \
  experiments.factored_representations_reproduction_PPO_2026_08.ppo_aux_ce.experiment \
  --smoke --hardware-profile cpu
```

## Scientific recipe

- Every factor is Appendix C.1.1 MESS3 with `alpha=0.6`, `x=0.15`, and
  `y=1-2x=0.7`.
- Factor priors, transitions, and emissions compose as exact Kronecker
  products. The visible token is the mixed-radix integer for the Cartesian
  product of factor subtokens: 9 actions/tokens for two factors and 27 for
  three.
- `delay=1` means the action predicts the current hidden joint token; that
  token becomes visible in the next observation.
- Episodes provide a BOS decision followed by eight token positions, matching
  the paper's length-8 analysis and `n_ctx=9`.
- PPO uses the controlled immediate-prediction recipe from
  `mess3_token_guess_cycle_2`: `gamma=0`, `lambda=0`, clipping `0.2`, six
  epochs, and no entropy bonus. Both the learner train batch and minibatch are
  32,768. On the three-factor PPO+CE arm, an RTX 4090 sustained 5,420 sampled
  environment steps/s with 76.7% of CUDA memory reserved; 65,536 OOMed.
- The transformer preserves the paper's four pre-LN blocks, ReLU MLP,
  `d_mlp=4*d_model`, learned absolute positions, and final-LN placement.
  Because 3 heads do not divide `d_model=64`, this study preregisters four
  16-dimensional heads. That is a necessary extra architectural choice, not a
  result-dependent tuning decision.

The direct-sum and joint predictions are:

| Factors | Factored dimensions | Joint dimensions | Residual capacity |
|---:|---:|---:|---:|
| 2 | 4 | 8 | 64 |
| 3 | 6 | 26 | 64 |

Thus both alternatives fit in the network. A low-dimensional result cannot be
explained by an inability to hold the joint geometry.

## Probe battery

Every run saves the exact initialized Algorithm, power-of-two training
iterations, and the final Tune checkpoint. The same analysis is run at each
checkpoint on the final block's residual after the MLP and before final
LayerNorm (`blocks.3.hook_resid_post` in the paper):

1. **Factor decodability.** One joint affine regression maps activations to all
   concatenated factor predictive vectors. The SVD cutoff is selected by
   10-fold cross-validation over the paper's candidate values. Unlike the
   paper's headline in-sample score, primary RMSE and R² use an independent
   process seed.
2. **CEV and effective dimension.** Activation PCA is compared with empirical
   CEV curves for both the concatenated factor beliefs and their full joint
   product, as well as the algebraic 4/8 and 6/26 predictions. CEV alone is not
   treated as evidence of factor identity.
3. **Vary-one identification.** For each factor, its subtoken sequences vary
   across an ensemble while all other factor sequences are frozen. Activations
   are centered separately for every frozen configuration *and sequence
   position* before PCA, matching the authors' released analysis.
4. **Two orthogonality tests.** The battery reports effective-dimension
   additivity for the union and basis-invariant squared principal-angle
   overlap, including overlap curves over subspace rank. Additivity alone is
   not called proof of orthogonality.
5. **Projected geometry.** Natural-process activations are projected onto each
   factor's top two vary-one PCs, then independently evaluated for held-out
   recovery of that factor's belief simplex.
6. **Regression subspaces.** The columns of each factor's block in the joint
   regression matrix provide a second subspace identification method, with
   rank-two principal-angle overlap reported separately.
7. **Token embedding.** The joint-token embedding receives its own CEV,
   vary-one additivity/overlap, and per-PC factor attribution
   (between-subtoken variance divided by total PC-score variance).
8. **Longitudinal controls.** Initialization, power-of-two checkpoints, final
   geometry, task accuracy, and objective metrics remain separate. Multiple
   run seeds are required for trained-seed uncertainty and an initialization
   overlap interval.

These analyses establish decodability and geometry. They do **not** establish
that PPO causally uses a decoded factor belief; that would require a separate
intervention study.

The batch benchmark estimates approximately 1,855 seconds (30.9 minutes) for
10 million sampled environment steps, including one measured build/compile
iteration. This is a training-only estimate; longitudinal checkpoint probes,
artifact upload, provider variance, and startup/bootstrap are additional.

## Compact and large outputs

Compact recipes, checkpoint probe JSON, trajectory plots, and Tune summaries
are written below each arm's `results/<run-id>/<factor-count>_factors/`.
RLlib checkpoints and Tune trees stay under ignored `artifacts/`.
