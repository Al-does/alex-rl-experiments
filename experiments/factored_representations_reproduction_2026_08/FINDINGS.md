# Findings: PPO factored-representation reproduction

## Scope and result provenance

These are the completed seed-42 results for the four preregistered cells:

| Objective | Factors | Environment steps | Result directory |
|---|---:|---:|---|
| PPO | 2 | 5,010,167 | `ppo_2_factors/results/20260826T183227Z-076e412c/` |
| PPO | 3 | 5,010,167 | `ppo_3_factors/results/20260826T183438Z-031b1396/` |
| PPO + next-token CE | 2 | 5,010,167 | `ppo_aux_ce_2_factors/results/20260826T183704Z-dbc6c542/` |
| PPO + next-token CE | 3 | 5,010,167 | `ppo_aux_ce_3_factors/results/20260826T184035Z-087b978a/` |

All reported probe regressions use 20,000 fit and 20,000 held-out process
samples. The PPO and PPO+CE comparisons share the same seed and initialization
within each factor count. This makes the objective comparison controlled, but
one seed does not provide trained-seed uncertainty.

## Task-performance baselines

Random guessing has accuracy \(1/3^F\), where \(F\) is the number of factors.
The Bayes policy chooses the largest component of the environment's exact
predictive distribution. A 50,000-episode Monte Carlo evaluation of those
diagnostic beliefs, with the same randomized initial episode length and BOS
exclusion as the probes, gave the following process-weighted ceilings:

Reproduce the estimate with:

```bash
uv run python -m \
  experiments.factored_representations_reproduction_2026_08.estimate_bayes_accuracy
```

| Factors | Joint classes | Chance | Estimated Bayes optimal |
|---:|---:|---:|---:|
| 2 | 9 | 0.1111 | 0.2304 |
| 3 | 27 | 0.0370 | 0.1106 |

Final held-out greedy policy accuracy was:

| Objective | Factors | Init | Final | Final / Bayes | Fraction of chance-to-Bayes gap closed |
|---|---:|---:|---:|---:|---:|
| PPO | 2 | 0.1099 | 0.1506 | 65.4% | 33.1% |
| PPO + CE | 2 | 0.1099 | 0.1517 | 65.8% | 34.0% |
| PPO | 3 | 0.0376 | 0.0541 | 48.9% | 23.1% |
| PPO + CE | 3 | 0.0376 | 0.0519 | 46.9% | 20.1% |

The auxiliary loss therefore changed representation geometry without improving
the task policy at this budget. All four agents remained substantially below
Bayes-optimal prediction.

## How to read the representation metrics

- **Belief R²** is held-out affine decodability of all concatenated exact factor
  predictive vectors. Initialization is an essential control: a random
  64-dimensional residual already gives R² 0.844 for two factors and 0.614 for
  three factors, so final decodability alone is not evidence of learned
  factorization.
- **Activation PR** is the PCA participation ratio. **CEV d90/d95/d99** gives
  the number of principal components needed to explain 90%, 95%, and 99% of
  activation variance. The algebraic factored/joint dimensions are 4/8 for two
  factors and 6/26 for three factors.
- **CEV RMSE factored/joint** compares the full activation CEV curve to the
  empirical factored-belief and joint-belief target curves. Lower is closer.
- **Vary overlap** is mean squared principal-angle overlap between rank-two
  subspaces identified by independently varying each factor while freezing all
  others. Zero means orthogonal.
- **Projected R²** is mean held-out factor-belief R² after projecting natural
  activations into the corresponding rank-two vary-one subspace. It checks
  whether the identified axes carry the expected factor geometry.
- **Embedding overlap** applies the rank-two vary-one overlap test to the joint
  token embedding. Zero means the leading factor directions are orthogonal.

## Final comparison

| Objective | Factors | Accuracy | Belief R² | Activation PR | CEV d90/d95/d99 | CEV RMSE factored/joint | Vary overlap | Projected R² | Embedding overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO | 2 | 0.1506 | 0.907 | 7.43 | 7/8/16 | 0.072/0.060 | 0.029 | 0.627 | 0.438 |
| PPO + CE | 2 | 0.1517 | 0.932 | 7.59 | 7/8/10 | 0.075/0.063 | 0.022 | 0.733 | 0.479 |
| PPO | 3 | 0.0541 | 0.824 | 7.83 | 9/11/27 | 0.048/0.027 | 0.286 | 0.340 | 0.346 |
| PPO + CE | 3 | 0.0519 | 0.906 | 5.78 | 6/7/18 | 0.032/0.047 | 0.099 | 0.654 | 0.009 |

### Two-factor interpretation

Both objectives learned factor-separated rank-two directions. Vary-one overlap
fell from 0.420 at initialization to 0.029 with PPO and 0.022 with PPO+CE.
Projected belief recovery rose from mean R² 0.226 to 0.627 and 0.733,
respectively. The auxiliary loss modestly improved factor decodability and
projected geometry.

The complete residual distribution was not purely four-dimensional, however:
both final activations required eight components for 95% CEV and their CEV
curves remained slightly closer to the empirical joint target. Token embeddings
also retained substantial factor-subspace overlap (0.438 and 0.479). The
defensible conclusion is that both agents developed a factored core embedded in
a higher-dimensional mixed representation.

### Three-factor interpretation

This is the clearest objective-dependent result. PPO-only remained globally
joint-like: its final CEV curve was closer to the joint target
(0.027 versus 0.048), needed 11 dimensions for 95% CEV, and retained vary-one
overlap 0.286. Its mean rank-two projected belief R² was only 0.340.

PPO+CE crossed from joint-like at initialization (CEV RMSE 0.137 factored
versus 0.106 joint) to factored-like by the end (0.032 factored versus 0.047
joint). Its final participation ratio was 5.78, close to the predicted factored
dimension 6; six and seven PCs explained 90% and 95% of activation variance.
Its vary-one overlap fell to 0.099, mean projected belief R² reached 0.654, and
the three token-embedding factor subspaces became nearly orthogonal (mean
overlap 0.009).

The representation is still not a mathematically pure direct sum. At 95%
variance, the three vary-one clouds individually required 7, 6, and 8
dimensions and their union required 9, rather than exactly two dimensions per
factor and six in total. The result supports a strong factored core plus
remaining context/interaction variance, not exclusively factored coding.

### What decodability does and does not show

Final factor decodability was high in all cells:

| Objective | Factors | Init R² | Final R² | Final per-factor R² |
|---|---:|---:|---:|---|
| PPO | 2 | 0.844 | 0.907 | 0.910, 0.905 |
| PPO + CE | 2 | 0.844 | 0.932 | 0.932, 0.932 |
| PPO | 3 | 0.614 | 0.824 | 0.821, 0.851, 0.802 |
| PPO + CE | 3 | 0.614 | 0.906 | 0.903, 0.911, 0.904 |

Because initialization already decodes well, the stronger evidence comes from
the longitudinal dimensional collapse, controlled vary-one separation,
projected belief recovery, and the three-factor CEV crossover. These probes
establish representation geometry and decodability; they do not establish that
the PPO action head causally uses the decoded factor beliefs.

## Longitudinal checkpoint results

### PPO, two factors

| Steps | Accuracy | Belief R² | Activation PR | CEV d90/d95/d99 | CEV RMSE factored/joint | Vary overlap | Projected R² | Embedding overlap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1099 | 0.844 | 12.39 | 17/25/44 | 0.141/0.132 | 0.420 | 0.226 | 0.796 |
| 32,959 | 0.1102 | 0.852 | 11.88 | 16/23/42 | 0.134/0.125 | 0.499 | 0.232 | 0.889 |
| 65,927 | 0.1096 | 0.855 | 11.61 | 15/22/41 | 0.131/0.121 | 0.520 | 0.232 | 0.922 |
| 131,852 | 0.1128 | 0.860 | 11.42 | 14/20/40 | 0.127/0.117 | 0.534 | 0.236 | 0.933 |
| 263,682 | 0.1182 | 0.866 | 10.84 | 14/19/39 | 0.120/0.110 | 0.450 | 0.263 | 0.921 |
| 527,378 | 0.1349 | 0.891 | 6.37 | 12/17/37 | 0.083/0.075 | 0.373 | 0.534 | 0.846 |
| 1,054,775 | 0.1455 | 0.898 | 5.80 | 7/13/33 | 0.048/0.040 | 0.066 | 0.716 | 0.471 |
| 2,109,548 | 0.1475 | 0.910 | 7.06 | 7/8/25 | 0.063/0.051 | 0.060 | 0.651 | 0.516 |
| 4,219,074 | 0.1503 | 0.910 | 7.17 | 7/8/18 | 0.067/0.055 | 0.037 | 0.621 | 0.493 |
| 5,010,167 | 0.1505 | 0.907 | 7.43 | 7/8/16 | 0.072/0.060 | 0.029 | 0.627 | 0.438 |

### PPO + CE, two factors

| Steps | Accuracy | Belief R² | Activation PR | CEV d90/d95/d99 | CEV RMSE factored/joint | Vary overlap | Projected R² | Embedding overlap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1099 | 0.844 | 12.39 | 17/25/44 | 0.141/0.132 | 0.420 | 0.226 | 0.796 |
| 32,959 | 0.1097 | 0.855 | 11.88 | 16/23/42 | 0.136/0.127 | 0.583 | 0.178 | 0.846 |
| 65,927 | 0.1109 | 0.865 | 11.72 | 15/21/41 | 0.133/0.123 | 0.649 | 0.162 | 0.878 |
| 131,852 | 0.1105 | 0.876 | 11.57 | 15/20/39 | 0.130/0.120 | 0.634 | 0.166 | 0.901 |
| 263,682 | 0.1100 | 0.886 | 11.45 | 14/19/37 | 0.127/0.117 | 0.500 | 0.280 | 0.722 |
| 527,378 | 0.1293 | 0.901 | 6.64 | 12/17/35 | 0.089/0.080 | 0.307 | 0.551 | 0.476 |
| 1,054,775 | 0.1416 | 0.905 | 4.30 | 6/11/27 | 0.039/0.036 | 0.322 | 0.619 | 0.436 |
| 2,109,548 | 0.1509 | 0.918 | 6.60 | 7/8/18 | 0.052/0.040 | 0.016 | 0.754 | 0.432 |
| 4,219,074 | 0.1517 | 0.929 | 7.34 | 7/8/12 | 0.068/0.057 | 0.011 | 0.748 | 0.455 |
| 5,010,167 | 0.1517 | 0.932 | 7.59 | 7/8/10 | 0.075/0.063 | 0.022 | 0.733 | 0.479 |

### PPO, three factors

| Steps | Accuracy | Belief R² | Activation PR | CEV d90/d95/d99 | CEV RMSE factored/joint | Vary overlap | Projected R² | Embedding overlap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0376 | 0.614 | 14.48 | 23/32/51 | 0.137/0.106 | 0.523 | 0.017 | 0.583 |
| 32,959 | 0.0389 | 0.626 | 13.50 | 21/30/50 | 0.127/0.095 | 0.525 | 0.028 | 0.586 |
| 65,927 | 0.0389 | 0.638 | 12.83 | 20/29/49 | 0.120/0.088 | 0.521 | 0.026 | 0.587 |
| 131,852 | 0.0398 | 0.649 | 12.31 | 20/28/48 | 0.115/0.083 | 0.500 | 0.030 | 0.583 |
| 263,682 | 0.0414 | 0.666 | 12.33 | 19/28/48 | 0.114/0.083 | 0.404 | 0.036 | 0.579 |
| 527,378 | 0.0377 | 0.684 | 12.42 | 19/27/47 | 0.112/0.080 | 0.447 | 0.054 | 0.577 |
| 1,054,775 | 0.0483 | 0.693 | 7.21 | 15/24/45 | 0.079/0.055 | 0.606 | 0.174 | 0.523 |
| 2,109,548 | 0.0534 | 0.782 | 6.18 | 8/15/37 | 0.046/0.042 | 0.376 | 0.421 | 0.443 |
| 4,219,074 | 0.0547 | 0.812 | 7.47 | 9/11/30 | 0.046/0.029 | 0.363 | 0.357 | 0.359 |
| 5,010,167 | 0.0541 | 0.824 | 7.83 | 9/11/27 | 0.048/0.027 | 0.286 | 0.340 | 0.346 |

### PPO + CE, three factors

| Steps | Accuracy | Belief R² | Activation PR | CEV d90/d95/d99 | CEV RMSE factored/joint | Vary overlap | Projected R² | Embedding overlap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0376 | 0.614 | 14.48 | 23/32/51 | 0.137/0.106 | 0.523 | 0.017 | 0.583 |
| 32,959 | 0.0372 | 0.632 | 14.03 | 21/30/50 | 0.130/0.098 | 0.365 | 0.036 | 0.584 |
| 65,927 | 0.0360 | 0.652 | 13.76 | 21/29/49 | 0.127/0.095 | 0.351 | 0.042 | 0.582 |
| 131,852 | 0.0380 | 0.679 | 13.33 | 20/28/48 | 0.124/0.092 | 0.422 | 0.042 | 0.580 |
| 263,682 | 0.0382 | 0.710 | 13.25 | 20/28/47 | 0.124/0.092 | 0.428 | 0.051 | 0.572 |
| 527,378 | 0.0362 | 0.739 | 13.18 | 20/28/47 | 0.124/0.091 | 0.434 | 0.098 | 0.537 |
| 1,054,775 | 0.0363 | 0.798 | 11.51 | 18/26/45 | 0.108/0.077 | 0.373 | 0.286 | 0.427 |
| 2,109,548 | 0.0440 | 0.874 | 4.08 | 9/15/34 | 0.069/0.071 | 0.381 | 0.434 | 0.245 |
| 4,219,074 | 0.0513 | 0.900 | 5.08 | 6/7/20 | 0.046/0.061 | 0.162 | 0.598 | 0.017 |
| 5,010,167 | 0.0519 | 0.906 | 5.78 | 6/7/18 | 0.032/0.047 | 0.099 | 0.654 | 0.009 |

## Bottom line

At two factors, both PPO objectives learned identifiable, nearly orthogonal
factor directions, but the full residual and token embedding remained mixed.
At three factors, PPO-only retained joint-like global geometry, whereas adding
next-token CE produced a low-dimensional, factor-aligned core across CEV,
vary-one, projected-belief, and token-embedding diagnostics. The auxiliary
objective therefore promoted factorization without improving policy accuracy.

This is a geometric and decodability result from one controlled seed, not yet a
seed-robust result or evidence that the policy causally relies on the factored
variables.
