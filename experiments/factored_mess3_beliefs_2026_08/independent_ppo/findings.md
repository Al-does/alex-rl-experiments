# Independent two-MESS3 PPO finding

Seed 42 completed 2,524,972 sampled environment steps with a 64-dimensional,
two-layer transformer. Mean training return was 372.03 per 512-step episode
(72.7% joint-state guessing accuracy; chance is 11.1%).

The final residual linearly tracks both factor beliefs:

| Diagnostic | Initialization | Final |
|---|---:|---:|
| Factor 0 held-out R² | 0.847 | 0.971 |
| Factor 1 held-out R² | 0.844 | 0.971 |
| Joint-belief held-out R² | 0.784 | 0.951 |
| Factor-subspace overlap | 0.016 | 0.064 |
| Activation dimensions at 95% CEV | 13 | 8 |

The current-token-only baselines reached factor R² of 0.806, but their
within-token fine-MSE ratios were approximately 1.001. The trained transformer's
fine-MSE ratios were 0.147 and 0.149. Thus its improvement is not merely a
linear recoding of the current token: the residual carries history-dependent
Bayesian information about both factors.

The regression readouts identify two rank-two factor subspaces whose union has
rank four. Their normalized principal-angle overlap is 0.064, close to
orthogonal. This is strong evidence that the model makes the two beliefs
available through separate linear channels.

The global CEV result is mixed. The paper-inspired factored prediction is four
dimensions and the full joint-simplex prediction is eight; the trained
activation reaches 95% CEV at eight dimensions. It compressed substantially
from initialization but not to the direct-sum prediction. The overall PPO
residual therefore contains more variance than the factor belief readouts
alone, potentially for token, policy, value, or other task computations.

The defensible conclusion from this single PPO seed is: the transformer tracks
the two beliefs in nearly orthogonal factor-specific subspaces, but the
residual stream as a whole is not a pure four-dimensional factored
representation. This differs from the paper's next-token-only setup and should
not be described as a full replication of its CEV result.
