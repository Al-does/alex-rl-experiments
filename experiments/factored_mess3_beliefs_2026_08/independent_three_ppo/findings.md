# Independent three-MESS3 PPO finding

Seed 42 completed 2,524,972 sampled environment steps with the same
64-dimensional, two-layer transformer used for the two-factor condition. Mean
training return was 320.97 per 512-step episode (62.7% joint-state guessing
accuracy; chance is 3.7%).

The final residual linearly tracks all three factor beliefs:

| Diagnostic | Initialization | Final |
|---|---:|---:|
| Factor 0 held-out R² | 0.786 | 0.955 |
| Factor 1 held-out R² | 0.799 | 0.958 |
| Factor 2 held-out R² | 0.801 | 0.953 |
| Joint-belief held-out R² | 0.653 | 0.870 |
| Mean factor-subspace overlap | 0.068 | 0.017 |
| Activation dimensions at 95% CEV | 31 | 24 |

The three factor readouts are rank two each. Their union has rank six, and
pairwise normalized principal-angle overlaps are 0.004, 0.031, and 0.015.
Thus the model exposes the three beliefs through distinct, nearly orthogonal
six-dimensional linear channels.

The named joint-versus-factor probes agree:

- **PCJR:** multiplying the three decoded marginals reconstructs the exact
  27-state joint belief substantially better than the much larger direct joint
  affine probe (R² 0.920 versus 0.870; MSE ratio 0.610). The paired
  episode-bootstrap MSE difference is -0.001011, with 95% CI
  [-0.001030, -0.000993] over 97 episode segments.
- **CRD:** the correlation residual is degenerate at numerical precision, as
  required for three independent HMMs.
- **JRES:** 78.7% of direct-joint weight energy lies outside the six-dimensional
  factor union, and these directions help a linear joint readout. This is
  expected when a linear probe approximates multiplicative tensor-product
  coordinates. PCJR shows that the lower-dimensional marginal channels retain
  more useful joint-belief information once the correct product operation is
  supplied.

The global CEV result is again not the paper's ideal direct-sum result. The
factored prediction is six dimensions and the full joint-simplex prediction is
26; the trained activation reaches 95% CEV at 24 dimensions. Scaling from two
to three factors therefore moves global CEV close to the joint prediction even
while the factor-specific probes become more orthogonal and the constrained
factored reconstruction decisively outperforms direct joint decoding.

The defensible single-seed conclusion is that global residual CEV is strongly
contaminated by other PPO computations or by features that linearize products.
For these models, factor readout rank, principal-angle overlap, and PCJR provide
more direct evidence about belief factorization than whole-residual CEV alone.
