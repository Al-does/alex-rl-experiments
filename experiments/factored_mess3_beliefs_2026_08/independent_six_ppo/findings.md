# Independent six-MESS3 PPO finding

> Historical result: this run used the former delay-zero hidden-state-guess
> recipe. The current leaf uses delay-one joint-token prediction and the
> 120-dimensional paper transformer; new runs are not directly comparable.

Seed 42 completed 2,506,752 sampled environment steps with the same
64-dimensional, two-layer transformer used for the smaller conditions. Mean
training return was 5.375 per 512-step episode (1.05% joint-state guessing
accuracy; chance is 0.137%).

This condition does not support the factorization hypothesis:

| Diagnostic | Initialization | Final |
|---|---:|---:|
| Factor held-out R² range | -0.011–0.007 | 0.035–0.056 |
| Joint-belief held-out R² | -0.005 | 0.005 |
| Mean factor-subspace overlap | 0.073 | 0.056 |
| Activation dimensions at 95% CEV | 35 | 20 |

The current-token-only baselines reached factor R² of 0.787–0.797 and
joint-belief R² of 0.430. The trained residual therefore discarded most of the
belief information that was linearly available in the observation. Its
within-token factor fine-MSE ratios were 5.05–5.38, so it also failed to encode
useful history-dependent belief variation.

The low factor-subspace overlap is not evidence of factoring here. Principal
angles become scientifically meaningful only after the corresponding factor
beliefs are demonstrably decodable. The six rank-two fitted readout bases have
a twelve-dimensional union, but their held-out predictions are near the global
mean baseline.

The joint-versus-factor controls are likewise null:

- **PCJR:** the direct and product-constrained joint probes both have
  essentially zero held-out explanatory power (R² 0.0047 and 0.0030).
  The statistically resolved MSE difference is tiny and does not rescue either
  decoder.
- **CRD:** the exact correlation residual is degenerate, as required by the
  independent product-state generator. This validates the target construction,
  not the learned representation.
- **JRES:** excess direct-joint directions produce only a
  0.00000127 absolute MSE improvement over the factor-union-restricted probe.
  Both predictions remain near the mean-belief baseline, so the excess has no
  useful belief-decoding interpretation.

The CEV comparison is capacity-limited. A 64-dimensional residual cannot span
the 728-dimensional joint simplex, so observing 20 dimensions at 95% CEV does
not distinguish the twelve-dimensional direct-sum geometry from a compressed
joint representation. It only shows that training reduced whole-residual
variance dimension from 35 to 20.

The defensible single-seed conclusion is that sparse 729-way joint-state reward
did not train this model into a useful belief tracker at 2.5 million steps.
Unlike the two- and three-factor conditions, the six-factor run is a failed
representation-learning condition and should not be used as evidence for
native factoring. A follow-up would need a scientifically revised task or
learning signal, not merely another geometry interpretation of this checkpoint.
