# Limitations

- Affine belief decoding measures held-out linear accessibility. It does not show causal use and does not uniquely identify the network's internal representation.
- Initialization is replayed on the exact same sampled histories. High initialization scores weaken any claim that training created the geometry.
- Exact predictive controls can explain apparent belief decoding; probe surplus over them is descriptive rather than a causal mediation result.
- Fit and test contain 16 independent continuing environment trajectories each. Bootstrap intervals resample complete held-out trajectories with fixed probes; they do not measure training-seed variation.
- Checkpoints induce different stochastic-policy, process-weighted sampling distributions. Their scores are not evaluations on one common occupancy distribution.
- Task reward and probe fit are separate quantities. The myopic oracle is evaluated on visited beliefs and is not a certified Bayes-optimal POMDP bound.
- History shuffling deliberately breaks the original target alignment and changes the input distribution. It is a stress control, not a null required to yield zero R².
- Only the actual initialization and final two available checkpoints were probed; intermediate geometry is unmeasured.
- Variant 2 has an exact coarse quotient; the reported prediction-map null direction exposes one-step predictive indistinguishability of M1/M2.
