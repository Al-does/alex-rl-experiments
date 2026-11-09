# History-dependent Reward Relevance

This new study uses two independent Full-State Quotient factors with the same
sticky baseline and emission accuracy 0.75 as
`belief_quotient_history_REINFORCE_2026_11`. Each factor is changed independently
before forming Kronecker-product controlled transitions; there is no shared
hold event coupling the factors. Joint action ordering and occupancy rewards
are unchanged from the historical Reward Relevance study.

The eight leaves are
`{reward_both,reward_factor_1}_{transformer,memoryless,gru128,gru92}`.
Use Joint Reward and Unrewarded Factor 2 in prose. The MLP observes only the
current joint-token one-hot; the transformer and GRUs also receive the previous
joint action. Architectures match the companion ladder except that the
transformer retains the source study's per-layer context 10, not 32.

Full runs target 30M steps with baseline-free REINFORCE, learning rate 4.2e-4,
gamma 0.99, and no entropy bonus. Seeds remain runtime `--seed` values; no
campaign or number of seeds is launched automatically. Fresh runs only.

Requires RL-Harness PR 87's model support (`9e4fc16` or a later descendant).
From the experiment repository:

```bash
uv run rl-harness experiments.reward_factors_history_REINFORCE_2026_11.reward_both_transformer.experiment --smoke --hardware cpu --no-upload-artifacts
uv run rl-harness experiments.reward_factors_history_REINFORCE_2026_11.reward_both_transformer.experiment --seed 42 --hardware cuda4090
```

Smoke targets 4,096 steps and performs real initialization and trained probes;
its 128-row splits can produce unstable fit values and are wiring checks only.
Full probes retain 20K independent fit/test rows, warmup 8, and the
pre-final-LayerNorm representation site. The analysis reads the restored
checkpoint's environment and validates it against this study, rather than
calling the old alpha-0.85 environment builder. It checks that exact posterior
marginals reconstruct the joint belief.

The new task adapter replaces the old task's hardcoded transition matrices.
`process.py` passes the alpha and baseline explicitly to both factor model
factories. The two reward conditions differ only in whether factor 2 contributes
to reward; factor 2 remains visible, controlled, and dynamic.

Initialization, powers-of-two iterations, 2M-step checkpoints, and the final
checkpoint are retained and analyzed. Compact reports stay in `results/`;
checkpoints/raw outputs stay ignored in `artifacts/`. Compare probe changes
within architecture, keeping reward and affine accessibility separate.

Single-factor memoryless references in the companion design do not establish
the optimum of a policy observing the joint token: such a reactive policy can
couple factor actions. Do not label their sum as a certified joint memoryless
ceiling. Historical studies and manuscript files are intentionally unchanged.

Local CPU verification passed all eight leaves at seed 42 using harness
`9e4fc16`. Each smoke produced initialization and trained reports at actual
steps 0, 2,970, and 5,018. The ignored `.smoke/verification.json` indexes the
individual smoke summaries. No full training runs or broader test suite were
executed.
