# Critic-aware Belief-Quotient Ladder audit

Frozen final checkpoints at 2,012,217,344 environment steps for REINFORCE, shared-trunk PPO, and automatic-alpha SAC (actor plus both Q critics). The three seeds, 42, 44, and 46, match the original critic-aware analysis. Every checkpoint's size and SHA-256 are checked against its canonical Backblaze B2 durability manifest. No new training is required for this audit.

## Reproduce

From the repository root with the JAX CPU group installed and B2 credentials available:

```sh
export audit_root=experiments/critic_aware_ladder_audit_2026_11
.venv/bin/python -m experiments.critic_aware_ladder_audit_2026_11.audit \
  --workers 6 --output "$audit_root/artifacts/matched"
.venv/bin/python -m experiments.critic_aware_ladder_audit_2026_11.audit \
  --hard-reference "$audit_root/results/pbvi_reference.zip" \
  --workers 6 --output "$audit_root/artifacts/hard"
.venv/bin/python -m experiments.critic_aware_ladder_audit_2026_11.audit \
  --soft --workers 4 --output "$audit_root/artifacts/soft"
.venv/bin/python -m experiments.critic_aware_ladder_audit_2026_11.history_support
.venv/bin/python -m experiments.critic_aware_ladder_audit_2026_11.summarize
.venv/bin/python -m pytest -q "$audit_root/test_audit.py"
```

Pass `--smoke` to an audit command for one Reward-State run with small sample sizes. Use clean output directories when rerunning. Reduce `--workers` on hosts with limited RAM, especially for the soft continuations.

## Matched histories

For each condition and training seed, each trained actor in turn generates complete 1,024-step greedy episodes. The RNG keys are the same by seed and split across all history donors. Fit and test use independent episodes.

Every trunk replays the same observations and recorded preceding actions within each donor, and the self-replay is checked against the collected embedding. The filter and models use token and action histories. The post-final-LayerNorm embedding is the feature for a fresh affine probe with ridge $10^{-6}$. The test rows and target variance are identical across compared trunks. Trunks may be evaluated on histories their own policies would seldom visit.

The original critic-aware protocol scores steps $[64,600)$ in 32 fit and 40 test episodes (17,152 and 21,440 rows). The archived algorithm-generality protocol scores steps $[64,1024)$ and takes the first 60,000 fit and 80,000 test rows. The two protocols differ in episode count as well as the timestep window. All scores report unconstrained $1-R^2$. A scalar score centers its test target at its test mean. A vector score sums squared errors and centered sums of squares over all dimensions.

The original 0.3029 SAC Reward-State actor contrast score appeared in a chat table. The saved report and recomputation both give 0.1380 under the interior protocol. Recomputing the archived protocol gives 0.1303. The [original probe code and per-run results](https://github.com/Al-does/alex-rl-experiments/commit/bd61e6536ad736be91105558c483dff1331516ba) and [archived tables](../algorithm_generality_probes_eps02_2026_11/results/tables.md) identify the source runs.

## Approximate optimal-value references

The interior-window replay also fits probes to $V^*$, $Q^*$, and $A^*=Q^*-V^*$ and to their residuals after fitting an affine function of belief on the same fit histories. Target construction and denominators are shared across trunks within each donor. The three original point-based value-iteration alpha-vector arrays are in the compact reference archive. They use a 30-division simplex grid, discount 0.99, and tolerance $10^{-6}$. Maximum grid Bellman residuals are below $10^{-6}$ for Trivial and approximately 0.018 and 0.022 in unscaled return units for Reward-State and Full-State. Residuals with held-out variance below $10^{-12}$ have undefined normalized errors.

## Policy-specific SAC targets

For each SAC checkpoint, use 128 independent stochastic on-policy episodes up to decision 512. From each complete prefix, branch with each first action, then follow the frozen SAC actor through the final training-episode step. Sample future tokens from the predictive belief and integrate out the latent state in each reward. The two independent groups of 24 continuations per action reuse uniforms across first actions within a group to reduce the variance of action contrasts. They give a diagnostic estimate of the Monte Carlo noise remaining in the target mean.

Policy probabilities use the training logit temperature 1.5. The entropy bonus uses the learned alpha read from the same critic checkpoint. The soft target for a forced action includes its current reward and entropy from subsequent choices. The soft state value also includes the current policy entropy. All direct head scores use the training scale of $1-\gamma$. The minimum Q head is $\min(Q_1,Q_2)$, applied per action before averaging over the current policy. The entropy-omission control uses the same stochastic policy and continuations with the entropy reward set to zero.

Scores include raw $R^2$, RMSE and bias, and a Monte Carlo noise-adjusted $R^2$. The noise-adjusted metric estimates the conditional target variance using the two independent halves. It is undefined when the estimated signal variance is nonpositive and can be unstable for policy-relative action differences with very small signal. The result is a descriptive comparison across three training seeds. A forced action can be rare under the archived policy.

The target-network diagnostic uses the archived training curves. Start the scalar target at zero to match the initialized Q heads. For each optimizer update, give the scalar online predictor the exact one-step Bellman target, $q=(1-\gamma)(\bar r+\gamma\alpha\bar H)+\gamma\bar q$, then apply the recorded Polyak update $\bar q\gets (1-\tau)\bar q+\tau q$. The logged rollout mean reward, policy entropy, and alpha are held fixed within each training iteration. The scalar approximation ignores episodic truncation, replay distributions, and state dependence. Compare its iteration means with the saved replay-policy Q means in scaled units.

## Results

The result directory contains per-seed belief, approximate optimal-value, soft head, and target-lag CSVs. The compact summaries average the three seeds equally. The manifest records every frozen actor and critic hash and the value references.
