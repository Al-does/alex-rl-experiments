# Action-fixed replay and belief edits

**Scope.** Three paired training seeds (42–44), the final ε=0.02, temperature-1.5 REINFORCE checkpoints of the Reward-State Quotient and Full-State Quotient, and the actual step-zero checkpoints for Part A. Full-State is a positive control for Parts A–C; Part D asks what happens when the Reward-State policy is edited. No agents were retrained. No existing experiment code or tracked results were changed.

**Overlap verdict.** This is materially different from the twin-write study. That study replayed complete episodes with actions fixed, used a global A/B swap and 64-step emission-regime resampling, and refiltered both histories. It then fit a target-conditioned affine activation displacement in both directions and intervened at the current block-4 residual every step. Its held-out R², readout specificity, and goal support were not fidelity to the replay policy. We reuse the action-fixed filtering convention, paper environment, pinned historical rollout, sites, regression cutoff and Eq. (4) controls. We reran the overlapping A/B swap on the new evaluation splits because the prior results did not save a policy-invariance profile or initialization results. No prior displacement fit or twin result substitutes for a result here.

**Reproduce** from the configured JAX environment:

```sh
python -m experiments.belief_quotient_replay_edit_2026_11.run --smoke --seeds 42 43
python -m experiments.belief_quotient_replay_edit_2026_11.run
python -m experiments.belief_quotient_replay_edit_2026_11.summarize
```

## Protocol

- **A:** Compare the whole replay with the original. Both filters process full prefixes with previous executed actions. Full-State swaps past Tilt X and Tilt Y only in the positive-control swap. Its C/not-C posterior is the marginal of a three-state merged-observation filter, *not* a quotient. The probability metric is total variation (TV) under the checkpoint temperature 1.5; action changes use the greedy policy. Averages discard eight reset steps, as in Appendix D.1; full per-position profiles before exclusion are also saved. The relabel regression fits on independent episodes. Exact invariance of the Reward-State quotient posterior and the two symmetry relations on full posteriors are asserted numerically.
- **B:** Inject the replay residual into the original at one site. Query at every 256th step, with either one token changed at the listed lag or two tokens changed, one at that lag and one at the query. The three masks install at the query only, all edited-token positions, or edited-token positions strictly earlier than the query. Subsequent layers are recomputed. The historical hook installs at the latest position *at every decision*. Earlier writes survive only in downstream-layer caches; the block-4 and final-norm writes have no downstream attention and do not leave cached writes. Thus earlier-site every-step writes are not a one-off latest-only patch, and are not restricted to positions with changed input tokens either.
- **C:** Learn displacement from known belief-feature differences (with a separate linear two-column current-token contrast); test on independent episodes after selecting on a third episode split. Models optionally use log-odds, the quadratic products of the transformed features, or interactions with the original activation's top four training PCs. All maps use the paper's least-squares cutoff, `rcond=1e-6`. The fixed-token score asks what remains when the token contrast is zero, not whether the estimated directions are causal.
- **D:** Every step, act greedily from the steered pass and feed that action back to the environment, real-history filters, and next previous-action input. Fit Eq. (4) using the historical 12,000 retained rows, use seed 1042 with 16 episodes for strength selection and seed 2042 with 32 × 1,024-step episodes for confirmation, and give λ=1 results in the seed reports. Eq. (4) uses the archived block-4 strengths; the new methods apply the same selection rule on the same grid. Mirror is fixed at λ=1. Eq. (4) is rerun on CPU for a same-run comparator, not taken verbatim from the GPU archive. The exact encoder uses filtered full and quotient beliefs and the decoded encoder uses simplex-normalized clipped full estimates and a separate clipped quotient probe. A ρ target specifies the same desired state-3 probability in the winning feature coordinates; a δ target preserves the quotient coordinate. The fit's token contribution is zeroed when writing a belief.
- **D replay:** For each visited decision history, greedily minimize target-posterior error by working backwards over the last eight tokens; try retaining each token or toggling C status (both A and B for an original C). The bounded greedy search is not globally optimal and cannot in general reach point endpoints or mirror δ with C-only edits. It holds previous-action inputs fixed and starts from the unedited prefix posterior and transformer cache. The reference write installs the resulting activation at the latest position each step, just like the other write hooks. The full replay policy can still differ from a single-site write, particularly at an early site. Fidelity compares a write to that full replay policy on the same observed history at each decision, not to a globally achievable target. Closed-loop methods can visit different histories. The archived 256-episode exact-filter grid controller is a numerical performance reference, not the replay patch or a Bellman bound.
- **Fidelity and support:** `1 − Σ TV(write,replay) / Σ TV(original,replay)` on a common set of decisions. Zero means the replay error did not improve, negative values are worse, and a no-effect denominator is undefined. At very small replay TV, a numerically large fidelity need not be meaningful. Likelihood drop is the mean log-probability loss in nats per recorded executed action over the replay interval (Part A: the episode after warmup; Part B: earliest edit to query; Part D: recorded actions in the edited suffix before the current decision). A large drop is at least 0.5 nat/action; seed reports include the counts and outcomes or fidelity stratified by this cutoff. Fit, model selection and held-out trajectories use evaluation seeds 42, 43 and 44 respectively, distinct from training seeds and from closed-loop tuning and confirmation.

## Results

Cells report mean ± sample SD across the three training seeds unless otherwise labeled. In A the single-flip scale averages across the whole episode and is diluted by unedited positions; B measures the local effect at a known lag. Part B does not display fidelity when the replay TV is smaller than `1e-6`. In C, the three signs in each model label indicate log-odds, quadratic terms and PC interactions, in order; PC ΔR² toggles only the PC term for the selected feature family. See `results/v{2,3}_seed{42,43,44}.json` for all five sites, all lag/patch combinations, the per-position and likelihood-stratified measurements, all encoder fits, and decision budgets and checkpoint hashes.

### A. Invariance

| Condition | Checkpoint | Replay | Policy TV | Greedy changed | Block 4 rel. disp. | Norm rel. disp. | Full ρ MAE | Merged C posterior MAE | Drop (nat/step) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reward-State | init | swap | 0.0160 ± 0.0024 | 0.0766 ± 0.0599 | 0.1171 ± 0.0279 | 0.1172 ± 0.0278 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0096 ± 0.0074 |
| Reward-State | init | relabel | 0.0092 ± 0.0008 | 0.0434 ± 0.0339 | 0.0655 ± 0.0151 | 0.0655 ± 0.0152 | 0.0026 ± 0.0019 | 0.0000 ± 0.0000 | 0.0045 ± 0.0038 |
| Reward-State | init | flip | 0.0001 ± 0.0000 | 0.0004 ± 0.0004 | 0.0014 ± 0.0003 | 0.0014 ± 0.0003 | 0.0003 ± 0.0002 | 0.0003 ± 0.0002 | -0.0000 ± 0.0000 |
| Reward-State | final | swap | 0.0265 ± 0.0070 | 0.0226 ± 0.0122 | 0.1158 ± 0.0212 | 0.1052 ± 0.0194 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0304 ± 0.0168 |
| Reward-State | final | relabel | 0.0150 ± 0.0043 | 0.0131 ± 0.0071 | 0.0666 ± 0.0127 | 0.0604 ± 0.0116 | 0.0102 ± 0.0005 | 0.0000 ± 0.0000 | 0.0172 ± 0.0100 |
| Reward-State | final | flip | 0.0026 ± 0.0003 | 0.0026 ± 0.0003 | 0.0066 ± 0.0004 | 0.0058 ± 0.0005 | 0.0023 ± 0.0002 | 0.0024 ± 0.0003 | 0.0099 ± 0.0001 |
| Full-State | init | swap | 0.0831 ± 0.0431 | 0.5768 ± 0.3083 | 0.7083 ± 0.1923 | 0.7068 ± 0.1944 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1701 ± 0.0687 |
| Full-State | init | relabel | 0.0090 ± 0.0010 | 0.0431 ± 0.0300 | 0.0640 ± 0.0164 | 0.0640 ± 0.0164 | 0.0321 ± 0.0213 | 0.0000 ± 0.0000 | 0.0023 ± 0.0065 |
| Full-State | init | flip | 0.0001 ± 0.0000 | 0.0004 ± 0.0004 | 0.0014 ± 0.0003 | 0.0014 ± 0.0003 | 0.0004 ± 0.0001 | 0.0004 ± 0.0001 | -0.0000 ± 0.0000 |
| Full-State | final | swap | 0.3121 ± 0.0321 | 0.3295 ± 0.0352 | 0.7752 ± 0.0494 | 0.7611 ± 0.0448 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 1.6820 ± 0.1865 |
| Full-State | final | relabel | 0.1379 ± 0.0104 | 0.1434 ± 0.0111 | 0.3905 ± 0.0132 | 0.3890 ± 0.0111 | 0.0552 ± 0.0026 | 0.0000 ± 0.0000 | 0.4122 ± 0.0153 |
| Full-State | final | flip | 0.0029 ± 0.0002 | 0.0029 ± 0.0003 | 0.0077 ± 0.0003 | 0.0073 ± 0.0003 | 0.0022 ± 0.0002 | 0.0018 ± 0.0002 | 0.0088 ± 0.0006 |

- Reward-State relabel Δρ→Δlogit held-out R²: -0.002 ± 0.001.
- Reward-State block-4 probes, R² (ρ, signed δ, |δ|, merged C posterior): 0.974 ± 0.008; 0.737 ± 0.063; 0.692 ± 0.024; 0.976 ± 0.007.
- Full-State relabel Δρ→Δlogit held-out R²: -0.006 ± 0.001.
- Full-State block-4 probes, R² (ρ, signed δ, |δ|, merged C posterior): 0.979 ± 0.002; 0.964 ± 0.006; 0.922 ± 0.007; 0.966 ± 0.002.
- Full-State init swap TV after aligning Tilt X and Tilt Y: 0.0631 ± 0.0481.
- Full-State final swap TV after aligning Tilt X and Tilt Y: 0.0321 ± 0.0022.

### B. Positional patching

| Condition | Lag(s) | Replay TV | Block 1 latest | Block 1 changed | Block 1 past | Block 4 latest | Block 4 changed | Block 4 past |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reward-State | 0 | 0.596498 ± 0.060750 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| Reward-State | 1 | 0.397592 ± 0.171594 | 0.452 ± 0.458 | 0.309 ± 0.413 | 0.309 ± 0.413 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Reward-State | 2 | 0.396953 ± 0.179556 | 0.466 ± 0.440 | 0.331 ± 0.426 | 0.331 ± 0.426 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Reward-State | 4 | 0.375681 ± 0.118416 | 0.425 ± 0.424 | 0.366 ± 0.433 | 0.366 ± 0.433 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Reward-State | 8 | 0.095279 ± 0.035340 | 0.368 ± 0.426 | 0.367 ± 0.295 | 0.367 ± 0.295 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Reward-State | 16 | 0.011452 ± 0.008493 | 0.155 ± 0.219 | 0.354 ± 0.134 | 0.354 ± 0.134 | 1.000 ± 0.000 | -0.000 ± 0.000 | -0.000 ± 0.000 |
| Reward-State | 32 | 0.008009 ± 0.003122 | 0.205 ± 0.268 | 0.299 ± 0.421 | 0.299 ± 0.421 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Reward-State | 64 | 0.000081 ± 0.000035 | -0.000 ± 0.000 | 0.186 ± 0.205 | 0.186 ± 0.205 | 1.000 ± 0.000 | -0.000 ± 0.000 | -0.000 ± 0.000 |
| Reward-State | 128 | 0.000000 ± 0.000000 | — | — | — | — | — | — |
| Reward-State | 8, 0 | 0.637421 ± 0.022903 | 0.926 ± 0.127 | 0.980 ± 0.032 | 0.035 ± 0.034 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| Reward-State | 32, 0 | 0.593469 ± 0.054874 | 0.970 ± 0.043 | 0.979 ± 0.035 | 0.004 ± 0.004 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| Full-State | 0 | 0.275896 ± 0.034582 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| Full-State | 1 | 0.177004 ± 0.042915 | 0.712 ± 0.382 | 0.233 ± 0.354 | 0.233 ± 0.354 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Full-State | 2 | 0.191897 ± 0.048038 | 0.709 ± 0.384 | 0.228 ± 0.351 | 0.228 ± 0.351 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Full-State | 4 | 0.161021 ± 0.032720 | 0.687 ± 0.373 | 0.235 ± 0.355 | 0.235 ± 0.355 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Full-State | 8 | 0.042791 ± 0.019210 | 0.351 ± 0.116 | 0.390 ± 0.135 | 0.390 ± 0.135 | 1.000 ± 0.000 | -0.000 ± 0.000 | -0.000 ± 0.000 |
| Full-State | 16 | 0.022721 ± 0.002764 | 0.126 ± 0.119 | 0.542 ± 0.125 | 0.542 ± 0.125 | 1.000 ± 0.000 | -0.000 ± 0.000 | -0.000 ± 0.000 |
| Full-State | 32 | 0.008695 ± 0.001275 | 0.169 ± 0.047 | 0.264 ± 0.024 | 0.264 ± 0.024 | 1.000 ± 0.000 | -0.000 ± 0.000 | -0.000 ± 0.000 |
| Full-State | 64 | 0.000285 ± 0.000118 | 0.000 ± 0.000 | 0.185 ± 0.126 | 0.185 ± 0.126 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Full-State | 128 | 0.000000 ± 0.000000 | — | — | — | — | — | — |
| Full-State | 8, 0 | 0.468397 ± 0.108660 | 0.620 ± 0.067 | 0.862 ± 0.110 | 0.017 ± 0.024 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| Full-State | 32, 0 | 0.324808 ± 0.049775 | 0.844 ± 0.079 | 0.895 ± 0.056 | -0.002 ± 0.005 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |

### C. Encoding displacement

| Condition | Site | Selection winners (seeds 42, 43, 44) | Held-out R² | Held-out R², fixed token | Best c family | Best ρ family | Best ρ+δ family | Best ρ+|δ| family | PC ΔR² | High-drop pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Reward-State | 1 | rho_delta+++, rho_delta+++, c+++ | 0.838 ± 0.015 | 0.602 ± 0.110 | 0.828 ± 0.022 | 0.822 ± 0.022 | 0.838 ± 0.016 | 0.827 ± 0.021 | 0.074 ± 0.004 | 0.301 ± 0.049 |
| Reward-State | 2 | rho_delta+++, rho_delta+++, rho_delta−++ | 0.814 ± 0.026 | 0.671 ± 0.031 | 0.810 ± 0.030 | 0.801 ± 0.033 | 0.814 ± 0.026 | 0.808 ± 0.031 | 0.108 ± 0.024 | 0.301 ± 0.049 |
| Reward-State | 3 | rho_delta−++, rho_delta−++, c−++ | 0.820 ± 0.012 | 0.710 ± 0.023 | 0.817 ± 0.015 | 0.809 ± 0.014 | 0.818 ± 0.011 | 0.815 ± 0.014 | 0.087 ± 0.005 | 0.301 ± 0.049 |
| Reward-State | 4 | rho_delta+++, rho_delta−++, rho_delta−++ | 0.805 ± 0.012 | 0.700 ± 0.018 | 0.805 ± 0.015 | 0.793 ± 0.018 | 0.805 ± 0.012 | 0.800 ± 0.015 | 0.112 ± 0.044 | 0.301 ± 0.049 |
| Reward-State | Norm | rho_delta−++, rho_delta−++, rho_delta−++ | 0.808 ± 0.021 | 0.702 ± 0.027 | 0.804 ± 0.022 | 0.795 ± 0.024 | 0.808 ± 0.021 | 0.801 ± 0.023 | 0.081 ± 0.008 | 0.301 ± 0.049 |
| Full-State | 1 | rho_delta+++, rho_delta+++, rho_delta+++ | 0.809 ± 0.087 | 0.516 ± 0.032 | 0.763 ± 0.080 | 0.744 ± 0.094 | 0.809 ± 0.087 | 0.763 ± 0.096 | 0.047 ± 0.012 | 0.277 ± 0.037 |
| Full-State | 2 | rho_delta+++, rho_delta+++, rho_delta+++ | 0.781 ± 0.042 | 0.593 ± 0.033 | 0.676 ± 0.049 | 0.655 ± 0.041 | 0.781 ± 0.042 | 0.690 ± 0.045 | 0.077 ± 0.007 | 0.277 ± 0.037 |
| Full-State | 3 | rho_delta+++, rho_delta+++, rho_delta+++ | 0.781 ± 0.026 | 0.649 ± 0.020 | 0.608 ± 0.037 | 0.606 ± 0.032 | 0.781 ± 0.026 | 0.654 ± 0.033 | 0.091 ± 0.024 | 0.277 ± 0.037 |
| Full-State | 4 | rho_delta−++, rho_delta+++, rho_delta+++ | 0.786 ± 0.026 | 0.685 ± 0.023 | 0.548 ± 0.032 | 0.566 ± 0.030 | 0.786 ± 0.026 | 0.634 ± 0.028 | 0.072 ± 0.026 | 0.277 ± 0.037 |
| Full-State | Norm | rho_delta−++, rho_delta+++, rho_delta−++ | 0.742 ± 0.034 | 0.634 ± 0.037 | 0.484 ± 0.039 | 0.504 ± 0.045 | 0.742 ± 0.034 | 0.576 ± 0.042 | 0.082 ± 0.037 | 0.277 ± 0.037 |

### D. Closed-loop Reward-State writes

| Site | Edit | Write | λ | Occupancy | Actions changed | Fidelity | Fidelity (high drop) | Replay ρ MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Block 4 | rho_flip | eq4 | 0.933 ± 0.058 | 0.095 ± 0.018 | 0.483 ± 0.137 | 0.825 ± 0.096 | 0.419 ± 0.260 | 0.741 ± 0.019 |
| Block 4 | rho_flip | encoder_exact | 0.933 ± 0.058 | 0.103 ± 0.004 | 0.453 ± 0.157 | 0.875 ± 0.059 | 0.389 ± 0.430 | 0.723 ± 0.021 |
| Block 4 | rho_flip | encoder_decoded | 0.967 ± 0.058 | 0.108 ± 0.024 | 0.481 ± 0.116 | 0.803 ± 0.100 | 0.499 ± 0.305 | 0.711 ± 0.039 |
| Block 4 | rho_flip | reference | 0.750 ± 0.150 | 0.094 ± 0.005 | 0.432 ± 0.176 | 0.929 ± 0.049 | 0.687 ± 0.182 | 0.742 ± 0.006 |
| Block 4 | rho_point | eq4 | 0.750 ± 0.250 | 0.085 ± 0.011 | 0.623 ± 0.085 | 0.771 ± 0.050 | 0.473 ± 0.090 | 0.807 ± 0.018 |
| Block 4 | rho_point | encoder_exact | 0.783 ± 0.202 | 0.087 ± 0.010 | 0.611 ± 0.067 | 0.831 ± 0.090 | 0.527 ± 0.340 | 0.788 ± 0.008 |
| Block 4 | rho_point | encoder_decoded | 0.683 ± 0.257 | 0.099 ± 0.026 | 0.785 ± 0.113 | 0.616 ± 0.205 | 0.216 ± 0.703 | 0.779 ± 0.044 |
| Block 4 | rho_point | reference | 0.650 ± 0.087 | 0.076 ± 0.003 | 0.468 ± 0.163 | 0.938 ± 0.048 | 0.815 ± 0.106 | 0.818 ± 0.006 |
| Block 4 | mirror | eq4 | 1.000 ± 0.000 | 0.425 ± 0.021 | 0.006 ± 0.003 | -0.013 ± 0.010 | -0.001 ± 0.010 | 0.050 ± 0.004 |
| Block 4 | mirror | encoder_exact | 1.000 ± 0.000 | 0.432 ± 0.019 | 0.011 ± 0.007 | -0.040 ± 0.025 | -0.008 ± 0.017 | 0.049 ± 0.004 |
| Block 4 | mirror | encoder_decoded | 1.000 ± 0.000 | 0.427 ± 0.023 | 0.017 ± 0.018 | -0.103 ± 0.131 | -0.029 ± 0.057 | 0.050 ± 0.004 |
| Block 4 | mirror | reference | 1.000 ± 0.000 | 0.289 ± 0.011 | 0.183 ± 0.010 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.086 ± 0.002 |
| — | intact | Exact-filter grid policy | — | 0.436 | 0.000 | — | — | — |
| — | random | Exact-filter grid policy | — | 0.105 | 0.666 | — | — | — |
| — | rho_flip | Exact-filter grid policy | — | 0.124 | 0.789 | — | — | — |
| — | rho_point | Exact-filter grid policy | — | 0.077 | 0.872 | — | — | — |
| — | mirror | Exact-filter grid policy | — | 0.436 | 0.000 | — | — | — |

- Network intact occupancy: 0.425 ± 0.019.
- Network random occupancy: 0.106 ± 0.000.
- B-selected sites (42–44): Block 4, Block 4, Block 4.


Exit code: 0

## What held up and what did not

**A — approximate quotient invariance, not exact.** Relabeling changed full ρ by 0.0102 on average but left the quotient filter exactly unchanged and did not produce a predictive linear Δρ → Δlogits relation. Relabeling changed only 1.31% of actions, versus 14.34% in the Full-State control. Nevertheless, both Reward-State replays moved the policy and late activations. Even the global swap, which preserves both ρ and c, changed 2.26% of greedy actions. The strict invariance and zero-displacement predictions failed, and the data do not identify a literal quotient-filter implementation. The Full-State swap mirrored the policy approximately: TV fell from 0.312 unaligned to 0.032 after aligning the tilts, while late displacement remained large. None of the Reward-State windows exceeded the likelihood-drop threshold; all 42 Full-State swap windows and 9 of 42 Full-State relabel windows exceeded it.

**B — a latest block-4 patch suffices, not because history is absent.** Where the replay affected policy, installing its latest block-4 residual reproduced its effect to numerical precision; this is expected because only the final normalization and head remain. Before block 4 the past mattered: for Reward-State lag 16, the block-1 past patch beat that site's latest patch (0.354 versus 0.155). However, at lag 16 the unpatched behavioral effect was already small, and at lag 128 it was below the reporting threshold. Blocks 4 and final norm gave equivalent latest-position reference fidelity; the selector chose block 4 in all seeds. This does not locate a unique belief state.

**C — state-dependent fits help, but the best Reward-State families are almost tied.** Signed-δ models won block-4 selection, but their held-out score (0.805) was essentially tied with the best quotient-only belief features plus the same token/PC controls (0.805). Full-State had a clear signed-δ advantage (0.786 versus 0.548). The interaction terms improved the selected block-4 families by 0.112 R² in Reward-State and 0.072 in Full-State. Fixed-token scores were lower. In high-drop pairs, the corresponding selected fits scored 0.757 ± 0.021 and 0.786 ± 0.036; report these separately rather than treating these pairs as on-policy. Predicting displacement is not evidence that the agent uses all of the fitted features for control.

**D — the exact-belief encoder improves replay fidelity, not occupancy.** At selected strengths, exact encoder fidelity exceeded Eq. (4) for both ρ flip (0.875 vs 0.825) and ρ point (0.831 vs 0.771), with slightly higher occupancy (less suppression). At λ=1 the fidelity advantage remained (0.870 vs 0.820 and 0.811 vs 0.701). The decoded encoder lost that advantage (selected 0.803 and 0.616), and the high-drop comparisons were variable. The reference patch achieved 1.0 fidelity at λ=1, by construction at block 4. Its selected-strength fidelity was higher than either fitted method, but the constructed histories were **poor posterior matches**: mean ρ errors of 0.742 for flip and 0.818 for point on the reference rollout. We therefore improved reproduction of a specific attainable replay-policy change, not fidelity to a correctly installed target posterior. Low recorded-action drop does not solve that target mismatch. The negative control left Eq. (4) and learned-write occupancy close to intact, though neither method left every action unchanged. The C-toggle reference failed the mirror negative control behaviorally (18.3% of actions changed); its residual match is not evidence of a faithful δ mirror.

## Deviations and limits

- Used seeds 42–44 per the revised scope, not all 15. There was no initialization analysis beyond A, no Full-State control steering in D, and no XOR rerun or training.
- The exact-filter grid policy is reused verbatim from the historical report. CPU reruns exactly matched archived intact and random occupancy; the Eq. (4) steering rerun differed from the archive by at most 0.0032 in occupancy and 0.0046 in action-change fraction. A separate seed-42 diagnostic matched the pinned historical rollout exactly on CPU for all three writes, so bitwise parity with the GPU archive was not an appropriate acceptance gate. Both archive and CPU values are saved.
- We did not search all 128 lags exhaustively for Part D token edits or certify exact goal reachability. The eight-token replay search, the fixed-action history construction, and the linear-in-features family all limit the conclusions. The replays and causal interventions do not establish a unique implementation of the filters.
