# Post-hoc transducer belief probes (cycle 1, seed 42)

Held-out affine probes from the post-final-norm residual stream to the
controlled (action-conditioned) Bayes belief over the three Pusher-B latent
states, evaluated at every saved checkpoint (initial, iterations 1–128
log-spaced, final) of the eight full 50M-step runs.

Protocol (`probe_analysis.py`): 32 parallel environments, stochastic policy
sampling, 20,000 fit steps and 20,000 independent held-out test steps per
checkpoint (separate seed streams), ridge 1e-6, no warmup drop. The target is
`b_t = normalize(s_{t-1} @ K_{a_{t-1}}[y_{t-1}]) @ T_{a_t}` filtered over
complete episodes; it equals the environment's `belief_current` diagnostic to
about 1e-16 on every checkpoint. Checkpoints were restored from the B2 URIs
recorded in each per-checkpoint JSON (`transducer_probes/*.json` beside the
run's `run_manifest.json`). Regenerate the figure and summary with
`python -m experiments.pusher_b_reward_state_action_symmetry_cycle_1.report_probes`.

| arm | R² initial | R² final | 1 − R² final |
|---|---:|---:|---:|
| b90_reward_a.variant_2 | 0.703 | 0.215 | 0.785 |
| b90_reward_a.variant_3 | 0.696 | 0.545 | 0.455 |
| b90_reward_b.variant_2 | 0.674 | 0.927 | 0.073 |
| b90_reward_b.variant_3 | 0.683 | 0.918 | 0.082 |
| b10_reward_a.variant_2 | 0.318 | 0.903 | 0.097 |
| b10_reward_a.variant_3 | 0.204 | 0.521 | 0.479 |
| b10_reward_b.variant_2 | 0.384 | 0.765 | 0.235 |
| b10_reward_b.variant_3 | 0.259 | 0.748 | 0.252 |

Observations (descriptive; one training seed per arm, no confidence intervals):
both `b90_reward_a` arms become *less* linearly decodable with training, while
every `reward_b` arm and `b10_reward_a.variant_2` become more decodable;
`b10_reward_a.variant_2` dips to R² ≈ 0.02 at 4M steps before rising to 0.90.
Affine decodability does not establish causal policy use.
