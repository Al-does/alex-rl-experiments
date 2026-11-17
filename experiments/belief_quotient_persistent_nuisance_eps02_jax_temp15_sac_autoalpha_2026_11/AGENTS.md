# belief_quotient_persistent_nuisance_eps02_jax_temp15_sac_autoalpha_2026_11

Entropy-regularised discrete SAC, separate from the unchanged fixed-alpha
eps02 SAC leaves. The human user selected automatic alpha tuning with
target entropy 0.3*log(|A|), resolved from the environment action count,
initial alpha 0.01 and Adam lr=3e-4. Ladder leaves have three actions,
so the resolved target is 0.3295836866004329 nats.
Alpha loss is log_alpha * stop_gradient(H(pi)-target_entropy); one update
per gradient step. Current stop-gradiented alpha enters both the actor loss
and the soft critic target. Temperature 1.5 applies to sampling and all losses.
Actor Adam lr=3e-5; critic Adam lr=3e-4; no gradient clipping.
Both independent critic linear heads start at zero (weights and bias);
target copies start identically. Heads predict (1-gamma)*Q, gamma=0.99,
and the two scaled MSE losses are summed. Next-step values shift complete
episode outputs, final steps are terminal, and Polyak tau=0.005 per update.

Environment, 2B-step budget, 19,456 synchronized 1,024-step episodes,
architecture, checkpoint schedule, probe sites and 4,096-step smoke budget
match the corresponding fixed-alpha SAC study. Epsilon-greedy is zero.
Replay retains four rollouts as uint8 tokens/actions and float32 rewards:
478,150,664 bytes at full scale. Uniform sampling with replacement retains
whole episodes. Each rollout is followed by 64 updates of 304 episodes each;
smoke reduces the minibatch to one episode, keeping 64 updates.
Probes read post-final-LayerNorm embeddings.

Curves include alpha, log_alpha, alpha_loss, entropy, policy_entropy,
mean_policy_q (policy expectation of min(Q1,Q2)), existing all-action mean_q,
actor_loss, critic_loss and rollout metrics. Update metrics average over
the iteration and use the pre-update alpha used in that gradient step.
Actor-only step_*.npz files keep the parent layout for greedy probes;
final critic_step_*.npz also stores targets, log_alpha, alpha and alpha Adam
state. Resolved recipes record tuning, initialization and learning rates.
Implementation is shared from
../belief_quotient_persistent_nuisance_eps02_jax_temp15_sac_2026_11/sac.py.
Smoke scores are verification only, not research findings.
