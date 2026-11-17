# belief_quotient_persistent_nuisance_eps02_jax_temp15_sac_2026_11

Discrete-action SAC extension of the matching eps02 temperature-1.5 parent.
Environment, 2B-step budget, 19,456 synchronized complete 1,024-step episodes,
architecture, checkpoint schedule and probes are unchanged. Smoke retains
4,096 steps. Policy softmax temperature is 1.5 in sampling and every loss;
epsilon-greedy exploration is zero.

The human user chose fixed alpha=0.0: this is a soft-free actor-critic,
without entropy regularization or automatic temperature tuning.
Two independent transformer Q-networks match the actor architecture and
predict all actions through linear heads scaled by (1-gamma). Critics use
gamma=0.99 and the expected next-action minimum of two Polyak target critics,
with tau=0.005 after every gradient step. Final episode steps are terminal,
with no bootstrap; next-step values shift full-episode network outputs.
Both scaled critic MSEs are summed; targets are stop-gradiented.
Actor loss is the policy expectation of alpha*log(pi)-min(Q1,Q2), with
stop-gradiented critics. Actor checkpoints have the unchanged parent layout.

Each rollout enters an on-device four-rollout episode ring; tokens/actions
are uint8 and rewards float32. Previous-action one-hots reconstruct exactly,
including zero action at reset. Full replay costs 478,150,664 bytes
(456 MiB plus 8 bytes of counters). Sampling is uniform with replacement
over filled whole episodes, preserving transformer histories.
Each iteration takes 64 gradient steps on 304-episode batches, with one
Adam update each for actor and critics, lr=4.2e-4, no gradient clipping.
Smoke retains 64 updates but reduces the minibatch to one episode.
Curves log actor_loss, critic_loss, mean_q, policy_entropy and rollout metrics.
Final critic and target parameters are saved separately as critic_step_*.npz;
only actor parameters enter step_*.npz and the greedy probe battery.
Probes keep reading post-final-LayerNorm embeddings.
Implementation is shared from ../belief_quotient_persistent_nuisance_eps02_jax_temp15_sac_2026_11/sac.py.
Smoke scores are verification only, not research findings.
