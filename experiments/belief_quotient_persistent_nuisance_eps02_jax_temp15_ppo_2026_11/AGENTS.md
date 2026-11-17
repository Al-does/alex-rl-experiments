# belief_quotient_persistent_nuisance_eps02_jax_temp15_ppo_2026_11

PPO extension of the matching eps02 temperature-1.5 parent study.
Environment, 2B-step budget, 19,456 complete 1,024-step episodes per rollout,
64 whole-episode minibatches, architecture, checkpoint schedule and probes
are unchanged. Smoke retains 4,096 steps. Temperature is 1.5 in sampling
and likelihoods; entropy and epsilon-greedy exploration remain zero.

PPO uses four independently shuffled epochs, gamma 0.99, GAE lambda 0.95,
clip 0.2, Adam lr 4.2e-4, no gradient clipping or KL penalty.
Episode end is terminal (zero bootstrap), as in parent reward-to-go.
Advantages are batch-standardized with std floor 1e-4; value targets use
the unstandardized GAE advantages plus rollout-time values.
A separate PyTorch-default linear critic reads the shared post-final-LayerNorm
embedding. Its output predicts (1-gamma)*V; loss is 0.5 times scaled-target
MSE, without value clipping. Curves include vf_loss and explained_variance.
Probes continue reading the post-final-LayerNorm embedding.
PPO implementation is shared from ../belief_quotient_persistent_nuisance_eps02_jax_temp15_ppo_2026_11/ppo.py.
Smoke results are verification only, not research findings.
