# Cycle 6 REINFORCE design

Cycle 6 isolates the learning algorithm while retaining cycle 5's scientific
controls: the same sticky-state MESS3 environment, three action variants,
64-dimensional four-layer transformer with one attention head and context
length 10, seed policy, and held-out affine belief probes. Full training stops
after 2.5 million sampled environment steps.

RLlib 2.56 no longer ships its former `PG` algorithm. This recipe therefore
uses the maintained PPO collection, checkpoint, and Torch Learner path in the
configuration where its policy update is REINFORCE:

- complete episodes and `lambda=1` produce Monte Carlo discounted returns;
- an identically zero value function removes the critic baseline;
- normalized returns provide the usual batch variance reduction;
- one epoch over the full train batch produces one policy-gradient update;
- value, KL, and entropy losses are disabled.

At the only optimizer step, current and behavior policies are identical, so
PPO's likelihood ratio is one and its clipping branch has exactly the vanilla
score-function gradient. The reusable transformer and probing code remains
unchanged, and the resolved recipe records that PPO is the RLlib execution
engine rather than describing the scientific algorithm as PPO.
