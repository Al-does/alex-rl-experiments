# Cycle 4 baseline-free REINFORCE design

Cycle 4 repeats cycle 3's two-factor process, reward conditions, 64-dimensional
transformer, 5-million-step budget, checkpoint schedule, and belief probes. It
changes only the policy-gradient implementation to match
`mess3_reward_state_action_symmetry_cycle_6`.

The recipe uses complete 1,024-step episodes, an identically zero value
function, `lambda=1`, normalized discounted returns, and exactly one optimizer
step over each full collected batch. Value, entropy, and KL losses are disabled.
RLlib's maintained PPO Learner remains the execution engine, but its sole
first-policy update has the vanilla REINFORCE gradient because the current and
behavior policies are identical.

RLlib's new PPO stack still routes return computation through its connector
named `GeneralAdvantageEstimation`. With complete episodes, `lambda=1`, and
all value predictions fixed to zero, that connector reduces to Monte Carlo
discounted returns and performs no critic bootstrapping. It also standardizes
returns within the batch. The allocated value-head parameters are deliberately
disconnected: `compute_values()` returns device-native zeros and the value-loss
coefficient is zero.
