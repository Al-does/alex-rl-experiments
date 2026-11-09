# Reward Relevance: memoryless

Architecture-only control of `two_factor_reward_state_REINFORCE_cycle_4`.

Residual MLP: width 64, six pre-LN residual GELU blocks; current-token one-hot only, excluding previous actions.

All other settings, seed policy, budgets, checkpoint schedule, and probe battery
are inherited unchanged. Probes retain the pre-final-LayerNorm residual representation.

Total parameters including the value head: **200,714**, versus **201,930** for the source transformer (within 1%).
The value head is retained but the REINFORCE baseline is identically zero.
Resolved recipes record study, architecture, exact model config and parameter count.
