# Reward Relevance: gru92

Architecture-only control of `two_factor_reward_state_REINFORCE_cycle_4`.

GRU: hidden width 92, four layers, truncated-BPTT chunks of 32; full token-plus-previous-action observation.
This deeper variant complements the unchanged `reward_relevance_gru_2026_11` (width 128, two layers).

All other settings, seed policy, budgets, checkpoint schedule, and probe battery
are inherited unchanged. Probes retain the pre-final-LayerNorm residual representation.

Total parameters including the value head: **208,206**, versus **201,930** for the source transformer (3.1% above).
Width 92 is the closest parameter-matched deeper option.
The value head is retained but the REINFORCE baseline is identically zero.
Resolved recipes record study, architecture, exact model config and parameter count.
