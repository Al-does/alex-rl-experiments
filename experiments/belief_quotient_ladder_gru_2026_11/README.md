# Belief-Quotient Ladder: gru

Architecture-only control of `mess3_reward_state_action_symmetry_cycle_7/variant_{1,2,3}_ctx32_ent`.

GRU: hidden width 128, two layers, truncated-BPTT chunks of 32; full token-plus-previous-action observation.

All other settings, seed policy, budgets, checkpoint schedule, and probe battery
are inherited unchanged. The entropy bonus is 0.03 -> 0, ending 1M steps before the target; temperature is 1.0. Probes use post-final-LayerNorm embeddings.

Total parameters including the value head: **199,812**, versus **200,772** for the source transformer (within 1%).
The value head is retained but the REINFORCE baseline is identically zero.
Resolved recipes record study, architecture, exact model config and parameter count.
