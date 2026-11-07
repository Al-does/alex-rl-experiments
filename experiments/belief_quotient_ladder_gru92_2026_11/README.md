# Belief-Quotient Ladder: gru92

Architecture-only control of `mess3_reward_state_action_symmetry_cycle_7/variant_{1,2,3}_ctx32_ent`.

GRU: hidden width 92, four layers, truncated-BPTT chunks of 32; full token-plus-previous-action observation.
This deeper variant complements the unchanged `belief_quotient_ladder_gru_2026_11` (width 128, two layers).

All other settings, seed policy, budgets, checkpoint schedule, and probe battery
are inherited unchanged. The entropy bonus is 0.03 -> 0, ending 1M steps before the target; temperature is 1.0. Probes use post-final-LayerNorm embeddings.

Total parameters including the value head: **206,544**, versus **200,772** for the source transformer (2.9% above).
Width 92 is the closest parameter-matched deeper option.
The value head is retained but the REINFORCE baseline is identically zero.
Resolved recipes record study, architecture, exact model config and parameter count.
