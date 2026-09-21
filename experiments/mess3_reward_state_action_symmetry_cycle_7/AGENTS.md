# `mess3_reward_state_action_symmetry_cycle_7` — naming caveat

This folder is a self-contained continuation of the cycle-6 REINFORCE
action-symmetry study (`mess3_reward_state_action_symmetry_cycle_6`), relocated
here so its merge stayed purely additive after main drifted. All code was moved
verbatim except import paths, which now reference `cycle_7`.

## Result directory names do not match this folder

Every run in this campaign was launched while the code still lived under
cycle_6, so per-run result directories and `run_manifest.json` `condition`
fields retain the `mess3_reward_state_action_symmetry_cycle_6-*` prefix, e.g.

```text
variant_2_ctx32_ent/results/mess3_reward_state_action_symmetry_cycle_6-variant_2_ctx32_ent-seed42-10m/
```

The prefix is launch-time provenance — do not rewrite it. New runs queued
through this folder's `seed_queue.py` use `STUDY =
"mess3_reward_state_action_symmetry_cycle_7"` and will produce `cycle_7-*`
result names, so both prefixes may coexist here.
