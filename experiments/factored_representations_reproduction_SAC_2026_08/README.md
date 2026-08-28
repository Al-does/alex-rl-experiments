# Discrete-SAC factored-representation reproduction

This study keeps the 2-factor and 3-factor MESS3 tasks and probe definitions
from the matched
[`factored_representations_reproduction_PPO_2026_08`](../factored_representations_reproduction_PPO_2026_08/)
study, but replaces PPO with RLlib's discrete Soft Actor-Critic.

Each factor count has two objective arms:

- `sac`: correctness-reward discrete SAC;
- `sac_aux_ce`: the same SAC objective plus coefficient-one cross entropy for
  the joint token revealed in the next observation.

Run both factor counts for an objective:

```bash
uv run rl-harness \
  experiments.factored_representations_reproduction_SAC_2026_08.sac.experiment \
  --smoke --hardware-profile cpu

uv run rl-harness \
  experiments.factored_representations_reproduction_SAC_2026_08.sac_aux_ce.experiment \
  --smoke --hardware-profile cpu
```

The four single-cell entrypoints are `sac_2_factors`, `sac_3_factors`,
`sac_aux_ce_2_factors`, and `sac_aux_ce_3_factors`.

## Architecture

RLlib SAC supplies the discrete actor/critic losses, entropy-temperature
optimization, replay, target networks, and soft target updates. A custom
`SACCatalog` replaces only its neural architecture:

- actor: paper-style four-block transformer → linear categorical-policy head;
- critic: independent transformer → linear Q head;
- twin critic: another independent transformer → linear Q head.

No parameters are shared between actor and critics. Because RLlib SAC does not
support recurrent RLModules, the generic HMM environment emits its existing
nine-position token-history window. This contains the same BOS-plus-eight-token
information as the PPO module's recurrent state without adding actions,
beliefs, or hidden state to the policy observation.

The auxiliary classifier is attached only to the actor encoder. Its loss is
added to SAC's named actor loss, so it updates the actor and classifier but
neither critic.

## Probes

Checkpoint analysis uses only the actor's final transformer-block residual
after the MLP and before final LayerNorm. The critic and target networks are
never passed to the probe battery. Sampling, held-out regression, CEV,
vary-one, principal-angle, and token-embedding definitions are shared with the
PPO study to keep the algorithm comparison matched.

Full runs use the same 50-million-environment-step budget as the completed PPO
continuations. SAC uses `gamma=0`, one-step targets, twin Q networks,
`tau=0.005`, actor learning rate `3e-5`, critic/temperature learning rate
`3e-4`, batch size 256, and 1,500 replay warm-up steps. These are explicit
preregistered defaults, not tuned results.
