# Ladder Eq. (4) encoder-only ablation

The shared runner, audit, full protocol and cross-test findings are in
`../reward_factors_encoder_nonlinearity_2026_11/findings.md`; shared tables and
hash manifests are in that folder's `results/`. Ladder and Full-State compact
per-arm reports live here in `results/`.

Historical ladder encoder training uses exact-filter `belief[:, :2]`, not
probe predictions (`algorithm_generality_probes_eps02_2026_11/steer_core.py:120–127`).
Write-time current belief remains probe-decoded (`steer_core.py:31–38`). The new
arms vary only encoder architecture (affine / Linear–tanh–Linear with hidden
width 128 > activation width 64) and fitting input (exact / decoded).

Use the fixed-archived-lambda comparison for strict encoder-only effects. The
separate reselected comparison uses the same historical random-floor selection
rule on 16 episodes for all four arms. Mirror strength stays one; confirmation
uses the same paired 32 × 1,024-step streams as the fresh Eq. (4) controls.
Full-State is a positive control, not a target for low mirror spillover.

At the shared archived lambda (mean ± sample SD across seeds 42–44):

| Encoder / fitting inputs | Reward-State occupancy / mirror changed | Full-State occupancy / mirror changed |
|---|---:|---:|
| Affine / exact (historical) | .0949±.0178 / .0056±.0026 | .0774±.0039 / .4907±.0357 |
| Affine / decoded | .0877±.0154 / .0086±.0045 | .0775±.0035 / .4943±.0315 |
| MLP / exact | .1252±.0289 / .0078±.0045 | .0911±.0064 / .4903±.0319 |
| MLP / decoded | .1217±.0141 / .0145±.0118 | .0869±.0052 / .4921±.0237 |

Decoded-source affine lowers Reward-State occupancy for all three seeds, but
increases mirror action changes for all three. Both MLPs worsen both Reward-State
metrics for all three seeds, and worsen Full-State occupancy for all three,
despite higher held-out encoder R². The 200-epoch cap is an important limit:
convergence is not established. All ladder historical-control metrics match
their published fresh controls exactly.

Improved activation reconstruction R² does not by itself establish better belief
steering. The previous contextual-full donor is a retrieval/lookup controller,
not a belief write; this experiment does not use it.
