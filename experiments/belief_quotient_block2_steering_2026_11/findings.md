# Ladder block-2 steering

Shared runner, fitting contract and cache semantics are documented in
`../reward_factors_block2_steering_2026_11/findings.md`.
Reward-State and Full-State controls use the same site-specific pipeline.
Results belong in this folder's `results/`; checkpoints remain untracked.

Seed-42 pilot: Eq.(4) Reward-State occupancy/mirror is .0974/.0115 at B2
versus .1151/.0086 at B4. Full-State is .1305/.4660 versus .0737/.5156.
Earlier writing improves the first occupancy metric but not spillover or the
Full-State occupancy control. The B2 policy-implied mirror condition has
.012909 total action changes versus only .000702 immediate/local changes,
showing why past-write cache effects cannot be omitted from specificity.

Full donors remain whole-activation references, not belief-coordinate writes.
Unlike B4 unit replacements, B2 unit replacements do not exactly reproduce
donor policies: Reward-State exact/policy mirror TV is .014748/.006958 and
Full-State mirror TV is .086631/.083822. These are conditional means over
supported matches. The shared findings contain all paired tables, the control
audit caveat, historical references, runtime estimate and verified cleanup.

## Approved three-seed result (42–44)

The pilot-only stopping condition was superseded by approval of seeds 43/44.
Both completed on fresh, parallel rentals at unchanged 16/32×1,024 budgets.
Full tables, R², strict per-seed layer signs and the 732-file validated manifest
are in the shared study's `results/campaign_tables.md` / `campaign_summary.json`.
All uncertainties below are sample SDs across three agents.

The pilot Eq.(4) Reward-State occupancy advantage **does not replicate**:
B2 .1044±.0244 versus B4 .0949±.0178, only 1/3 strict B2 wins. Mirror changes
are .0058±.0050 versus .0056±.0026 (2/3 lower at B2, but not a lower mean).
Full-State occupancy is worse at B2 in all three agents: .1039±.0236 versus
.0774±.0039. Policy-implied occupancy improves in all agents (.1589±.0000
versus .1768±.0179), but mirror specificity worsens in all agents
(.0045±.0073 versus .0002±.0004). Action-subspace occupancy worsens in all
agents (.2221±.0745 versus .1087±.0158).

Unrestricted filter-assisted behavioural targets improve both Reward-State and
Full-State occupancy in all three agents, but are not activation-only decoder
solutions. Contextual subspace writes remain near intact. Full donors remain
whole-activation references; mean B2 unit TV for the Reward-State ρ condition
is .005594±.003668 exact / .047214±.061073 policy, so they are not B4-style
lookups through only a norm and head. Cache effects and the intact-shadow
specificity metric remain essential.

MLP encoders have higher activation R² than affine encoders without universal
steering gains. Across all tests, 44/48 B2 MLP fits reached the 200-epoch cap.
The shared findings report the full encoder R²/steering comparison and the
unresolved low-level B4 control discrepancy, including 48 diagnostic controls.
Both owned continuation rentals (61893278/61893277) published compact results
and were authoritatively confirmed absent from the provider.
