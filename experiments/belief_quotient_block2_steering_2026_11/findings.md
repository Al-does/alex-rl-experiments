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
