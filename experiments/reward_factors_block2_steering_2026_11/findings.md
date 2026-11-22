# Block-2 steering: seed-42 pilot findings

This study leaves prior experiment code and results unchanged. Site 1 is the
residual after block 2; site 3 is the residual after block 4. Writes affect the
latest position **every decision**. The keys/values of later blocks retain past
writes; no past residual band or donor KV history is directly transplanted.

Action-change metrics compare against an independent intact shadow cache on
the **same executed observations/actions**, so they include the effects of
past B2 writes. `local_actions_changed` additionally saves the immediate write
effect holding the recipient's already-intervened cache fixed. These coincide
at B4 but need not at B2. Archived factor-task B2 action-change numbers used
the local definition, whereas archived ladder used the intact shadow.

Every B2 decoder, encoder, donor activation bank, readout complement, Jacobian,
action basis and DAS basis is fitted at B2. Policy-implied forward maps are
refitted on the fitting streams' beliefs and actual final-head logits. Gradients
through the B2 activation continue through blocks 3–4 with the actual current
cache and position, including differentiation through their attention and head.
Offline Jacobians/DAS use pre-query caches reconstructed from complete intact
episode residuals, not truncated zero-history continuations.

All non-reference B2 methods reselect lambda on the historical grid/rule using
16 episodes, followed by 32 independent 1,024-step confirmation episodes. Ladder
mirror controls keep lambda one. Decoder/donor fits use 24 complete episodes and
12 held-out episodes. Historical Eq.(4)/encoder fitting retains its 12,000-row
ladder and 60,000-row factor-task cutoffs, collecting enough complete episodes.
For the pilot, expensive real-cache action/DAS objectives use 512 evenly spaced
fit contexts (DAS uses four full paired episodes per split, ranks 1/2/4/8).
Both receive fresh B4 controls with the identical context budget. Eq.(4)
factored/joint also receive fresh, reselected B4 controls; this avoids retaining
the previous causal campaign's archived lambda for that comparison. Random and
exact-map references are fresh at both sites. Remaining B4 rows are saved
16/32-budget campaign reports, including the encoder study's reselected primary
metrics (not its fixed-lambda companions). This is not an entirely fresh,
single-backend B2/B4 experiment. The affine/MLP encoder arms retain raw probe-decoded
write-time current beliefs and their original fitting/early-stopping protocol.

Contextual full donors at B2 are **not assumed to be lookup controllers**:
blocks 3–4 still attend to the recipient history. Reports save conditional mean
TV to the selected donor moment's actual action distribution, both for a unit
replacement and the selected lambda, on supported decisions. Unsupported
matches abstain; no coverage means undefined TV. At B4, a supported unit
replacement reproduces the donor distribution up to roundoff, as checked by
the unit test; archived B4 reports did not save empirical TV.

## Completed pilot and interpretation

Only agent seed 42 ran. `seed_pilot` has no three-seed path. The full pilot
contains **152 fresh confirmation reports** (122 B2, 30 B4), with 92 saved B4
reports supplying the remaining comparisons. The audit validates 244 unique
inputs, consistent checkpoint hashes, fresh implementation hashes, sites,
positions, action-change definitions and full episode budgets. The 52 logical
method/site rows include repeated ladder joint encoder entries. There is no
across-agent SD from this pilot.

Complete paired tables, moving specificity, donor TV and encoder R² are in
[`results/pilot_tables.md`](results/pilot_tables.md); exact values and input
hashes are in [`results/pilot_summary.json`](results/pilot_summary.json).

- **No blanket benefit from moving to B2.** Factored Eq.(4) improves Reward-State
  occupancy (.0974 versus .1151), with worse mirror spillover (.0115 versus
  .0086). Full-State occupancy is worse (.1305 versus .0737). Action-subspace
  occupancy control weakens (.2269 versus .1245), and its v2 reward falls from
  .6150 at B4 to .5081 at B2.
- **Cache history matters.** For the B2 policy-implied mirror control, primary
  action changes are .012909, versus .000702 for the immediate/local effect.
  B4's corresponding primary value is .000641. A near-preserved current-step
  policy is not evidence of preserved closed-loop behavior before downstream
  attention.
- **Probe-coordinate methods do not become a universal solution.** Contextual
  subspace donors and band writes remain at/near intact. MLP decoder writes
  gain some moving reward at B2 (.9095 versus .8672 intact), with .0322 factor-1
  action changes. This pilot does not measure online decoded-target attainment,
  so the earlier B4 target-error conclusion must not simply be assumed at B2.
- **Policy geometry remains task-specific.** Policy-implied B2 moving reward
  is 1.0141 with .0295 factor-1 changes, versus B4's 1.0355/.0279. B2 is not an
  improvement on that B4 specificity result. Behavioural-target writes remain
  filter-assisted references, not certified bounds or activation-only decoders.
- **B2 full donors are not exact lookup controllers.** Conditional unit-write
  TV is nonzero: moving exact/policy donors .07809/.27226, eps02
  .05527/.08183, v2 .04649/.17838. Full-State mirror TVs are .08663/.08382.
  These are whole-activation retrieval interventions followed by history-aware
  processing, not better belief decoders. Policy-full moving reward improves
  to 1.0555 at B2 from 1.0458 at B4, but that remains a controller-reference
  result. Policy-full coverage is only .1142 for eps02 and .2125 for v2; TVs
  are conditional on support. Unit TVs are counterfactual current-step unit
  replacements along the selected-lambda trajectory, not separate unit-lambda
  rollout estimates.
- **Encoder fit quality still does not imply universal steering gains.** Joint
  MLP/decoded moving reward rises to 1.0294 at B2 from .9885 at B4, but its B2
  v2 selection is the zero write (.5081 intact reward). MLP/exact joint v2 is
  .4940 at B2 versus .5329 at B4. MLP encoder R² exceeds affine R², but **14/16
  B2 MLP fits reached the 200-epoch cap**; convergence is not established.

## Control audit limitation

All **8/8 B2 affine/exact encoder controls** match fresh Eq.(4) exactly in
selected strength, reward, factor-1 reward and action-change metrics. At B4,
5/8 match; three saved-encoder versus fresh-causal XOR controls differ:

| Test / coordinates | Fresh causal B4 reward | Saved affine/exact B4 reward | Selection |
|---|---:|---:|---|
| eps02 / marginal | .459900 | .444214 | lambda .85 versus 1 |
| v2 / marginal | .552856 | .555573 | same lambda |
| v2 / joint | .564575 | .564240 | same lambda |

Their precise numerical/execution cause was **not isolated**. The same-lambda
v2 discrepancies show that lambda selection alone is not a complete
explanation. Both implementations use `(target-current) @ encoder`, followed
by lambda, so a different affine bias-cancellation formula does not explain
the checked path. Checkpoint and budget mismatches are excluded by the audit.
Both sets are retained without a tolerance waiver; per-field discrepancies are
recorded in `pilot_summary.json:baseline_consistency`. Do not silently pool
their baselines or attribute these differences entirely to intervention layer.

## Historical B2 Eq.(4) reference (not paired fresh controls)

Exact archive paths, revisions, hashes, strengths and protocols are saved in
[`historical_B2_seed42.json`](results/historical_B2_seed42.json).

| Coordinates | Ladder occupancy / mirror | Moving reward / gap / F1 changed | eps02 | v2 | Full-State occupancy / mirror |
|---|---:|---:|---:|---:|---:|
| Factored | .0982 / .0116 | .9899 / .5115 / .0692 | .4876 | .5645 | .1277 / .4534 |
| Joint | same ladder row | 1.0304 / .7055 / .0652 | .4807 | .4636 | same Full-State row |

Historical ladder used 16/32 episodes. Historical factor-task rows used 64/256,
selected lambda at B4 then reused it at B2, and used the **local** F1-change
definition. Their moving gap uses their own intact/map streams. They are
reference rows, not matched estimates of the differences in the fresh table.

## Cheap archived-B4-lambda companions

These were already evaluated by the new encoder runner. Because B2
affine/exact agrees with Eq.(4), its fixed-lambda companions supply this extra
comparison without another scientific run. All encoder arms also retain their
own `matched_lambda_metrics` in the raw compact reports. Primary tables always
use reselected lambda; confirmation reward was not used to choose between
primary and companion rows.

| B2 affine/exact arm | Archived B4 lambda | Reward / occupancy | F1 changed, or ladder mirror changed |
|---|---:|---:|---:|
| Ladder | 1 | .0974 | .0115 |
| Full-State | 1 | .1305 | .4660 |
| Moving factored | 1 | .9363 | .0815 |
| Moving joint | 1.25 | 1.0150 | .0728 |
| eps02 factored | .5 | .4490 | .0677 |
| eps02 joint | 0 | .4774 | .0000 |
| v2 factored | 1.25 | .5574 | .2661 |
| v2 joint | 1.25 | .4193 | .4809 |

Moving joint's archived-lambda confirmation reward exceeds its reselected
primary .9686. This exposes selection uncertainty from 16 episodes; it is not
permission to retrospectively replace the selected result.

## Runtime, publication and stopping point

The CPU phase ran **23:01:02–01:47:49 UTC, 2h46m47s**, on one 64-core RTX4090
rental with eight causal workers and a short disjoint-CPU B4 Eq.(4) job. It
resumed four completed GPU ladder rows and two local CPU rows. Policy-implied
rows dominated: ladder 35.6 min, Full-State 66.1, moving 60.3, eps02 105.9,
v2 90.2. The initial GPU configuration left policy/null unfinished during the
profiling window; local full-budget ladder null took 48.4s and policy 18.0min.
This is a mixed-backend operational timing, not a clean hardware benchmark.
No inverse/write optimization steps or episode budgets were reduced.

For the same calibration cap and CPU-rich hardware, budget approximately
**3–3.5 hours of science per remaining agent seed**, plus provisioning and
publication; two fresh rentals could run 43/44 in parallel after approval.
This is an extrapolation, not a measured three-seed runtime. The pilot's
512-context action/DAS cap is a disclosed calibration choice; its B4 controls
were refit with the same cap rather than compared only to the denser old fit.

Remote compact results were published in `432fad6f`. Both instances rented
for this task, `61867358` (aborted while loading) and `61868041` (successful),
were confirmed **absent from the provider**. No other instance was destroyed.
No agent-seed 43/44 confirmation run was launched. Stop here for pilot review.
