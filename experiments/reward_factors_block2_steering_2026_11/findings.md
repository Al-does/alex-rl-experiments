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

## Approved three-seed campaign: seeds 42–44

The pilot-only stopping condition above was superseded by approval to run seeds
43 and 44. Both completed on **fresh, parallel Vast rentals**, with no scientific
budget reductions: 16 selection episodes and 32×1,024 confirmation episodes.
The seed-42 action/DAS calibration cap (512 contexts, ranks 1/2/4/8) was retained;
the corresponding fresh B4 controls use the same cap. All primary B2 strengths
were reselected. Cheap archived-B4-strength companions remain in the per-seed
reports and are not substituted retrospectively for the primary selection.

Full tables for every method/test, moving gap/F1, encoder R², and strict per-seed
sign counts are in [campaign_tables.md](results/campaign_tables.md). The machine
summary [campaign_summary.json](results/campaign_summary.json) retains individual
seed values/differences, exact wins/ties/losses, checkpoint identities, source
hashes and the 732-file input manifest. Means and **sample SDs** are across three
agent seeds, not episode standard errors. Three seeds do not establish
statistical significance; a strict numerical win can be a very small effect.

### Layer comparison

There is **no general B2 advantage**, and the pilot's Reward-State Eq.(4)
occupancy advantage does not replicate: factored Eq.(4) wins occupancy in only
1/3 seeds, with mean .1044±.0244 at B2 versus .0949±.0178 at B4. Full-State
occupancy is worse at B2 in all three seeds (.1039±.0236 versus .0774±.0039).
Factored moving Eq.(4) does improve in 3/3 seeds: reward .9833±.0376 versus
.9598±.0401; gap .5481±.1621 versus .4189±.1604. It does not win XOR eps02
reward in any seed. Joint Eq.(4) loses XOR-v2 reward in all three seeds.

| Method / metric | B2 mean ± sample SD | B4 mean ± sample SD | Strict B2 wins / 3 |
|---|---:|---:|---:|
| Policy-implied: Reward-State occupancy ↓ | .1589±.0000 | .1768±.0179 | 3 |
| Policy-implied: mirror actions changed ↓ | .0045±.0073 | .0002±.0004 | 0 |
| Policy-implied: moving reward ↑ | 1.0106±.0073 | 1.0339±.0228 | 1 |
| Policy-implied: moving F1 changed ↓ | .0551±.0229 | .0304±.0127 | 0 |
| Action-subspace: Reward-State occupancy ↓ | .2221±.0745 | .1087±.0158 | 0 |
| Action-subspace: moving reward ↑ | .9611±.0407 | .9264±.0241 | 2 |
| Behavioural target: Reward-State occupancy ↓ | .1628±.0037 | .2696±.0673 | 3 |
| Behavioural target: Full-State occupancy ↓ | .1669±.0106 | .2002±.0435 | 3 |
| Action-restricted behavioural: XOR-v2 reward ↑ | .5843±.0144 | .5415±.0211 | 3 |
| Marginal MLP/decoded encoder: moving reward ↑ | 1.0300±.0347 | 1.0011±.0179 | 3 |
| Marginal MLP/decoded encoder: XOR-v2 reward ↑ | .5817±.0083 | .5745±.0094 | 3 |
| Joint MLP/exact encoder: moving reward ↑ | .9928±.0291 | .9649±.0584 | 3 |
| Joint MLP/exact encoder: moving F1 changed ↓ | .0851±.0293 | .1589±.0723 | 3 |

The policy-implied Reward-State occupancy gain trades against greater mirror
spillover in every seed; its moving specificity is worse at B2. Action-subspace
writes lose their B4 ladder occupancy advantage. Some consistent B2 gains occur
for **filter-assisted behavioural references**, not a recovered activation-only
belief decoder. Contextual subspace donors and band writes remain near intact;
MLP-decoder gradient effects remain task/seed dependent. In particular, a 3/3
win for a contextual subspace donor on XOR-v2 need not mean useful steering:
the rewards remain near intact. Null controls also have task-specific changes.

### Encoder quality does not select a universally better write

MLP encoders reconstruct held-out activations better than affine encoders at
B2, but neither layer nor encoder dominates steering across tests. Marginal
MLP/decoded improves moving and XOR-v2 versus B4 in all three agents, but loses
Full-State occupancy in all three. Joint MLP/exact improves moving reward and
F1 specificity in all three agents versus its B4 counterpart, yet its mean B2
moving reward is still below B2 joint affine/exact (.9928 versus .9995).

B2 encoder R² is often below B4 despite these gains: moving marginal
MLP/decoded .8494±.0444 versus .8808±.0210; joint MLP/exact .8388±.0293 versus
.8587±.0134. XOR-v2 marginal affine R² is higher at B2 in all three agents,
without a universal XOR steering gain. Full R² tables and their per-seed
layer signs are in the campaign table. **44/48 B2 MLP fits reached the
200-epoch cap** (14/16, 16/16, 14/16 by agent); convergence is not established.
The historical exact-belief fitting source and raw, unclipped probe-decoded
write-time current belief remain unchanged.

### Cache semantics and full-donor policy reproduction

All writes remain **latest-position, every decision**. At B2, blocks 3–4 retain
the effects of past writes in their KV caches. Gradients traverse those blocks,
final norm and head with the real cache; action-change metrics use an independent
intact shadow cache on the same executed observations/actions. Local immediate
changes are saved separately and are not the primary specificity metric.

Full donors are **whole-activation controller references, not better belief
decoders**. Unlike B4, a B2 unit replacement has downstream attention and does
not exactly reproduce the donor policy. Conditional unit-replacement TV,
mean ± sample SD over agents, along each selected-strength trajectory:

| Test / condition | Exact full donor | Policy full donor |
|---|---:|---:|
| Reward-State ρ | .005594±.003668 | .047214±.061073 |
| Full-State ρ | .000115±.000058 | .002175±.001300 |
| Moving | .073086±.008513 | .243820±.113094 |
| XOR eps02 | .051021±.013326 | .150297±.080591 |
| XOR v2 | .047675±.007161 | .177808±.033538 |
| Full-State mirror | .077546±.008730 | .125882±.036784 |

Per-seed coverage and selected-λ TV are retained in `campaign_summary.json:donor_tv`.
These are conditional on supported matches, not an unconditional coverage score.

### Historical B2 reference (unpaired)

Read-only archived extraction now covers all three agents. Values below are
mean ± sample SD. Factor-task archives used 64 selection / 256 confirmation
episodes, B4-selected λ reused at B2, and **local** F1 action changes. They
are contextual references, not paired estimates of the current layer effect.

| Coordinates | Ladder occupancy / mirror | Moving reward / gap / local F1 changed | eps02 reward | v2 reward | Full-State occupancy / mirror |
|---|---:|---:|---:|---:|---:|
| Factored | .1079±.0247 / .0051±.0056 | 1.0096±.0207 / .5825±.0886 / .0767±.0295 | .4846±.0053 | .5656±.0105 | .0991±.0250 / .4005±.0486 |
| Joint | same | 1.0415±.0132 / .7443±.0574 / .0843±.0223 | .4852±.0039 | .4981±.0450 | same |

### Control discrepancies: limitation retained

**The exact low-level cause of the three seed-42 B4 XOR discrepancies remains
unexplained.** A 48-control investigation reproduces all three implementations
exactly within each host, but reproduces different original values across hosts.
Exact fitting beliefs and checkpoints match; collected activations and fitted
float32 parameters differ slightly. Fixed-map transplants demonstrate that
fitting differences alone are insufficient; an AVX2 cap and a Python-3.14
switch do not resolve the remaining eps02 difference. No particular CPU ISA,
Python ABI, or argmax tie mechanism was established. See the full
[audit and counterfactual table](results/control_discrepancy_notes.md).

Across the completed campaign, **24/24 B2** affine/exact controls match fresh
Eq.(4) exactly, versus **17/24 B4**. The seven B4 differences comprise the
original three seed-42 XOR cases, seed-43 eps02/v2 marginal action-change-only
differences, seed-44 moving marginal reward/action changes, and seed-44 eps02
marginal action changes. The seed-44 moving reward difference is .0001220703125.
Both versions are retained: the fresh Eq.(4) rows and saved encoder-affine rows
must not be silently collapsed. Saved-versus-fresh B4 provenance and numerical
sensitivity limit attribution of small differences solely to layer placement.

### Validation, publication and cleanup

Completed 456 fresh full-budget confirmation reports across three agents
(366 B2, 90 B4), plus 276 saved B4 inputs. Per-seed names retain the legacy
`pilot_site` prefix; `smoke: false` and budget fields distinguish the full runs.
The separate four-row smoke reports per new seed are not aggregate inputs.

```
36 passed in 31.49s
Validated 156 method/site/seed rows; 732 compact input files; seeds 42–44
Aggregate: mean ± sample SD; strict per-seed wins/ties/losses and encoder R² saved
Baseline B2: 24/24 exact matches
Baseline B4: 17/24 exact matches
```

The focused test command covers block-2 steering, the seed campaign/reporting,
causal decoding and encoder nonlinearity. Report validation checks scientific
budgets, site, latest-every-step positions, intact-shadow metric reference,
fresh implementation hashes and checkpoint consistency. No existing study
code/results were edited. Compact numeric outputs are committed; raw profiles,
portable fitted maps and checkpoints remain ignored artifacts.

Seed 44 finished science at approximately 04:35 UTC and seed 43 approximately
05:18 UTC on 2026-11-22, after a 02:07 UTC rental launch: approximately 2½ and
3¼ hours including setup, respectively. These are approximate wall times,
not sums of parallel per-method timers. Both published compact results to the
launch branch before self-destruction. Authoritative provider checks returned:

```
61893278: absent from provider
61893277: absent from provider
```

Only these two owned rentals were used for the approved continuation. No
unrelated instance was destroyed and no rental remains for this task.
