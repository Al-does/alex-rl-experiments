# Stage 3: completed three-agent evaluation

## Scope and evidence

All 17 methods, all five tests, frozen agent seeds 42–44: **255/255** complete
reports. Fit budgets are 24 episodes, diagnostic budgets 12, selection budgets
16, confirmation budgets 32, with 1,024 decisions per episode. The wrapper ran
the all-method one-episode smoke before each full command. The 255 smoke files
are separately prefixed and are never aggregated as scientific results.

Generate the complete table and hash manifest without loading checkpoints:

```sh
uv run --group jax python -m experiments.reward_factors_causal_decoding_2026_11.campaign_report
```

[campaign_tables.md](campaign_tables.md) contains all metrics. Values are mean
± **sample SD across three agent seeds**, not standard errors, confidence
intervals, or population-level significance tests. Moving gap closed is
normalized against the paired intact and exact-filter-map values separately
for each agent seed before averaging. The map controller has zero across-agent
SD because it is agent-independent and shares environment streams; that is not
a claim of zero evaluation uncertainty. Ladder map rows duplicate intact.

## Main findings

The contextual full donor is a **retrieval/lookup controller, not a better
decoder or a belief write**. At lambda one, a supported match replaces the
entire final-block latest activation and reproduces the donor moment's action
distribution. Its reward gains belong with controller references such as the
exact-belief map, not evidence of successful belief steering. Patching only the
probe-readable coordinates from the same donor bank left behavior at or near
intact across tests, as did the band/MLP gradient writes despite near-zero target
error in the reported held-out optimization diagnostics. The evidence at this
site is that probe-readable belief coordinates are not what the policy reads.
The positive steering results instead follow policy geometry: action-subspace
writes reduce ladder spillover, and policy-implied gradient writes improve
moving-transfer specificity. Neither is a universal replacement for Eq. (4).

### Why full donors are controller references

This follows from the executed code path, not just the observed reward scores:

- `decoders.contextual_donor(full=True)` returns `bank_h[index] - h` for
  supported matches; the subspace variant projects that same difference into
  the readout row-space. Both variants share the bank and matching rule, though
  their different closed-loop histories can retrieve different donors.
- `run.py` builds the bank from `fit["sites"][:, :, 3]`, reads the current
  `sites[:, 3]`, and applies `strength * delta` through `base.forward(site=3)`.
  All 15 archived campaign checkpoint recipes specify four transformer blocks.
- `steer_moving_base.forward` adds the shift after block 4's attention and MLP.
  The only remaining activation-to-logit operations are `final_norm` and the
  policy dense layer. Thus at lambda one, `h + (h_donor - h) = h_donor`, and
  logits and action probabilities equal the donor moment's at the same
  temperature, up to floating-point roundoff. Rollout uses their greedy argmax.

No donor history or KV cache is transplanted: the block's keys and values were
computed before the shift. The equality concerns the current decision, not a
reconstruction of the donor's belief or history. Unsupported matches abstain;
non-unit strengths interpolate/extrapolate whole activations rather than exactly
copying a donor distribution. The reported selected-strength scores include
those variants and remain retrieval-based controller references, not belief
writes.

### Measurements: steering versus controller references

- **Ladder specificity trade-off:** action-subspace rho occupancy is
  .1019 ± .0149 versus Eq. (4)'s .0949 ± .0178, with mirror spillover
  .0013 ± .0015 versus .0056 ± .0026. Its paired occupancy difference is
  +.0070 ± .0146 (lower is better), not an occupancy improvement. DAS reaches
  .0948 ± .0135 but leaks .1201 ± .0493 of mirror actions. Band and MLP remain
  near intact. Full-State remains a positive control for the existing write:
  Eq. (4) lowers occupancy to .0774 ± .0039 and changes .4907 ± .0357 of
  mirror actions; action reaches .1596 ± .0012, a weaker positive-control write.
- **Moving controller reference:** policy-contextual full donors score
  1.0404 ± .0094 versus joint Eq. (4)'s 1.0177 ± .0116; the paired reward
  difference is +.0227 ± .0190,
  positive for all three agents. Gap closed is .8458 ± .0691 versus .7274 ± .0377.
  Factor-1 reward is .3719 ± .0184 versus .3649 ± .0035; factor-1 actions
  changed are .1077 ± .0172 versus .1134 ± .0091. This is a retrieval-controller
  reward comparison, not an improvement in decoding or belief writing.
  **Policy-implied gradient steering** reaches 1.0339 ± .0228 with much smaller
  action spillover, .0304 ± .0127, but loses on one agent. Action itself is
  worse on all three agents: paired reward difference −.0915 ± .0241 versus
  joint Eq. (4).
- **XOR controller reference:** exact-contextual full donors score above
  factored Eq. (4) for every agent in both cohorts: eps02 .5277 ± .0145 versus
  .4866 ± .0224, paired difference +.0411 ± .0219; v2 .5975 ± .0053 versus
  .5785 ± .0192, paired difference +.0190 ± .0179. These are retrieval/lookup
  gains, not evidence of
  a better belief decoder or write. The corresponding contextual subspace
  patches stay near intact on both XOR cohorts and the other tests. The v2
  action result .5854 ± .0184 is not a consistent improvement: its paired
  difference +.0069 ± .0232 changes sign.
- **Behavioural reference:** unrestricted direct-policy optimization gives
  .5255 ± .0324 on eps02 and .6039 ± .0072 on v2. The seed-42 pilot's .7251 v2
  value does not survive the larger budget. These remain filter-assisted,
  upper-bound-style references, not certified upper bounds or evidence of
  activation-only belief decoding. Action-restricted behavioural optimization
  is much weaker. Neither variant rescues ladder occupancy.

The action-subspace decoder retains much less correlational information:

| Test | Full affine R² | Action-subspace R² |
|---|---:|---:|
| Reward-State | .8772 ± .0367 | .5364 ± .0610 |
| Full-State | .9738 ± .0025 | .8907 ± .0130 |
| Moving | .9127 ± .0058 | .4175 ± .0922 |
| XOR eps02 | .9127 ± .0058 | .8142 ± .0185 |
| XOR v2 | .9397 ± .0025 | .8468 ± .0139 |

The global bank-Jacobian basis is rank-truncated and is not a pointwise exact
non-target null space. No conclusion about all policy-ignored directions, or
whether agents internally track exact Bayes belief, follows from these R² values.

## Historical comparison limits

Historical ladder selected rho occupancy is .0959 ± .0179 for old Eq. (4) and
.0895 ± .0083 for twin; their full-strength mirror spillovers are .0056 ± .0026
and .1871 ± .1323. At full rho strength, occupancy is .0949 ± .0193 and
.0759 ± .0052 respectively. Thus action's low spillover is unlike the twin's
strong specificity failure, but it does not establish superior rho control.

Historical XOR factored old/twin means are .4899/.4824 (eps02) and .5851/.5760
(v2); historical joint means are .4908/.4867 and .5626/.5667. The complete
historical SDs and per-file hashes are in the generated table and
`results/historical_baselines.json`. Those XOR runs used **64 selection / 256
confirmation episodes**, not this campaign's user-requested 16/32. Historical
agent-independent environment samples and probe fits also differ; use the
fresh same-campaign Eq. (4) rows for paired comparisons. No moving or Full-State
twin run exists in those historical result files. Existing Eq. (4) strengths
remain archived, while candidate strengths are selected on this campaign's
selection episodes. XOR Jacobian-null has the same direction as Eq. (4);
differences there are lambda retuning, not a new causal direction.

## Paid execution and cleanup

| Seed | Successful instance | Machine | Observed $/hour | Compact-results commit |
|---|---:|---:|---:|---|
| 42 | 61712892 | 8749 | .4764 | a7cb55a1 |
| 43 | 61715409 | 9204 | .5022 | 3400bb73 |
| 44 | 61715539 | 10447 | .5033 | 4dc5aa8e |

Two earlier seed-43/44 hosts, instances 61713035 and 61713137 on machines
51618/51617, timed out during dependency installation at the provisioner's
1,200-second readiness limit. The provisioner destroyed them before any
scientific run. Both machines were quarantined; replacements were pinned to
different US hosts. There were five rental attempts total, never more than
three active slots. The successful replacements were staggered about 40 seconds.
The requested `--min-cpu-ram` option is absent from this harness revision;
the equivalent 64-GB SDK gate and returned-value check were used instead.

The model/run/decoder source is identical across the successful launches.
Seed 42 launched f4f8cc45; replacements launched 96a3a382, which adds only
historical-object preparation before the smoke and its test adaptation.
Harness revision: a267508fdeb523e95b3ef3fc42119102540bbe38. Runtime is explicitly
JAX CPU, with 12 workers sharing up to 32 cores. Longest full job wall times
were 56.5, 31.3, and 30.9 minutes respectively; job wall times are not box uptime.

All three successful boxes published through the configured self-destruct hook
to `devin/1795210916-causal-decoding`. The workstation verified that every one
of the 510 remote-added paths is under `experiments/**/results/**` (255 smoke,
255 confirmation), and that all five owned instance IDs are absent from the
provider. No git command was run over SSH in a completed box checkout. No
checkpoints, raw rollout arrays, artifact trees, or secrets were committed.
