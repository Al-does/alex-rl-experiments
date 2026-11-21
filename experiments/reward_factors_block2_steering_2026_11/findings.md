# Block-2 steering: seed-42 pilot protocol

This study leaves prior experiment code and results unchanged. Site 1 is the
residual after block 2; site 3 is the residual after block 4. Writes affect the
latest position **every decision**. The keys/values of later blocks retain past
writes; no past residual band or donor KV history is directly transplanted.

Every B2 decoder, encoder, donor activation bank, readout complement, Jacobian,
action basis and DAS basis is fitted at B2. Policy-implied forward maps are
refitted on the fitting streams' beliefs and actual final-head logits. Gradients
through the B2 activation continue through blocks 3–4 with the actual current
cache and position, including differentiation through their attention and head.
Offline Jacobians/DAS use pre-query caches reconstructed from complete intact
episode residuals, not truncated zero-history continuations.

All non-reference B2 methods reselect lambda on the historical grid/rule using
16 episodes, followed by 32 independent 1,024-step confirmation episodes. Ladder
mirror controls keep lambda one. Fits use 24 independent complete episodes and
12 held-out episodes; historical Eq.(4)/encoder fitting row cutoffs are retained.
For the pilot, expensive real-cache action/DAS objectives use 512 evenly spaced
fit contexts (DAS uses four full paired episodes per split, ranks 1/2/4/8).
Both receive fresh B4 controls with the identical context budget. Other B4
comparison rows come from the saved campaign, whose Eq.(4) lambda was archived
rather than reselected. These are explicit comparison limitations, not paired
layer-only estimates. The affine/MLP encoder arms retain raw probe-decoded
write-time current beliefs and their original fitting/early-stopping protocol.

Contextual full donors at B2 are **not assumed to be lookup controllers**:
blocks 3–4 still attend to the recipient history. Reports save conditional mean
TV to the selected donor moment's actual action distribution, both for a unit
replacement and the selected lambda, on supported decisions. Unsupported
matches abstain; no coverage means undefined TV. Fresh B4 full donors should
reproduce the donor distribution at lambda one up to roundoff.

Only seed 42 is authorized in this pilot. `seed_pilot` has no three-seed path.
No seeds 43–44 or final campaign are launched before reporting this pilot.
Tables, measured timings, historical B2 Eq.(4) references, limitations and
runtime extrapolation will accompany the completed compact pilot results.
