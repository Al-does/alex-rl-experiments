# Six-agent stable Vast campaign

N = no ε training; E = ε=.05 training. Variants 1/2/3 are Trivial/Reward-State/Full-State. All evaluations are greedy, seed-42 final checkpoints. Occupancy is across 32 complete 1024-step confirmation episodes, seed 2042; one training seed per agent, not a training-seed confidence interval.

## Important qualifications

- The first GPU campaign is superseded: reduced GPU affine-matmul precision caused catastrophic post-LN cancellation. Stable runs explicitly use HIGHEST affine matmul precision and least-squares rcond=1e-6. Never mix original seed42 reports with seed42_stable.
- Scalar norm calibration is not guaranteed. Some attempted `normmatched` conditions fail even on tuning data; others fail on confirmation. Ratios below compare achieved confirmation norm with the same-agent/site rho_flip λ=1 confirmation norm. Large departures are NOT equal-strength controls.
- All six GPU confirmation occupancy vectors exactly match fresh CPU evaluations. No-ε FSQ has one different action among 32768 (GPU Noop −1, Y +1); the other five action-share vectors match exactly. This is not bitwise policy equivalence.
- Reference-grid convergence is empirical on 32 matched reference episodes, not an optimality certificate. FSQ headline uses barycentric γ=.995/N=300, not the original nearest-grid N=100.
- Per-agent action-agreement metrics retain the original nearest-grid N=100 policy; they are not recomputed against the refined headline policy.
- Early-site pulse writes pass through later blocks before the lag-0 block-4 readout. Attenuation/amplification before that readout and imperfect encoder/decoder inversion prevent a clean inference about an exact belief-update law from amplitude mismatch alone.
- Reward-State δ steering preserves occupancy across the requested λ≤1 grid, but approximate norm matching at effective λ≈14–32 changes actions and reduces occupancy substantially. Those are large-gain extrapolations beyond validity-preserving target interpolation, not a demonstrated equal-strength causal null.
- Full metrics, episode SEs, all sites, all orthogonal controls, both probe exclusions, all pulse coordinates and paired ε differences: [aggregate](belief_steering_v2_six_agents.md). Durable object/hash audit: [audit](belief_steering_v2_campaign_audit.json).

## Infrastructure

| Agent | Instance | GPU | Analysis seconds | Listed $/hour |
|---|---:|---|---:|---:|
| N1 | 60575158 | RTX 5090 | 95.057 | 0.535 |
| N2 | 60575159 | RTX 5090 | 66.264 | 0.496 |
| N3 | 60574997 | RTX 4090 | 77.599 | 0.393 |
| E1 | 60574999 | RTX 4090 | 66.638 | 0.482 |
| E2 | 60575000 | RTX 4090 | 77.272 | 0.537 |
| E3 | 60575163 | RTX 5090 | 67.253 | 0.537 |

All 12 owned instances are gone (11 tracked records report gone; stalled original 60573740 was explicitly destroyed). Six replacements completed B2 durability, branch publication and self-destruction. Listed hourly rates are not total charges. The billing endpoint returned HTTP 400 `contract_ids must be a non-empty list`; no verified invoice total is available. Analysis runtime excludes paid bootstrap/publication time.

## Intact / uniform-random occupancy

| Agent | Intact | Random |
|---|---:|---:|
| N1 | 0.942780 | 0.188049 |
| N2 | 0.506287 | 0.105743 |
| N3 | 0.473358 | 0.096100 |
| E1 | 0.942780 | 0.188049 |
| E2 | 0.525879 | 0.105743 |
| E3 | 0.471222 | 0.096100 |

## Main steering

Each occupancy cell is λ=1 / selected-λ. Remaining metrics are block-4 λ=1. All post-LN metrics appear in the full aggregate. Norm is mean ‖Δh‖ with relative-to-‖h‖ in parentheses.

| Agent | Target | Selected λ | B4 occupancy | Post-LN occupancy | Noop\|S3 | Action change | Norm (relative) |
|---|---|---:|---|---|---:|---:|---|
| N1 | point | 0 | 0.942780 / 0.942780 | 0.942780 / 0.942780 | 0.00000 | 0.00000 | 0.0397 (0.003) |
| N1 | mirror | 0 | 0.942780 / 0.942780 | 0.942780 / 0.942780 | 0.00000 | 0.00000 | 0.0734 (0.006) |
| N1 | rho_flip | 0.75 | 0.941040 / 0.941040 | 0.904907 / 0.936890 | 0.00000 | 0.00317 | 33.4806 (2.637) |
| N1 | rho_point | 0 | 0.942780 / 0.942780 | 0.942780 / 0.942780 | 0.00000 | 0.00009 | 32.3146 (2.549) |
| N2 | point | 0 | 0.506287 / 0.506287 | 0.506287 / 0.506287 | 0.92700 | 0.00046 | 1.0446 (0.069) |
| N2 | mirror | 0 | 0.506287 / 0.506287 | 0.506287 / 0.506287 | 0.92682 | 0.00003 | 0.7499 (0.049) |
| N2 | rho_flip | 0.9 | 0.052307 / 0.053650 | 0.060577 / 0.060944 | 0.48541 | 0.80737 | 21.2706 (1.208) |
| N2 | rho_point | 0.75 | 0.052490 / 0.059113 | 0.050079 / 0.059570 | 0.40581 | 0.83176 | 24.2552 (1.374) |
| N3 | point | 0.4 | 0.209320 / 0.213348 | 0.207184 / 0.211761 | 0.94693 | 0.56735 | 19.9491 (1.342) |
| N3 | mirror | 0.75 | 0.190338 / 0.186066 | 0.173157 / 0.177887 | 0.95847 | 0.51669 | 14.7692 (0.996) |
| N3 | rho_flip | 1 | 0.083282 / 0.083282 | 0.083527 / 0.083527 | 0.85782 | 0.68747 | 18.1702 (1.270) |
| N3 | rho_point | 0.9 | 0.063568 / 0.080688 | 0.063507 / 0.077972 | 0.67211 | 0.73447 | 20.5918 (1.432) |
| E1 | point | 0 | 0.942780 / 0.942780 | 0.942780 / 0.942780 | 0.00000 | 0.00000 | 0.0557 (0.005) |
| E1 | mirror | 0 | 0.942780 / 0.942780 | 0.942780 / 0.942780 | 0.00000 | 0.00000 | 0.0651 (0.006) |
| E1 | rho_flip | 0 | 0.942780 / 0.942780 | 0.849762 / 0.942780 | 0.00010 | 0.00012 | 1.6542 (0.141) |
| E1 | rho_point | 0 | 0.942780 / 0.942780 | 0.942780 / 0.942780 | 0.00010 | 0.00012 | 1.6734 (0.143) |
| E2 | point | 0 | 0.525879 / 0.525879 | 0.525879 / 0.525879 | 0.94121 | 0.00000 | 0.8342 (0.061) |
| E2 | mirror | 0 | 0.525879 / 0.525879 | 0.525879 / 0.525879 | 0.94121 | 0.00000 | 0.6049 (0.044) |
| E2 | rho_flip | 0.75 | 0.067871 / 0.082703 | 0.077240 / 0.099976 | 0.75315 | 0.75159 | 26.1144 (1.644) |
| E2 | rho_point | 0.75 | 0.057892 / 0.063690 | 0.063049 / 0.062958 | 0.61782 | 0.80496 | 25.9668 (1.616) |
| E3 | point | 0.75 | 0.203003 / 0.203888 | 0.192383 / 0.184570 | 0.94708 | 0.54816 | 17.3000 (1.151) |
| E3 | mirror | 0.9 | 0.188751 / 0.187714 | 0.183838 / 0.184753 | 0.95247 | 0.50458 | 11.6375 (0.776) |
| E3 | rho_flip | 0.9 | 0.061554 / 0.068298 | 0.058624 / 0.063019 | 0.75359 | 0.71887 | 24.4297 (1.701) |
| E3 | rho_point | 0.75 | 0.048126 / 0.071045 | 0.047577 / 0.065857 | 0.59226 | 0.75220 | 25.8193 (1.801) |

## Controls

Orthogonal control below uses rho_flip’s selected λ. δ cells are occupancy (effective λ; confirmation norm ratio). Other orthogonal controls and all norms are in the aggregate.

| Agent | Site | Orthogonal occupancy | δ point | δ mirror |
|---|---|---:|---|---|
| N1 | B4 | 0.938202 | 0.942780 (72.203; 0.086×) | 0.942780 (41.680; 0.091×) |
| N1 | post-LN | 0.942780 | 0.888214 (131.281; 64.171×) | 0.005280 (7.390; 146.682×) |
| N2 | B4 | 0.511292 | 0.366455 (13.990; 0.985×) | 0.377991 (18.543; 1.034×) |
| N2 | post-LN | 0.527374 | 0.256592 (14.998; 1.023×) | 0.316528 (17.965; 1.134×) |
| N3 | B4 | 0.471527 | 0.205353 (0.853; 0.940×) | 0.192810 (1.192; 0.967×) |
| N3 | post-LN | 0.452789 | 0.207611 (1.038; 0.904×) | 0.198029 (1.412; 0.930×) |
| E1 | B4 | 0.942780 | 0.918304 (35.602; 29.742×) | 0.005280 (23.848; 1639.108×) |
| E1 | post-LN | 0.942780 | 0.942780 (33.086; 0.106×) | 0.893433 (31.270; 7.831×) |
| E2 | B4 | 0.525879 | 0.415863 (23.918; 1.036×) | 0.380676 (27.332; 1.029×) |
| E2 | post-LN | 0.525879 | 0.305115 (26.676; 1.061×) | 0.300690 (32.414; 1.148×) |
| E3 | B4 | 0.470306 | 0.204193 (1.395; 0.989×) | 0.199371 (2.239; 1.072×) |
| E3 | post-LN | 0.471436 | 0.192139 (1.440; 0.995×) | 0.188843 (2.251; 1.104×) |

## Coordinate R²

Exclusion 8; tuples are (b1,b2,ρ,δ). All five sites and exclusion-128 supplement are in the aggregate.

| Agent | B4 | Post-LN |
|---|---|---|
| N1 | 0.8910 / 0.8672 / 0.9577 / 0.7493 | 0.8916 / 0.8680 / 0.9603 / 0.7468 |
| N2 | 0.9327 / 0.9311 / 0.9969 / 0.8739 | 0.9294 / 0.9273 / 0.9975 / 0.8667 |
| N3 | 0.9763 / 0.9771 / 0.9916 / 0.9632 | 0.9768 / 0.9774 / 0.9919 / 0.9637 |
| E1 | 0.8979 / 0.8375 / 0.9761 / 0.6976 | 0.8975 / 0.8404 / 0.9748 / 0.7020 |
| E2 | 0.9407 / 0.9396 / 0.9950 / 0.8912 | 0.9402 / 0.9377 / 0.9964 / 0.8878 |
| E3 | 0.9661 / 0.9688 / 0.9878 / 0.9483 | 0.9665 / 0.9698 / 0.9858 / 0.9515 |

## Pulse

Mean absolute intended-coordinate effects at block 1, then fitted intended-coordinate multipliers for blocks 1/2/3. These tail fits exclude lag 0 and numerical-floor values; they are not fractions of the initial write retained, and the curves are not generally exponential. Fit R² and both-coordinate curves are in JSON.

| Agent | Target | Block-1 lag 0 → 1 | Measured multipliers B1/B2/B3 | Full-filter multiplier | Last nonzero lag B1/B2/B3 |
|---|---|---|---|---:|---|
| N1 | mirror | 0.0515309 → 0.0086906 | 0.9460 / 0.9172 / 0.9469 | 0.8375653680518528 | 95 / 64 / 32 |
| N1 | rho_flip | 0.421602 → 0.260021 | 0.9686 / 0.9855 / 0.9468 | 0.8353786746028957 | 96 / 64 / 32 |
| N2 | mirror | 0.0668325 → 0.0256444 | 0.9281 / 0.9081 / 0.9230 | 0.9560452693611678 | 96 / 64 / 32 |
| N2 | rho_flip | 0.105186 → 0.306755 | 0.9048 / 0.8958 / 0.9526 | 0.857230832902451 | 96 / 64 / 32 |
| N3 | mirror | 0.0758631 → 0.0193462 | 0.9076 / 0.8840 / 0.9833 | 0.9644637640478931 | 96 / 64 / 32 |
| N3 | rho_flip | 0.191402 → 0.145523 | 0.8924 / 0.8534 / 0.9690 | 0.9321095267401156 | 96 / 64 / 32 |
| E1 | mirror | 0.0472599 → 0.00847638 | 0.9474 / 0.9343 / 0.9962 | 0.8375653671739363 | 96 / 64 / 32 |
| E1 | rho_flip | 0.508996 → 0.00713897 | 0.9738 / 0.9743 / 0.9735 | 0.8353786744697114 | 96 / 64 / 32 |
| E2 | mirror | 0.148825 → 0.0204525 | 0.9283 / 0.9180 / 0.9835 | 0.9567880453403669 | 96 / 64 / 32 |
| E2 | rho_flip | 0.354 → 0.357843 | 0.9059 / 0.9289 / 0.9901 | 0.857238697874517 | 96 / 64 / 32 |
| E3 | mirror | 0.16639 → 0.0227424 | 0.9123 / 0.9100 / 0.9463 | 0.9636522651951552 | 96 / 64 / 32 |
| E3 | rho_flip | 0.542881 → 0.207627 | 0.8978 / 0.9233 / 0.9692 | 0.9341799521798871 | 96 / 64 / 32 |

Pulse effects are exactly zero after lag 96/64/32 for post-block-1/2/3 writes respectively, in all 36 conditions. Input receptive field is 128, but a post-block-1 pulse has only three downstream 32-step cache bands. Quotient-filter ρ for a variant-2 mirror pulse is exactly zero; quotient and full filters give the same ρ evolution for rho_flip.

## ε=.05 minus no-ε, λ=1 occupancy

| Variant | Target | B4 | Post-LN |
|---:|---|---:|---:|
| 1 | point | +0.000000 | +0.000000 |
| 1 | mirror | +0.000000 | +0.000000 |
| 1 | rho_flip | +0.001740 | -0.055145 |
| 1 | rho_point | +0.000000 | +0.000000 |
| 2 | point | +0.019592 | +0.019592 |
| 2 | mirror | +0.019592 | +0.019592 |
| 2 | rho_flip | +0.015564 | +0.016663 |
| 2 | rho_point | +0.005402 | +0.012970 |
| 3 | point | -0.006317 | -0.014801 |
| 3 | mirror | -0.001587 | +0.010681 |
| 3 | rho_flip | -0.021729 | -0.024902 |
| 3 | rho_point | -0.015442 | -0.015930 |

Paired episode SEs and selected-λ comparisons are in the aggregate; training-seed generalization is untested.

## FSQ references

| γ | N150→200 maximum Δ occupancy | N200→300 maximum Δ occupancy |
|---:|---:|---:|
| 0.97 | 0.00057983 | 0.00057983 |
| 0.99 | 0.00106812 | 0.00268555 |
| 0.995 | 0.00140381 | 0.00000000 |

All three γ values pass the <.01 tested-resolution criterion. Headline γ=.995/N=300 curves:

| λ | δ point | δ mirror | ρ flip | ρ point |
|---:|---:|---:|---:|---:|
| 0.0 | 0.506104 | 0.506104 | 0.506104 | 0.506104 |
| 0.25 | 0.497864 | 0.491638 | 0.209869 | 0.209869 |
| 0.4 | 0.444946 | 0.508575 | 0.209869 | 0.209869 |
| 0.45 | 0.424744 | 0.503174 | 0.209869 | 0.209869 |
| 0.5 | 0.111420 | 0.226471 | 0.209869 | 0.209869 |
| 0.55 | 0.111420 | 0.117310 | 0.209869 | 0.209869 |
| 0.6 | 0.111420 | 0.117310 | 0.209869 | 0.209869 |
| 0.75 | 0.111420 | 0.117188 | 0.209869 | 0.181000 |
| 0.9 | 0.109253 | 0.115601 | 0.209869 | 0.063446 |
| 1.0 | 0.098358 | 0.115265 | 0.124969 | 0.063446 |
