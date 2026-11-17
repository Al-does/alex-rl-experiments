# Seed-42 architecture pulse comparison

CPU refits; each cell is at-pulse → +1 (retained fraction). GRU sites are carried hidden states; Transformer sites are post-block, decoded after block 4.

| Agent | Condition | Pulse | Site 1 | Site 2 | Site 3 | Site 4 |
|---|---|---|---:|---:|---:|---:|
| transformer | Full-State Quotient | δ mirror | 0.076360 → 0.020088 (0.263) | 0.266476 → 0.025262 (0.095) | 0.476079 → 0.007037 (0.015) | — |
| transformer | Full-State Quotient | ρ flip | 0.254188 → 0.069403 (0.273) | 0.450994 → 0.054149 (0.120) | 0.546340 → 0.001838 (0.003) | — |
| transformer | Reward-State Quotient | δ mirror | 0.080377 → 0.023099 (0.287) | 0.287934 → 0.006019 (0.021) | 0.299877 → 0.004019 (0.013) | — |
| transformer | Reward-State Quotient | ρ flip | 0.335898 → 0.145071 (0.432) | 0.496968 → 0.017786 (0.036) | 0.636462 → 0.001608 (0.003) | — |
| gru128 | Full-State Quotient | δ mirror | 0.075162 → 0.075706 (1.007) | 0.553048 → 0.544448 (0.984) | — | — |
| gru128 | Full-State Quotient | ρ flip | 0.105681 → 0.146354 (1.385) | 0.609643 → 0.498276 (0.817) | — | — |
| gru128 | Reward-State Quotient | δ mirror | 0.195107 → 0.238784 (1.224) | 0.442613 → 0.259966 (0.587) | — | — |
| gru128 | Reward-State Quotient | ρ flip | 0.524308 → 0.381879 (0.728) | 0.626506 → 0.173387 (0.277) | — | — |
| gru92 | Full-State Quotient | δ mirror | 0.077653 → 0.136609 (1.759) | 0.205038 → 0.219763 (1.072) | 0.437597 → 0.299745 (0.685) | 0.587766 → 0.145565 (0.248) |
| gru92 | Full-State Quotient | ρ flip | 0.061798 → 0.074045 (1.198) | 0.120356 → 0.087065 (0.723) | 0.188102 → 0.252566 (1.343) | 0.604575 → 0.533496 (0.882) |
| gru92 | Reward-State Quotient | δ mirror | 0.046480 → 0.069499 (1.495) | 0.132086 → 0.155616 (1.178) | 0.240924 → 0.183768 (0.763) | 0.356821 → 0.121595 (0.341) |
| gru92 | Reward-State Quotient | ρ flip | 0.186707 → 0.133334 (0.714) | 0.178078 → 0.132996 (0.747) | 0.467405 → 0.298956 (0.640) | 0.643188 → 0.582771 (0.906) |

## Later lags

| Agent | Condition | Pulse/site | +2 | +8 | +32 | +128 |
|---|---|---|---:|---:|---:|---:|
| transformer | Reward-State Quotient | mirror block1 | 0.021024 | 0.020162 | 0.010080 | 0.000000 |
| transformer | Reward-State Quotient | mirror block2 | 0.005876 | 0.004170 | 0.008566 | 0.000000 |
| transformer | Reward-State Quotient | mirror block3 | 0.003195 | 0.004062 | 0.009759 | 0.000000 |
| transformer | Reward-State Quotient | rho_flip block1 | 0.149262 | 0.085558 | 0.013808 | 0.000000 |
| transformer | Reward-State Quotient | rho_flip block2 | 0.020523 | 0.017426 | 0.007350 | 0.000000 |
| transformer | Reward-State Quotient | rho_flip block3 | 0.001532 | 0.000948 | 0.003303 | 0.000000 |
| transformer | Full-State Quotient | mirror block1 | 0.019132 | 0.008299 | 0.003320 | 0.000000 |
| transformer | Full-State Quotient | mirror block2 | 0.023272 | 0.015337 | 0.005843 | 0.000000 |
| transformer | Full-State Quotient | mirror block3 | 0.004350 | 0.007552 | 0.004887 | 0.000000 |
| transformer | Full-State Quotient | rho_flip block1 | 0.064354 | 0.018766 | 0.008605 | 0.000000 |
| transformer | Full-State Quotient | rho_flip block2 | 0.052756 | 0.036854 | 0.005776 | 0.000000 |
| transformer | Full-State Quotient | rho_flip block3 | 0.001233 | 0.002958 | 0.001560 | 0.000000 |
| gru128 | Reward-State Quotient | mirror layer1 | 0.267730 | 0.255484 | 0.061422 | 0.002724 |
| gru128 | Reward-State Quotient | mirror layer2 | 0.169844 | 0.060668 | 0.011609 | 0.000097 |
| gru128 | Reward-State Quotient | rho_flip layer1 | 0.285971 | 0.164960 | 0.052602 | 0.002049 |
| gru128 | Reward-State Quotient | rho_flip layer2 | 0.293035 | 0.394957 | 0.047155 | 0.001474 |
| gru128 | Full-State Quotient | mirror layer1 | 0.072516 | 0.152637 | 0.067287 | 0.004395 |
| gru128 | Full-State Quotient | mirror layer2 | 0.528394 | 0.231898 | 0.031433 | 0.000501 |
| gru128 | Full-State Quotient | rho_flip layer1 | 0.162148 | 0.141794 | 0.056415 | 0.002851 |
| gru128 | Full-State Quotient | rho_flip layer2 | 0.439917 | 0.285742 | 0.051298 | 0.001293 |
| gru92 | Reward-State Quotient | mirror layer1 | 0.076191 | 0.051572 | 0.005798 | 0.000210 |
| gru92 | Reward-State Quotient | mirror layer2 | 0.142672 | 0.058682 | 0.006479 | 0.000242 |
| gru92 | Reward-State Quotient | mirror layer3 | 0.143522 | 0.088254 | 0.022237 | 0.002741 |
| gru92 | Reward-State Quotient | mirror layer4 | 0.068626 | 0.017841 | 0.003265 | 0.003406 |
| gru92 | Reward-State Quotient | rho_flip layer1 | 0.079037 | 0.039815 | 0.014554 | 0.000999 |
| gru92 | Reward-State Quotient | rho_flip layer2 | 0.094593 | 0.050758 | 0.014618 | 0.001054 |
| gru92 | Reward-State Quotient | rho_flip layer3 | 0.200364 | 0.063515 | 0.030853 | 0.003954 |
| gru92 | Reward-State Quotient | rho_flip layer4 | 0.432077 | 0.181563 | 0.029792 | 0.002691 |
| gru92 | Full-State Quotient | mirror layer1 | 0.183667 | 0.206781 | 0.076339 | 0.002991 |
| gru92 | Full-State Quotient | mirror layer2 | 0.182972 | 0.054828 | 0.005691 | 0.000059 |
| gru92 | Full-State Quotient | mirror layer3 | 0.155656 | 0.044547 | 0.009831 | 0.000174 |
| gru92 | Full-State Quotient | mirror layer4 | 0.092499 | 0.125106 | 0.013958 | 0.000119 |
| gru92 | Full-State Quotient | rho_flip layer1 | 0.075921 | 0.078289 | 0.028670 | 0.001473 |
| gru92 | Full-State Quotient | rho_flip layer2 | 0.066846 | 0.099025 | 0.012710 | 0.000339 |
| gru92 | Full-State Quotient | rho_flip layer3 | 0.397089 | 0.156333 | 0.013507 | 0.000178 |
| gru92 | Full-State Quotient | rho_flip layer4 | 0.355440 | 0.059721 | 0.011675 | 0.000454 |

## Held-out decoding R²

Site order is bottom to top; Transformer additionally includes final LayerNorm (site 5).

| Agent | Condition | ρ by site | δ by site |
|---|---|---|---|
| transformer | Reward-State Quotient | 0.963817, 0.978178, 0.979662, 0.978791, 0.979070 | 0.572667, 0.666048, 0.657092, 0.732931, 0.712582 |
| transformer | Full-State Quotient | 0.952602, 0.971609, 0.979445, 0.980837, 0.974062 | 0.909345, 0.941524, 0.950718, 0.954212, 0.955057 |
| gru128 | Reward-State Quotient | 0.989239, 0.992952 | 0.858528, 0.862319 |
| gru128 | Full-State Quotient | 0.975653, 0.986118 | 0.941190, 0.953305 |
| gru92 | Reward-State Quotient | 0.989624, 0.993369, 0.992898, 0.992911 | 0.775287, 0.781246, 0.758454, 0.600168 |
| gru92 | Full-State Quotient | 0.980102, 0.985619, 0.988406, 0.988791 | 0.947319, 0.965036, 0.970269, 0.971090 |

## Transformer sanity: CPU refit versus original GPU stable reports

| Condition | Pulse/site | Original at/+1 | CPU at/+1 | Max curve difference (requested coordinate) |
|---|---|---:|---:|---:|
| Reward-State Quotient | mirror_block1 | 0.081200428/0.023188902 | 0.080376722/0.023099409 | 0.000823706 |
| Reward-State Quotient | mirror_block2 | 0.288073570/0.006433917 | 0.287934154/0.006019469 | 0.000544447 |
| Reward-State Quotient | mirror_block3 | 0.300086766/0.004069077 | 0.299877018/0.004019434 | 0.000277271 |
| Reward-State Quotient | rho_flip_block1 | 0.327982336/0.141064107 | 0.335897535/0.145071387 | 0.007915199 |
| Reward-State Quotient | rho_flip_block2 | 0.496899992/0.017685762 | 0.496967912/0.017786445 | 0.000100683 |
| Reward-State Quotient | rho_flip_block3 | 0.636387527/0.001617417 | 0.636461735/0.001607740 | 0.000074208 |
| Full-State Quotient | mirror_block1 | 0.076655813/0.020068249 | 0.076360427/0.020087881 | 0.000295386 |
| Full-State Quotient | mirror_block2 | 0.266478330/0.025244994 | 0.266476035/0.025261505 | 0.000057392 |
| Full-State Quotient | mirror_block3 | 0.476026982/0.007073729 | 0.476078987/0.007036861 | 0.000052005 |
| Full-State Quotient | rho_flip_block1 | 0.253028125/0.068863325 | 0.254187554/0.069402739 | 0.001159430 |
| Full-State Quotient | rho_flip_block2 | 0.450876743/0.054216806 | 0.450994492/0.054149494 | 0.000117749 |
| Full-State Quotient | rho_flip_block3 | 0.546114385/0.001839551 | 0.546339691/0.001837723 | 0.000225306 |

Exact numerical reproduction is **not** established. The pinned pulse/core source hashes match the original report's hashes. Exact-filter curves agree within 1.8e-6 across all lags, but fitted probe scores differ slightly; the contribution of CPU/GPU activations versus fitting is not isolated. Do not call this an exact sanity pass.

## Paper references (not these agents' own-history filter estimates)

| Condition | Pulse | Exact filter at/+1 | 15-seed Transformer block 3 at/+1 |
|---|---|---:|---:|
| Full-State Quotient | mirror | 0.589/0.570 | 0.463/0.012 |
| Full-State Quotient | rho_flip | 0.643/0.586 | 0.555/0.007 |
| Reward-State Quotient | mirror | 0.603/0.577 | 0.350/0.006 |
| Reward-State Quotient | rho_flip | 0.641/0.582 | 0.581/0.011 |
