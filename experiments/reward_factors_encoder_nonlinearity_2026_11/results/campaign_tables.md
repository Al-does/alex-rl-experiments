# Eq. (4) encoder-only experiment

Mean ± sample SD across agent seeds 42–44 (pilot: seed 42 only).
Ladder/Full-State cells: occupancy / mirror actions changed. Gap normalization uses paired fresh intact/map references.
Fixed-λ affine/exact is the local historical control; any discrepancies from previously published remote controls require an exact unchanged-runner CPU recheck (recorded in the manifest).
Full-State mirror changes are a positive-control response, not a spillover-minimization objective.

## Held-out encoder R²

| Test / coordinates | Affine exact | Affine decoded | MLP exact | MLP decoded |
|---|---:|---:|---:|---:|
| ladder / marginal | 0.7080±0.0837 | 0.7240±0.0882 | 0.8549±0.0369 | 0.8443±0.0695 |
| ladder_full / marginal | 0.7923±0.0137 | 0.8109±0.0161 | 0.8971±0.0031 | 0.8980±0.0015 |
| moving / marginal | 0.6485±0.0095 | 0.7021±0.0059 | 0.8576±0.0132 | 0.8808±0.0210 |
| moving / joint | 0.7180±0.0239 | 0.7937±0.0202 | 0.8587±0.0134 | 0.9163±0.0154 |
| xor_eps02 / marginal | 0.6485±0.0095 | 0.7021±0.0059 | 0.8576±0.0132 | 0.8808±0.0210 |
| xor_eps02 / joint | 0.7180±0.0239 | 0.7937±0.0202 | 0.8587±0.0134 | 0.9163±0.0154 |
| xor_v2 / marginal | 0.6681±0.0158 | 0.7126±0.0112 | 0.9053±0.0112 | 0.9235±0.0035 |
| xor_v2 / joint | 0.7804±0.0112 | 0.8491±0.0063 | 0.9074±0.0119 | 0.9460±0.0028 |

## Archived λ held fixed: strict encoder-only comparison

| Encoder / source / coordinates | Ladder ↓ / ↓ | Moving reward ↑ | Moving gap ↑ | Moving F1 changed ↓ | XOR eps02 ↑ | XOR v2 ↑ | Full-State |
|---|---:|---:|---:|---:|---:|---:|---:|
| affine/exact/marginal | 0.0949±0.0178 / 0.0056±0.0026 | 0.9671±0.0427 | 0.4574±0.1800 | 0.1379±0.0359 | 0.4867±0.0225 | 0.5785±0.0192 | 0.0774±0.0039 / 0.4907±0.0357 |
| affine/decoded/marginal | 0.0877±0.0154 / 0.0086±0.0045 | 0.9608±0.0412 | 0.4237±0.1876 | 0.1545±0.0476 | 0.4899±0.0313 | 0.5768±0.0239 | 0.0775±0.0035 / 0.4943±0.0315 |
| mlp/exact/marginal | 0.1252±0.0289 / 0.0078±0.0045 | 1.0012±0.0149 | 0.6322±0.0930 | 0.1144±0.0260 | 0.5009±0.0218 | 0.5809±0.0059 | 0.0911±0.0064 / 0.4903±0.0319 |
| mlp/decoded/marginal | 0.1217±0.0141 / 0.0145±0.0118 | 0.9884±0.0322 | 0.5714±0.1355 | 0.0972±0.0262 | 0.5062±0.0146 | 0.5786±0.0023 | 0.0869±0.0052 / 0.4921±0.0237 |
| affine/exact/joint | 0.0949±0.0178 / 0.0056±0.0026 | 1.0177±0.0116 | 0.7274±0.0377 | 0.1134±0.0091 | 0.4771±0.0030 | 0.5676±0.0106 | 0.0774±0.0039 / 0.4907±0.0357 |
| affine/decoded/joint | 0.0877±0.0154 / 0.0086±0.0045 | 1.0068±0.0343 | 0.6709±0.1797 | 0.1086±0.0215 | 0.4817±0.0055 | 0.5664±0.0105 | 0.0775±0.0035 / 0.4943±0.0315 |
| mlp/exact/joint | 0.1252±0.0289 / 0.0078±0.0045 | 0.9860±0.0203 | 0.5560±0.0665 | 0.1782±0.0219 | 0.4814±0.0050 | 0.5631±0.0185 | 0.0911±0.0064 / 0.4903±0.0319 |
| mlp/decoded/joint | 0.1217±0.0141 / 0.0145±0.0118 | 0.9763±0.0213 | 0.4938±0.1298 | 0.1363±0.0403 | 0.4898±0.0194 | 0.5611±0.0077 | 0.0869±0.0052 / 0.4921±0.0237 |

### Paired differences versus historical affine/exact at its archived λ

Signs are raw new−historical differences (negative favors occupancy/spillover; positive favors reward/gap). Full-State mirror is nondirectional.

| Arm | Metric | Δ seed42 | Δ seed43 | Δ seed44 | Mean ± SD |
|---|---|---:|---:|---:|---:|
| affine/decoded/marginal | ladder_occupancy | -0.009674 | -0.002228 | -0.009827 | -0.0072±0.0043 |
| affine/decoded/marginal | ladder_mirror | +0.005188 | +0.002045 | +0.001740 | 0.0030±0.0019 |
| affine/decoded/marginal | moving_reward | +0.019196 | +0.017792 | -0.055939 | -0.0063±0.0430 |
| affine/decoded/marginal | moving_gap_closed | +0.095665 | +0.108688 | -0.305500 | -0.0337±0.2355 |
| affine/decoded/marginal | moving_f1_changed | +0.010437 | +0.008911 | +0.030548 | 0.0166±0.0121 |
| affine/decoded/marginal | xor_eps02_reward | -0.005157 | +0.015015 | +0.000000 | 0.0033±0.0105 |
| affine/decoded/marginal | xor_v2_reward | +0.001343 | -0.007568 | +0.001129 | -0.0017±0.0051 |
| affine/decoded/marginal | full_occupancy | +0.006653 | -0.003326 | -0.002960 | 0.0001±0.0057 |
| affine/decoded/marginal | full_mirror | -0.002197 | +0.008118 | +0.004883 | 0.0036±0.0053 |
| mlp/exact/marginal | ladder_occupancy | +0.033508 | +0.052826 | +0.004639 | 0.0303±0.0243 |
| mlp/exact/marginal | ladder_mirror | +0.004364 | +0.001068 | +0.001160 | 0.0022±0.0019 |
| mlp/exact/marginal | moving_reward | +0.074097 | +0.002655 | +0.025421 | 0.0341±0.0365 |
| mlp/exact/marginal | moving_gap_closed | +0.369278 | +0.016219 | +0.138833 | 0.1748±0.1793 |
| mlp/exact/marginal | moving_f1_changed | -0.021210 | -0.014771 | -0.034363 | -0.0234±0.0100 |
| mlp/exact/marginal | xor_eps02_reward | +0.020325 | +0.022247 | +0.000000 | 0.0142±0.0123 |
| mlp/exact/marginal | xor_v2_reward | +0.003601 | +0.025330 | -0.021942 | 0.0023±0.0237 |
| mlp/exact/marginal | full_occupancy | +0.020050 | +0.006805 | +0.014252 | 0.0137±0.0066 |
| mlp/exact/marginal | full_mirror | -0.006927 | +0.003571 | +0.001923 | -0.0005±0.0056 |
| mlp/decoded/marginal | ladder_occupancy | +0.016815 | +0.024200 | +0.039307 | 0.0268±0.0115 |
| mlp/decoded/marginal | ladder_mirror | +0.019470 | +0.005829 | +0.001495 | 0.0089±0.0094 |
| mlp/decoded/marginal | moving_reward | +0.033539 | +0.013855 | +0.016479 | 0.0213±0.0107 |
| mlp/decoded/marginal | moving_gap_closed | +0.167148 | +0.084638 | +0.090000 | 0.1139±0.0462 |
| mlp/decoded/marginal | moving_f1_changed | -0.040985 | -0.030640 | -0.050476 | -0.0407±0.0099 |
| mlp/decoded/marginal | xor_eps02_reward | +0.037628 | +0.021088 | +0.000000 | 0.0196±0.0189 |
| mlp/decoded/marginal | xor_v2_reward | -0.004486 | +0.023315 | -0.018555 | 0.0001±0.0213 |
| mlp/decoded/marginal | full_occupancy | +0.011261 | +0.005951 | +0.011353 | 0.0095±0.0031 |
| mlp/decoded/marginal | full_mirror | -0.005035 | +0.015564 | -0.006348 | 0.0014±0.0123 |
| affine/decoded/joint | moving_reward | +0.002228 | +0.015045 | -0.049896 | -0.0109±0.0344 |
| affine/decoded/joint | moving_gap_closed | +0.011103 | +0.091909 | -0.272500 | -0.0565±0.1914 |
| affine/decoded/joint | moving_f1_changed | -0.010254 | -0.014069 | +0.009705 | -0.0049±0.0128 |
| affine/decoded/joint | xor_eps02_reward | +0.000000 | +0.000000 | +0.014008 | 0.0047±0.0081 |
| affine/decoded/joint | xor_v2_reward | -0.000366 | -0.008972 | +0.005890 | -0.0011±0.0075 |
| mlp/exact/joint | moving_reward | -0.038757 | -0.019897 | -0.036530 | -0.0317±0.0103 |
| mlp/exact/joint | moving_gap_closed | -0.193156 | -0.121551 | -0.199500 | -0.1714±0.0433 |
| mlp/exact/joint | moving_f1_changed | +0.074951 | +0.044617 | +0.074768 | 0.0648±0.0175 |
| mlp/exact/joint | xor_eps02_reward | +0.000000 | +0.000000 | +0.013092 | 0.0044±0.0076 |
| mlp/exact/joint | xor_v2_reward | -0.012634 | -0.010193 | +0.009460 | -0.0045±0.0121 |
| mlp/decoded/joint | moving_reward | -0.011169 | -0.042847 | -0.070190 | -0.0414±0.0295 |
| mlp/decoded/joint | moving_gap_closed | -0.055665 | -0.261745 | -0.383333 | -0.2336±0.1656 |
| mlp/decoded/joint | moving_f1_changed | -0.008698 | +0.022827 | +0.054352 | 0.0228±0.0315 |
| mlp/decoded/joint | xor_eps02_reward | +0.000000 | +0.000000 | +0.038269 | 0.0128±0.0221 |
| mlp/decoded/joint | xor_v2_reward | -0.002289 | -0.007385 | -0.009674 | -0.0064±0.0038 |

## All four arms reselected with the historical grid/rule on 16 episodes

| Encoder / source / coordinates | Ladder ↓ / ↓ | Moving reward ↑ | Moving gap ↑ | Moving F1 changed ↓ | XOR eps02 ↑ | XOR v2 ↑ | Full-State |
|---|---:|---:|---:|---:|---:|---:|---:|
| affine/exact/marginal | 0.0949±0.0178 / 0.0056±0.0026 | 0.9598±0.0402 | 0.4191±0.1604 | 0.1224±0.0103 | 0.4592±0.0174 | 0.5804±0.0245 | 0.0774±0.0039 / 0.4907±0.0357 |
| affine/decoded/marginal | 0.0877±0.0154 / 0.0086±0.0045 | 0.9754±0.0359 | 0.5032±0.1450 | 0.1341±0.0080 | 0.4836±0.0304 | 0.5828±0.0276 | 0.0898±0.0104 / 0.4943±0.0315 |
| mlp/exact/marginal | 0.1254±0.0290 / 0.0078±0.0045 | 1.0139±0.0212 | 0.7009±0.1291 | 0.0976±0.0141 | 0.4918±0.0194 | 0.5778±0.0191 | 0.0911±0.0064 / 0.4903±0.0319 |
| mlp/decoded/marginal | 0.1114±0.0183 / 0.0145±0.0118 | 1.0011±0.0179 | 0.6376±0.0649 | 0.1012±0.0320 | 0.5016±0.0287 | 0.5745±0.0094 | 0.0869±0.0052 / 0.4921±0.0237 |
| affine/exact/joint | 0.0949±0.0178 / 0.0056±0.0026 | 1.0014±0.0502 | 0.6521±0.2476 | 0.0990±0.0551 | 0.4748±0.0201 | 0.5561±0.0198 | 0.0774±0.0039 / 0.4907±0.0357 |
| affine/decoded/joint | 0.0877±0.0154 / 0.0086±0.0045 | 1.0001±0.0394 | 0.6387±0.1987 | 0.1132±0.0456 | 0.4660±0.0073 | 0.5652±0.0065 | 0.0898±0.0104 / 0.4943±0.0315 |
| mlp/exact/joint | 0.1254±0.0290 / 0.0078±0.0045 | 0.9649±0.0584 | 0.4451±0.3064 | 0.1589±0.0723 | 0.4660±0.0204 | 0.5454±0.0115 | 0.0911±0.0064 / 0.4903±0.0319 |
| mlp/decoded/joint | 0.1114±0.0183 / 0.0145±0.0118 | 0.9992±0.0335 | 0.6315±0.1676 | 0.1010±0.0486 | 0.4789±0.0218 | 0.5539±0.0203 | 0.0869±0.0052 / 0.4921±0.0237 |

### Paired differences versus historical affine/exact at its archived λ

Signs are raw new−historical differences (negative favors occupancy/spillover; positive favors reward/gap). Full-State mirror is nondirectional.

| Arm | Metric | Δ seed42 | Δ seed43 | Δ seed44 | Mean ± SD |
|---|---|---:|---:|---:|---:|
| affine/exact/marginal | ladder_occupancy | +0.000000 | +0.000000 | +0.000000 | 0.0000±0.0000 |
| affine/exact/marginal | ladder_mirror | +0.000000 | +0.000000 | +0.000000 | 0.0000±0.0000 |
| affine/exact/marginal | moving_reward | +0.000000 | +0.007263 | -0.029175 | -0.0073±0.0193 |
| affine/exact/marginal | moving_gap_closed | +0.000000 | +0.044370 | -0.159333 | -0.0383±0.1071 |
| affine/exact/marginal | moving_f1_changed | +0.000000 | +0.006287 | -0.052856 | -0.0155±0.0325 |
| affine/exact/marginal | xor_eps02_reward | -0.016663 | -0.046814 | -0.018799 | -0.0274±0.0168 |
| affine/exact/marginal | xor_v2_reward | -0.026001 | +0.023224 | +0.008545 | 0.0019±0.0253 |
| affine/exact/marginal | full_occupancy | +0.000000 | +0.000000 | +0.000000 | 0.0000±0.0000 |
| affine/exact/marginal | full_mirror | +0.000000 | +0.000000 | +0.000000 | 0.0000±0.0000 |
| affine/decoded/marginal | ladder_occupancy | -0.009674 | -0.002228 | -0.009827 | -0.0072±0.0043 |
| affine/decoded/marginal | ladder_mirror | +0.005188 | +0.002045 | +0.001740 | 0.0030±0.0019 |
| affine/decoded/marginal | moving_reward | +0.019196 | +0.017578 | -0.012054 | 0.0082±0.0176 |
| affine/decoded/marginal | moving_gap_closed | +0.095665 | +0.107383 | -0.065833 | 0.0457±0.0968 |
| affine/decoded/marginal | moving_f1_changed | +0.010437 | +0.021454 | -0.043304 | -0.0038±0.0346 |
| affine/decoded/marginal | xor_eps02_reward | -0.003418 | +0.015015 | -0.020752 | -0.0031±0.0179 |
| affine/decoded/marginal | xor_v2_reward | -0.023315 | +0.019409 | +0.016724 | 0.0043±0.0239 |
| affine/decoded/marginal | full_occupancy | +0.025360 | +0.014648 | -0.002960 | 0.0123±0.0143 |
| affine/decoded/marginal | full_mirror | -0.002197 | +0.008118 | +0.004883 | 0.0036±0.0053 |
| mlp/exact/marginal | ladder_occupancy | +0.033508 | +0.053406 | +0.004639 | 0.0305±0.0245 |
| mlp/exact/marginal | ladder_mirror | +0.004364 | +0.001068 | +0.001160 | 0.0022±0.0019 |
| mlp/exact/marginal | moving_reward | +0.086029 | +0.008850 | +0.045349 | 0.0467±0.0386 |
| mlp/exact/marginal | moving_gap_closed | +0.428745 | +0.054064 | +0.247667 | 0.2435±0.1874 |
| mlp/exact/marginal | moving_f1_changed | -0.038788 | -0.019012 | -0.063019 | -0.0403±0.0220 |
| mlp/exact/marginal | xor_eps02_reward | +0.025452 | +0.011353 | -0.021484 | 0.0051±0.0241 |
| mlp/exact/marginal | xor_v2_reward | +0.013245 | -0.000793 | -0.014526 | -0.0007±0.0139 |
| mlp/exact/marginal | full_occupancy | +0.020050 | +0.006805 | +0.014252 | 0.0137±0.0066 |
| mlp/exact/marginal | full_mirror | -0.006927 | +0.003571 | +0.001923 | -0.0005±0.0056 |
| mlp/decoded/marginal | ladder_occupancy | +0.016815 | +0.015198 | +0.017578 | 0.0165±0.0012 |
| mlp/decoded/marginal | ladder_mirror | +0.019470 | +0.005829 | +0.001495 | 0.0089±0.0094 |
| mlp/decoded/marginal | moving_reward | +0.062622 | +0.021240 | +0.018066 | 0.0340±0.0249 |
| mlp/decoded/marginal | moving_gap_closed | +0.312091 | +0.129754 | +0.098667 | 0.1802±0.1153 |
| mlp/decoded/marginal | moving_f1_changed | -0.035126 | -0.033691 | -0.041290 | -0.0367±0.0040 |
| mlp/decoded/marginal | xor_eps02_reward | +0.039124 | +0.029053 | -0.023407 | 0.0149±0.0336 |
| mlp/decoded/marginal | xor_v2_reward | -0.017578 | +0.023834 | -0.018188 | -0.0040±0.0241 |
| mlp/decoded/marginal | full_occupancy | +0.011261 | +0.005951 | +0.011353 | 0.0095±0.0031 |
| mlp/decoded/marginal | full_mirror | -0.005035 | +0.015564 | -0.006348 | 0.0014±0.0123 |
| affine/exact/joint | moving_reward | -0.046783 | +0.029999 | -0.032227 | -0.0163±0.0408 |
| affine/exact/joint | moving_gap_closed | -0.233156 | +0.183259 | -0.176000 | -0.0753±0.2257 |
| affine/exact/joint | moving_f1_changed | +0.055817 | -0.035736 | -0.063293 | -0.0144±0.0624 |
| affine/exact/joint | xor_eps02_reward | -0.024719 | +0.000000 | +0.018005 | -0.0022±0.0215 |
| affine/exact/joint | xor_v2_reward | +0.008362 | -0.042908 | +0.000000 | -0.0115±0.0275 |
| affine/decoded/joint | moving_reward | -0.022064 | +0.019043 | -0.049896 | -0.0176±0.0347 |
| affine/decoded/joint | moving_gap_closed | -0.109962 | +0.116331 | -0.272500 | -0.0887±0.1953 |
| affine/decoded/joint | moving_f1_changed | +0.039032 | -0.049347 | +0.009705 | -0.0002±0.0450 |
| affine/decoded/joint | xor_eps02_reward | -0.019287 | -0.012299 | -0.001495 | -0.0110±0.0090 |
| affine/decoded/joint | xor_v2_reward | +0.002167 | -0.005615 | -0.003815 | -0.0024±0.0041 |
| mlp/exact/joint | moving_reward | -0.040863 | -0.002136 | -0.115417 | -0.0528±0.0576 |
| mlp/exact/joint | moving_gap_closed | -0.203650 | -0.013050 | -0.630333 | -0.2823±0.3161 |
| mlp/exact/joint | moving_f1_changed | +0.031738 | -0.010925 | +0.115753 | 0.0455±0.0645 |
| mlp/exact/joint | xor_eps02_reward | -0.034790 | +0.000000 | +0.001770 | -0.0110±0.0206 |
| mlp/exact/joint | xor_v2_reward | -0.022949 | -0.020905 | -0.022797 | -0.0222±0.0011 |
| mlp/decoded/joint | moving_reward | -0.015900 | +0.010712 | -0.050171 | -0.0185±0.0305 |
| mlp/decoded/joint | moving_gap_closed | -0.079240 | +0.065436 | -0.274000 | -0.0959±0.1703 |
| mlp/decoded/joint | moving_f1_changed | -0.036652 | -0.034027 | +0.033417 | -0.0124±0.0397 |
| mlp/decoded/joint | xor_eps02_reward | -0.020813 | +0.000000 | +0.026306 | 0.0018±0.0236 |
| mlp/decoded/joint | xor_v2_reward | -0.024414 | -0.005310 | -0.011475 | -0.0137±0.0098 |
