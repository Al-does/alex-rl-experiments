# FSQ single-step pulse and decay

Open-loop intact histories; no branch action enters the environment or its action features.
Coordinate differences are decoded at block 4. Signed and absolute lag curves are in JSON.
Rate is exp(slope) from log mean absolute effect; <1 decay, >1 growth; null means insufficient nonzero lags.
Fits exclude values below max(1e-12, peak*1e-8). This descriptive exponential fit is not a dynamical model.

| Pulse/site | Coordinate | Measured rate | Full filter rate | Quotient filter rate |
|---|---|---:|---:|---:|
| mirror/block_1 | rho | 0.981234 | 0.210398 | — |
| mirror/block_1 | delta | 0.948915 | 0.21595 | — |
| mirror/block_2 | rho | 0.970402 | 0.210398 | — |
| mirror/block_2 | delta | 0.983886 | 0.21595 | — |
| mirror/block_3 | rho | 1.00038 | 0.210398 | — |
| mirror/block_3 | delta | 1.00159 | 0.21595 | — |
| rho_flip/block_1 | rho | 0.96614 | 0.233243 | — |
| rho_flip/block_1 | delta | 0.962718 | 0.243097 | — |
| rho_flip/block_2 | rho | 0.969263 | 0.233243 | — |
| rho_flip/block_2 | delta | 0.982078 | 0.243097 | — |
| rho_flip/block_3 | rho | 1.00032 | 0.233243 | — |
| rho_flip/block_3 | delta | 1.00145 | 0.243097 | — |
