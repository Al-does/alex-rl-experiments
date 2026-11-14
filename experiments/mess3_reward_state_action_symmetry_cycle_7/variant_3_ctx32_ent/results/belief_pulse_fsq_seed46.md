# FSQ single-step pulse and decay

Open-loop intact histories; no branch action enters the environment or its action features.
Coordinate differences are decoded at block 4. Signed and absolute lag curves are in JSON.
Rate is exp(slope) from log mean absolute effect; <1 decay, >1 growth; null means insufficient nonzero lags.
Fits exclude values below max(1e-12, peak*1e-8). This descriptive exponential fit is not a dynamical model.

| Pulse/site | Coordinate | Measured rate | Full filter rate | Quotient filter rate |
|---|---|---:|---:|---:|
| mirror/block_1 | rho | 1.00482 | 0.210398 | — |
| mirror/block_1 | delta | 1.00942 | 0.21595 | — |
| mirror/block_2 | rho | 0.954448 | 0.210398 | — |
| mirror/block_2 | delta | 0.961297 | 0.21595 | — |
| mirror/block_3 | rho | 1.00068 | 0.210398 | — |
| mirror/block_3 | delta | 1.00068 | 0.21595 | — |
| rho_flip/block_1 | rho | 0.984564 | 0.233243 | — |
| rho_flip/block_1 | delta | 0.993828 | 0.243097 | — |
| rho_flip/block_2 | rho | 0.99348 | 0.233243 | — |
| rho_flip/block_2 | delta | 0.994443 | 0.243097 | — |
| rho_flip/block_3 | rho | 1.00016 | 0.233243 | — |
| rho_flip/block_3 | delta | 1.0027 | 0.243097 | — |
