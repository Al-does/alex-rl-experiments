# FSQ single-step pulse and decay

Open-loop intact histories; no branch action enters the environment or its action features.
Coordinate differences are decoded at block 4. Signed and absolute lag curves are in JSON.
Rate is exp(slope) from log mean absolute effect; <1 decay, >1 growth; null means insufficient nonzero lags.
Fits exclude values below max(1e-12, peak*1e-8). This descriptive exponential fit is not a dynamical model.

| Pulse/site | Coordinate | Measured rate | Full filter rate | Quotient filter rate |
|---|---|---:|---:|---:|
| mirror/block_1 | rho | 0.999081 | 0.210398 | — |
| mirror/block_1 | delta | 0.972113 | 0.21595 | — |
| mirror/block_2 | rho | 0.962444 | 0.210398 | — |
| mirror/block_2 | delta | 0.988872 | 0.21595 | — |
| mirror/block_3 | rho | 0.996393 | 0.210398 | — |
| mirror/block_3 | delta | 0.996523 | 0.21595 | — |
| rho_flip/block_1 | rho | 1.00578 | 0.233243 | — |
| rho_flip/block_1 | delta | 0.962234 | 0.243097 | — |
| rho_flip/block_2 | rho | 1.00941 | 0.233243 | — |
| rho_flip/block_2 | delta | 0.980596 | 0.243097 | — |
| rho_flip/block_3 | rho | 0.996825 | 0.233243 | — |
| rho_flip/block_3 | delta | 0.997957 | 0.243097 | — |
