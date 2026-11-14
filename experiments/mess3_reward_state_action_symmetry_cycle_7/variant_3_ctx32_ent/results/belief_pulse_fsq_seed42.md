# FSQ single-step pulse and decay

Open-loop intact histories; no branch action enters the environment or its action features.
Coordinate differences are decoded at block 4. Signed and absolute lag curves are in JSON.
Rate is exp(slope) from log mean absolute effect; <1 decay, >1 growth; null means insufficient nonzero lags.
Fits exclude values below max(1e-12, peak*1e-8). This descriptive exponential fit is not a dynamical model.

| Pulse/site | Coordinate | Measured rate | Full filter rate | Quotient filter rate |
|---|---|---:|---:|---:|
| mirror/block_1 | rho | 1.00247 | 0.210398 | — |
| mirror/block_1 | delta | 0.968164 | 0.21595 | — |
| mirror/block_2 | rho | 0.984659 | 0.210398 | — |
| mirror/block_2 | delta | 0.939037 | 0.21595 | — |
| mirror/block_3 | rho | 1.00396 | 0.210398 | — |
| mirror/block_3 | delta | 0.997918 | 0.21595 | — |
| rho_flip/block_1 | rho | 0.992672 | 0.233243 | — |
| rho_flip/block_1 | delta | 1.00177 | 0.243097 | — |
| rho_flip/block_2 | rho | 0.953943 | 0.233243 | — |
| rho_flip/block_2 | delta | 0.935412 | 0.243097 | — |
| rho_flip/block_3 | rho | 0.997837 | 0.233243 | — |
| rho_flip/block_3 | delta | 0.998252 | 0.243097 | — |
