# FSQ single-step pulse and decay

Open-loop intact histories; no branch action enters the environment or its action features.
Coordinate differences are decoded at block 4. Signed and absolute lag curves are in JSON.
Rate is exp(slope) from log mean absolute effect; <1 decay, >1 growth; null means insufficient nonzero lags.
Fits exclude values below max(1e-12, peak*1e-8). This descriptive exponential fit is not a dynamical model.

| Pulse/site | Coordinate | Measured rate | Full filter rate | Quotient filter rate |
|---|---|---:|---:|---:|
| mirror/block_1 | rho | 1.0161 | 0.210398 | — |
| mirror/block_1 | delta | 1.02767 | 0.21595 | — |
| mirror/block_2 | rho | 0.983232 | 0.210398 | — |
| mirror/block_2 | delta | 0.995117 | 0.21595 | — |
| mirror/block_3 | rho | 1.00245 | 0.210398 | — |
| mirror/block_3 | delta | 0.99902 | 0.21595 | — |
| rho_flip/block_1 | rho | 0.981958 | 0.233243 | — |
| rho_flip/block_1 | delta | 0.99206 | 0.243097 | — |
| rho_flip/block_2 | rho | 1.00464 | 0.233243 | — |
| rho_flip/block_2 | delta | 0.972508 | 0.243097 | — |
| rho_flip/block_3 | rho | 1.00255 | 0.233243 | — |
| rho_flip/block_3 | delta | 0.998152 | 0.243097 | — |
