# FSQ single-step pulse and decay

Open-loop intact histories; no branch action enters the environment or its action features.
Coordinate differences are decoded at block 4. Signed and absolute lag curves are in JSON.
Rate is exp(slope) from log mean absolute effect; <1 decay, >1 growth; null means insufficient nonzero lags.
Fits exclude values below max(1e-12, peak*1e-8). This descriptive exponential fit is not a dynamical model.

| Pulse/site | Coordinate | Measured rate | Full filter rate | Quotient filter rate |
|---|---|---:|---:|---:|
| mirror/block_1 | rho | 1.00148 | 0.210398 | — |
| mirror/block_1 | delta | 0.992657 | 0.21595 | — |
| mirror/block_2 | rho | 0.974238 | 0.210398 | — |
| mirror/block_2 | delta | 0.971621 | 0.21595 | — |
| mirror/block_3 | rho | 1.00033 | 0.210398 | — |
| mirror/block_3 | delta | 0.999273 | 0.21595 | — |
| rho_flip/block_1 | rho | 0.986975 | 0.233243 | — |
| rho_flip/block_1 | delta | 0.985485 | 0.243097 | — |
| rho_flip/block_2 | rho | 0.98994 | 0.233243 | — |
| rho_flip/block_2 | delta | 0.976958 | 0.243097 | — |
| rho_flip/block_3 | rho | 0.999479 | 0.233243 | — |
| rho_flip/block_3 | delta | 0.999982 | 0.243097 | — |
