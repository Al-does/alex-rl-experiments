# RSQ single-step pulse and decay

Open-loop intact histories; no branch action enters the environment or its action features.
Coordinate differences are decoded at block 4. Signed and absolute lag curves are in JSON.
Rate is exp(slope) from log mean absolute effect; <1 decay, >1 growth; null means insufficient nonzero lags.
Fits exclude values below max(1e-12, peak*1e-8). This descriptive exponential fit is not a dynamical model.

| Pulse/site | Coordinate | Measured rate | Full filter rate | Quotient filter rate |
|---|---|---:|---:|---:|
| mirror/block_1 | rho | 0.979581 | 0.191427 | — |
| mirror/block_1 | delta | 0.990105 | 0.193226 | — |
| mirror/block_2 | rho | 1.01617 | 0.191427 | — |
| mirror/block_2 | delta | 0.943265 | 0.193226 | — |
| mirror/block_3 | rho | 1.00398 | 0.191427 | — |
| mirror/block_3 | delta | 1.00503 | 0.193226 | — |
| rho_flip/block_1 | rho | 0.976124 | 0.191202 | 0.100037 |
| rho_flip/block_1 | delta | 0.978774 | 0.225084 | — |
| rho_flip/block_2 | rho | 0.960474 | 0.191202 | 0.100037 |
| rho_flip/block_2 | delta | 0.962165 | 0.225084 | — |
| rho_flip/block_3 | rho | 0.997072 | 0.191202 | 0.100037 |
| rho_flip/block_3 | delta | 0.999149 | 0.225084 | — |
