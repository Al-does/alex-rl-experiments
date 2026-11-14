# RSQ single-step pulse and decay

Open-loop intact histories; no branch action enters the environment or its action features.
Coordinate differences are decoded at block 4. Signed and absolute lag curves are in JSON.
Rate is exp(slope) from log mean absolute effect; <1 decay, >1 growth; null means insufficient nonzero lags.
Fits exclude values below max(1e-12, peak*1e-8). This descriptive exponential fit is not a dynamical model.

| Pulse/site | Coordinate | Measured rate | Full filter rate | Quotient filter rate |
|---|---|---:|---:|---:|
| mirror/block_1 | rho | 0.998245 | 0.191427 | — |
| mirror/block_1 | delta | 1.00787 | 0.193226 | — |
| mirror/block_2 | rho | 0.92692 | 0.191427 | — |
| mirror/block_2 | delta | 0.899984 | 0.193226 | — |
| mirror/block_3 | rho | 0.997534 | 0.191427 | — |
| mirror/block_3 | delta | 0.997574 | 0.193226 | — |
| rho_flip/block_1 | rho | 0.996911 | 0.191202 | 0.100037 |
| rho_flip/block_1 | delta | 0.991273 | 0.225084 | — |
| rho_flip/block_2 | rho | 0.960164 | 0.191202 | 0.100037 |
| rho_flip/block_2 | delta | 0.955378 | 0.225084 | — |
| rho_flip/block_3 | rho | 0.996888 | 0.191202 | 0.100037 |
| rho_flip/block_3 | delta | 0.994261 | 0.225084 | — |
