# RSQ single-step pulse and decay

Open-loop intact histories; no branch action enters the environment or its action features.
Coordinate differences are decoded at block 4. Signed and absolute lag curves are in JSON.
Rate is exp(slope) from log mean absolute effect; <1 decay, >1 growth; null means insufficient nonzero lags.
Fits exclude values below max(1e-12, peak*1e-8). This descriptive exponential fit is not a dynamical model.

| Pulse/site | Coordinate | Measured rate | Full filter rate | Quotient filter rate |
|---|---|---:|---:|---:|
| mirror/block_1 | rho | 0.977479 | 0.191427 | — |
| mirror/block_1 | delta | 0.986377 | 0.193226 | — |
| mirror/block_2 | rho | 0.976405 | 0.191427 | — |
| mirror/block_2 | delta | 0.945772 | 0.193226 | — |
| mirror/block_3 | rho | 0.997311 | 0.191427 | — |
| mirror/block_3 | delta | 1.00009 | 0.193226 | — |
| rho_flip/block_1 | rho | 0.981802 | 0.191202 | 0.100037 |
| rho_flip/block_1 | delta | 0.988982 | 0.225084 | — |
| rho_flip/block_2 | rho | 0.99358 | 0.191202 | 0.100037 |
| rho_flip/block_2 | delta | 0.981458 | 0.225084 | — |
| rho_flip/block_3 | rho | 0.996119 | 0.191202 | 0.100037 |
| rho_flip/block_3 | delta | 0.997443 | 0.225084 | — |
