Final-checkpoint action shares (2.01B steps, seed 42), actions 0/1/2 = Noop / Tilt X / Tilt Y. Greedy is the argmax evaluation policy; sampled is the last training iteration's rollouts (T=1.5, plus the ε mixture in the ε arms).

| Arm | Greedy Noop / Tilt X / Tilt Y | Sampled Noop / Tilt X / Tilt Y |
|---|---|---|
| no ε v1 | 0.000 / 1.000 / 0.000 | 0.000 / 1.000 / 0.000 |
| no ε v2 | 0.594 / 0.406 / 0.000 | 0.615 / 0.384 / 0.000 |
| no ε v3 | 0.608 / 0.184 / 0.208 | 0.618 / 0.191 / 0.190 |
| ε=0.05 v1 | 0.000 / 1.000 / 0.000 | 0.017 / 0.967 / 0.017 |
| ε=0.05 v2 | 0.620 / 0.380 / 0.000 | 0.566 / 0.417 / 0.017 |
| ε=0.05 v3 | 0.609 / 0.205 / 0.186 | 0.552 / 0.235 / 0.213 |
