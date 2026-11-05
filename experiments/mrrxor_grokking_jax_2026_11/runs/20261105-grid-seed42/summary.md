# MRRXOR PPO sweep

Sorted descending by final held-out random-bit accuracy, then held-out expected reward.
One context seed; descriptive comparisons, not confirmation of grokking.

| Arm | Steps | Train random | Held-out random | Held-out reward | Train excess CE | Held-out excess CE | NTP-PC 1−R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| n16384 | 100003840 | 0.529834 | 0.528594 | 0.633746 | 1.491297 | 1.487415 | 0.467201 |
| n4096 | 100003840 | 0.525165 | 0.527892 | 0.635865 | 1.498468 | 1.482273 | 0.428230 |
| n256 | 100003840 | 0.540330 | 0.527246 | 0.632664 | 1.517696 | 1.545607 | 0.432183 |
| n1024 | 100003840 | 0.502897 | 0.501637 | 0.501121 | 0.383447 | 0.391553 | 0.957573 |
