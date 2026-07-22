# MESS3 reward-state Kelly/IQN battery

All conditions use 30 million environment steps and seed 42. Belief R² uses the action-aware predictive transducer target.

| arm | gamma | reward % | greedy reward % | global R² | fine R² |
|---|---:|---:|---:|---:|---:|
| PPO | 0 | 2.31 | 2.50 | 0.9148 | 0.2955 |
| PPO | 0.99 | 74.76 | 74.66 | 0.8526 | 0.7581 |
| IQN | 0 | 31.51 | 31.65 | 0.7923 | -0.6667 |
| IQN | 0.99 | 74.87 | 74.66 | 0.8523 | 0.7576 |
| Kelly | 0 | 6.44 | 6.19 | 0.9894 | 0.9431 |
| Kelly | 0.99 | 74.41 | 76.71 | 0.9228 | 0.8769 |
| Kelly + IQN | 0 | 2.81 | 1.45 | 0.9798 | 0.9105 |
| Kelly + IQN | 0.99 | 78.78 | 80.78 | 0.8653 | 0.7894 |
