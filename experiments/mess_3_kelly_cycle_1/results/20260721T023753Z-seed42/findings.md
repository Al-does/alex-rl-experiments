# MESS3 Kelly cycle 1

All four conditions train from scratch without predictive auxiliary loss.

| condition | belief R² | token accuracy | mean wager | expected log growth | wager RMSE | collapse |
|---|---:|---:|---:|---:|---:|---:|
| fixed_full | 0.9281 | 0.6754 | 0.9999 | -2.240269 | 0.5278 | false |
| policy_implied_kelly | 0.8829 | 0.6489 | 0.6761 | 0.042845 | 0.3393 | false |
| learned_kelly | 0.9490 | 0.5210 | 0.3628 | 0.184927 | 0.0960 | false |
| bayes_oracle | 0.8770 | 0.6729 | 0.5108 | 0.287087 | 0.0000 | false |
