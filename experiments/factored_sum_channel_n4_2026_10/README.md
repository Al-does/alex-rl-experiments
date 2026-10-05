# Factored sum channel, N=4 — held-out ladder, CE vs CE+Kelly

Identical to `factored_sum_channel_n3_2026_10` (same trainer, channel, leaves
and metrics) with 4 factors: 81 tuple + 3 sum tokens, vocab 85 incl. BOS;
joint belief 81 states (80 dims, still under d128) vs 8 dims for per-factor
beliefs. The factored filter's gap to Bayes is smaller (~0.005 nats/token vs
~0.008 at N=3).

This is the replication arm of the shared pre-registration:
[../factored_sum_channel_n3_2026_10/PREREGISTRATION.md](../factored_sum_channel_n3_2026_10/PREREGISTRATION.md).
