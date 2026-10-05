# Factored sum channel, N=3 — held-out ladder, CE vs CE+Kelly

Repeats `pusher_b_jax_2026_10/heldout_ladder` on a factored process where a
factored representation is not enough: 3 sharp Mess3 factors (alpha=0.85,
x=0.05), observed as the sub-token tuple (27 tokens) w.p. 1-eps or only as
the sum mod 3 (3 tokens) w.p. eps=0.5. Vocab 31 incl. BOS; joint belief 27
states (26 dims) vs 6 dims for per-factor beliefs.

- `process.py` — the channel, exact Bayes filter and factored
  (product-of-marginals) filter.
- `heldout.py` — trainer and metrics (shared with the N=4 study).
- `plot_ladder.py` — overlays CE/Kelly per rung.
- Leaves: `h20|h40|h80 _ ce|kelly` (eps=0.5) and control `eps0_h20_ce|kelly`.
- [PREREGISTRATION.md](PREREGISTRATION.md) — success criteria.

Each run writes `curve.png` (joint probe 1-R^2 + excess CE) and
`diagnostics.png` (joint / marginal / residual probes, effective dimension,
excess CE, perplexity with Bayes and factored-filter lines, guess hit rate,
Kelly wager). Smoke: `uv run pytest -q tests/test_factored_sum_channel_2026_10.py`.
