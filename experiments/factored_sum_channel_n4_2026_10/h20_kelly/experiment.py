"""N=4 sum channel (eps=0.5), 20% of the sequence pool held out, next-token CE + decoupled Kelly wager head, 400M env steps."""

from experiments.factored_sum_channel_n3_2026_10.heldout import run_heldout


def run(context):
    return run_heldout(context, n_factors=4, heldout_fraction=0.20, kelly=True)
