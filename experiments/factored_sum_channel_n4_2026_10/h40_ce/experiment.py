"""N=4 sum channel (eps=0.5), 40% of the sequence pool held out, plain next-token CE, 400M env steps."""

from experiments.factored_sum_channel_n3_2026_10.heldout import run_heldout


def run(context):
    return run_heldout(context, n_factors=4, heldout_fraction=0.40, kelly=False)
