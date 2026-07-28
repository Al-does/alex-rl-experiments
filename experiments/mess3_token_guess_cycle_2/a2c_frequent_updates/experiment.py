"""One-million-step A2C pilot with frequent fresh-data optimizer updates.

This keeps the original gamma-zero A2C objective, LR, model, reward, and
strictly on-policy one-epoch training. It changes the fresh train batch from
32,768 to 128, the intervention that reduced held-out belief MSE from 0.00541
to 0.000853 by ~131k sampled steps in the diagnostic run. It also uses
``vf_loss_coeff=1.0`` so A2C's conventional half-MSE has the same effective
critic-gradient scale as PPO's raw-MSE loss with coefficient 0.5.
"""

from harness.context import RunContext

from experiments.mess3_token_guess_cycle_2.shared import run_condition


def run(context: RunContext):
    return run_condition(context, "a2c_frequent_updates")
