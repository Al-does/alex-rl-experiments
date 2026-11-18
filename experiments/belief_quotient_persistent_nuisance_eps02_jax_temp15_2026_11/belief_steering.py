"""Belief steering and pulse tests on the epsilon-0.02 ladder transformers.

Reuses the persistent-nuisance v2 protocol (affine maps at four post-block
residuals and the final LayerNorm, point/mirror/rho-flip/rho-point targets,
decoder-orthogonal controls, norm-matched delta, single-write pulses) with this
study's emission matrix for every exact-filter reference. One invocation runs
one variant over several training seeds so the compiled rollouts are reused.
"""
import argparse
from pathlib import Path

from experiments.belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11 import environment
from experiments.belief_quotient_persistent_nuisance_v2_jax_temp15_2026_11 import belief_steering as v2

STUDIES = ('belief_quotient_persistent_nuisance_eps02_jax_temp15_2026_11',
           'belief_quotient_persistent_nuisance_eps02_jax_temp15_cont4b_2026_11')
ROOT = Path(__file__).resolve().parents[2]


def report_path(study, variant, seed):
    return ROOT/'experiments'/study/f'variant_{variant}_transformer'/'results'/f'belief_steering_v2_seed{seed}_stable.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=STUDIES, default=STUDIES[0])
    parser.add_argument('--variant', choices=(1, 2, 3), type=int, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    for seed in args.seeds:
        if not args.smoke and report_path(args.study, args.variant, seed).exists():
            print(f'seed {seed}: compact report exists, skipping', flush=True)
            continue
        print(f'=== {args.study} variant {args.variant} seed {seed} ===', flush=True)
        v2.run(args.study, args.variant, args.smoke, seed=seed, environment=environment)


if __name__ == '__main__':
    main()
