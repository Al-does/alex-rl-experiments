# Cycle 4 belief-symmetry probe campaign

Primary comparisons use held-out global MSE ratio. Raw MSE and R² are retained in `campaign_summary.json`; error bars in the figure are seed SD.

The initial checkpoint is a random-network floor, not a zero-information control. Each target also has its own held-out permutation null.

## Symmetry decomposition

- Variant 1, `symmetric_b2`: final normalized MSE 0.0445 ± 0.0045, at or above the random-network value 0.0258 ± 0.0128; permutation-null median ratio 1.0278 ± 0.0077.
- Variant 1, `antisymmetric_b0_minus_b1`: final normalized MSE 0.0657 ± 0.0053, at or above the random-network value 0.0393 ± 0.0152; permutation-null median ratio 1.0294 ± 0.0220.
- Variant 2, `symmetric_b2`: final normalized MSE 0.0048 ± 0.0008, below the random-network value 0.0227 ± 0.0152; permutation-null median ratio 1.0263 ± 0.0139.
- Variant 2, `antisymmetric_b0_minus_b1`: final normalized MSE 0.0456 ± 0.0060, at or above the random-network value 0.0372 ± 0.0166; permutation-null median ratio 1.0131 ± 0.0058.
- Variant 3, `symmetric_b2`: final normalized MSE 0.0074 ± 0.0017, below the random-network value 0.0503 ± 0.0432; permutation-null median ratio 1.0237 ± 0.0144.
- Variant 3, `antisymmetric_b0_minus_b1`: final normalized MSE 0.0104 ± 0.0027, below the random-network value 0.0345 ± 0.0137; permutation-null median ratio 1.0160 ± 0.0048.

## Coarse filter versus projected full filter

Deltas are `coarse - symmetric`, paired by model seed. Negative values favor the cheap coarse-filter target; positive values favor the symmetric projection of the full filter. With five seeds these comparisons are descriptive.

- Variant 1: normalized delta -0.0036 (bootstrap 95% CI [-0.0037, -0.0035]); raw-MSE delta -0.000571. The point estimate favors the coarse filter.
- Variant 2: normalized delta -0.0032 (bootstrap 95% CI [-0.0034, -0.0030]); raw-MSE delta -0.000431. The point estimate favors the coarse filter.
