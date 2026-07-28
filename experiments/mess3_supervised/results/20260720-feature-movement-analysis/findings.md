# SGD versus Muon movement from initialization

Both completed large-batch runs used seed 0. Their step-0 model state
dictionaries are bit-identical, so all differences below start from the same
parameter initialization.

## Finding

The lower Muon belief-probe R² is not explained by Muon remaining in a lazier,
near-initialization regime. At the analyzed update 61,446:

- Relative displacement of all 2D parameters was 0.2523 for SGD and 146.9924
  for Muon. Muon therefore moved approximately 582.6 times farther by this
  measure.
- The combined 2D parameter norm was 1.0406 times its initial value for SGD and
  147.0078 times its initial value for Muon.
- Cosine similarity of the flattened 2D parameters to initialization was
  0.9702 for SGD and 0.0188 for Muon. The Muon result is therefore not merely
  radial norm growth.
- Centered linear CKA between the block-3 residual representation and its
  initialization was 0.6082 for SGD and 0.3438 for Muon. This scale-invariant
  metric also shows greater feature change under Muon.

Muon's held-out belief-probe R² peaked at 0.9921 near update 20,000 and ended at
0.9841. Over the same interval, its relative 2D displacement increased from
71.87 to 148.57. The observed trajectory is more consistent with excessive
continued movement under the constant Muon learning rate than with lazy
feature learning.

## Method

Parameter metrics use every model tensor in each retained checkpoint. The
representation metric uses the block-3 residual stream before the final
LayerNorm on 4,096 fixed contexts selected at evenly spaced indices from all
length-10 MESS3 contexts.

The machine-readable trajectory is in `feature_movement_analysis.json`.
`feature_movement.canvas.tsx` is a Cursor Canvas containing the charts and
interpretation. Reproduce the JSON with:

```bash
uv run python -m experiments.mess3_supervised.analyze_feature_movement \
  --output experiments/mess3_supervised/results/20260720-feature-movement-analysis/feature_movement_analysis.json
```

Parameter displacement and linear CKA are descriptive rather than a formal
neural-tangent-kernel test. A follow-up Muon run with a smaller learning rate,
learning-rate decay, or weight decay would more directly test whether its
late-training feature drift caused the R² regression.
