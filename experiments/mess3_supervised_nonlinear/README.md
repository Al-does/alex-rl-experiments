# MESS3 supervised nonlinear decoders

These three conditions keep the paper-faithful supervised MESS3 transformer
and SGD recipe fixed while varying its next-token head:

- `linear_decoder_control`: `64 -> 3`, matching the paper
- `two_layer_decoder`: `64 -> 64 -> 3`
- `four_layer_decoder`: `64 -> 64 -> 64 -> 64 -> 3`

Every nonlinear hidden decoder layer uses Swish (`torch.nn.SiLU`); output
layers have no activation. All conditions retain the paper analysis: an affine OLS belief
probe fitted to the final transformer-block residual before the final
LayerNorm. Thus prediction-head nonlinearity varies while the diagnostic belief
decoder remains linear.

Smoke the conditions from the experiment repository:

```bash
uv run rl-harness \
  experiments.mess3_supervised_nonlinear.linear_decoder_control.experiment \
  --smoke --hardware-profile cpu

uv run rl-harness \
  experiments.mess3_supervised_nonlinear.two_layer_decoder.experiment \
  --smoke --hardware-profile cpu

uv run rl-harness \
  experiments.mess3_supervised_nonlinear.four_layer_decoder.experiment \
  --smoke --hardware-profile cpu
```
