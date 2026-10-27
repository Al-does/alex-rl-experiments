"""Strict mapping from the RLlib PPO encoder state to the JAX trunk."""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np

from experiments.pusher_b_jax_2026_10.model import ModelSpec


def torch_encoder_to_jax(
    state: Mapping[str, np.ndarray],
    spec: ModelSpec,
) -> dict:
    """Convert only the trunk; non-encoder keys are intentionally ignored."""

    encoder_state = {key: value for key, value in state.items() if key.startswith("encoder.")}
    expected = {"encoder.input_embedding.weight", "encoder.bos_embedding", "encoder.final_norm.weight"}
    for layer in range(spec.n_layers):
        prefix = f"encoder.blocks.{layer}."
        expected.update(
            {
                f"{prefix}attention_norm.weight",
                f"{prefix}attention.qkv.weight",
                f"{prefix}attention.qkv.bias",
                f"{prefix}attention.output.weight",
                f"{prefix}attention.output.bias",
                f"{prefix}mlp_norm.weight",
                f"{prefix}mlp.gate.weight",
                f"{prefix}mlp.gate.bias",
                f"{prefix}mlp.value.weight",
                f"{prefix}mlp.value.bias",
                f"{prefix}mlp.output.weight",
                f"{prefix}mlp.output.bias",
            }
        )
    missing = expected.difference(encoder_state)
    if missing:
        raise ValueError(f"source encoder is missing keys: {sorted(missing)}")
    unexpected = set(encoder_state).difference(expected)
    if unexpected:
        raise ValueError(f"unconsumed source encoder keys: {sorted(unexpected)}")

    def array(key: str, shape: tuple[int, ...]) -> jnp.ndarray:
        value = np.asarray(encoder_state[key])
        if value.shape != shape:
            raise ValueError(
                f"encoder shape mismatch for {key}: {value.shape} != {shape}"
            )
        return jnp.asarray(value)

    d, m = spec.d_model, spec.d_mlp
    embedding = np.zeros((spec.vocab, d), dtype=np.float32)
    embedding[:2] = array("encoder.input_embedding.weight", (d, 2)).T
    embedding[2] = array("encoder.bos_embedding", (d,))
    if spec.vocab != 3:
        raise ValueError(f"guess-or-pass encoder requires vocab=3, got {spec.vocab}")

    blocks = []
    for layer in range(spec.n_layers):
        prefix = f"encoder.blocks.{layer}."

        def linear(
            name: str,
            fan_in: int,
            fan_out: int,
            *,
            prefix: str = prefix,
        ) -> dict[str, jnp.ndarray]:
            return {
                "w": array(f"{prefix}{name}.weight", (fan_out, fan_in)).T,
                "b": array(f"{prefix}{name}.bias", (fan_out,)),
            }

        blocks.append(
            {
                "attn_norm": array(f"{prefix}attention_norm.weight", (d,)),
                "qkv": linear("attention.qkv", d, 3 * d),
                "out": linear("attention.output", d, d),
                "mlp_norm": array(f"{prefix}mlp_norm.weight", (d,)),
                "gate": linear("mlp.gate", d, m),
                "up": linear("mlp.value", d, m),
                "down": linear("mlp.output", m, d),
            }
        )
    return {
        "embed": jnp.asarray(embedding),
        "blocks": blocks,
        "final_norm": array("encoder.final_norm.weight", (d,)),
    }
