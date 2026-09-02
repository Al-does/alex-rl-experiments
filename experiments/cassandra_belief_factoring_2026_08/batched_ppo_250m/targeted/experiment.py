"""250M-step device-batched PPO with component-targeted actions."""

from __future__ import annotations

from harness.context import RunContext

from ..shared import build_config as build_recipe_config
from ..shared import run_recipe


ACTION_SCOPE = "targeted"
CONDITION = "batched_ppo_250m_symbol_targeted"


def build_config(context: RunContext):
    return build_recipe_config(context, action_scope=ACTION_SCOPE)


def run(context: RunContext):
    return run_recipe(
        context,
        action_scope=ACTION_SCOPE,
        condition=CONDITION,
    )


__all__ = ["ACTION_SCOPE", "CONDITION", "build_config", "run"]
