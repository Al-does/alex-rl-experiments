"""Paper MESS3 transformer with a 64 -> 64 -> 3 Swish decoder."""

from __future__ import annotations

from harness.context import RunContext

from ..shared import run_decoder_replication


def run(context: RunContext):
    return run_decoder_replication(context, decoder_depth=2)
