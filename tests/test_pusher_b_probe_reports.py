from __future__ import annotations

import json

import pytest

from experiments.pusher_b.probe_analysis import plot_combined


def test_combined_probe_data_contains_six_series_per_preset():
    combined = plot_combined.load_combined_data()

    assert tuple(combined) == ("b10", "b90")
    for series in combined.values():
        assert tuple(series) == plot_combined.SERIES_ORDER
        assert all(series[key] for key in plot_combined.SERIES_ORDER)

    b10_token_guess = combined["b10"]["token_guess_rl"]
    assert len(b10_token_guess) == 13
    assert b10_token_guess[-1]["tokens_consumed"] == 115_901_216
    assert b10_token_guess[-1]["one_minus_r_squared"] == pytest.approx(
        0.03834404294707883
    )
    assert combined["b90"]["reward_b_variant_3"][-1][
        "one_minus_r_squared"
    ] == pytest.approx(0.08236850000169138)


def test_combined_probe_report_writes_both_plots_and_summary(tmp_path):
    combined = plot_combined.load_combined_data()

    for preset in ("b10", "b90"):
        output = tmp_path / f"{preset}.png"
        plot_combined.plot_preset(preset, combined[preset], output)
        assert output.stat().st_size > 0

    summary = tmp_path / "summary.json"
    plot_combined.write_summary(combined, summary)
    written = json.loads(summary.read_text())
    assert tuple(written["presets"]) == ("b10", "b90")
    assert written["x_axis"] == "tokens_consumed"
    assert written["y_axis"] == "one_minus_r_squared"
