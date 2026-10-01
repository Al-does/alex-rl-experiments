"""Run a small checkpoint probe set while full training is still in flight."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from harness.seeding import named_seed_sequences, seed_sequence_to_int

from .analysis import analyze_checkpoint
from .shared import _STREAM_KEYS


def _largest_power_of_two_at_most(value: int) -> int:
    if value <= 0:
        return 0
    update = 1
    while update * 2 <= value:
        update *= 2
    return update


def _resolve_updates(
    *,
    current_update: int,
    checkpoint_dir: Path,
    include_init: bool,
    include_midpoint: bool,
    include_latest: bool,
) -> list[int]:
    available = sorted(
        int(path.stem.removeprefix("update_"))
        for path in checkpoint_dir.glob("update_*.pt")
    )
    if not available:
        raise FileNotFoundError(f"no checkpoints under {checkpoint_dir}")

    chosen: list[int] = []
    if include_init:
        chosen.append(0)
    if include_midpoint:
        midpoint = _largest_power_of_two_at_most(current_update // 2)
        if midpoint not in available:
            midpoint = max(
                (update for update in available if update <= current_update // 2),
                default=available[0],
            )
        chosen.append(midpoint)
    if include_latest:
        latest_path = checkpoint_dir / "latest.pt"
        if latest_path.is_file():
            payload = torch.load(
                latest_path,
                map_location="cpu",
                weights_only=False,
            )
            latest = int(payload["update"])
        else:
            latest = max(available)
        if latest not in available:
            latest = max(update for update in available if update <= latest)
        chosen.append(latest)

    deduped = sorted(set(chosen))
    missing = [update for update in deduped if not (checkpoint_dir / f"update_{update:06d}.pt").is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing checkpoint file(s) for updates {missing} in {checkpoint_dir}"
        )
    return deduped


def run_midrun_probes(
    *,
    factor_count: int,
    seed: int,
    checkpoint_dir: Path,
    results_dir: Path,
    current_update: int,
    device: str,
) -> list[dict]:
    """Probe initialization, midpoint, and latest retained checkpoints."""

    probe_seed = seed_sequence_to_int(
        named_seed_sequences(seed, _STREAM_KEYS)["checkpoint_probes"]
    )
    torch_device = torch.device(device)
    updates = _resolve_updates(
        current_update=current_update,
        checkpoint_dir=checkpoint_dir,
        include_init=True,
        include_midpoint=True,
        include_latest=True,
    )
    reports = []
    for update in updates:
        probe_dir = results_dir / "checkpoint_probes" / f"midrun_updates_{update:06d}"
        reports.append(
            analyze_checkpoint(
                checkpoint=checkpoint_dir / f"update_{update:06d}.pt",
                factor_count=factor_count,
                update=update,
                seed=probe_seed,
                smoke=False,
                device=torch_device,
                results_dir=probe_dir,
            )
        )
    summary_path = results_dir / "midrun_probe_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    summary_path.write_text(
        json.dumps(
            {
                "factor_count": factor_count,
                "seed": seed,
                "current_update": current_update,
                "probed_updates": updates,
                "reports": [
                    {
                        "update": report["update"],
                        "path": str(
                            results_dir
                            / "checkpoint_probes"
                            / f"midrun_updates_{report['update']:06d}"
                            / "probe_battery.json"
                        ),
                    }
                    for report in reports
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-count", type=int, choices=(2, 3), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--current-update", type=int, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args(argv)
    reports = run_midrun_probes(
        factor_count=args.factor_count,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        current_update=args.current_update,
        device=args.device,
    )
    for report in reports:
        print(
            f"update={report['update']:6d} "
            f"loss={report['next_token_prediction']['loss_nats']:.4f} "
            f"acc={report['next_token_prediction']['accuracy']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
