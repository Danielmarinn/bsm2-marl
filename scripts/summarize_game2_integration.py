"""
Summarise and sanity-check a Game2 integration log.

This is an integration-test helper, not a thesis metric exporter. It checks
whether the MATLAB/Python bridge produced bounded four-action rows, stable
augmented-state dimensions, and monotonically increasing simulation time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_DIMS = {
    "game2_context_dim": 28,
    "qec_aug_dim": 33,
    "qint_aug_dim": 32,
    "do_aug_dim": 33,
    "qw_aug_dim": 33,
}

ACTION_BOUNDS = {
    "Qec": (0.0, 5.0),
    "Qint": (5_000.0, 61_944.0),
    "SO4ref": (0.0, 10.0),
    "Qw": (0.0, 450.0),
}


def parse_args() -> argparse.Namespace:
    default_log = Path(__file__).resolve().parents[1] / "logs" / "game2_integration_log.csv"
    parser = argparse.ArgumentParser(description="Summarise Game2 integration log.")
    parser.add_argument(
        "log",
        nargs="?",
        default=str(default_log),
        help="Path to game2_integration_log.csv.",
    )
    parser.add_argument(
        "--latest-run",
        action="store_true",
        help="If the log contains multiple run_id values, validate only the last run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.log)
    if not path.exists():
        print(f"[GAME2-CHECK] missing log: {path}")
        return 2

    log = pd.read_csv(path)
    if log.empty:
        print(f"[GAME2-CHECK] empty log: {path}")
        return 2

    if args.latest_run and "run_id" in log.columns:
        run_ids = log["run_id"].dropna()
        if not run_ids.empty:
            latest_run_id = run_ids.iloc[-1]
            log = log[log["run_id"] == latest_run_id].copy()
            print(f"[GAME2-CHECK] filtered to latest run_id: {latest_run_id}")

    print(f"[GAME2-CHECK] file: {path}")
    print(f"[GAME2-CHECK] rows: {len(log)}")

    bad = False

    if "run_id" in log.columns:
        run_ids = log["run_id"].dropna().unique()
        print(f"[GAME2-CHECK] run ids: {', '.join(map(str, run_ids))}")

    if "time" in log.columns:
        times = pd.to_numeric(log["time"], errors="coerce")
        print(
            "[GAME2-CHECK] time range: "
            f"{times.min():.6g} -> {times.max():.6g} days"
        )
        nonmonotonic = int((times.diff().dropna() < 0).sum())
        if nonmonotonic:
            print(f"[GAME2-CHECK] FAIL: non-monotonic time jumps: {nonmonotonic}")
            bad = True
        else:
            print("[GAME2-CHECK] OK: simulation time is monotonic")

    for col, expected in EXPECTED_DIMS.items():
        values = sorted(pd.to_numeric(log[col], errors="coerce").dropna().unique())
        if values != [expected]:
            print(f"[GAME2-CHECK] FAIL: {col} values {values}, expected {expected}")
            bad = True
        else:
            print(f"[GAME2-CHECK] OK: {col} = {expected}")

    for col, (low, high) in ACTION_BOUNDS.items():
        values = pd.to_numeric(log[col], errors="coerce")
        out_of_bounds = values[(values < low) | (values > high)]
        print(
            f"[GAME2-CHECK] {col}: min={values.min():.6g}, "
            f"max={values.max():.6g}, bounds=[{low:.6g}, {high:.6g}]"
        )
        if len(out_of_bounds):
            print(f"[GAME2-CHECK] FAIL: {col} has {len(out_of_bounds)} out-of-bounds rows")
            bad = True

    if "mode" in log.columns:
        modes = ", ".join(map(str, log["mode"].dropna().unique()))
        print(f"[GAME2-CHECK] modes: {modes}")

    if np.isfinite(pd.to_numeric(log.get("time"), errors="coerce")).all():
        approx_hours = (
            pd.to_numeric(log["time"], errors="coerce").iloc[-1]
            - pd.to_numeric(log["time"], errors="coerce").iloc[0]
        ) * 24.0
        print(f"[GAME2-CHECK] simulated elapsed span: {approx_hours:.3f} h")

    if bad:
        print("[GAME2-CHECK] result: FAIL")
        return 1

    print("[GAME2-CHECK] result: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
