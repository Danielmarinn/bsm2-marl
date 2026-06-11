"""
Compute CTRL-3 state normalisation constants from a CSV file.

Expected columns:
    SO_4, SNH_4, SNH_5, SNO_3, TSS_3

Example:
    python prog_RL/scripts/compute_ctrl3_state_stats.py path/to/ctrl3_baseline.csv
"""

import argparse
import pathlib
import sys

import pandas as pd


CTRL3_COLUMNS = ["SO_4", "SNH_4", "SNH_5", "SNO_3", "TSS_3"]


def format_vector(values):
    return "[" + ", ".join(f"{value:.6g}" for value in values) + "]"


def main():
    parser = argparse.ArgumentParser(
        description="Compute STATE_MEAN and STATE_STD for CTRL-3 from a CSV."
    )
    parser.add_argument("csv_path", help="CSV file containing the CTRL-3 state columns.")
    args = parser.parse_args()

    csv_path = pathlib.Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [column for column in CTRL3_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "Missing required CTRL-3 columns: " + ", ".join(missing)
        )

    state_df = df[CTRL3_COLUMNS].dropna()
    if len(state_df) < 10:
        raise ValueError(
            f"Need at least 10 valid rows to compute stable statistics; got {len(state_df)}."
        )

    mean = state_df.mean(axis=0)
    std = state_df.std(axis=0, ddof=0)
    if (std <= 0).any():
        bad = [name for name, value in std.items() if value <= 0]
        raise ValueError(
            "Non-positive standard deviation for: " + ", ".join(bad)
        )

    print("CTRL-3 columns:", ", ".join(CTRL3_COLUMNS))
    print(f"Rows used: {len(state_df)}")
    print()
    print(f"STATE_MEAN = {format_vector(mean.values)}")
    print(f"STATE_STD  = {format_vector(std.values)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[CTRL3 stats] ERROR: {exc}", file=sys.stderr)
        raise
