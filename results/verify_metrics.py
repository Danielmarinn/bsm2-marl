"""Small helper for checking exported BSM2 log summaries.

Example:
    python results/verify_metrics.py \
        --log results/bsm2_manual_baseline_summary.csv:baseline

For full training logs, pass CSV files with a time column and metric columns
such as EQI, AE, PE, and reward.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def analyze_log(path: Path, label: str) -> dict[str, float | int | str]:
    df = pd.read_csv(path)

    if {"metric", "value"}.issubset(df.columns):
        # Compact summary tables are already aggregated.
        values = dict(zip(df["metric"], df["value"]))
        return {
            "label": label,
            "path": str(path),
            "status": "summary table",
            "rows": len(df),
            "mean_EQI": values.get("mean_EQI", values.get("EQI_kg_pollution_units_per_d", "")),
            "mean_AE": values.get("mean_AE", values.get("airenergy_kWh_per_d", "")),
            "mean_PE": values.get("mean_PE", values.get("pumpenergy_kWh_per_d", "")),
        }

    time_col = next((c for c in ("sim_time", "time", "time_d", "start_day") if c in df.columns), None)
    if time_col is None:
        return {
            "label": label,
            "path": str(path),
            "status": "no time column",
            "columns": ", ".join(df.columns),
        }

    df_eval = df[(df[time_col] >= 245) & (df[time_col] <= 609)]
    if df_eval.empty:
        return {"label": label, "path": str(path), "status": "no rows in 245-609 day window"}

    def mean_if_present(column: str) -> float | str:
        return float(df_eval[column].mean()) if column in df_eval.columns else ""

    return {
        "label": label,
        "path": str(path),
        "status": "ok",
        "samples": len(df_eval),
        "start_time": float(df_eval[time_col].min()),
        "end_time": float(df_eval[time_col].max()),
        "mean_EQI": mean_if_present("EQI"),
        "mean_AE": mean_if_present("AE"),
        "mean_PE": mean_if_present("PE"),
        "mean_reward": mean_if_present("reward"),
    }


def parse_log_arg(raw: str) -> tuple[Path, str]:
    if ":" in raw:
        path, label = raw.rsplit(":", 1)
        return Path(path), label
    path = Path(raw)
    return path, path.stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        action="append",
        required=True,
        help="CSV path, optionally followed by :label. Can be passed multiple times.",
    )
    args = parser.parse_args()

    rows = []
    for raw in args.log:
        path, label = parse_log_arg(raw)
        rows.append(analyze_log(path, label))

    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
