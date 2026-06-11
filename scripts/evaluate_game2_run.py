"""
Evaluate Game2 multi-agent SAC logs.

This is a diagnostic helper for the current Game2/G2ANet development loop.
It summarizes reward scale, action saturation, and whether rewards are aligned
with the previous action that actually affected the plant.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
CORE_DIR = ROOT / "core"
sys.path.insert(0, str(CORE_DIR))

try:
    from game2_abstraction import ACTION_BOUNDS
except Exception:  # pragma: no cover - fallback for standalone use
    ACTION_BOUNDS = {
        "qec": (0.0, 5.0),
        "qint": (5_000.0, 61_944.0),
        "do": (0.0, 10.0),
        "qw": (0.0, 450.0),
    }


@dataclass(frozen=True)
class ActionSpec:
    agent: str
    command_col: str
    applied_col: str
    reward_col: str
    low: float
    high: float

    @property
    def span(self) -> float:
        return self.high - self.low


ACTION_SPECS = (
    ActionSpec("qec", "Qec", "Qec_applied_for_reward", "reward_qec", *ACTION_BOUNDS["qec"]),
    ActionSpec("qint", "Qint", "Qint_applied_for_reward", "reward_qint", *ACTION_BOUNDS["qint"]),
    ActionSpec("do", "SO4ref", "SO4ref_applied_for_reward", "reward_do", *ACTION_BOUNDS["do"]),
    ActionSpec("qw", "Qw", "Qw_applied_for_reward", "reward_qw", *ACTION_BOUNDS["qw"]),
)

JOINT_REWARD_EXTRA_COLUMNS = [
    "J_joint",
    "EQI_joint",
    "AE_joint",
    "PE_joint",
    "EC_joint",
    "SRT_penalty",
]


@dataclass
class RunEvaluation:
    summary: dict[str, object]
    command_saturation: pd.DataFrame
    applied_saturation: pd.DataFrame
    alignment: pd.DataFrame
    correlations: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Game2 SAC diagnostic logs.")
    parser.add_argument(
        "logs",
        nargs="*",
        help="CSV log path(s). Defaults to logs/game2_sac_training_log.csv.",
    )
    parser.add_argument(
        "--latest-by-mode",
        action="store_true",
        help="Discover the newest sac-local and sac-game2 CSVs in prog_RL/logs.",
    )
    parser.add_argument(
        "--logs-dir",
        default=str(LOG_DIR),
        help="Directory to scan when --latest-by-mode is used.",
    )
    parser.add_argument(
        "--latest-run",
        action="store_true",
        help="If a CSV contains multiple run_id values, evaluate only the last run.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["sac-local", "sac-game2"],
        help="Modes to discover with --latest-by-mode.",
    )
    parser.add_argument(
        "--summary-csv",
        default="",
        help="Optional path for the one-row-per-run summary CSV.",
    )
    return parser.parse_args()


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def finite(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def pct(mask: pd.Series) -> float:
    if len(mask) == 0:
        return math.nan
    return float(mask.mean() * 100.0)


def corr(a: pd.Series, b: pd.Series) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return math.nan
    aa = a[mask]
    bb = b[mask]
    if aa.nunique(dropna=True) < 2 or bb.nunique(dropna=True) < 2:
        return math.nan
    return float(aa.corr(bb))


def safe_last(series: pd.Series, default: object = "") -> object:
    values = series.dropna()
    if values.empty:
        return default
    return values.iloc[-1]


def _read_mixed_schema_log(path: Path) -> pd.DataFrame:
    """
    Recover logs whose first reward-less row wrote the legacy header before
    later joint-reward rows added the six CTDE diagnostic columns.
    """
    rows: list[dict[str, str]] = []
    with path.open(newline="") as f:
        reader = csv.reader(f)
        legacy_columns = next(reader, None)
        if legacy_columns is None:
            return pd.DataFrame()

        if "reward_total" in legacy_columns and "J_joint" not in legacy_columns:
            insert_at = legacy_columns.index("reward_total") + 1
            joint_columns = (
                legacy_columns[:insert_at]
                + JOINT_REWARD_EXTRA_COLUMNS
                + legacy_columns[insert_at:]
            )
        else:
            joint_columns = legacy_columns

        for line_no, row in enumerate(reader, start=2):
            if len(row) == len(joint_columns):
                rows.append(dict(zip(joint_columns, row)))
            elif len(row) == len(legacy_columns):
                rows.append(dict(zip(legacy_columns, row)))
            elif len(row) < len(joint_columns):
                padded = row + [""] * (len(joint_columns) - len(row))
                rows.append(dict(zip(joint_columns, padded)))
            else:
                raise ValueError(
                    f"cannot recover {path}: line {line_no} has {len(row)} fields, "
                    f"expected {len(legacy_columns)} or {len(joint_columns)}"
                )

    return pd.DataFrame(rows)


def load_log(path: Path, latest_run: bool) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError:
        df = _read_mixed_schema_log(path)
    if df.empty:
        return df
    if latest_run and "run_id" in df.columns:
        run_id = safe_last(df["run_id"])
        if run_id != "":
            df = df[df["run_id"] == run_id].copy()
    return df


def read_modes(path: Path) -> set[str]:
    try:
        cols = pd.read_csv(path, usecols=lambda col: col == "mode")
    except Exception:
        return set()
    if "mode" not in cols.columns:
        return set()
    return set(cols["mode"].dropna().astype(str).unique())


def discover_latest_by_mode(logs_dir: Path, modes: list[str]) -> list[Path]:
    candidates = sorted(logs_dir.glob("game2_sac*training_log*.csv"))
    selected: dict[str, tuple[float, Path]] = {}
    for path in candidates:
        path_modes = read_modes(path)
        if not path_modes:
            continue
        mtime = path.stat().st_mtime
        for mode in modes:
            if mode not in path_modes:
                continue
            previous = selected.get(mode)
            if previous is None or mtime > previous[0]:
                selected[mode] = (mtime, path)

    missing = [mode for mode in modes if mode not in selected]
    if missing:
        print(f"[GAME2-EVAL] warning: no log found for modes: {', '.join(missing)}")
    return [selected[mode][1] for mode in modes if mode in selected]


def applied_action(df: pd.DataFrame, spec: ActionSpec) -> tuple[pd.Series, str]:
    if spec.applied_col in df.columns:
        return numeric(df, spec.applied_col), "explicit_applied_column"
    return numeric(df, spec.command_col).shift(1), "derived_previous_command"


def summarize_action_series(
    df: pd.DataFrame,
    spec: ActionSpec,
    series: pd.Series,
    mode: str,
    run_id: str,
    file_name: str,
    kind: str,
) -> dict[str, object]:
    values = finite(series)
    if values.empty:
        return {
            "mode": mode,
            "run_id": run_id,
            "file": file_name,
            "kind": kind,
            "action": spec.command_col,
            "n": 0,
        }

    lower_1 = spec.low + 0.01 * spec.span
    upper_1 = spec.high - 0.01 * spec.span
    lower_5 = spec.low + 0.05 * spec.span
    upper_5 = spec.high - 0.05 * spec.span
    norm = (values - spec.low) / spec.span

    return {
        "mode": mode,
        "run_id": run_id,
        "file": file_name,
        "kind": kind,
        "action": spec.command_col,
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "p05": float(values.quantile(0.05)),
        "p50": float(values.quantile(0.50)),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
        "norm_mean": float(norm.mean()),
        "near_low_1pct": pct(values <= lower_1),
        "near_high_1pct": pct(values >= upper_1),
        "near_low_5pct": pct(values <= lower_5),
        "near_high_5pct": pct(values >= upper_5),
        "out_of_bounds": int(((values < spec.low) | (values > spec.high)).sum()),
    }


def summarize_actions(
    df: pd.DataFrame,
    mode: str,
    run_id: str,
    file_name: str,
    kind: str,
) -> pd.DataFrame:
    rows = []
    reward_mask = np.isfinite(numeric(df, "reward_total"))
    for spec in ACTION_SPECS:
        if kind == "commanded":
            series = numeric(df, spec.command_col)
        elif kind == "applied_for_reward":
            series, _source = applied_action(df, spec)
            series = series.where(reward_mask)
        else:
            raise ValueError(f"Unknown action summary kind: {kind}")
        rows.append(summarize_action_series(df, spec, series, mode, run_id, file_name, kind))
    return pd.DataFrame(rows)


def summarize_alignment(df: pd.DataFrame, mode: str, run_id: str, file_name: str) -> pd.DataFrame:
    rows = []
    reward = numeric(df, "reward_total")
    reward_mask = np.isfinite(reward)
    uses_previous = numeric(df, "reward_uses_previous_action")
    uses_mask = reward_mask & np.isfinite(uses_previous)
    uses_previous_pct = pct(uses_previous[uses_mask] == 1) if int(uses_mask.sum()) else math.nan

    for spec in ACTION_SPECS:
        command = numeric(df, spec.command_col)
        previous_command = command.shift(1)
        applied, source = applied_action(df, spec)

        compare_mask = reward_mask & np.isfinite(applied) & np.isfinite(previous_command)
        lag_mask = reward_mask & np.isfinite(applied) & np.isfinite(command)

        if source == "explicit_applied_column" and int(compare_mask.sum()):
            diff = (applied[compare_mask] - previous_command[compare_mask]).abs()
            tolerance = max(1e-6, spec.span * 1e-9)
            matches = diff <= tolerance
            previous_match_pct = pct(matches)
            previous_max_abs_error = float(diff.max())
            previous_mismatch_count = int((~matches).sum())
        else:
            previous_match_pct = math.nan
            previous_max_abs_error = math.nan
            previous_mismatch_count = 0

        if int(lag_mask.sum()):
            same_row_gap = (command[lag_mask] - applied[lag_mask]).abs()
            mean_same_row_gap = float(same_row_gap.mean())
            max_same_row_gap = float(same_row_gap.max())
        else:
            mean_same_row_gap = math.nan
            max_same_row_gap = math.nan

        rows.append(
            {
                "mode": mode,
                "run_id": run_id,
                "file": file_name,
                "action": spec.command_col,
                "applied_source": source,
                "reward_rows": int(reward_mask.sum()),
                "uses_previous_action_pct": uses_previous_pct,
                "previous_command_match_pct": previous_match_pct,
                "previous_command_mismatches": previous_mismatch_count,
                "previous_command_max_abs_error": previous_max_abs_error,
                "mean_abs_same_row_command_gap": mean_same_row_gap,
                "max_abs_same_row_command_gap": max_same_row_gap,
            }
        )
    return pd.DataFrame(rows)


def summarize_correlations(df: pd.DataFrame, mode: str, run_id: str, file_name: str) -> pd.DataFrame:
    rows = []
    total_reward = numeric(df, "reward_total")
    for spec in ACTION_SPECS:
        command = numeric(df, spec.command_col)
        applied, _source = applied_action(df, spec)
        agent_reward = numeric(df, spec.reward_col)
        rows.append(
            {
                "mode": mode,
                "run_id": run_id,
                "file": file_name,
                "action": spec.command_col,
                "agent_reward": spec.reward_col,
                "corr_agent_applied": corr(agent_reward, applied),
                "corr_agent_command": corr(agent_reward, command),
                "corr_total_applied": corr(total_reward, applied),
                "corr_total_command": corr(total_reward, command),
            }
        )
    return pd.DataFrame(rows)


def summarize_run(path: Path, df: pd.DataFrame) -> dict[str, object]:
    mode = str(safe_last(df["mode"])) if "mode" in df.columns else ""
    run_id = str(safe_last(df["run_id"])) if "run_id" in df.columns else ""
    reward = finite(numeric(df, "reward_total"))
    times = finite(numeric(df, "time"))

    if times.empty:
        time_start = math.nan
        time_end = math.nan
        elapsed_hours = math.nan
    else:
        time_start = float(times.iloc[0])
        time_end = float(times.iloc[-1])
        elapsed_hours = float((time_end - time_start) * 24.0)

    summary = {
        "mode": mode,
        "run_id": run_id,
        "file": path.name,
        "rows": int(len(df)),
        "reward_rows": int(reward.size),
        "time_start_d": time_start,
        "time_end_d": time_end,
        "elapsed_h": elapsed_hours,
        "reward_mean": float(reward.mean()) if not reward.empty else math.nan,
        "reward_last": float(reward.iloc[-1]) if not reward.empty else math.nan,
        "reward_min": float(reward.min()) if not reward.empty else math.nan,
        "reward_max": float(reward.max()) if not reward.empty else math.nan,
    }

    for spec in ACTION_SPECS:
        values = finite(numeric(df, spec.command_col))
        summary[f"{spec.command_col}_mean"] = float(values.mean()) if not values.empty else math.nan
    return summary


def evaluate_path(path: Path, latest_run: bool) -> RunEvaluation:
    df = load_log(path, latest_run=latest_run)
    if df.empty:
        raise ValueError(f"empty log: {path}")
    summary = summarize_run(path, df)
    mode = str(summary["mode"])
    run_id = str(summary["run_id"])
    file_name = str(summary["file"])
    return RunEvaluation(
        summary=summary,
        command_saturation=summarize_actions(df, mode, run_id, file_name, "commanded"),
        applied_saturation=summarize_actions(df, mode, run_id, file_name, "applied_for_reward"),
        alignment=summarize_alignment(df, mode, run_id, file_name),
        correlations=summarize_correlations(df, mode, run_id, file_name),
    )


def comparison_frame(evals: list[RunEvaluation]) -> pd.DataFrame:
    by_mode = {str(ev.summary["mode"]): ev for ev in evals}
    if "sac-local" not in by_mode or "sac-game2" not in by_mode:
        return pd.DataFrame()

    local = by_mode["sac-local"]
    game2 = by_mode["sac-game2"]
    rows = []

    def add(metric: str, local_value: float, game2_value: float, unit: str = "") -> None:
        rows.append(
            {
                "metric": metric,
                "sac-local": local_value,
                "sac-game2": game2_value,
                "game2_minus_local": game2_value - local_value,
                "unit": unit,
            }
        )

    add("reward_mean", float(local.summary["reward_mean"]), float(game2.summary["reward_mean"]))
    add("reward_last", float(local.summary["reward_last"]), float(game2.summary["reward_last"]))

    local_sat = local.command_saturation.set_index("action")
    game2_sat = game2.command_saturation.set_index("action")
    for spec in ACTION_SPECS:
        action = spec.command_col
        add(
            f"{action}_mean",
            float(local_sat.loc[action, "mean"]),
            float(game2_sat.loc[action, "mean"]),
        )
        add(
            f"{action}_near_low_5pct",
            float(local_sat.loc[action, "near_low_5pct"]),
            float(game2_sat.loc[action, "near_low_5pct"]),
            "percentage points",
        )
        add(
            f"{action}_near_high_5pct",
            float(local_sat.loc[action, "near_high_5pct"]),
            float(game2_sat.loc[action, "near_high_5pct"]),
            "percentage points",
        )
    return pd.DataFrame(rows)


def print_frame(title: str, frame: pd.DataFrame, columns: list[str] | None = None) -> None:
    print(f"\n[GAME2-EVAL] {title}")
    if frame.empty:
        print("(no rows)")
        return
    shown = frame.copy()
    if columns is not None:
        shown = shown[columns]
    print(shown.to_string(index=False, na_rep="", float_format=lambda x: f"{x:.4g}"))


def main() -> int:
    args = parse_args()

    if args.latest_by_mode:
        paths = discover_latest_by_mode(Path(args.logs_dir), args.modes)
        for path in paths:
            print(f"[GAME2-EVAL] selected latest log: {path}")
    elif args.logs:
        paths = [Path(item) for item in args.logs]
    else:
        paths = [LOG_DIR / "game2_sac_training_log.csv"]

    if not paths:
        print("[GAME2-EVAL] no logs to evaluate")
        return 2

    evals: list[RunEvaluation] = []
    for path in paths:
        if not path.exists():
            print(f"[GAME2-EVAL] missing log: {path}")
            return 2
        evals.append(evaluate_path(path, latest_run=args.latest_run))

    summary = pd.DataFrame([ev.summary for ev in evals])
    command_saturation = pd.concat([ev.command_saturation for ev in evals], ignore_index=True)
    applied_saturation = pd.concat([ev.applied_saturation for ev in evals], ignore_index=True)
    alignment = pd.concat([ev.alignment for ev in evals], ignore_index=True)
    correlations = pd.concat([ev.correlations for ev in evals], ignore_index=True)
    comparison = comparison_frame(evals)

    print_frame(
        "Run summary",
        summary,
        [
            "mode",
            "run_id",
            "file",
            "rows",
            "reward_rows",
            "elapsed_h",
            "reward_mean",
            "reward_last",
            "Qec_mean",
            "Qint_mean",
            "SO4ref_mean",
            "Qw_mean",
        ],
    )
    print_frame(
        "Commanded action saturation",
        command_saturation,
        [
            "mode",
            "action",
            "n",
            "mean",
            "min",
            "p05",
            "p95",
            "max",
            "near_low_5pct",
            "near_high_5pct",
            "out_of_bounds",
        ],
    )
    print_frame(
        "Reward-causal action saturation",
        applied_saturation,
        [
            "mode",
            "action",
            "n",
            "mean",
            "min",
            "p05",
            "p95",
            "max",
            "near_low_5pct",
            "near_high_5pct",
            "out_of_bounds",
        ],
    )
    print_frame(
        "Reward/action alignment",
        alignment,
        [
            "mode",
            "action",
            "applied_source",
            "reward_rows",
            "uses_previous_action_pct",
            "previous_command_match_pct",
            "previous_command_mismatches",
            "previous_command_max_abs_error",
            "mean_abs_same_row_command_gap",
        ],
    )
    print_frame(
        "Reward/action correlations",
        correlations,
        [
            "mode",
            "action",
            "agent_reward",
            "corr_agent_applied",
            "corr_agent_command",
            "corr_total_applied",
            "corr_total_command",
        ],
    )
    print_frame("sac-game2 minus sac-local comparison", comparison)

    if args.summary_csv:
        out_path = Path(args.summary_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_path, index=False)
        print(f"\n[GAME2-EVAL] wrote summary CSV: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
