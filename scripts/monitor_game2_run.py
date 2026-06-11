"""
Read-only live monitor for Game2 SAC/CTDE BSM2 runs.

The script never writes flags, never edits logs, and never stops MATLAB or
Python. It only reads the current training CSV and prints an early warning
summary so a long run can be stopped by the user before wasting days.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / "logs" / "game2_sac_training_log.csv"
DEFAULT_BASELINE = ROOT / "results" / "bsm2_manual_baseline_official_summary.csv"

EFF_LIMITS = {
    "TN": ("Ntot_eff_inst", 18.0),
    "SNH": ("SNH_eff_inst", 4.0),
    "TSS": ("TSS_eff_inst", 30.0),
    "COD": ("COD_eff_inst", 100.0),
    "BOD5": ("BOD5_eff_inst", 10.0),
}

ACTION_LIMITS = {
    "Qec": (0.0, 5.0),
    "Qint": (5_000.0, 61_944.0),
    "SO4ref": (0.0, 10.0),
    "Qw": (0.0, 450.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only monitor for Game2 BSM2 runs.")
    parser.add_argument("--log", default=str(DEFAULT_LOG), help="Game2 SAC training log CSV.")
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE), help="Manual BSM2 official summary CSV.")
    parser.add_argument("--no-baseline", action="store_true", help="Skip manual-baseline comparisons.")
    parser.add_argument("--latest-run", action="store_true", help="Only analyse the last run_id in the CSV.")
    parser.add_argument("--eval-start", type=float, default=245.0)
    parser.add_argument("--eval-stop", type=float, default=609.0)
    parser.add_argument("--rolling-days", type=float, nargs="*", default=[7.0, 14.0, 30.0])
    return parser.parse_args()


def numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col not in {"schema_version", "run_id", "mode", "action_space"}:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_baseline(path: Path) -> dict[str, float]:
    if not path.exists():
        print(f"WARNING: baseline file not found, continuing without comparisons: {path}")
        return {}
    data = pd.read_csv(path)
    values = dict(zip(data["metric"], data["value"]))
    eqi = float(values["EQI_kg_pollution_units_per_d"])
    ae = float(values["airenergy_kWh_per_d"])
    pe = float(values["pumpenergy_kWh_per_d"])
    ec = float(values["carbon_kgCOD_per_d"])
    return {
        "EQI": eqi,
        "AE": ae,
        "PE": pe,
        "EC": ec,
        "Jbase": 200.0 * eqi + 40.0 * ae + 3.0 * pe + ec,
        "Jjoint": 200.0 * eqi + 40.0 * ae + 3.0 * pe + ec,
        "OCIc": ae + pe + 3.0 * ec,
        "TN_violation": float(values["TN_violation_percent"]),
        "SNH_violation": float(values["SNH_violation_percent"]),
        "TSS_violation": float(values["TSS_violation_percent"]),
        "COD_violation": float(values["COD_violation_percent"]),
        "BOD5_violation": float(values["BOD5_violation_percent"]),
    }


def pct(mask: pd.Series) -> float:
    if len(mask) == 0:
        return math.nan
    return float(mask.mean() * 100.0)


def p95(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return math.nan
    return float(clean.quantile(0.95))


def summarize(part: pd.DataFrame) -> dict[str, float]:
    if part.empty:
        return {}
    out: dict[str, float] = {
        "rows": float(len(part)),
        "start": float(part["time"].min()),
        "stop": float(part["time"].max()),
        "days": float(part["time"].max() - part["time"].min()),
    }
    if all(c in part.columns for c in ("EQI_bsm2_inst", "AE_bsm2_inst", "PE_bsm2_inst", "EC_bsm2_inst")):
        jbase = (
            200.0 * part["EQI_bsm2_inst"]
            + 40.0 * part["AE_bsm2_inst"]
            + 3.0 * part["PE_bsm2_inst"]
            + part["EC_bsm2_inst"]
        )
        out["Jbase"] = float(jbase.mean())
        ocic = part["AE_bsm2_inst"] + part["PE_bsm2_inst"] + 3.0 * part["EC_bsm2_inst"]
        out["OCIc"] = float(ocic.mean())
    for label, col in [
        ("reward", "reward_total"),
        ("Jjoint", "J_joint"),
        ("EQI", "EQI_bsm2_inst"),
        ("AE", "AE_bsm2_inst"),
        ("PE", "PE_bsm2_inst"),
        ("EC", "EC_bsm2_inst"),
        ("SRT", "SRT_3d_state"),
    ]:
        if col in part.columns:
            out[label] = float(pd.to_numeric(part[col], errors="coerce").mean())
    for name, (col, limit) in EFF_LIMITS.items():
        if col in part.columns:
            out[f"{name}_viol"] = pct(pd.to_numeric(part[col], errors="coerce") > limit)
            out[f"{name}_p95"] = p95(part[col])
            out[f"{name}_max"] = float(pd.to_numeric(part[col], errors="coerce").max())
    for col, (low, high) in ACTION_LIMITS.items():
        if col in part.columns:
            values = pd.to_numeric(part[col], errors="coerce")
            out[f"{col}_low_pct"] = pct(values <= low + 1e-9)
            out[f"{col}_high_pct"] = pct(values >= high - 1e-9)
            out[f"{col}_mean"] = float(values.mean())
    if "reward_total" in part.columns:
        out["reward_floor_pct"] = pct(pd.to_numeric(part["reward_total"], errors="coerce") <= -19.999)
    if "reward_nam_clipped" in part.columns:
        out["nam_clip_floor_pct"] = pct(pd.to_numeric(part["reward_nam_clipped"], errors="coerce") <= -0.999)
    return out


def status_from_metrics(
    eval_summary: dict[str, float],
    rolling: dict[float, dict[str, float]],
    eval_days: float,
    baseline: dict[str, float] | None = None,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    status = "OK"

    def escalate(level: str, reason: str) -> None:
        nonlocal status
        order = {"OK": 0, "WATCH": 1, "STOP-AND-REVIEW": 2}
        if order[level] > order[status]:
            status = level
        reasons.append(reason)

    if eval_days < 1.0:
        return "PRE-EVAL", ["official BSM2 window has not started yet"]

    official_snh = eval_summary.get("SNH_viol", math.nan)
    official_snh_p95 = eval_summary.get("SNH_p95", math.nan)
    if eval_days >= 14.0 and official_snh > 5.0:
        escalate("STOP-AND-REVIEW", f"official partial SNH violation is {official_snh:.2f}% (>5%)")
    elif eval_days >= 14.0 and (official_snh > 1.5 or official_snh_p95 > 4.0):
        escalate("WATCH", f"official partial SNH risk: violation={official_snh:.2f}%, p95={official_snh_p95:.2f}")

    for days, metrics in rolling.items():
        snh = metrics.get("SNH_viol", math.nan)
        snh_p95 = metrics.get("SNH_p95", math.nan)
        tn = metrics.get("TN_viol", math.nan)
        tss = metrics.get("TSS_viol", math.nan)
        if snh > 10.0:
            escalate("STOP-AND-REVIEW", f"last {days:g}d SNH violation is {snh:.2f}% (>10%)")
        elif snh > 3.0 or snh_p95 > 4.0:
            escalate("WATCH", f"last {days:g}d SNH risk: violation={snh:.2f}%, p95={snh_p95:.2f}")
        if tn > 5.0:
            escalate("WATCH", f"last {days:g}d TN violation is {tn:.2f}%")
        if tss > 3.0:
            escalate("WATCH", f"last {days:g}d TSS violation is {tss:.2f}%")

    if eval_summary.get("reward_floor_pct", 0.0) > 20.0:
        escalate("WATCH", f"reward floor occupancy is {eval_summary['reward_floor_pct']:.2f}%")

    baseline = baseline or {}
    manual_j = baseline.get("Jjoint", baseline.get("Jbase", math.nan))
    manual_ec = baseline.get("EC", math.nan)
    if eval_days >= 60.0 and np.isfinite(manual_j) and manual_j > 0:
        official_j_ratio = eval_summary.get("Jjoint", math.nan) / manual_j
        if official_j_ratio > 1.05:
            escalate("STOP-AND-REVIEW", f"official partial Jjoint/manual is {official_j_ratio:.3f} (>1.05)")
        elif official_j_ratio > 1.0:
            escalate("WATCH", f"official partial Jjoint/manual is {official_j_ratio:.3f} (>1.0)")

    for days, metrics in rolling.items():
        if not (np.isfinite(manual_j) and manual_j > 0 and np.isfinite(manual_ec) and manual_ec > 0):
            continue
        j_ratio = metrics.get("Jjoint", math.nan) / manual_j
        ec_ratio = metrics.get("EC", math.nan) / manual_ec
        qec_high = metrics.get("Qec_high_pct", math.nan)
        so4_low = metrics.get("SO4ref_low_pct", math.nan)
        if days >= 7.0 and j_ratio > 1.15 and ec_ratio > 2.0 and qec_high > 80.0:
            escalate(
                "STOP-AND-REVIEW",
                f"last {days:g}d expensive-carbon attractor: Jjoint/manual={j_ratio:.3f}, "
                f"EC/manual={ec_ratio:.2f}, Qec_high={qec_high:.1f}%",
            )
        elif days >= 7.0 and j_ratio > 1.0 and ec_ratio > 1.8 and qec_high > 80.0:
            escalate(
                "WATCH",
                f"last {days:g}d carbon saturation risk: Jjoint/manual={j_ratio:.3f}, "
                f"EC/manual={ec_ratio:.2f}, Qec_high={qec_high:.1f}%",
            )
        elif days >= 7.0 and 1.35 < ec_ratio < 1.75 and so4_low > 80.0 and j_ratio > 1.0:
            escalate(
                "WATCH",
                f"last {days:g}d low-aeration carbon plateau: Jjoint/manual={j_ratio:.3f}, "
                f"EC/manual={ec_ratio:.2f}, SO4ref_low={so4_low:.1f}%",
            )

    if not reasons:
        reasons.append("no rolling legal-risk trigger fired")
    return status, reasons


def print_summary(label: str, summary: dict[str, float], baseline: dict[str, float] | None = None) -> None:
    if not summary:
        print(f"\n[{label}] no data")
        return
    print(f"\n[{label}] rows={summary['rows']:.0f} time={summary['start']:.2f}-{summary['stop']:.2f} days={summary['days']:.2f}")
    for key in ("reward", "Jbase", "Jjoint", "OCIc", "EQI", "AE", "PE", "EC"):
        if key not in summary:
            continue
        text = f"  {key}: {summary[key]:.3f}"
        if baseline and key in baseline:
            ratio = summary[key] / baseline[key]
            text += f"  vs manual ratio={ratio:.3f} improvement={(1-ratio)*100:.1f}%"
        print(text)
    for name in ("TN", "SNH", "TSS", "COD", "BOD5"):
        if f"{name}_viol" in summary:
            print(
                f"  {name}: viol={summary[f'{name}_viol']:.3f}% "
                f"p95={summary[f'{name}_p95']:.3f} max={summary[f'{name}_max']:.3f}"
            )
    for col in ("Qec", "Qint", "SO4ref", "Qw"):
        if f"{col}_mean" in summary:
            print(
                f"  {col}: mean={summary[f'{col}_mean']:.3f} "
                f"low={summary[f'{col}_low_pct']:.1f}% high={summary[f'{col}_high_pct']:.1f}%"
            )
    if "reward_floor_pct" in summary:
        print(f"  reward floor(-20)={summary['reward_floor_pct']:.2f}% nam clip(-1)={summary.get('nam_clip_floor_pct', math.nan):.2f}%")


def main() -> None:
    args = parse_args()
    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(f"log not found: {log_path}")

    baseline = {} if args.no_baseline else load_baseline(Path(args.baseline))
    df = numeric(pd.read_csv(log_path))
    if args.latest_run and "run_id" in df.columns and not df["run_id"].dropna().empty:
        last_run = str(df["run_id"].dropna().iloc[-1])
        df = df[df["run_id"].astype(str) == last_run].copy()
    df = df.sort_values("time").reset_index(drop=True)

    latest_time = float(df["time"].max())
    run_id = str(df["run_id"].dropna().iloc[-1]) if "run_id" in df.columns and not df["run_id"].dropna().empty else "unknown"
    mode = str(df["mode"].dropna().iloc[-1]) if "mode" in df.columns and not df["mode"].dropna().empty else "unknown"
    action_space = str(df["action_space"].dropna().iloc[-1]) if "action_space" in df.columns and not df["action_space"].dropna().empty else "unknown"
    print(f"Run: {run_id} | mode={mode} | action_space={action_space}")
    print(f"Log: {log_path}")
    print(f"Rows={len(df)} latest_time={latest_time:.3f} d rows/day={len(df)/(df.time.max()-df.time.min()):.3f}")

    eval_part = df[(df["time"] >= args.eval_start) & (df["time"] < min(args.eval_stop, latest_time + 1e-9))]
    eval_days = max(0.0, min(latest_time, args.eval_stop) - args.eval_start)
    eval_summary = summarize(eval_part)
    print_summary(f"official partial {args.eval_start:g}-{min(latest_time, args.eval_stop):.2f}", eval_summary, baseline)

    rolling: dict[float, dict[str, float]] = {}
    for days in args.rolling_days:
        start = max(args.eval_start, latest_time - days)
        part = df[(df["time"] >= start) & (df["time"] < min(latest_time + 1e-9, args.eval_stop))]
        rolling[days] = summarize(part)
        print_summary(f"rolling last {days:g}d inside official window", rolling[days])

    status, reasons = status_from_metrics(eval_summary, rolling, eval_days, baseline)
    print(f"\nRECOMMENDATION: {status}")
    for reason in reasons:
        print(f"- {reason}")
    print("\nThis script is read-only. It does not stop or modify the running simulation.")


if __name__ == "__main__":
    main()
