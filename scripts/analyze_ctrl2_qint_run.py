"""Analyze the ctrl2 Qint SAC run and produce metrics for the comparison document."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROG_RL = ROOT / "prog_RL"
LOG = PROG_RL / "logs" / "ctrl2_qint_training.csv"
BASELINE_TS = PROG_RL / "results" / "bsm2_manual_baseline_timeseries.csv"
BASELINE_OFFICIAL = PROG_RL / "results" / "bsm2_manual_baseline_official_summary.csv"
CTRL1_SUMMARY = ROOT / "outputs" / "_archive" / "2026-04-25" / "ctrl1_qec_analysis_2026-04-25" / "ctrl1_summary_metrics.csv"
CTRL1_CLEAN = ROOT / "outputs" / "ctrl1_qec_clean_summary_2026-04-25" / "ctrl1_clean_comparison.csv"


def summarize(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {}
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p05": float(s.quantile(0.05)),
        "p95": float(s.quantile(0.95)),
        "min": float(s.min()),
        "max": float(s.max()),
    }


def main():
    df = pd.read_csv(LOG)
    df["mode"] = df["mode"].astype(str)
    df["is_sac"] = df["mode"].str.upper() == "SAC"
    sac = df[df["is_sac"]]

    print("=== CTRL-2 QINT SAC RUN ===")
    print(f"Rows total / SAC: {len(df):,} / {len(sac):,}")
    step_min = df["step"].min()
    step_max = df["step"].max()
    print(f"Step range: {step_min:.0f} to {step_max:.0f}")
    sim_min = df["sim_time"].min()
    sim_max = df["sim_time"].max()
    print(f"Sim time: {sim_min:.2f} to {sim_max:.2f} days")
    mode_counts = df["mode"].value_counts().to_dict()
    print(f"Mode counts: {mode_counts}")
    print()

    cols = ["reward", "EQI", "J", "ratio", "SNO_2", "SNO_1", "SNO_3", "SNH", "Qint_new"]
    print("=== ALL ROWS ===")
    all_stats = {}
    for col in cols:
        s = summarize(df[col])
        all_stats[col] = s
        if s:
            print(f"  {col:12s}: mean={s['mean']:.4f}, median={s['median']:.4f}, p05={s['p05']:.4f}, p95={s['p95']:.4f}")

    print()
    print("=== SAC-ONLY ===")
    sac_stats = {}
    for col in cols:
        s = summarize(sac[col])
        sac_stats[col] = s
        if s:
            print(f"  {col:12s}: mean={s['mean']:.4f}, median={s['median']:.4f}")

    print()
    print("=== LAST 1000 STEPS ===")
    last1k = df.tail(1000)
    for col in ["reward", "J", "ratio", "SNO_2", "SNH"]:
        s = summarize(last1k[col])
        if s:
            print(f"  {col:12s}: mean={s['mean']:.4f}, median={s['median']:.4f}")

    print()
    print("=== CHECKS ===")
    reward_s = pd.to_numeric(df["reward"], errors="coerce")
    clipped = int((reward_s <= -1 + 1e-9).sum())
    total_r = int(reward_s.notna().sum())
    print(f"  Reward clipped at -1: {clipped:,} / {total_r:,} ({clipped/total_r*100:.1f}%)")
    qint_s = pd.to_numeric(df["Qint_new"], errors="coerce").dropna()
    print(f"  Qint_new: min={qint_s.min():.0f}, max={qint_s.max():.0f}, mean={qint_s.mean():.0f}")
    ratio_s = pd.to_numeric(df["ratio"], errors="coerce")
    below = int((ratio_s < 1.0).sum())
    total_ratio = int(ratio_s.notna().sum())
    print(f"  J ratio < 1.0: {below:,} of {total_ratio:,} ({below/total_ratio*100:.1f}%)")

    last = df.iloc[-1]
    print(f"  Final row: step={last['step']:.0f}, reward={last['reward']:.4f}, ratio={last['ratio']:.4f}, Qint={last['Qint_new']:.0f}")

    print()
    print("=== BASELINE (timeseries proxy) ===")
    if BASELINE_TS.exists():
        bl = pd.read_csv(BASELINE_TS)
        for col in bl.columns:
            c = pd.to_numeric(bl[col], errors="coerce")
            if c.notna().any():
                bl[col] = c
        for col in ["reward", "EQI", "J", "ratio", "SNO_2", "SNH_2", "Qec"]:
            if col in bl.columns:
                s = summarize(bl[col])
                if s:
                    print(f"  {col:12s}: mean={s['mean']:.4f}, median={s['median']:.4f}")

    print()
    print("=== CTRL-1 SUMMARY (from CSV) ===")
    if CTRL1_SUMMARY.exists():
        c1 = pd.read_csv(CTRL1_SUMMARY)
        c1 = c1.set_index("metric")
        for metric in ["reward", "EQI", "J", "ratio"]:
            if metric in c1.index:
                print(f"  {metric:12s}: mean={c1.loc[metric, 'mean']:.4f}, median={c1.loc[metric, 'median']:.4f}")


if __name__ == "__main__":
    main()
