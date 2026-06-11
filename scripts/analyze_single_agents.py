"""
Unified evolution analysis across the four single-agent training logs.

For each agent reports:
  - Total SAC steps
  - Reward by quintile (early vs converged) and first/last 1k means
  - Action range, saturation near bounds
  - End-of-training mean of the plant outputs the agent's reward depends on
  - Cost ratio J(t)/J_manual quintile means

CTRL-1's CSV has an evolving 15->24->29-column schema. This script
hand-parses the dominant 29-column rows so the post-warmup window is
recoverable end-to-end.

Usage
-----
    python prog_RL/scripts/analyze_single_agents.py
    python prog_RL/scripts/analyze_single_agents.py --matched 46400

The optional --matched flag truncates the three fast-decision logs to
the same SAC-step count for direct cross-agent comparison (CTRL-4 is
left untruncated because its decision interval is one day).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
LOG_DIR = ROOT / "logs"


def read_ctrl1(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) == 29:
                rows.append(parts)
    cols = [
        "episode", "step", "mode",
        "SNO_2", "SNO_1", "SNO_3", "CODTN",
        "prev_Qec_raw", "prev_Qec_norm", "SNH", "Flow",
        "Qec_prev", "Qec_new", "reward", "buffer",
        "EQI", "AE", "PE", "EC", "J", "J_manual", "ratio",
        "Qint_used", "Qec_used",
        "extra_24", "extra_25", "extra_26", "extra_27", "extra_28",
    ]
    df = pd.DataFrame(rows, columns=cols)
    for c in df.columns:
        if c != "mode":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def read_other(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, engine="python", on_bad_lines="skip")


SPECS = {
    "CTRL-1 / Qec":  {"path": LOG_DIR / "ctrl1_qec_training_optionB.csv",
                        "action": "Qec_new",  "low": 0.0, "high": 5.0,
                        "outputs": ["SNO_2", "SNH", "EQI", "J", "ratio"]},
    "CTRL-2 / Qint": {"path": LOG_DIR / "ctrl2_qint_training.csv",
                        "action": "Qint_new", "low": 5_000.0, "high": 61_944.0,
                        "outputs": ["SNO_2", "SNH", "EQI", "J", "ratio"]},
    "CTRL-3 / DO":   {"path": LOG_DIR / "ctrl3_do_training.csv",
                        "action": "SO4ref_new", "low": 0.0, "high": 10.0,
                        "outputs": ["SO_4", "SNH_4", "SNH_5", "SNO_3", "EQI", "J", "ratio"]},
    "CTRL-4 / Qw":   {"path": LOG_DIR / "ctrl4_qw_training.csv",
                        "action": "Qw_new",   "low": 0.0, "high": 450.0,
                        "outputs": ["TSS_5_1d", "SRT_3d", "EQI_1d", "J", "ratio"]},
}


def quintile_means(s: pd.Series, n: int = 5) -> list[float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return [float("nan")] * n
    out = []
    chunk = max(1, len(s) // n)
    for i in range(n):
        sl = s.iloc[i * chunk : (i + 1) * chunk]
        out.append(float(sl.mean()) if len(sl) else float("nan"))
    return out


def saturation_pct(values: pd.Series, low: float, high: float, edge: float = 0.05) -> tuple[float, float]:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if len(v) == 0:
        return float("nan"), float("nan")
    span = high - low
    return (
        float((v <= low + edge * span).mean() * 100.0),
        float((v >= high - edge * span).mean() * 100.0),
    )


def analyse(label: str, spec: dict, matched: int | None) -> None:
    path: Path = spec["path"]
    if "ctrl1" in path.name:
        df = read_ctrl1(path)
    else:
        df = read_other(path)

    if "mode" in df.columns:
        sac = df[df["mode"].astype(str).str.upper() == "SAC"].copy()
    else:
        sac = df.copy()

    if matched and "CTRL-4" not in label:
        sac = sac.head(matched)

    print(f"\n=== {label} ===")
    print(f"  SAC rows={len(sac)}  step range={int(sac['step'].min()) if 'step' in sac.columns else '-'}-"
          f"{int(sac['step'].max()) if 'step' in sac.columns else '-'}")

    r = pd.to_numeric(sac.get("reward", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(r):
        q = quintile_means(r, 5)
        print(f"  reward quintiles: {[f'{x:.3f}' for x in q]}")
        print(f"  reward first1k={r.head(1000).mean():.3f}  last1k={r.tail(1000).mean():.3f}  "
              f"min={r.min():.3f}  max={r.max():.3f}  clip(-1)={(r<=-0.999).mean()*100:.2f}%")

    a_col = spec["action"]
    if a_col in sac.columns:
        a = pd.to_numeric(sac[a_col], errors="coerce").dropna()
        if len(a):
            nl, nh = saturation_pct(a, spec["low"], spec["high"])
            print(f"  action {a_col}: mean={a.mean():.3f}  median={a.median():.3f}  "
                  f"min={a.min():.3f}  max={a.max():.3f}  near_low_5%={nl:.1f}%  near_high_5%={nh:.1f}%")
            print(f"  action quintile means: {[f'{x:.3f}' for x in quintile_means(a, 5)]}")

    if "ratio" in sac.columns:
        rs = pd.to_numeric(sac["ratio"], errors="coerce").dropna()
        if len(rs):
            print(f"  ratio quintile means: {[f'{x:.4g}' for x in quintile_means(rs, 5)]}")

    for col in spec["outputs"]:
        if col in sac.columns:
            s = pd.to_numeric(sac[col], errors="coerce").dropna()
            if len(s):
                fst = s.head(1000).mean()
                lst = s.tail(1000).mean()
                print(f"  {col}: first1k={fst:.4g}  last1k={lst:.4g}  delta={(lst-fst):+.4g}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-agent training analysis")
    parser.add_argument("--matched", type=int, default=0,
                        help="If >0, truncate fast-decision logs to this many SAC steps for matched comparison.")
    args = parser.parse_args()

    matched = args.matched if args.matched > 0 else None
    if matched:
        print(f"=== MATCHED COMPARISON (first ~{matched} SAC steps for fast agents) ===")
    else:
        print("=== FULL COMPARISON (each agent uses its complete log) ===")

    for label, spec in SPECS.items():
        analyse(label, spec, matched)

    print("\n=== Manual baseline (days 245-609, official) ===")
    print("  EQI=5576.665  OCI=9450.032  AE=4225.43  PE=445.45  SNH95=1.54")


if __name__ == "__main__":
    main()
