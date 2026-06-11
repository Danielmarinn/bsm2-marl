"""
Controller-1 Qec SAC run analysis.

This script intentionally reads only the CTRL-1 log plus static baseline/result
files. It does not inspect CTRL-2 logs or communication files used by live runs.
"""

from __future__ import annotations

import html
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROG_RL = ROOT / "prog_RL"
LOG_PATH = PROG_RL / "logs" / "ctrl1_qec_training_optionB.csv"
DIAG_PATH = PROG_RL / "logs" / "diag_qec_sno2_corr.csv"
BASELINE_TS_PATH = PROG_RL / "results" / "bsm2_manual_baseline_timeseries.csv"
OUT_DIR = ROOT / "outputs" / "ctrl1_qec_analysis_2026-04-25"

EXTENDED_COLUMNS = [
    "episode",
    "step",
    "mode",
    "SNO_2",
    "SNO_1",
    "SNO_3",
    "CODTN",
    "prev_Qec",
    "prev_Qec_norm",
    "SNH",
    "Flow",
    "Qec_prev",
    "Qec_new",
    "reward",
    "buffer",
    "EQI",
    "AE",
    "PE",
    "EC",
    "J",
    "J_manual",
    "ratio",
    "Qint_used",
    "Qec_used",
    "loss_critic",
    "loss_actor",
    "alpha",
    "critic_gnorm",
    "actor_gnorm",
]

NUMERIC_COLUMNS = [c for c in EXTENDED_COLUMNS if c != "mode"]

NEW_SCHEMA_NUMERIC_COLUMNS = [
    "schema_version",
    "episode",
    "episode_start_day",
    "episode_stop_day",
    "step",
    "sim_time",
    "SNO_2",
    "SNO_1",
    "SNO_3",
    "CODTN",
    "prev_Qec",
    "prev_Qec_norm",
    "SNH",
    "Flow",
    "Temp",
    "Qec_prev",
    "Qec_new",
    "Qec_applied_for_reward",
    "Qec_command_next",
    "reward_uses_previous_action",
    "reward",
    "buffer",
    "EQI",
    "AE",
    "PE",
    "EC",
    "J",
    "J_manual",
    "ratio",
    "Qint_used",
    "Qec_used",
    "loss_critic",
    "loss_actor",
    "alpha",
    "critic_gnorm",
    "actor_gnorm",
]


def load_ctrl1() -> pd.DataFrame:
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"CTRL-1 log not found: {LOG_PATH}")

    header = pd.read_csv(LOG_PATH, nrows=0).columns.tolist()
    if header and header[0] == "schema_version":
        df = pd.read_csv(LOG_PATH)
        for col in NEW_SCHEMA_NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        df = pd.read_csv(
            LOG_PATH,
            header=None,
            skiprows=1,
            names=EXTENDED_COLUMNS,
            engine="python",
        )
        for col in NUMERIC_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["Qec_applied_for_reward"] = df["Qec_prev"]
        df["Qec_command_next"] = df["Qec_new"]
        df["reward_uses_previous_action"] = df["Qec_prev"].notna().astype(int)
    df["mode"] = df["mode"].astype(str)
    df["is_sac"] = df["mode"].str.upper().eq("SAC")
    return df


def summarize_series(s: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(s, errors="coerce").dropna()
    if clean.empty:
        return {
            "n": 0,
            "mean": math.nan,
            "min": math.nan,
            "p05": math.nan,
            "median": math.nan,
            "p95": math.nan,
            "max": math.nan,
        }
    return {
        "n": int(clean.shape[0]),
        "mean": float(clean.mean()),
        "min": float(clean.min()),
        "p05": float(clean.quantile(0.05)),
        "median": float(clean.median()),
        "p95": float(clean.quantile(0.95)),
        "max": float(clean.max()),
    }


def downsample(df: pd.DataFrame, max_points: int = 3000) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    step = (len(df) - 1) / (max_points - 1)
    idx = sorted({round(i * step) for i in range(max_points)})
    return df.iloc[idx]


def finite_series(x: pd.Series, y: pd.Series, max_points: int = 2500) -> list[tuple[float, float]]:
    temp = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")})
    temp = downsample(temp.dropna(), max_points=max_points)
    return [(float(row.x), float(row.y)) for row in temp.itertuples()]


def rolling_pairs(
    x: pd.Series,
    y: pd.Series,
    *,
    window: int = 500,
    max_points: int = 2500,
) -> list[tuple[float, float]]:
    min_periods = 1 if window <= 1 else max(10, window // 10)
    roll = pd.to_numeric(y, errors="coerce").rolling(window, min_periods=min_periods).mean()
    return finite_series(x, roll, max_points=max_points)


def nice_bounds(values: list[float]) -> tuple[float, float]:
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return 0.0, 1.0
    lo = min(clean)
    hi = max(clean)
    if lo == hi:
        pad = abs(lo) * 0.1 or 1.0
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def map_point(
    x: float,
    y: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    left: float,
    top: float,
    width: float,
    height: float,
) -> tuple[float, float]:
    px = left + (x - x_min) / (x_max - x_min or 1.0) * width
    py = top + height - (y - y_min) / (y_max - y_min or 1.0) * height
    return px, py


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def svg_line_panels(
    panels: list[dict],
    out_path: Path,
    *,
    title: str,
    width: int = 1200,
    panel_height: int = 215,
) -> None:
    margin_left = 82
    margin_right = 28
    margin_top = 66
    margin_bottom = 48
    gap = 36
    plot_width = width - margin_left - margin_right
    height = margin_top + margin_bottom + len(panels) * panel_height + (len(panels) - 1) * gap

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Segoe UI,Arial,sans-serif;fill:#1f2933} .grid{stroke:#d7dde5;stroke-width:1} .axis{stroke:#6b7280;stroke-width:1.1} .thin{stroke-width:1.1;fill:none;opacity:.38} .thick{stroke-width:2.2;fill:none} .ref{stroke:#555;stroke-width:1.2;stroke-dasharray:5 5;fill:none}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2:.0f}" y="34" text-anchor="middle" font-size="24" font-weight="700">{html.escape(title)}</text>',
    ]

    for panel_idx, panel in enumerate(panels):
        top = margin_top + panel_idx * (panel_height + gap)
        series = panel["series"]
        x_vals = [p[0] for item in series for p in item["points"]]
        y_vals = [p[1] for item in series for p in item["points"]]
        for hline in panel.get("hlines", []):
            y_vals.append(float(hline["y"]))
        x_min, x_max = nice_bounds(x_vals)
        y_min, y_max = nice_bounds(y_vals)

        parts.append(f'<text x="16" y="{top + 18}" font-size="16" font-weight="700">{html.escape(panel["label"])}</text>')
        for tick in range(5):
            frac = tick / 4
            y = top + frac * panel_height
            val = y_max - frac * (y_max - y_min)
            parts.append(f'<line class="grid" x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_width}" y2="{y:.1f}"/>')
            parts.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{val:.3g}</text>')
        for tick in range(5):
            frac = tick / 4
            x = margin_left + frac * plot_width
            val = x_min + frac * (x_max - x_min)
            parts.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + panel_height}"/>')
            if panel_idx == len(panels) - 1:
                parts.append(f'<text x="{x:.1f}" y="{top + panel_height + 20}" text-anchor="middle" font-size="11">{val:.0f}</text>')
        parts.append(f'<rect x="{margin_left}" y="{top}" width="{plot_width}" height="{panel_height}" fill="none" class="axis"/>')

        for hline in panel.get("hlines", []):
            _, py = map_point(x_min, float(hline["y"]), x_min, x_max, y_min, y_max, margin_left, top, plot_width, panel_height)
            parts.append(f'<line class="ref" x1="{margin_left}" y1="{py:.1f}" x2="{margin_left + plot_width}" y2="{py:.1f}"/>')
            parts.append(f'<text x="{margin_left + plot_width - 4}" y="{py - 5:.1f}" text-anchor="end" font-size="11">{html.escape(hline["label"])}</text>')

        legend_x = margin_left + 10
        legend_y = top + 20
        for item in series:
            mapped = [
                map_point(px, py, x_min, x_max, y_min, y_max, margin_left, top, plot_width, panel_height)
                for px, py in item["points"]
            ]
            if mapped:
                cls = "thick" if item.get("main") else "thin"
                parts.append(f'<polyline class="{cls}" stroke="{item["color"]}" points="{polyline(mapped)}"/>')
            parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 22}" y2="{legend_y}" stroke="{item["color"]}" stroke-width="{2.2 if item.get("main") else 1.1}"/>')
            parts.append(f'<text x="{legend_x + 28}" y="{legend_y + 4}" font-size="12">{html.escape(item["label"])}</text>')
            legend_x += 180

    parts.append(f'<text x="{width/2:.0f}" y="{height - 14}" text-anchor="middle" font-size="13">Training step</text>')
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def panel_from_series(
    df: pd.DataFrame,
    y_col: str,
    *,
    color: str,
    label: str | None = None,
    x_col: str = "step",
    window: int = 500,
    hlines: list[dict] | None = None,
) -> dict:
    label = label or y_col
    return {
        "label": label,
        "hlines": hlines or [],
        "series": [
            {
                "label": f"{label} raw",
                "color": color,
                "points": finite_series(df[x_col], df[y_col]),
                "main": False,
            },
            {
                "label": f"{label} rolling mean",
                "color": color,
                "points": rolling_pairs(df[x_col], df[y_col], window=window),
                "main": True,
            },
        ],
    }


def histogram_panel(values: pd.Series, *, bins: int, color: str, label: str) -> dict:
    clean = [float(v) for v in pd.to_numeric(values, errors="coerce").dropna()]
    if not clean:
        points = []
    else:
        lo, hi = min(clean), max(clean)
        width = (hi - lo) / bins if hi != lo else 1.0
        counts = [0] * bins
        for val in clean:
            idx = min(bins - 1, max(0, int((val - lo) / width)))
            counts[idx] += 1
        points = [(lo + (i + 0.5) * width, count) for i, count in enumerate(counts)]
    return {"label": label, "series": [{"label": label, "color": color, "points": points, "main": True}]}


def save_summary_tables(df: pd.DataFrame, baseline: pd.DataFrame | None) -> None:
    metrics = [
        "reward",
        "Qec_new",
        "SNO_2",
        "SNO_1",
        "SNO_3",
        "CODTN",
        "SNH",
        "EQI",
        "J",
        "ratio",
        "loss_critic",
        "loss_actor",
        "alpha",
        "critic_gnorm",
        "actor_gnorm",
    ]
    metric_rows = []
    for metric in metrics:
        row = {"metric": metric}
        row.update(summarize_series(df[metric]))
        metric_rows.append(row)
    pd.DataFrame(metric_rows).to_csv(OUT_DIR / "ctrl1_summary_metrics.csv", index=False)

    sac = df[df["is_sac"]]
    windows = {
        "first_900_random": df.head(900),
        "first_1000_sac": sac.head(1000),
        "last_1000": df.tail(1000),
        "last_5000": df.tail(5000),
        "all_sac": sac,
        "all_rows": df,
    }
    window_rows = []
    for name, wdf in windows.items():
        window_rows.append(
            {
                "window": name,
                "rows": len(wdf),
                "step_min": wdf["step"].min(),
                "step_max": wdf["step"].max(),
                "reward_mean": wdf["reward"].mean(),
                "reward_median": wdf["reward"].median(),
                "Qec_mean": wdf["Qec_new"].mean(),
                "Qec_median": wdf["Qec_new"].median(),
                "J_mean": wdf["J"].mean(),
                "ratio_mean": wdf["ratio"].mean(),
                "alpha_mean": wdf["alpha"].mean(),
            }
        )
    pd.DataFrame(window_rows).to_csv(OUT_DIR / "ctrl1_window_summary.csv", index=False)

    corr_cols = ["reward", "SNO_2", "SNO_1", "SNO_3", "SNH", "J", "ratio"]
    corr_rows = [
        {"pair": f"Qec_new_vs_{col}", "pearson_r": df[["Qec_new", col]].corr().iloc[0, 1]}
        for col in corr_cols
    ]
    pd.DataFrame(corr_rows).to_csv(OUT_DIR / "ctrl1_correlations.csv", index=False)

    if baseline is not None:
        baseline_map = {
            "reward": "reward",
            "Qec": "Qec",
            "SNO_2": "SNO_2",
            "SNO_1": "SNO_1",
            "SNO_3": "SNO_3",
            "SNH": "SNH_2",
            "EQI": "EQI",
            "J": "J",
            "ratio": "ratio",
        }
        comparison_rows = []
        for ctrl_col, base_col in baseline_map.items():
            df_col = "Qec_new" if ctrl_col == "Qec" else ctrl_col
            if df_col in df.columns and base_col in baseline.columns:
                comparison_rows.append(
                    {
                        "metric": ctrl_col,
                        "ctrl1_mean": df[df_col].mean(),
                        "ctrl1_median": df[df_col].median(),
                        "baseline_mean": baseline[base_col].mean(),
                        "baseline_median": baseline[base_col].median(),
                    }
                )
        pd.DataFrame(comparison_rows).to_csv(
            OUT_DIR / "ctrl1_vs_manual_baseline_proxy_comparison.csv", index=False
        )


def make_plots(df: pd.DataFrame, baseline: pd.DataFrame | None) -> None:
    svg_line_panels(
        [
            panel_from_series(df, "reward", color="#1f77b4", label="Reward"),
            panel_from_series(
                df,
                "Qec_new",
                color="#d55e00",
                label="Qec new",
                hlines=[{"y": 0.0, "label": "min"}, {"y": 5.0, "label": "max"}],
            ),
        ],
        OUT_DIR / "ctrl1_reward_and_qec.svg",
        title="CTRL-1 Qec SAC: Reward and Action",
    )

    svg_line_panels(
        [
            panel_from_series(df, "SNO_2", color="#0072b2"),
            panel_from_series(df, "SNO_1", color="#009e73"),
            panel_from_series(df, "SNO_3", color="#cc79a7"),
            panel_from_series(df, "SNH", color="#e69f00"),
        ],
        OUT_DIR / "ctrl1_state_trends.svg",
        title="CTRL-1 State Trends",
    )

    objective_panels = []
    for col, color in [("J", "#1f77b4"), ("ratio", "#009e73"), ("EQI", "#d55e00")]:
        hlines = []
        if baseline is not None and col in baseline.columns:
            hlines.append({"y": float(baseline[col].mean()), "label": "manual baseline mean"})
        objective_panels.append(panel_from_series(df, col, color=color, hlines=hlines))
    svg_line_panels(
        objective_panels,
        OUT_DIR / "ctrl1_objective_components.svg",
        title="CTRL-1 Objective Components",
    )

    sac = df[df["is_sac"]].copy()
    svg_line_panels(
        [
            panel_from_series(sac, "loss_critic", color="#0072b2"),
            panel_from_series(sac, "loss_actor", color="#d55e00"),
            panel_from_series(sac, "alpha", color="#009e73"),
            panel_from_series(sac, "critic_gnorm", color="#cc79a7"),
            panel_from_series(sac, "actor_gnorm", color="#e69f00"),
        ],
        OUT_DIR / "ctrl1_training_diagnostics.svg",
        title="CTRL-1 SAC Training Diagnostics",
    )

    svg_line_panels(
        [
            histogram_panel(df["Qec_new"], bins=60, color="#d55e00", label="Qec distribution"),
            histogram_panel(df["reward"], bins=60, color="#1f77b4", label="Reward distribution"),
        ],
        OUT_DIR / "ctrl1_distributions.svg",
        title="CTRL-1 Distributions",
    )

    if DIAG_PATH.exists():
        diag = pd.read_csv(DIAG_PATH)
        for col in ["step", "corr_qec_sno2", "mean_qec", "mean_sno2"]:
            diag[col] = pd.to_numeric(diag[col], errors="coerce")
        svg_line_panels(
            [
                panel_from_series(
                    diag,
                    "corr_qec_sno2",
                    color="#0072b2",
                    label="corr(Qec, SNO2)",
                    window=1,
                    hlines=[{"y": 0.0, "label": "zero"}],
                ),
                panel_from_series(diag, "mean_qec", color="#d55e00", label="mean Qec", window=1),
                panel_from_series(diag, "mean_sno2", color="#009e73", label="mean SNO2", window=1),
            ],
            OUT_DIR / "ctrl1_qec_sno2_diagnostics.svg",
            title="CTRL-1 Diagnostic Correlation Snapshots",
        )


def write_report(df: pd.DataFrame, baseline: pd.DataFrame | None) -> None:
    mode_counts = df["mode"].value_counts().to_dict()
    first_sac_step = df.loc[df["is_sac"], "step"].min()
    last = df.iloc[-1]
    last_1000 = df.tail(1000)
    last_5000 = df.tail(5000)
    qec_bounds_bad = int(((df["Qec_new"] < -1e-9) | (df["Qec_new"] > 5 + 1e-9)).sum())
    sno3_low = int((df["SNO_3"] < 0.5).sum())
    reward_nan = int(df["reward"].isna().sum())
    reward_clipped = int((df["reward"] <= -1.0 + 1e-9).sum())
    ratio_below_manual = int((df["ratio"] < 1.0).sum())
    qec_at_min = int((df["Qec_new"] <= 1e-9).sum())
    qec_at_max = int((df["Qec_new"] >= 5.0 - 1e-9).sum())
    corr_qec_reward = df[["Qec_new", "reward"]].corr().iloc[0, 1]
    corr_qec_snh = df[["Qec_new", "SNH"]].corr().iloc[0, 1]

    lines = [
        "# CTRL-1 Qec SAC Analysis",
        "",
        f"Source log: `{LOG_PATH}`",
        "",
        "## Scope",
        "",
        "This analysis reads only the completed CTRL-1 Qec log and static baseline/result files. It does not read CTRL-2 logs or live communication files.",
        "",
        "## Run Coverage",
        "",
        f"- Rows: {len(df):,}",
        f"- Step range: {df['step'].min():.0f} to {df['step'].max():.0f}",
        f"- Mode counts: {mode_counts}",
        f"- First SAC step: {first_sac_step:.0f}",
        f"- Final buffer value: {last['buffer']:.0f}",
        "",
        "## Key Results",
        "",
        f"- Overall mean reward: {df['reward'].mean():.4f}; median: {df['reward'].median():.4f}",
        f"- Last 1,000-step mean reward: {last_1000['reward'].mean():.4f}; median: {last_1000['reward'].median():.4f}",
        f"- Last 5,000-step mean reward: {last_5000['reward'].mean():.4f}; median: {last_5000['reward'].median():.4f}",
        f"- Overall mean Qec: {df['Qec_new'].mean():.4f} m3/d; median: {df['Qec_new'].median():.4f} m3/d",
        f"- Last 1,000-step mean Qec: {last_1000['Qec_new'].mean():.4f} m3/d",
        f"- Overall mean J ratio: {df['ratio'].mean():.4f}; last 1,000-step mean ratio: {last_1000['ratio'].mean():.4f}",
        "",
        "## Checks",
        "",
        f"- Qec out-of-bounds rows: {qec_bounds_bad}",
        f"- Qec exactly at minimum rows: {qec_at_min}; exactly at maximum rows: {qec_at_max}",
        f"- Missing reward rows: {reward_nan}",
        f"- Reward clipped at -1 rows: {reward_clipped} ({reward_clipped / len(df) * 100:.1f}%)",
        f"- J ratio below manual baseline rows: {ratio_below_manual} ({ratio_below_manual / df['ratio'].notna().sum() * 100:.1f}%)",
        f"- SNO_3 below 0.5 rows: {sno3_low}",
        f"- Final row: step {last['step']:.0f}, reward {last['reward']:.4f}, Qec {last['Qec_new']:.4f}, ratio {last['ratio']:.4f}",
        f"- Pearson corr(Qec, reward): {corr_qec_reward:.3f}; corr(Qec, SNH): {corr_qec_snh:.3f}",
        "",
        "## Manual Baseline Context",
        "",
    ]

    if baseline is not None:
        lines.extend(
            [
                "The baseline comparison uses the static `bsm2_manual_baseline_timeseries.csv` proxy metrics. The run lengths differ, so use this as a contextual comparison rather than a paired time-step comparison.",
                "",
                f"- CTRL-1 mean reward: {df['reward'].mean():.4f}; manual baseline mean reward: {baseline['reward'].mean():.4f}",
                f"- CTRL-1 mean J: {df['J'].mean():.2f}; manual baseline mean J: {baseline['J'].mean():.2f}",
                f"- CTRL-1 mean ratio: {df['ratio'].mean():.4f}; manual baseline mean ratio: {baseline['ratio'].mean():.4f}",
                f"- CTRL-1 median J: {df['J'].median():.2f}; manual baseline median J: {baseline['J'].median():.2f}",
                "",
                "Interpretation note: the reward is clipped at -1, so mean reward and mean J do not always rank runs in the same way. For this run, CTRL-1 has lower mean J/ratio than the manual baseline, while the manual baseline has better median J and higher mean reward.",
            ]
        )
    else:
        lines.append("Manual baseline timeseries was not available.")

    lines.extend(
        [
            "",
            "## Generated Files",
            "",
            "- `ctrl1_summary_metrics.csv`",
            "- `ctrl1_window_summary.csv`",
            "- `ctrl1_correlations.csv`",
            "- `ctrl1_vs_manual_baseline_proxy_comparison.csv`",
            "- `ctrl1_reward_and_qec.svg`",
            "- `ctrl1_state_trends.svg`",
            "- `ctrl1_objective_components.svg`",
            "- `ctrl1_training_diagnostics.svg`",
            "- `ctrl1_distributions.svg`",
            "- `ctrl1_qec_sno2_diagnostics.svg`",
            "",
        ]
    )
    (OUT_DIR / "ctrl1_qec_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_ctrl1()

    baseline = None
    if BASELINE_TS_PATH.exists():
        baseline = pd.read_csv(BASELINE_TS_PATH)
        for col in baseline.columns:
            converted = pd.to_numeric(baseline[col], errors="coerce")
            if converted.notna().any():
                baseline[col] = converted

    save_summary_tables(df, baseline)
    make_plots(df, baseline)
    write_report(df, baseline)

    print(f"CTRL-1 analysis complete: {OUT_DIR}")


if __name__ == "__main__":
    main()
