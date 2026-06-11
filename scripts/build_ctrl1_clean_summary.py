"""
Build a concise CTRL-1 Qec summary against the BSM2 closed-loop baseline.

The output is intentionally small: one markdown summary, one comparison CSV,
and a few plots that answer "better or worse?" before showing diagnostics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROG_RL = ROOT / "prog_RL"
LOG_PATH = PROG_RL / "logs" / "ctrl1_qec_training_optionB.csv"
BASELINE_PATH = PROG_RL / "results" / "bsm2_manual_baseline_timeseries.csv"
OUT_DIR = ROOT / "outputs" / "ctrl1_qec_clean_summary_2026-04-25"
PLOTS_DIR = OUT_DIR / "plots"

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


plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 180,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
    }
)


def load_ctrl1() -> pd.DataFrame:
    if not LOG_PATH.exists():
        raise FileNotFoundError(LOG_PATH)

    df = pd.read_csv(
        LOG_PATH,
        header=None,
        skiprows=1,
        names=EXTENDED_COLUMNS,
        engine="python",
    )
    for col in EXTENDED_COLUMNS:
        if col != "mode":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["mode"] = df["mode"].astype(str)
    df["is_sac"] = df["mode"].str.upper().eq("SAC")
    return df


def load_baseline() -> pd.DataFrame:
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(BASELINE_PATH)
    df = pd.read_csv(BASELINE_PATH)
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            df[col] = converted
    return df


def pct_change(ctrl: float, baseline: float, *, higher_is_better: bool) -> float:
    if pd.isna(ctrl) or pd.isna(baseline) or baseline == 0:
        return float("nan")
    raw = (ctrl - baseline) / abs(baseline) * 100
    return raw if higher_is_better else -raw


def build_comparison(ctrl: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "metric": "Reward",
            "direction": "higher is better",
            "ctrl1_mean": ctrl["reward"].mean(),
            "baseline_mean": baseline["reward"].mean(),
            "improvement_percent": pct_change(ctrl["reward"].mean(), baseline["reward"].mean(), higher_is_better=True),
        },
        {
            "metric": "Objective cost J",
            "direction": "lower is better",
            "ctrl1_mean": ctrl["J"].mean(),
            "baseline_mean": baseline["J"].mean(),
            "improvement_percent": pct_change(ctrl["J"].mean(), baseline["J"].mean(), higher_is_better=False),
        },
        {
            "metric": "J ratio",
            "direction": "lower is better",
            "ctrl1_mean": ctrl["ratio"].mean(),
            "baseline_mean": baseline["ratio"].mean(),
            "improvement_percent": pct_change(ctrl["ratio"].mean(), baseline["ratio"].mean(), higher_is_better=False),
        },
        {
            "metric": "EQI",
            "direction": "lower is better",
            "ctrl1_mean": ctrl["EQI"].mean(),
            "baseline_mean": baseline["EQI"].mean(),
            "improvement_percent": pct_change(ctrl["EQI"].mean(), baseline["EQI"].mean(), higher_is_better=False),
        },
        {
            "metric": "Median J",
            "direction": "lower is better",
            "ctrl1_mean": ctrl["J"].median(),
            "baseline_mean": baseline["J"].median(),
            "improvement_percent": pct_change(ctrl["J"].median(), baseline["J"].median(), higher_is_better=False),
        },
    ]
    return pd.DataFrame(rows)


def save_better_or_worse_plot(comparison: pd.DataFrame) -> None:
    plot_df = comparison.copy()
    colors = ["#2a9d8f" if value >= 0 else "#d55e00" for value in plot_df["improvement_percent"]]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(plot_df["metric"], plot_df["improvement_percent"], color=colors)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel("Improvement vs BSM2 closed-loop baseline (%)")
    ax.set_title("CTRL-1 Qec: better or worse than normal BSM2 closed-loop?")
    for index, row in plot_df.iterrows():
        value = row["improvement_percent"]
        label_x = value + (1.5 if value >= 0 else -1.5)
        ha = "left" if value >= 0 else "right"
        ax.text(label_x, index, f"{value:+.1f}%", va="center", ha=ha, fontweight="bold")
    ax.set_xlim(min(-45, plot_df["improvement_percent"].min() - 8), max(15, plot_df["improvement_percent"].max() + 8))
    ax.text(
        0.01,
        -0.17,
        "Positive means better for that metric. Reward is better when higher; J, ratio, and EQI are better when lower.",
        transform=ax.transAxes,
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_better_or_worse.png", bbox_inches="tight")
    plt.close(fig)


def save_training_plot(ctrl: pd.DataFrame, baseline: pd.DataFrame) -> None:
    x = ctrl["step"]
    roll_reward = ctrl["reward"].rolling(1000, min_periods=50).mean()
    roll_ratio = ctrl["ratio"].rolling(1000, min_periods=50).mean()
    roll_qec = ctrl["Qec_new"].rolling(1000, min_periods=50).mean()

    fig, axes = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True)
    panels = [
        (axes[0], roll_reward, baseline["reward"].mean(), "Reward", "higher is better", "#0072b2"),
        (axes[1], roll_ratio, baseline["ratio"].mean(), "J ratio", "lower is better", "#009e73"),
        (axes[2], roll_qec, baseline["Qec"].mean(), "Qec command (m3/d)", "baseline fixed at 2", "#d55e00"),
    ]

    for ax, y, baseline_value, ylabel, note, color in panels:
        ax.plot(x, y, color=color, linewidth=1.8, label="CTRL-1 rolling mean")
        ax.axhline(baseline_value, color="#444444", linestyle="--", linewidth=1.2, label="BSM2 baseline mean")
        ax.axvline(1000, color="#777777", linestyle=":", linewidth=1)
        ax.set_ylabel(ylabel)
        ax.text(0.01, 0.86, note, transform=ax.transAxes, fontsize=9, color="#4b5563")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Training step")
    fig.suptitle("CTRL-1 learning trend compared with BSM2 baseline", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_training_vs_baseline.png", bbox_inches="tight")
    plt.close(fig)


def save_process_plot(ctrl: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        ("SNO_2", "SNO_2"),
        ("SNO_1", "SNO_1"),
        ("SNO_3", "SNO_3"),
        ("SNH", "SNH_2"),
    ]
    rows = []
    for ctrl_col, base_col in metrics:
        rows.append(
            {
                "state": ctrl_col,
                "ctrl1_mean": ctrl[ctrl_col].mean(),
                "baseline_mean": baseline[base_col].mean(),
                "percent_difference": (ctrl[ctrl_col].mean() - baseline[base_col].mean()) / baseline[base_col].mean() * 100,
            }
        )
    state_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    positions = range(len(state_df))
    width = 0.36
    ax.bar([p - width / 2 for p in positions], state_df["baseline_mean"], width=width, label="BSM2 baseline", color="#9ca3af")
    ax.bar([p + width / 2 for p in positions], state_df["ctrl1_mean"], width=width, label="CTRL-1", color="#0072b2")
    ax.set_xticks(list(positions), state_df["state"])
    ax.set_ylabel("Mean concentration")
    ax.set_title("Main process states: CTRL-1 stays very close to baseline")
    for index, row in state_df.iterrows():
        top = max(row["ctrl1_mean"], row["baseline_mean"])
        ax.text(index, top * 1.03, f"{row['percent_difference']:+.2f}%", ha="center", fontsize=9, color="#374151")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_process_states_vs_baseline.png", bbox_inches="tight")
    plt.close(fig)
    return state_df


def save_distribution_plot(ctrl: pd.DataFrame, baseline: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))

    axes[0].hist(ctrl["Qec_new"].dropna(), bins=50, color="#d55e00", alpha=0.82)
    axes[0].axvline(baseline["Qec"].mean(), color="#333333", linestyle="--", linewidth=1.4, label="baseline Qec")
    axes[0].set_title("CTRL-1 uses variable Qec")
    axes[0].set_xlabel("Qec command (m3/d)")
    axes[0].set_ylabel("Rows")
    axes[0].legend()

    axes[1].hist(baseline["reward"].dropna(), bins=50, alpha=0.62, color="#9ca3af", label="BSM2 baseline")
    axes[1].hist(ctrl["reward"].dropna(), bins=50, alpha=0.68, color="#0072b2", label="CTRL-1")
    axes[1].axvline(-1, color="#333333", linestyle=":", linewidth=1)
    axes[1].set_title("Reward is worse and often clipped")
    axes[1].set_xlabel("Reward")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_qec_and_reward_distribution.png", bbox_inches="tight")
    plt.close(fig)


def write_readme(ctrl: pd.DataFrame, baseline: pd.DataFrame, comparison: pd.DataFrame, state_df: pd.DataFrame) -> None:
    last_1000 = ctrl.tail(1000)
    clipped = int((ctrl["reward"] <= -1 + 1e-9).sum())
    below_baseline_ratio = int((ctrl["ratio"] < 1.0).sum())
    total_ratio = int(ctrl["ratio"].notna().sum())
    qec_at_limits = int((ctrl["Qec_new"].le(1e-9) | ctrl["Qec_new"].ge(5 - 1e-9)).sum())

    def fmt_metric(name: str) -> str:
        row = comparison.loc[comparison["metric"].eq(name)].iloc[0]
        return f"{row['ctrl1_mean']:.3g} vs {row['baseline_mean']:.3g} ({row['improvement_percent']:+.1f}%)"

    lines = [
        "# CTRL-1 Qec clean summary",
        "",
        "This is the short version of the 2026-04-25 CTRL-1 analysis. Baseline means the normal BSM2 closed-loop run saved in `prog_RL/results/bsm2_manual_baseline_timeseries.csv`.",
        "",
        "## Main answer",
        "",
        "- **Cost objective improved on average:** mean `J` is lower than the BSM2 baseline, and mean `ratio` is also lower.",
        "- **Reward did not improve:** CTRL-1 mean reward is lower than baseline, and many rows are clipped at `-1`.",
        "- **Process states barely changed:** SNO and SNH averages are almost identical to baseline, so the controller mostly changes cost/reward behavior through `Qec`, not through a clearly different nitrogen profile.",
        "- **Conclusion:** call this run *promising but not solved*. It looks better by mean objective cost, worse by reward, and not clearly better by stable process quality.",
        "",
        "## Better or worse table",
        "",
        "| Metric | CTRL-1 vs BSM2 baseline | Direction |",
        "|---|---:|---|",
        f"| Reward | {fmt_metric('Reward')} | higher is better |",
        f"| Objective cost J | {fmt_metric('Objective cost J')} | lower is better |",
        f"| J ratio | {fmt_metric('J ratio')} | lower is better |",
        f"| EQI | {fmt_metric('EQI')} | lower is better |",
        f"| Median J | {fmt_metric('Median J')} | lower is better |",
        "",
        "## Simple takeaways",
        "",
        f"- Last 1,000 steps: reward mean `{last_1000['reward'].mean():.3f}` vs baseline reward mean `{baseline['reward'].mean():.3f}`.",
        f"- Last 1,000 steps: ratio mean `{last_1000['ratio'].mean():.3f}` vs baseline ratio mean `{baseline['ratio'].mean():.3f}`.",
        f"- CTRL-1 had `ratio < 1.0` in `{below_baseline_ratio:,}` of `{total_ratio:,}` rows (`{below_baseline_ratio / total_ratio * 100:.1f}%`), meaning the objective cost was lower than the row-wise manual reference in those rows.",
        f"- Reward was clipped at `-1` in `{clipped:,}` rows (`{clipped / len(ctrl) * 100:.1f}%`), which explains why reward and objective cost tell different stories.",
        f"- `Qec` stayed within bounds; it touched the exact 0 or 5 limits only `{qec_at_limits}` times.",
        "- Mean process-state differences vs baseline: "
        + ", ".join(f"{row.state} {row.percent_difference:+.2f}%" for row in state_df.itertuples())
        + ".",
        "",
        "## Plots",
        "",
        "1. ![Better or worse](plots/01_better_or_worse.png)",
        "2. ![Training trend](plots/02_training_vs_baseline.png)",
        "3. ![Process states](plots/03_process_states_vs_baseline.png)",
        "4. ![Qec and reward distribution](plots/04_qec_and_reward_distribution.png)",
        "",
        "## Files in this clean version",
        "",
        "- `ctrl1_clean_comparison.csv`: compact numbers behind the first plot.",
        "- `ctrl1_process_state_comparison.csv`: compact process-state comparison.",
        "- `plots/`: only four readable plots.",
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    ctrl = load_ctrl1()
    baseline = load_baseline()

    comparison = build_comparison(ctrl, baseline)
    comparison.to_csv(OUT_DIR / "ctrl1_clean_comparison.csv", index=False)

    save_better_or_worse_plot(comparison)
    save_training_plot(ctrl, baseline)
    state_df = save_process_plot(ctrl, baseline)
    state_df.to_csv(OUT_DIR / "ctrl1_process_state_comparison.csv", index=False)
    save_distribution_plot(ctrl, baseline)
    write_readme(ctrl, baseline, comparison, state_df)

    print(f"Clean CTRL-1 summary written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
