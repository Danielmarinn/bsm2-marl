from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "logs" / "ctrl2_qint_training.csv"
OUT_PNG = ROOT / "results" / "ctrl2_qint_training_overview.png"
OUT_MD = ROOT / "results" / "ctrl2_qint_summary.md"


def main() -> None:
    df = pd.read_csv(LOG_PATH)
    plot_df = df.dropna(subset=["ratio"]).copy()
    plot_df["reward_ma"] = plot_df["reward"].rolling(10, min_periods=1).mean()
    plot_df["ratio_ma"] = plot_df["ratio"].rolling(10, min_periods=1).mean()
    plot_df["qint_ma"] = plot_df["Qint"].rolling(10, min_periods=1).mean()

    start = plot_df.iloc[0]
    end = plot_df.iloc[-1]
    recent = plot_df.tail(min(10, len(plot_df)))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    fig.patch.set_facecolor("#fbf8f3")

    for ax in axes:
        ax.set_facecolor("#fffdf8")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    x = plot_df["step"]

    axes[0].plot(x, plot_df["reward"], color="#c47c2c", alpha=0.25, linewidth=2, label="Raw logged reward")
    axes[0].plot(x, plot_df["reward_ma"], color="#8c4f16", linewidth=2.5, label="10-step moving average")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("CTRL-2 (Qint) SAC Training Snapshot", fontsize=16, weight="bold")
    axes[0].text(
        0.01,
        0.92,
        f"Reward: {start['reward']:.3f} -> {end['reward']:.3f}",
        transform=axes[0].transAxes,
        fontsize=10,
        color="#5b3a14",
    )
    axes[0].legend(loc="upper right", frameon=False)

    axes[1].plot(x, plot_df["ratio"], color="#2b7a78", alpha=0.25, linewidth=2, label="Raw logged ratio")
    axes[1].plot(x, plot_df["ratio_ma"], color="#14514f", linewidth=2.5, label="10-step moving average")
    axes[1].set_ylabel("J / J_manual")
    axes[1].text(
        0.01,
        0.92,
        f"Ratio: {start['ratio']:.3f} -> {end['ratio']:.3f}",
        transform=axes[1].transAxes,
        fontsize=10,
        color="#0f403f",
    )
    axes[1].legend(loc="upper right", frameon=False)

    axes[2].plot(x, plot_df["Qint"], color="#4e79a7", alpha=0.25, linewidth=2, label="Raw logged Qint")
    axes[2].plot(x, plot_df["qint_ma"], color="#1f4e79", linewidth=2.5, label="10-step moving average")
    axes[2].set_ylabel("Qint (m3/d)")
    axes[2].set_xlabel("Training Step")
    axes[2].text(
        0.01,
        0.92,
        f"Qint: {start['Qint']:.0f} -> {end['Qint']:.0f} m3/d",
        transform=axes[2].transAxes,
        fontsize=10,
        color="#183b59",
    )
    axes[2].legend(loc="upper right", frameon=False)

    fig.suptitle(
        "Ongoing result from the tracked example log in logs/ctrl2_qint_training.csv",
        fontsize=11,
        y=0.98,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = f"""# CTRL-2 Qint Ongoing Result

This summary is generated from `logs/ctrl2_qint_training.csv`.

## Snapshot

- Logged steps with valid reward ratio: {len(plot_df)}
- Step range: {int(start['step'])} to {int(end['step'])}
- Reward improved from `{start['reward']:.3f}` to `{end['reward']:.3f}`
- `J / J_manual` decreased from `{start['ratio']:.3f}` to `{end['ratio']:.3f}`
- `Qint` moved from `{start['Qint']:.0f}` to `{end['Qint']:.0f}` m3/d

## Recent window

- Mean reward over the last {len(recent)} logged steps: `{recent['reward'].mean():.3f}`
- Mean `J / J_manual` over the last {len(recent)} logged steps: `{recent['ratio'].mean():.3f}`
- Mean `Qint` over the last {len(recent)} logged steps: `{recent['Qint'].mean():.0f}` m3/d

## Interpretation

This is an ongoing, non-final training snapshot, but it already shows the controller moving away from the default high recirculation setting while improving the reward and reducing the normalized operating-cost metric.
"""
    OUT_MD.write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
