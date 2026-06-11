"""
Generate supervisor-facing validation plots and summary tables for the
four single-agent controllers.

Outputs are written to:
    prog_RL/docs/validation/
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover - optional
    torch = None


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
CKPT_DIR = ROOT / "checkpoints"
OUT_DIR = ROOT / "docs" / "validation"


@dataclass(frozen=True)
class ControllerSpec:
    name: str
    label: str
    log_path: Path
    checkpoint_path: Path
    action_candidates: tuple[str, ...]
    action_bounds: tuple[float, float]
    state_columns: tuple[str, ...]
    notes: str = ""


SPECS = (
    ControllerSpec(
        name="ctrl1_qec",
        label="CTRL-1 / Qec",
        log_path=LOG_DIR / "ctrl1_qec_training_optionB.csv",
        checkpoint_path=CKPT_DIR / "ctrl1_qec_sac_optionB.pt",
        action_candidates=("Qec_new", "Qec"),
        action_bounds=(0.0, 5.0),
        state_columns=("SNO_2", "SNO_1", "SNO_3", "CODTN"),
        notes="Restored full validated run from the archived BSM2 workspace.",
    ),
    ControllerSpec(
        name="ctrl2_qint",
        label="CTRL-2 / Qint",
        log_path=LOG_DIR / "ctrl2_qint_training.csv",
        checkpoint_path=CKPT_DIR / "ctrl2_qint_sac.pt",
        action_candidates=("Qint_new", "Qint"),
        action_bounds=(5000.0, 61944.0),
        state_columns=("SNO_2", "SNO_1", "SNO_3", "CODTN"),
    ),
    ControllerSpec(
        name="ctrl3_do",
        label="CTRL-3 / DO / SO4ref",
        log_path=LOG_DIR / "ctrl3_do_training.csv",
        checkpoint_path=CKPT_DIR / "ctrl3_do_sac.pt",
        action_candidates=("SO4ref_new",),
        action_bounds=(0.0, 10.0),
        state_columns=("SO_4", "SNH_4", "SNH_5", "SNO_3", "TSS_3"),
    ),
    ControllerSpec(
        name="ctrl4_qw",
        label="CTRL-4 / Qw",
        log_path=LOG_DIR / "ctrl4_qw_training.csv",
        checkpoint_path=CKPT_DIR / "ctrl4_qw_sac.pt",
        action_candidates=("Qw_new",),
        action_bounds=(0.0, 450.0),
        state_columns=("TSS_5_1d", "SND_5_1d", "SRT_3d", "SNH_5_1d", "EQI_1d"),
        notes="Uses the clean resumed segment so plant time is continuous.",
    ),
)


def pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"None of the candidate columns exist: {tuple(candidates)}")


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    step = max(1, len(x) // max_points)
    return x[::step], y[::step]


def checkpoint_total_steps(path: Path) -> float:
    if not path.exists() or torch is None:
        return math.nan
    try:
        ck = torch.load(path, map_location="cpu")
        return float(ck.get("total_steps", math.nan))
    except Exception:
        return math.nan


def make_controller_plot(spec: ControllerSpec, df: pd.DataFrame, action_col: str, summary_row: dict) -> Path:
    x = numeric_series(df, "step")
    if x.isna().all():
        x = pd.Series(np.arange(len(df), dtype=float))

    reward = numeric_series(df, "reward")
    action = numeric_series(df, action_col)
    mode = df.get("mode", pd.Series([""] * len(df))).astype(str).str.upper()
    first_sac = numeric_series(df.loc[mode == "SAC"], "step")
    first_sac_step = first_sac.iloc[0] if not first_sac.empty else math.nan

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(spec.label, fontsize=15, fontweight="bold")

    x_reward, y_reward = downsample_xy(x.to_numpy(), reward.to_numpy())
    axes[0].plot(x_reward, y_reward, color="#1f77b4", linewidth=1.2)
    axes[0].set_ylabel("Reward")
    axes[0].grid(True, alpha=0.25)
    axes[0].set_title(
        f"Rows={summary_row['rows']}  Last step={summary_row['last_step']}  "
        f"SAC rows={summary_row['sac_rows']}  Checkpoint step={summary_row['checkpoint_total_steps']}"
    )

    x_action, y_action = downsample_xy(x.to_numpy(), action.to_numpy())
    axes[1].plot(x_action, y_action, color="#ff7f0e", linewidth=1.2)
    axes[1].axhline(spec.action_bounds[0], color="tab:red", linestyle="--", linewidth=1.0)
    axes[1].axhline(spec.action_bounds[1], color="tab:red", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Action")
    axes[1].grid(True, alpha=0.25)
    axes[1].set_title(f"Observed action range: [{summary_row['action_min']:.4g}, {summary_row['action_max']:.4g}]")

    loss_critic = numeric_series(df, "loss_critic") if "loss_critic" in df.columns else pd.Series(dtype=float)
    loss_actor = numeric_series(df, "loss_actor") if "loss_actor" in df.columns else pd.Series(dtype=float)
    alpha = numeric_series(df, "alpha") if "alpha" in df.columns else pd.Series(dtype=float)

    valid = ~(loss_critic.isna() & loss_actor.isna() & alpha.isna())
    if valid.any():
        x_loss = x[valid].to_numpy()
        lc = loss_critic[valid].to_numpy()
        la = loss_actor[valid].to_numpy()
        al = alpha[valid].to_numpy()
        x_loss, lc = downsample_xy(x_loss, lc)
        _, la = downsample_xy(x[valid].to_numpy(), la)
        _, al = downsample_xy(x[valid].to_numpy(), al)
        axes[2].plot(x_loss, lc, label="critic", color="#2ca02c", linewidth=1.1)
        axes[2].plot(x_loss, la, label="actor", color="#d62728", linewidth=1.1)
        ax2 = axes[2].twinx()
        ax2.plot(x_loss, al, label="alpha", color="#9467bd", linewidth=1.1)
        ax2.set_ylabel("Alpha")
        axes[2].legend(loc="upper left")
        ax2.legend(loc="upper right")
    else:
        axes[2].text(0.5, 0.5, "No SAC loss data yet", ha="center", va="center", transform=axes[2].transAxes)

    axes[2].set_ylabel("Loss")
    axes[2].set_xlabel("Training step")
    axes[2].grid(True, alpha=0.25)

    if not math.isnan(first_sac_step):
        for ax in axes:
            ax.axvline(first_sac_step, color="0.35", linestyle=":", linewidth=1.0)

    if spec.notes:
        fig.text(0.01, 0.01, f"Note: {spec.notes}", fontsize=9)

    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    out_path = OUT_DIR / f"{spec.name}_validation.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def make_overview_plot(summary_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(len(SPECS), 2, figsize=(14, 3.6 * len(SPECS)))
    if len(SPECS) == 1:
        axes = np.array([axes])

    for row_idx, spec in enumerate(SPECS):
        df = pd.read_csv(spec.log_path)
        action_col = pick_column(df, spec.action_candidates)
        x = pd.to_numeric(df["step"], errors="coerce")
        if x.isna().all():
            x = pd.Series(np.arange(len(df), dtype=float))
        reward = pd.to_numeric(df["reward"], errors="coerce")
        action = pd.to_numeric(df[action_col], errors="coerce")
        xr, yr = downsample_xy(x.to_numpy(), reward.to_numpy(), max_points=1200)
        xa, ya = downsample_xy(x.to_numpy(), action.to_numpy(), max_points=1200)

        ax_r = axes[row_idx, 0]
        ax_a = axes[row_idx, 1]
        ax_r.plot(xr, yr, color="#1f77b4", linewidth=1.0)
        ax_r.set_title(f"{spec.label} - Reward")
        ax_r.grid(True, alpha=0.25)

        ax_a.plot(xa, ya, color="#ff7f0e", linewidth=1.0)
        ax_a.axhline(spec.action_bounds[0], color="tab:red", linestyle="--", linewidth=0.9)
        ax_a.axhline(spec.action_bounds[1], color="tab:red", linestyle="--", linewidth=0.9)
        ax_a.set_title(f"{spec.label} - Action")
        ax_a.grid(True, alpha=0.25)

    axes[-1, 0].set_xlabel("Training step")
    axes[-1, 1].set_xlabel("Training step")
    fig.suptitle("Single-Agent Controller Validation Overview", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = OUT_DIR / "single_agent_validation_overview.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def summarize_controller(spec: ControllerSpec) -> tuple[dict, Path]:
    df = pd.read_csv(spec.log_path)
    action_col = pick_column(df, spec.action_candidates)
    reward = numeric_series(df, "reward")
    action = numeric_series(df, action_col)
    mode = df.get("mode", pd.Series([""] * len(df))).astype(str).str.upper()
    sac_rows = int((mode == "SAC").sum())
    state_cols_present = all(col in df.columns for col in spec.state_columns)
    bounds_ok = True
    if action.notna().any():
        bounds_ok = bool((action.dropna() >= spec.action_bounds[0] - 1e-9).all() and (action.dropna() <= spec.action_bounds[1] + 1e-9).all())

    recent_reward = reward.dropna().tail(50)
    summary = {
        "controller": spec.label,
        "log_file": spec.log_path.name,
        "rows": int(len(df)),
        "last_step": int(pd.to_numeric(df["step"], errors="coerce").dropna().iloc[-1]),
        "last_mode": str(df["mode"].iloc[-1]) if "mode" in df.columns else "",
        "sac_rows": sac_rows,
        "checkpoint_total_steps": int(checkpoint_total_steps(spec.checkpoint_path))
        if not math.isnan(checkpoint_total_steps(spec.checkpoint_path))
        else "",
        "action_column": action_col,
        "action_min": float(action.dropna().min()),
        "action_max": float(action.dropna().max()),
        "action_bounds": f"[{spec.action_bounds[0]}, {spec.action_bounds[1]}]",
        "action_within_bounds": bounds_ok,
        "reward_min": float(reward.dropna().min()),
        "reward_max": float(reward.dropna().max()),
        "reward_last": float(reward.dropna().iloc[-1]),
        "reward_mean_last50": float(recent_reward.mean()) if not recent_reward.empty else math.nan,
        "state_columns_present": state_cols_present,
        "checkpoint_exists": spec.checkpoint_path.exists(),
        "notes": spec.notes,
    }
    plot_path = make_controller_plot(spec, df, action_col, summary)
    return summary, plot_path


def write_summary_files(summary_df: pd.DataFrame, overview_path: Path, plot_paths: list[Path]) -> None:
    summary_csv = OUT_DIR / "single_agent_validation_summary.csv"
    summary_md = OUT_DIR / "single_agent_validation_summary.md"
    summary_df.to_csv(summary_csv, index=False)

    lines = [
        "# Single-Agent Controller Validation Summary",
        "",
        "This folder contains the evidence pack used to show that the four single-agent controllers operate independently and produce valid logged data.",
        "",
        "## Main conclusions",
        "",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"- **{row['controller']}**: rows={row['rows']}, last_step={row['last_step']}, "
            f"SAC rows={row['sac_rows']}, action range=[{row['action_min']:.4g}, {row['action_max']:.4g}], "
            f"reward range=[{row['reward_min']:.4g}, {row['reward_max']:.4g}], bounds_ok={row['action_within_bounds']}."
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- Overview figure: `{overview_path.name}`",
        ]
    )
    for plot_path in plot_paths:
        lines.append(f"- Controller plot: `{plot_path.name}`")

    lines.extend(
        [
            f"- Summary table: `{summary_csv.name}`",
            "",
            "## Recommended presentation order",
            "",
            "1. Show the overview figure to prove all controllers reached active SAC operation.",
            "2. Use the summary table to highlight action bounds, SAC rows, and reward ranges.",
            "3. Open the controller-specific figure only if a supervisor asks for detail.",
            "",
        ]
    )

    summary_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    plot_paths = []
    for spec in SPECS:
        summary, plot_path = summarize_controller(spec)
        summary_rows.append(summary)
        plot_paths.append(plot_path)

    summary_df = pd.DataFrame(summary_rows)
    overview_path = make_overview_plot(summary_df)
    write_summary_files(summary_df, overview_path, plot_paths)

    print("Validation pack generated:")
    print(f"  Output dir: {OUT_DIR}")
    print(f"  Overview : {overview_path.name}")
    for plot_path in plot_paths:
        print(f"  Plot     : {plot_path.name}")
    print("  Summary  : single_agent_validation_summary.csv")
    print("  Report   : single_agent_validation_summary.md")


if __name__ == "__main__":
    main()
