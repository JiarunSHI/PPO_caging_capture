#!/usr/bin/env python3
"""Build a compact vector curriculum summary for the IEEE paper."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "artifacts/jailer_joint_v14_reward_v2/analysis/stage_summary.csv"
OUT = ROOT / "docs/paper/figures/curriculum_summary_ieee.pdf"


def main() -> None:
    frame = pd.read_csv(CSV)
    x = range(len(frame))
    labels = frame["short_stage"].tolist()
    colors = (["#4C78A8"] * 10) + (["#72B7B2"] * 10) + (["#F58518"] * 10)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(3, 1, figsize=(7.16, 4.65), sharex=True)

    axes[0].bar(x, frame["tail10_success_rate_pct"], color=colors, width=0.82)
    axes[0].axhline(75.0, color="#B22222", linestyle="--", linewidth=1.0, label="promotion target")
    axes[0].set_ylabel("Success (%)")
    axes[0].set_ylim(0, 100)
    axes[0].legend(loc="lower right", frameon=False)

    axes[1].plot(x, frame["tail10_position_mean_mm"], color="#1F4E79", marker="o", markersize=2.4, linewidth=1.0, label="terminal mean")
    axes[1].plot(x, frame["promotion_position_mm"], color="#B22222", linestyle="--", linewidth=1.0, label="promotion bound")
    axes[1].set_ylabel("Position (mm)")
    axes[1].legend(loc="upper left", ncol=2, frameon=False)

    pose = frame["promotion_orientation_deg"].notna()
    axes[2].plot(frame.index[pose], frame.loc[pose, "tail10_orientation_mean_deg"], color="#8C4A00", marker="o", markersize=2.4, linewidth=1.0, label="terminal mean")
    axes[2].plot(frame.index[pose], frame.loc[pose, "promotion_orientation_deg"], color="#B22222", linestyle="--", linewidth=1.0, label="promotion bound")
    axes[2].set_ylabel("Rotation (deg)")
    axes[2].set_xlabel("Curriculum stage")
    axes[2].legend(loc="upper left", ncol=2, frameon=False)

    for axis in axes:
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.45)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.axvline(9.5, color="#777777", linewidth=0.7)
        axis.axvline(19.5, color="#777777", linewidth=0.7)
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(labels, rotation=60, ha="right")
    axes[0].text(4.5, 96, "masked position", ha="center", va="top")
    axes[0].text(14.5, 96, "observed position", ha="center", va="top")
    axes[0].text(24.5, 96, "joint pose", ha="center", va="top")
    fig.tight_layout(pad=0.55, h_pad=0.35)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
