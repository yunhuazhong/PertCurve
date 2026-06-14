"""Plotting helpers for PertCurve outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .projection import project_points_to_curve_continuous


PLOT_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.5,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.0,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.grid": False,
}


def plot_projection(
    X,
    labels,
    curve,
    scores,
    control_label,
    title,
    out_path,
):
    """Plot PC1/PC2 cells, trajectory, and normalized pseudotime colors."""
    X = np.asarray(X)
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if X.shape[1] < 2:
        return

    with plt.rc_context(PLOT_RCPARAMS):
        is_control = labels == control_label
        perturbation_labels = sorted({str(label) for label in labels[~is_control]})
        perturbation_label = perturbation_labels[0] if len(perturbation_labels) == 1 else "Perturbation"
        fig, ax = plt.subplots(figsize=(2.8, 2.4), dpi=450)
        ax.set_facecolor("white")
        ax.set_axisbelow(True)

        vmin, vmax = -0.2, 1.2
        cmap = "viridis"

        ax.scatter(
            X[is_control, 0],
            X[is_control, 1],
            c=scores[is_control],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            marker="o",
            s=11,
            alpha=0.55,
            linewidths=0,
            label="Control",
            zorder=2,
        )
        scatter = ax.scatter(
            X[~is_control, 0],
            X[~is_control, 1],
            c=scores[~is_control],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            marker="^",
            s=14,
            alpha=0.80,
            linewidths=0,
            label=perturbation_label,
            zorder=3,
        )
        ax.plot(curve[:, 0], curve[:, 1], color="#FEA992", lw=1.6, label="Trajectory", zorder=5)

        plot_x = np.concatenate([X[:, 0], curve[:, 0]])
        plot_y = np.concatenate([X[:, 1], curve[:, 1]])
        x_lo, x_hi = np.percentile(plot_x, [1, 99])
        y_lo, y_hi = np.percentile(plot_y, [1, 99])
        x_pad = max((x_hi - x_lo) * 0.10, 1.0)
        y_pad = max((y_hi - y_lo) * 0.12, 1.0)
        ax.set_xlim(x_lo - x_pad / 5, x_hi + x_pad)
        ax.set_ylim(y_lo - y_pad / 5, y_hi + y_pad)

        center_ctrl = np.mean(X[is_control], axis=0)
        ctrl_proj, _, _ = project_points_to_curve_continuous(center_ctrl[None, :], curve)
        ax.plot(
            ctrl_proj[0, 0],
            ctrl_proj[0, 1],
            "o",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.2,
            label="Ctrl center",
            zorder=6,
        )

        center_pert = np.mean(X[~is_control], axis=0)
        pert_proj, _, _ = project_points_to_curve_continuous(center_pert[None, :], curve)
        ax.plot(
            pert_proj[0, 0],
            pert_proj[0, 1],
            "s",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.2,
            label="Pert center",
            zorder=6,
        )

        ax.set_title(title, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        cbar = fig.colorbar(scatter, ax=ax, pad=0.01, fraction=0.030, shrink=0.5)
        cbar.set_label("Normalized pseudotime", size=7)
        cbar.set_ticks([])
        ax.legend(loc="upper right", markerscale=1.0, handlelength=1.6, fontsize=6)

        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=450)
        plt.close(fig)
