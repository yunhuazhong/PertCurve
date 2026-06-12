"""Plotting helpers for PertCurve outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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

    is_control = labels == control_label
    fig, ax = plt.subplots(figsize=(4.0, 3.2), dpi=200)
    vmin = min(float(np.min(scores)), 0.0)
    vmax = max(float(np.max(scores)), 1.0)

    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=scores,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        s=np.where(is_control, 10, 14),
        marker="o",
        alpha=0.8,
        linewidths=0,
    )
    ax.scatter(X[~is_control, 0], X[~is_control, 1], facecolors="none", edgecolors="black", s=18, linewidths=0.4)
    ax.plot(curve[:, 0], curve[:, 1], color="#222222", linewidth=1.5, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label("Normalized pseudotime")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
