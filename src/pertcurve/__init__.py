"""PertCurve: trajectory-based perturbation scoring."""

from .curve import principal_curve_binned
from .projection import (
    get_polyline_cumulative_length,
    normalize_arc_lengths_unclipped,
    project_points_to_curve_continuous,
)
from .scoring import compute_distance_stats, score_perturbation
from .metrics import (
    balanced_subsample_indices,
    evaluate_trajectory_quality,
    neighbor_jaccard,
    pseudotime_mutual_information,
    smoothness_score,
    trajectory_reconstruction_mse,
)

__all__ = [
    "principal_curve_binned",
    "get_polyline_cumulative_length",
    "normalize_arc_lengths_unclipped",
    "project_points_to_curve_continuous",
    "compute_distance_stats",
    "score_perturbation",
    "balanced_subsample_indices",
    "evaluate_trajectory_quality",
    "neighbor_jaccard",
    "pseudotime_mutual_information",
    "smoothness_score",
    "trajectory_reconstruction_mse",
]
