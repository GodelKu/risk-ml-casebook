"""riskeval: honest evaluation of binary risk models."""

from .bootstrap import Interval, bootstrap_ci
from .drift import psi, temporal_metrics
from .metrics import brier, calibration_table, ece, log_loss, pr_auc, roc_auc
from .slices import slice_metrics
from .thresholds import (
    bayes_threshold,
    evaluate_threshold,
    threshold_at_fpr,
    threshold_by_cost,
)

__version__ = "0.1.0"

__all__ = [
    "Interval",
    "bayes_threshold",
    "bootstrap_ci",
    "brier",
    "calibration_table",
    "ece",
    "evaluate_threshold",
    "log_loss",
    "pr_auc",
    "psi",
    "roc_auc",
    "slice_metrics",
    "temporal_metrics",
    "threshold_at_fpr",
    "threshold_by_cost",
]
