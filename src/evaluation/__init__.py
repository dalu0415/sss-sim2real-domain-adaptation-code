"""Public evaluation utilities for the SSS sim-to-real experiments."""

from .core import (
    CLASS_NAMES,
    aggregate_seed_folds,
    classification_metrics,
    mean_checkpoint_probabilities,
    predict_labels,
)
from .statistics import (
    benjamini_hochberg,
    holm_adjust,
    paired_seed_differences,
    paired_t_difference,
)

__all__ = [
    "CLASS_NAMES",
    "aggregate_seed_folds",
    "benjamini_hochberg",
    "classification_metrics",
    "holm_adjust",
    "mean_checkpoint_probabilities",
    "paired_seed_differences",
    "paired_t_difference",
    "predict_labels",
]
