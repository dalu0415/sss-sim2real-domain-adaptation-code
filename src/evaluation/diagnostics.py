"""Read-only threshold, collapse, and target-label oracle diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .core import (
    AIRPLANE,
    SHIP,
    _metric_from_record,
    _validate_labels,
    _validate_probabilities,
    classification_metrics,
    predict_labels,
)


def _collapse_thresholds(low: float, high: float) -> tuple[float, float]:
    low_value = float(low)
    high_value = float(high)
    if (
        not np.isfinite(low_value)
        or not np.isfinite(high_value)
        or not 0.0 <= low_value < high_value <= 1.0
    ):
        raise ValueError("collapse thresholds must satisfy 0 <= low < high <= 1")
    return low_value, high_value


def _count(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a non-negative integer count")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0.0 or numeric != np.floor(numeric):
        raise ValueError(f"{name} must be a finite non-negative integer count")
    return int(numeric)


def collapse_from_counts(
    airplane_count: Any,
    ship_count: Any,
    *,
    low: float = 0.02,
    high: float = 0.60,
) -> dict[str, Any]:
    """Apply the study's closed collapse thresholds to a prediction histogram."""

    airplane = _count(airplane_count, name="airplane_count")
    ship = _count(ship_count, name="ship_count")
    total = airplane + ship
    if total == 0:
        raise ValueError("prediction counts must have a positive total")
    low_value, high_value = _collapse_thresholds(low, high)
    fraction = airplane / total
    return {
        "airplane_count": airplane,
        "ship_count": ship,
        "total": total,
        "airplane_fraction": float(fraction),
        "collapsed": bool(fraction <= low_value or fraction >= high_value),
        "low_threshold_inclusive": low_value,
        "high_threshold_inclusive": high_value,
    }


def collapse_from_predictions(
    y_pred: Any,
    *,
    low: float = 0.02,
    high: float = 0.60,
) -> dict[str, Any]:
    """Count fixed-order predictions and apply the study's collapse rule."""

    predicted = _validate_labels(y_pred, name="y_pred")
    return collapse_from_counts(
        int(np.sum(predicted == AIRPLANE)),
        int(np.sum(predicted == SHIP)),
        low=low,
        high=high,
    )


def epoch_prediction_collapse_routes(
    epoch_counts: Sequence[Mapping[str, Any]],
    *,
    low: float = 0.02,
    high: float = 0.60,
) -> dict[str, Any]:
    """Derive the last-epoch and any-epoch single-class collapse routes."""

    if (
        not isinstance(epoch_counts, Sequence)
        or isinstance(epoch_counts, (str, bytes))
        or not epoch_counts
    ):
        raise ValueError("epoch_counts must be a non-empty sequence")
    parsed: dict[int, dict[str, Any]] = {}
    expected_total: int | None = None
    for index, record in enumerate(epoch_counts):
        if not isinstance(record, Mapping):
            raise ValueError(f"epoch_counts[{index}] must be an object")
        epoch = record.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, (int, np.integer)):
            raise ValueError(f"epoch_counts[{index}] has a non-integer epoch")
        epoch_int = int(epoch)
        if epoch_int in parsed:
            raise ValueError(f"duplicate epoch {epoch_int}")
        route = collapse_from_counts(
            record.get("airplane_count"),
            record.get("ship_count"),
            low=low,
            high=high,
        )
        if expected_total is None:
            expected_total = route["total"]
        elif route["total"] != expected_total:
            raise ValueError(
                "all epoch prediction histograms must have the same positive total"
            )
        parsed[epoch_int] = route

    epochs = sorted(parsed)
    last_epoch = epochs[-1]
    exact_single_class_epochs = [
        epoch
        for epoch in epochs
        if parsed[epoch]["airplane_count"] == 0 or parsed[epoch]["ship_count"] == 0
    ]
    return {
        "epoch_prediction_histograms": [
            {
                "epoch": epoch,
                "airplane_count": parsed[epoch]["airplane_count"],
                "ship_count": parsed[epoch]["ship_count"],
            }
            for epoch in epochs
        ],
        "last_epoch_prediction_fraction": {
            "epoch": last_epoch,
            **parsed[last_epoch],
        },
        "any_epoch_exact_single_class": {
            "collapsed": bool(exact_single_class_epochs),
            "collapsed_epochs": exact_single_class_epochs,
            "n_epochs_scanned": len(epochs),
            "definition": "at least one epoch predicted exactly one class for every target sample",
        },
        "target_count_per_epoch": expected_total,
    }


def triangulate_collapse(routes: Sequence[bool | None]) -> dict[str, Any]:
    """Triangulate collapse evidence, requiring at least two valid routes."""

    if not isinstance(routes, Sequence) or isinstance(routes, (str, bytes)):
        raise ValueError("routes must be a sequence of booleans or None")
    valid: list[bool] = []
    for index, route in enumerate(routes):
        if route is None:
            continue
        if not isinstance(route, (bool, np.bool_)):
            raise ValueError(f"route {index} must be boolean or None")
        valid.append(bool(route))

    unanimous = bool(valid) and all(route == valid[0] for route in valid)
    if len(valid) < 2:
        verdict: bool | None = None
    else:
        true_votes = sum(valid)
        false_votes = len(valid) - true_votes
        if true_votes == false_votes:
            verdict = None
        else:
            verdict = bool(true_votes > false_votes)
    disagreement_warning = len(valid) < 2 or not unanimous
    return {
        "verdict": verdict,
        "unanimous": unanimous,
        "disagreement_warning": bool(disagreement_warning),
        "n_valid_routes": len(valid),
        "true_votes": int(sum(valid)),
        "false_votes": int(len(valid) - sum(valid)),
        "minimum_routes_for_verdict": 2,
    }


def _authorise_target_label_diagnostic(authorised: bool, reason: str) -> None:
    if authorised is not True:
        raise PermissionError("target-label diagnostic requires authorised=True")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("target-label diagnostic requires a non-empty reason")


def threshold_sweep(
    y_true: Any,
    probabilities: Any,
    thresholds: Any,
    *,
    authorised: bool = False,
    reason: str = "",
) -> dict[str, Any]:
    """Evaluate fixed thresholds without selecting or recommending one."""

    _authorise_target_label_diagnostic(authorised, reason)
    truth = _validate_labels(y_true, name="y_true")
    probs = _validate_probabilities(probabilities)
    if len(truth) != len(probs):
        raise ValueError("y_true and probabilities must have the same length")
    threshold_values = np.asarray(thresholds, dtype=np.float64)
    if threshold_values.ndim != 1 or threshold_values.size == 0:
        raise ValueError("thresholds must be a non-empty one-dimensional array")
    if not np.isfinite(threshold_values).all() or np.any(threshold_values < 0.0) or np.any(threshold_values > 1.0):
        raise ValueError("thresholds must be finite and within [0, 1]")
    if len(np.unique(threshold_values)) != len(threshold_values):
        raise ValueError("thresholds must be unique")

    rows: list[dict[str, Any]] = []
    for threshold in threshold_values:
        metrics = classification_metrics(
            truth,
            probs,
            prediction_mode="airplane_threshold",
            airplane_threshold=float(threshold),
        )
        predicted = predict_labels(
            probs,
            mode="airplane_threshold",
            airplane_threshold=float(threshold),
        )
        rows.append(
            {
                "threshold": float(threshold),
                "macro_f1": metrics["macro_f1"],
                "per_class": metrics["per_class"],
                "airplane_prediction_fraction": float(np.mean(predicted == AIRPLANE)),
            }
        )
    return {
        "rows": rows,
        "reason": reason.strip(),
        "read_only": True,
        "selection_performed": False,
        "warning": "diagnostic only; do not use target labels to select the deployed threshold",
    }


def oracle_best_epochs(
    epoch_records: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    *,
    authorised: bool = False,
    reason: str = "",
) -> dict[str, Any]:
    """Report an independent best epoch for each metric as an optimistic oracle."""

    _authorise_target_label_diagnostic(authorised, reason)
    if not isinstance(epoch_records, Sequence) or isinstance(epoch_records, (str, bytes)) or not epoch_records:
        raise ValueError("epoch_records must be a non-empty sequence")
    if not isinstance(metrics, Sequence) or isinstance(metrics, (str, bytes)) or not metrics:
        raise ValueError("metrics must be a non-empty sequence of dotted paths")
    metric_names = list(metrics)
    if not all(isinstance(metric, str) and metric for metric in metric_names):
        raise ValueError("metrics must contain non-empty strings")
    if len(metric_names) != len(set(metric_names)):
        raise ValueError("metrics must be unique")

    by_epoch: dict[int, Mapping[str, Any]] = {}
    for index, record in enumerate(epoch_records):
        if not isinstance(record, Mapping):
            raise ValueError(f"epoch record {index} must be an object")
        epoch = record.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, (int, np.integer)):
            raise ValueError(f"epoch record {index} has a non-integer epoch")
        epoch_int = int(epoch)
        if epoch_int in by_epoch:
            raise ValueError(f"duplicate epoch {epoch_int}")
        by_epoch[epoch_int] = record

    best: dict[str, Any] = {}
    for metric in metric_names:
        candidates = [
            (epoch, _metric_from_record(record, metric))
            for epoch, record in by_epoch.items()
        ]
        best_value = max(value for _, value in candidates)
        best_epoch = min(epoch for epoch, value in candidates if value == best_value)
        best[metric] = {
            "epoch": best_epoch,
            "value": float(best_value),
            "tie_break": "earliest epoch",
        }
    return {
        "best_by_metric": best,
        "reason": reason.strip(),
        "read_only": True,
        "single_deployable_checkpoint": False,
        "warning": "each metric may select a different epoch; this is an optimistic upper-bound diagnostic",
    }
