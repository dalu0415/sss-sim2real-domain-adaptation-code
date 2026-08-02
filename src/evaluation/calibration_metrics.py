"""Optional calibration, imbalance, and prediction-behaviour summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

from .core import AIRPLANE, CLASS_NAMES, SHIP, _validate_labels, _validate_probabilities


def brier_score(y_true: Any, probabilities: Any) -> dict[str, Any]:
    """Return the multiclass Brier score and true-class-conditioned scores."""

    truth = _validate_labels(y_true, name="y_true")
    probs = _validate_probabilities(probabilities)
    if len(truth) != len(probs):
        raise ValueError("y_true and probabilities must have the same length")

    one_hot = np.zeros_like(probs, dtype=np.float64)
    one_hot[np.arange(len(truth)), truth] = 1.0
    per_sample = np.sum((probs - one_hot) ** 2, axis=1)

    conditioned: dict[str, float | None] = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        mask = truth == class_index
        conditioned[class_name] = float(per_sample[mask].mean()) if mask.any() else None
    return {
        "brier": float(per_sample.mean()),
        "by_true_class": conditioned,
        "definition": "mean over samples of sum over classes (probability - one_hot)^2",
        "interpretation": "descriptive probability-quality metric; not a class-health verdict",
    }


def calibration_errors(
    y_true: Any,
    probabilities: Any,
    *,
    n_bins: int = 15,
) -> dict[str, Any]:
    """Compute equal-width top-label ECE and class-wise calibration error."""

    truth = _validate_labels(y_true, name="y_true")
    probs = _validate_probabilities(probabilities)
    if len(truth) != len(probs):
        raise ValueError("y_true and probabilities must have the same length")
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or int(n_bins) <= 0:
        raise ValueError("n_bins must be a positive integer")
    bins = int(n_bins)
    edges = np.linspace(0.0, 1.0, bins + 1)

    def binned_error(scores: np.ndarray, events: np.ndarray) -> tuple[float, list[int]]:
        assignments = np.digitize(scores, edges[1:-1], right=False)
        error = 0.0
        counts: list[int] = []
        for bin_index in range(bins):
            mask = assignments == bin_index
            count = int(mask.sum())
            if count == 0:
                continue
            counts.append(count)
            observed = float(events[mask].mean())
            expected = float(scores[mask].mean())
            error += (count / len(scores)) * abs(observed - expected)
        return float(error), counts

    predictions = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    top_label_ece, top_counts = binned_error(
        confidence,
        (predictions == truth).astype(np.float64),
    )

    by_class: dict[str, float] = {}
    by_class_counts: dict[str, list[int]] = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        error, counts = binned_error(
            probs[:, class_index],
            (truth == class_index).astype(np.float64),
        )
        by_class[class_name] = error
        by_class_counts[class_name] = counts

    sparse = {
        "top_label": bool(top_counts and min(top_counts) < 10),
        **{
            class_name: bool(counts and min(counts) < 10)
            for class_name, counts in by_class_counts.items()
        },
    }
    return {
        "ece": top_label_ece,
        "classwise_calibration_error": float(np.mean(list(by_class.values()))),
        "per_class_calibration_error": by_class,
        "n_bins": bins,
        "nonempty_bin_counts": {
            "top_label": top_counts,
            "per_class": by_class_counts,
        },
        "sparse_bin_warning": sparse,
        "binning": "equal-width bins over [0, 1]",
        "interpretation": "descriptive calibration diagnostic; pair with a proper score such as Brier",
    }


def imbalance_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """Return descriptive binary imbalance-aware aggregate metrics."""

    truth = _validate_labels(y_true, name="y_true")
    predicted = _validate_labels(y_pred, name="y_pred")
    if len(truth) != len(predicted):
        raise ValueError("y_true and y_pred must have the same length")

    _, recall, _, support = precision_recall_fscore_support(
        truth,
        predicted,
        labels=[AIRPLANE, SHIP],
        zero_division=0,
    )
    missing_classes = [
        class_name
        for class_name, count in zip(CLASS_NAMES, support)
        if int(count) == 0
    ]
    if missing_classes:
        balanced_accuracy: float | None = None
        gmean: float | None = None
        status = "undefined_class_recall"
    else:
        balanced_accuracy = float(np.mean(recall))
        gmean = float(np.sqrt(recall[AIRPLANE] * recall[SHIP]))
        status = "ok"

    mcc_value = float(matthews_corrcoef(truth, predicted))
    kappa_raw = float(cohen_kappa_score(truth, predicted))
    kappa: float | None = kappa_raw if np.isfinite(kappa_raw) else None
    if kappa is None and status == "ok":
        status = "cohen_kappa_undefined"
    return {
        "matthews_correlation_coefficient": mcc_value,
        "gmean_recall": gmean,
        "balanced_accuracy": balanced_accuracy,
        "cohen_kappa": kappa,
        "missing_true_classes": missing_classes,
        "status": status,
        "interpretation": "descriptive aggregates; minority-class claims require per-class metrics",
    }


def directed_confusion_rates(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """Return class-conditional off-diagonal error rates."""

    truth = _validate_labels(y_true, name="y_true")
    predicted = _validate_labels(y_pred, name="y_pred")
    if len(truth) != len(predicted):
        raise ValueError("y_true and y_pred must have the same length")

    result: dict[str, Any] = {}
    counts: dict[str, int] = {}
    for source_class, target_class, key in (
        (AIRPLANE, SHIP, "airplane_to_ship_rate"),
        (SHIP, AIRPLANE, "ship_to_airplane_rate"),
    ):
        mask = truth == source_class
        count = int(mask.sum())
        counts[CLASS_NAMES[source_class]] = count
        result[key] = (
            float(np.mean(predicted[mask] == target_class)) if count > 0 else None
        )
    result["true_class_counts"] = counts
    result["interpretation"] = (
        "descriptive directed confusion; None means the conditioning true class is absent"
    )
    return result


def confidence_entropy_summary(y_true: Any, probabilities: Any) -> dict[str, Any]:
    """Summarise maximum probability and Shannon entropy by class/correctness."""

    truth = _validate_labels(y_true, name="y_true")
    probs = _validate_probabilities(probabilities)
    if len(truth) != len(probs):
        raise ValueError("y_true and probabilities must have the same length")
    predicted = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    entropy_terms = np.zeros_like(probs, dtype=np.float64)
    positive = probs > 0.0
    entropy_terms[positive] = probs[positive] * np.log(probs[positive])
    entropy = -np.sum(entropy_terms, axis=1)

    def aggregate(mask: np.ndarray) -> dict[str, Any]:
        count = int(mask.sum())
        if count == 0:
            return {"mean_confidence": None, "mean_entropy_nats": None, "n": 0}
        return {
            "mean_confidence": float(confidence[mask].mean()),
            "mean_entropy_nats": float(entropy[mask].mean()),
            "n": count,
        }

    correct = predicted == truth
    return {
        "overall": aggregate(np.ones(len(truth), dtype=bool)),
        "by_true_class": {
            class_name: aggregate(truth == class_index)
            for class_index, class_name in enumerate(CLASS_NAMES)
        },
        "by_correctness": {
            "correct": aggregate(correct),
            "wrong": aggregate(~correct),
        },
        "prediction_rule": "argmax",
        "entropy_log_base": "e",
        "interpretation": "descriptive confidence/entropy summary; not a performance verdict",
    }
