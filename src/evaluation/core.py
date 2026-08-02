"""Core prediction, scoring, and fold-aggregation routines.

The paper uses the fixed class order ``airplane=0, ship=1``.  All routines
here are pure array/record transformations: they do not discover datasets,
read target labels, select checkpoints, or inspect experiment directories.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


CLASS_NAMES = ("airplane", "ship")
AIRPLANE = 0
SHIP = 1


def _validate_labels(values: Any, *, name: str, allow_empty: bool = False) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not allow_empty and raw.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f"{name} must contain numeric class labels")
    numeric = raw.astype(np.float64, copy=False)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} contains NaN or infinity")
    if not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{name} must contain integer class labels")
    labels = numeric.astype(np.int64)
    if not np.isin(labels, (AIRPLANE, SHIP)).all():
        raise ValueError(f"{name} may contain only class labels 0 and 1")
    return labels


def _validate_probabilities(
    values: Any,
    *,
    name: str = "probabilities",
    expected_ndim: int = 2,
    row_sum_atol: float = 1e-6,
) -> np.ndarray:
    probs = np.asarray(values, dtype=np.float64)
    if probs.ndim != expected_ndim or probs.shape[-1] != len(CLASS_NAMES):
        expected = "[N, 2]" if expected_ndim == 2 else "[K, N, 2]"
        raise ValueError(f"{name} must have shape {expected}")
    if any(size == 0 for size in probs.shape):
        raise ValueError(f"{name} must not have an empty dimension")
    if not np.isfinite(probs).all():
        raise ValueError(f"{name} contains NaN or infinity")
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError(f"{name} contains values outside [0, 1]")
    row_sums = probs.sum(axis=-1, dtype=np.float64)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=row_sum_atol):
        raise ValueError(f"rows of {name} must sum to 1")
    return probs


def mean_checkpoint_probabilities(
    checkpoint_probabilities: Any,
    *,
    expected_k: int | None = 5,
) -> np.ndarray:
    """Average per-checkpoint softmax probabilities in float64.

    ``checkpoint_probabilities`` must be ``[K, N, 2]`` and must already
    contain softmax probabilities, not logits.  The accumulator follows the
    supplied checkpoint order.  The paper path sets ``expected_k=5``.
    """

    probs = _validate_probabilities(
        checkpoint_probabilities,
        name="checkpoint_probabilities",
        expected_ndim=3,
    )
    if expected_k is not None:
        if isinstance(expected_k, bool) or not isinstance(expected_k, (int, np.integer)):
            raise ValueError("expected_k must be a positive integer or None")
        if int(expected_k) <= 0:
            raise ValueError("expected_k must be a positive integer or None")
        if probs.shape[0] != int(expected_k):
            raise ValueError(
                f"expected exactly {int(expected_k)} checkpoints, got {probs.shape[0]}"
            )

    total = np.zeros(probs.shape[1:], dtype=np.float64)
    for checkpoint_probs in probs:
        total += checkpoint_probs
    averaged = total / float(probs.shape[0])
    return _validate_probabilities(
        averaged,
        name="averaged probabilities",
        expected_ndim=2,
    )


def predict_labels(
    probabilities: Any,
    *,
    mode: str = "argmax",
    airplane_threshold: float = 0.5,
) -> np.ndarray:
    """Convert ``[N,2]`` probabilities to labels in the fixed class order.

    ``argmax`` is the paper's primary rule.  NumPy resolves an exact tie to
    class 0 (airplane).  ``airplane_threshold`` is diagnostic-only and is
    applied as ``p(airplane) >= threshold``.
    """

    probs = _validate_probabilities(probabilities)
    if mode == "argmax":
        return np.argmax(probs, axis=1).astype(np.int64)
    if mode != "airplane_threshold":
        raise ValueError("mode must be 'argmax' or 'airplane_threshold'")
    threshold = float(airplane_threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("airplane_threshold must be finite and within [0, 1]")
    return np.where(probs[:, AIRPLANE] >= threshold, AIRPLANE, SHIP).astype(np.int64)


def classification_metrics(
    y_true: Any,
    probabilities: Any,
    *,
    prediction_mode: str = "argmax",
    airplane_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute the paper's symmetric binary classification metrics.

    The legacy field ``airplane_pr_auc`` is retained for frozen-result
    compatibility.  Its value is scikit-learn Average Precision (AP), not
    trapezoidal area under a precision-recall curve.
    """

    truth = _validate_labels(y_true, name="y_true")
    probs = _validate_probabilities(probabilities)
    if len(truth) != len(probs):
        raise ValueError("y_true and probabilities must have the same length")
    predicted = predict_labels(
        probs,
        mode=prediction_mode,
        airplane_threshold=airplane_threshold,
    )

    precision, recall, class_f1, support = precision_recall_fscore_support(
        truth,
        predicted,
        labels=[AIRPLANE, SHIP],
        zero_division=0,
    )
    macro_f1 = f1_score(
        truth,
        predicted,
        labels=[AIRPLANE, SHIP],
        average="macro",
        zero_division=0,
    )
    matrix = confusion_matrix(truth, predicted, labels=[AIRPLANE, SHIP])

    airplane_binary = (truth == AIRPLANE).astype(np.int64)
    if 0 < int(airplane_binary.sum()) < len(airplane_binary):
        airplane_ap: float | None = float(
            average_precision_score(airplane_binary, probs[:, AIRPLANE])
        )
    else:
        airplane_ap = None

    per_class = {
        class_name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(class_f1[index]),
            "support": int(support[index]),
        }
        for index, class_name in enumerate(CLASS_NAMES)
    }
    return {
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion_matrix": matrix.astype(int).tolist(),
        "confusion_matrix_axes": "rows=true, columns=predicted",
        "class_order": list(CLASS_NAMES),
        "airplane_average_precision": airplane_ap,
        "airplane_pr_auc": airplane_ap,
        "airplane_pr_auc_definition": (
            "legacy alias for sklearn.metrics.average_precision_score "
            "(non-interpolated Average Precision)"
        ),
        "prediction_rule": {
            "mode": prediction_mode,
            "airplane_threshold": (
                float(airplane_threshold)
                if prediction_mode == "airplane_threshold"
                else None
            ),
        },
    }


def _metric_from_record(record: Mapping[str, Any], metric: str) -> float:
    if not isinstance(metric, str) or not metric:
        raise ValueError("metric must be a non-empty dotted path")
    current: Any = record.get("metrics", record)
    for part in metric.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise ValueError(f"record is missing metric '{metric}'")
        current = current[part]
    if isinstance(current, bool) or not isinstance(current, (int, float, np.number)):
        raise ValueError(f"metric '{metric}' must be numeric")
    value = float(current)
    if not np.isfinite(value):
        raise ValueError(f"metric '{metric}' must be finite")
    return value


def _normalise_expected_ids(values: Iterable[int], *, name: str) -> tuple[int, ...]:
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must contain integers")
        result.append(int(value))
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{name} must be non-empty and unique")
    return tuple(result)


def aggregate_seed_folds(
    records: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    expected_folds: Iterable[int] = range(5),
) -> list[dict[str, Any]]:
    """Average one metric over a complete fold set for every arm/seed.

    Duplicate, missing, or extra folds are rejected.  A record's optional
    ``collapsed`` flag is reported but never used to filter the average.
    """

    folds_expected = _normalise_expected_ids(expected_folds, name="expected_folds")
    expected_set = set(folds_expected)
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        raise ValueError("records must be a non-empty sequence")

    grouped: dict[tuple[str, int], dict[int, tuple[float, bool]]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"record {index} must be an object")
        arm = record.get("arm")
        seed = record.get("seed")
        fold = record.get("fold")
        if not isinstance(arm, str) or not arm.strip():
            raise ValueError(f"record {index} has an invalid arm")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError(f"record {index} has a non-integer seed")
        if isinstance(fold, bool) or not isinstance(fold, (int, np.integer)):
            raise ValueError(f"record {index} has a non-integer fold")
        key = (arm, int(seed))
        fold_int = int(fold)
        grouped.setdefault(key, {})
        if fold_int in grouped[key]:
            raise ValueError(f"duplicate record for arm={arm}, seed={seed}, fold={fold_int}")
        collapsed = record.get("collapsed", False)
        if not isinstance(collapsed, (bool, np.bool_)):
            raise ValueError(f"record {index} has a non-boolean collapsed flag")
        grouped[key][fold_int] = (_metric_from_record(record, metric), bool(collapsed))

    aggregated: list[dict[str, Any]] = []
    for (arm, seed), by_fold in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        actual_set = set(by_fold)
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        if missing or extra:
            raise ValueError(
                f"arm={arm}, seed={seed} has invalid folds; missing={missing}, extra={extra}"
            )
        values = [by_fold[fold][0] for fold in folds_expected]
        collapsed_folds = [fold for fold in folds_expected if by_fold[fold][1]]
        aggregated.append(
            {
                "arm": arm,
                "seed": seed,
                "metric": metric,
                "mean": float(np.mean(values, dtype=np.float64)),
                "n_folds": len(folds_expected),
                "folds": list(folds_expected),
                "fold_values": [float(value) for value in values],
                "collapsed_folds": collapsed_folds,
                "collapsed_folds_included": True,
            }
        )
    return aggregated
