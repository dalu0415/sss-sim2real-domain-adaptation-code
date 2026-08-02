"""Descriptive feature-distribution and label-shift diagnostics.

These quantities do not adjudicate model performance.  In particular, a
smaller domain discrepancy does not imply better minority-class recall.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core import AIRPLANE, CLASS_NAMES, SHIP, _validate_labels, _validate_probabilities


def _feature_matrix(values: Any, *, name: str, min_samples: int = 1) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < min_samples or matrix.shape[1] == 0:
        raise ValueError(
            f"{name} must have shape [N, D] with N >= {min_samples} and D >= 1"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return matrix


def _feature_pair(
    features_source: Any,
    features_target: Any,
    *,
    min_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    source = _feature_matrix(features_source, name="features_source", min_samples=min_samples)
    target = _feature_matrix(features_target, name="features_target", min_samples=min_samples)
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target feature dimensions differ")
    return source, target


def proxy_a_distance(
    features_source: Any,
    features_target: Any,
    *,
    seed: int = 0,
    holdout_fraction: float = 0.3,
) -> dict[str, Any]:
    """Estimate Proxy-A-distance with a held-out linear domain classifier.

    The returned estimate follows ``2 * (1 - 2 * error)`` without clipping;
    finite-sample estimates can therefore be negative when held-out error is
    above chance.  The raw domain error is always reported alongside it.
    """

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    source, target = _feature_pair(features_source, features_target, min_samples=2)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    fraction = float(holdout_fraction)
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("holdout_fraction must be strictly between 0 and 1")

    features = np.vstack((source, target))
    domain_labels = np.concatenate(
        (
            np.zeros(len(source), dtype=np.int64),
            np.ones(len(target), dtype=np.int64),
        )
    )
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            domain_labels,
            test_size=fraction,
            random_state=int(seed),
            stratify=domain_labels,
        )
    except ValueError as exc:
        raise ValueError(f"insufficient samples for stratified domain holdout: {exc}") from exc

    scaler = StandardScaler().fit(x_train)
    classifier = LogisticRegression(
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=int(seed),
        solver="liblinear",
    )
    classifier.fit(scaler.transform(x_train), y_train)
    error = 1.0 - float(accuracy_score(y_test, classifier.predict(scaler.transform(x_test))))
    estimate = 2.0 * (1.0 - 2.0 * error)
    return {
        "proxy_a_distance": float(estimate),
        "domain_classifier_error": float(error),
        "n_source": int(len(source)),
        "n_target": int(len(target)),
        "holdout_fraction": fraction,
        "seed": int(seed),
        "convention": "unclipped finite-sample estimate 2*(1-2*error)",
        "interpretation": "descriptive only; lower domain discrepancy does not imply higher class recall",
    }


def _squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.sum(left * left, axis=1)[:, None]
    right_norm = np.sum(right * right, axis=1)[None, :]
    return np.maximum(left_norm + right_norm - 2.0 * (left @ right.T), 0.0)


def rbf_mmd2(
    features_source: Any,
    features_target: Any,
    *,
    bandwidth: float | None = None,
) -> dict[str, Any]:
    """Compute unbiased squared MMD with an RBF kernel.

    For cross-arm comparisons, calculate one bandwidth and pass that same
    explicit value to every call.  The unbiased estimate may be slightly
    negative when the distributions are close.
    """

    source, target = _feature_pair(features_source, features_target, min_samples=2)
    if bandwidth is None:
        pooled = np.vstack((source, target))
        squared = _squared_distances(pooled, pooled)
        upper = np.triu_indices(len(pooled), k=1)
        positive_distances = np.sqrt(squared[upper])
        positive_distances = positive_distances[positive_distances > 0.0]
        if positive_distances.size == 0:
            raise ValueError("automatic MMD bandwidth is undefined because all pooled features coincide")
        sigma = float(np.median(positive_distances))
        bandwidth_source = "pooled median pairwise Euclidean distance"
    else:
        sigma = float(bandwidth)
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("bandwidth must be finite and strictly positive")
        bandwidth_source = "explicit"

    gamma = 1.0 / (2.0 * sigma * sigma)

    def kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.exp(-gamma * _squared_distances(left, right))

    k_source = kernel(source, source)
    k_target = kernel(target, target)
    k_cross = kernel(source, target)
    m = len(source)
    n = len(target)
    term_source = (k_source.sum() - np.trace(k_source)) / (m * (m - 1))
    term_target = (k_target.sum() - np.trace(k_target)) / (n * (n - 1))
    term_cross = 2.0 * k_cross.sum() / (m * n)
    estimate = float(term_source + term_target - term_cross)
    return {
        "mmd2": estimate,
        "kernel": "rbf",
        "bandwidth": sigma,
        "bandwidth_source": bandwidth_source,
        "estimator": "unbiased squared MMD",
        "interpretation": "descriptive only; lower domain discrepancy does not imply higher class recall",
    }


def coral_distance(features_source: Any, features_target: Any) -> dict[str, Any]:
    """Compute CORAL covariance distance and the first-moment L2 gap."""

    source, target = _feature_pair(features_source, features_target, min_samples=2)
    dimension = source.shape[1]
    covariance_source = np.cov(source, rowvar=False)
    covariance_target = np.cov(target, rowvar=False)
    covariance_source = np.atleast_2d(covariance_source)
    covariance_target = np.atleast_2d(covariance_target)
    distance = float(
        np.sum((covariance_source - covariance_target) ** 2)
        / (4.0 * dimension * dimension)
    )
    mean_difference = float(np.linalg.norm(source.mean(axis=0) - target.mean(axis=0)))
    return {
        "coral_distance": distance,
        "mean_difference_l2": mean_difference,
        "feature_dimension": int(dimension),
        "covariance": "sample covariance (N-1 denominator)",
        "interpretation": "descriptive only; lower domain discrepancy does not imply higher class recall",
    }


def bbse_hard_source_soft_target_binary(
    source_labels: Any,
    source_probabilities: Any,
    target_probabilities: Any,
    *,
    reference_airplane_prior: float | None = None,
    condition_limit: float = 1e8,
) -> dict[str, Any]:
    """Reproduce the paper's hard-source/soft-target binary BBSE variant.

    Source predictions form a hard joint confusion matrix while target
    probabilities form a soft prediction marginal.  Negative solved weights
    are clipped to zero and explicitly reported.  Singular or severely
    ill-conditioned systems fail rather than silently substituting another
    estimator.
    """

    labels = _validate_labels(source_labels, name="source_labels")
    source_probs = _validate_probabilities(
        source_probabilities,
        name="source_probabilities",
    )
    target_probs = _validate_probabilities(
        target_probabilities,
        name="target_probabilities",
    )
    if len(labels) != len(source_probs):
        raise ValueError("source_labels and source_probabilities must have the same length")
    source_counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    if np.any(source_counts == 0):
        raise ValueError("source_labels must contain both classes for BBSE")
    limit = float(condition_limit)
    if not np.isfinite(limit) or limit <= 1.0:
        raise ValueError("condition_limit must be finite and greater than 1")

    source_predictions = np.argmax(source_probs, axis=1)
    joint_confusion = np.zeros((2, 2), dtype=np.float64)
    for predicted in (AIRPLANE, SHIP):
        for truth in (AIRPLANE, SHIP):
            joint_confusion[predicted, truth] = np.sum(
                (source_predictions == predicted) & (labels == truth)
            )
    joint_confusion /= float(len(labels))

    condition_number = float(np.linalg.cond(joint_confusion))
    if not np.isfinite(condition_number) or condition_number > limit:
        raise ValueError(
            "BBSE source confusion matrix is singular or ill-conditioned "
            f"(condition_number={condition_number})"
        )

    target_prediction_marginal = target_probs.mean(axis=0, dtype=np.float64)
    raw_weights = np.linalg.solve(joint_confusion, target_prediction_marginal)
    clipped_weights = np.clip(raw_weights, 0.0, None)
    clipped = bool(np.any(raw_weights < 0.0))
    source_prior = source_counts.astype(np.float64) / float(len(labels))
    target_prior_unnormalised = clipped_weights * source_prior
    normaliser = float(target_prior_unnormalised.sum())
    if not np.isfinite(normaliser) or normaliser <= 0.0:
        raise ValueError("BBSE produced no positive target-prior mass after clipping")
    target_prior = target_prior_unnormalised / normaliser

    reference: float | None
    gap: float | None
    if reference_airplane_prior is None:
        reference = None
        gap = None
    else:
        reference = float(reference_airplane_prior)
        if not np.isfinite(reference) or not 0.0 <= reference <= 1.0:
            raise ValueError("reference_airplane_prior must be finite and within [0, 1]")
        gap = float(target_prior[AIRPLANE] - reference)

    return {
        "class_order": list(CLASS_NAMES),
        "estimated_target_prior": [float(value) for value in target_prior],
        "estimated_airplane_fraction": float(target_prior[AIRPLANE]),
        "raw_weights": [float(value) for value in raw_weights],
        "clipped_weights": [float(value) for value in clipped_weights],
        "negative_weights_clipped": clipped,
        "status": "negative_weights_clipped" if clipped else "ok",
        "source_joint_confusion": joint_confusion.tolist(),
        "target_soft_prediction_marginal": [
            float(value) for value in target_prediction_marginal
        ],
        "source_prior": [float(value) for value in source_prior],
        "condition_number": condition_number,
        "reference_airplane_prior": reference,
        "airplane_prior_gap": gap,
        "variant": "source hard joint confusion + target soft probability marginal",
        "interpretation": "descriptive label-shift diagnostic; not a performance metric",
    }
