"""Optional image-quality and precomputed-covariate diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from .core import AIRPLANE, CLASS_NAMES, SHIP, _validate_labels


def niqe_scores(
    image_paths: list[str] | tuple[str, ...],
    *,
    backend: str,
    cache_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compute or load NIQE scores using an explicitly selected backend.

    NIQE was designed for natural optical images and is only a coarse
    covariate for side-scan sonar.  It is not part of the default scoring
    chain.  Supported backends are ``cache``, ``skvideo``, and ``pyiqa``.
    Individual image failures are reported by index without aborting the
    entire batch or exposing a path in the result.
    """

    paths = list(image_paths)
    if not paths:
        raise ValueError("image_paths must not be empty")
    if not all(isinstance(path, str) and path for path in paths):
        raise ValueError("image_paths must contain non-empty strings")
    if backend not in {"cache", "skvideo", "pyiqa"}:
        raise ValueError("backend must be 'cache', 'skvideo', or 'pyiqa'")

    values: list[float | None] = []
    failures: list[dict[str, Any]] = []
    implementation = backend

    if backend == "cache":
        if cache_path is None:
            raise ValueError("cache_path is required for backend='cache'")
        payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        by_path = payload.get("niqe_by_path")
        if not isinstance(by_path, dict):
            raise ValueError("NIQE cache must contain an object named 'niqe_by_path'")
        implementation = f"cache:{payload.get('impl', 'unspecified')}"
        for index, path in enumerate(paths):
            raw = by_path.get(path)
            if raw is None:
                values.append(None)
                failures.append({"index": index, "error": "cache_miss"})
                continue
            value = float(raw)
            if not np.isfinite(value):
                raise ValueError(f"NIQE cache value at image index {index} is non-finite")
            values.append(value)
    elif backend == "skvideo":
        try:
            import skvideo.measure
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("backend='skvideo' requires scikit-video and Pillow") from exc
        for index, path in enumerate(paths):
            try:
                image = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
                value = float(np.asarray(skvideo.measure.niqe(image[None, ...])).ravel()[0])
                if not np.isfinite(value):
                    raise ValueError("non-finite NIQE score")
                values.append(value)
            except Exception as exc:  # one unreadable image must not hide other valid scores
                values.append(None)
                failures.append({"index": index, "error": type(exc).__name__})
    else:
        try:
            import pyiqa
        except ImportError as exc:
            raise RuntimeError("backend='pyiqa' requires pyiqa") from exc
        metric = pyiqa.create_metric("niqe")
        for index, path in enumerate(paths):
            try:
                value = float(metric(path).item())
                if not np.isfinite(value):
                    raise ValueError("non-finite NIQE score")
                values.append(value)
            except Exception as exc:  # one unreadable image must not hide other valid scores
                values.append(None)
                failures.append({"index": index, "error": type(exc).__name__})

    finite = np.asarray([value for value in values if value is not None], dtype=np.float64)
    return {
        "backend": implementation,
        "scores": values,
        "n_images": len(paths),
        "n_scored": int(finite.size),
        "failures": failures,
        "mean": float(finite.mean()) if finite.size else None,
        "std": float(finite.std(ddof=0)) if finite.size else None,
        "interpretation": "optional optical-IQA covariate; descriptive only for side-scan sonar",
    }


def image_area_covariate(image_paths: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Read image dimensions and return area values without returning paths."""

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("image_area_covariate requires Pillow") from exc

    paths = list(image_paths)
    if not paths:
        raise ValueError("image_paths must not be empty")
    values: list[float | None] = []
    failures: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        if not isinstance(path, str) or not path:
            raise ValueError(f"image path at index {index} is invalid")
        try:
            with Image.open(path) as image:
                width, height = image.size
            values.append(float(width * height))
        except Exception as exc:
            values.append(None)
            failures.append({"index": index, "error": type(exc).__name__})
    return {"name": "image_area_pixels", "values": values, "failures": failures}


def stratified_classification_metrics(
    y_true: Any,
    y_pred: Any,
    covariate: Any,
    *,
    n_bins: int = 3,
) -> dict[str, Any]:
    """Compute per-class P/R/F1 within quantile bins of a precomputed covariate."""

    truth = _validate_labels(y_true, name="y_true")
    predicted = _validate_labels(y_pred, name="y_pred")
    raw_values = np.asarray(covariate, dtype=object)
    if raw_values.ndim != 1:
        raise ValueError("covariate must be one-dimensional")
    try:
        values = np.asarray(
            [np.nan if value is None else float(value) for value in raw_values],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("covariate must contain numeric values or None") from exc
    if len(truth) != len(predicted) or len(truth) != len(values):
        raise ValueError("y_true, y_pred, and covariate must have the same length")
    if np.isinf(values).any():
        raise ValueError("covariate contains infinity")
    if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or int(n_bins) <= 0:
        raise ValueError("n_bins must be a positive integer")

    valid = np.isfinite(values)
    excluded = int((~valid).sum())
    if not valid.any():
        return {
            "requested_bins": int(n_bins),
            "effective_bins": 0,
            "excluded_nonfinite": excluded,
            "bins": [],
            "status": "no_valid_covariates",
            "interpretation": "descriptive stratification; not a performance verdict",
        }

    requested = int(n_bins)
    quantiles = np.quantile(values[valid], np.linspace(0.0, 1.0, requested + 1))
    unique_edges = np.unique(quantiles)
    if len(unique_edges) == 1:
        cut_points = np.asarray([], dtype=np.float64)
        ranges = [(float(unique_edges[0]), float(unique_edges[0]))]
    else:
        cut_points = unique_edges[1:-1]
        ranges = [
            (float(unique_edges[index]), float(unique_edges[index + 1]))
            for index in range(len(unique_edges) - 1)
        ]
    assignments = np.full(len(values), -1, dtype=np.int64)
    assignments[valid] = np.searchsorted(cut_points, values[valid], side="right")

    bins: list[dict[str, Any]] = []
    sparse_class_warning = False
    for bin_index, bounds in enumerate(ranges):
        mask = assignments == bin_index
        precision, recall, class_f1, support = precision_recall_fscore_support(
            truth[mask],
            predicted[mask],
            labels=[AIRPLANE, SHIP],
            zero_division=0,
        )
        if np.any(support < 10):
            sparse_class_warning = True
        bins.append(
            {
                "range": [bounds[0], bounds[1]],
                "n": int(mask.sum()),
                "per_class": {
                    class_name: {
                        "precision": float(precision[class_index]),
                        "recall": float(recall[class_index]),
                        "f1": float(class_f1[class_index]),
                        "support": int(support[class_index]),
                    }
                    for class_index, class_name in enumerate(CLASS_NAMES)
                },
            }
        )

    return {
        "requested_bins": requested,
        "effective_bins": len(bins),
        "quantile_edges_after_deduplication": [float(value) for value in unique_edges],
        "excluded_nonfinite": excluded,
        "bins": bins,
        "sparse_class_warning": bool(sparse_class_warning),
        "status": "degenerate_equal_covariate" if len(unique_edges) == 1 else "ok",
        "interpretation": "descriptive stratification; not a performance verdict",
    }
