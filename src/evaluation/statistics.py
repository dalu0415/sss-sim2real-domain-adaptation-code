"""Seed-matched statistics and multiplicity corrections."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy import stats


def _finite_vector(values: Any, *, name: str, min_size: int = 1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < min_size:
        raise ValueError(f"{name} must be a one-dimensional array with at least {min_size} values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def _seed_map(values: Any, *, name: str) -> dict[int, float]:
    if isinstance(values, Mapping):
        items = values.items()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        parsed: list[tuple[Any, Any]] = []
        for index, record in enumerate(values):
            if not isinstance(record, Mapping) or "seed" not in record or "mean" not in record:
                raise ValueError(f"{name}[{index}] must contain seed and mean")
            parsed.append((record["seed"], record["mean"]))
        items = parsed
    else:
        raise ValueError(f"{name} must be a seed-to-value mapping or aggregate records")

    result: dict[int, float] = {}
    for seed, value in items:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError(f"{name} contains a non-integer seed")
        seed_int = int(seed)
        if seed_int in result:
            raise ValueError(f"{name} contains duplicate seed {seed_int}")
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ValueError(f"{name} contains a non-numeric value for seed {seed_int}")
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"{name} contains a non-finite value for seed {seed_int}")
        result[seed_int] = number
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def paired_seed_differences(
    arm_a: Any,
    arm_b: Any,
    *,
    expected_seeds: Iterable[int] | None = range(5),
) -> dict[str, Any]:
    """Return ``A-B`` differences after strict seed matching."""

    values_a = _seed_map(arm_a, name="arm_a")
    values_b = _seed_map(arm_b, name="arm_b")
    if set(values_a) != set(values_b):
        raise ValueError(
            "arm seed sets differ; "
            f"only_in_a={sorted(set(values_a) - set(values_b))}, "
            f"only_in_b={sorted(set(values_b) - set(values_a))}"
        )

    if expected_seeds is None:
        seeds = tuple(sorted(values_a))
    else:
        seeds_list: list[int] = []
        for seed in expected_seeds:
            if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
                raise ValueError("expected_seeds must contain integers")
            seeds_list.append(int(seed))
        if not seeds_list or len(seeds_list) != len(set(seeds_list)):
            raise ValueError("expected_seeds must be non-empty and unique")
        seeds = tuple(seeds_list)
        if set(values_a) != set(seeds):
            raise ValueError(
                f"paired arms must contain exactly expected seeds {list(seeds)}; "
                f"got {sorted(values_a)}"
            )

    a_ordered = [values_a[seed] for seed in seeds]
    b_ordered = [values_b[seed] for seed in seeds]
    deltas = [a - b for a, b in zip(a_ordered, b_ordered)]
    return {
        "seeds": list(seeds),
        "arm_a_values": [float(value) for value in a_ordered],
        "arm_b_values": [float(value) for value in b_ordered],
        "differences": [float(value) for value in deltas],
        "direction": "arm_a_minus_arm_b",
    }


def paired_t_difference(
    differences: Any,
    *,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Summarise paired differences with a two-sided paired-t interval/test."""

    values = _finite_vector(differences, name="differences", min_size=2)
    confidence_value = float(confidence)
    if not np.isfinite(confidence_value) or not 0.0 < confidence_value < 1.0:
        raise ValueError("confidence must be finite and strictly between 0 and 1")

    n = int(values.size)
    df = n - 1
    mean = float(values.mean(dtype=np.float64))
    sd = float(values.std(ddof=1, dtype=np.float64))
    se = float(sd / np.sqrt(n))
    alpha = 1.0 - confidence_value
    t_critical = float(stats.t.ppf(1.0 - alpha / 2.0, df))

    if sd == 0.0:
        if mean == 0.0:
            t_statistic = 0.0
            p_value = 1.0
        else:
            t_statistic = float(np.copysign(np.inf, mean))
            p_value = 0.0
        ci_lower = mean
        ci_upper = mean
    else:
        t_statistic = float(mean / se)
        p_value = float(2.0 * stats.t.sf(abs(t_statistic), df))
        margin = t_critical * se
        ci_lower = float(mean - margin)
        ci_upper = float(mean + margin)

    return {
        "n": n,
        "df": df,
        "mean_difference": mean,
        "sample_sd": sd,
        "standard_error": se,
        "t_statistic": t_statistic,
        "p_value_two_sided": p_value,
        "confidence": confidence_value,
        "t_critical": t_critical,
        "confidence_interval": [ci_lower, ci_upper],
        "raw_differences": [float(value) for value in values],
        "method": "paired-t interval and two-sided paired-t test",
    }


def _validate_p_values(p_values: Any) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if not np.isfinite(values).all():
        raise ValueError("p_values contains NaN or infinity")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("p_values must be within [0, 1]")
    return values


def holm_adjust(p_values: Any, *, alpha: float = 0.05) -> dict[str, Any]:
    """Holm family-wise error correction, restored to input order."""

    values = _validate_p_values(p_values)
    alpha_value = float(alpha)
    if not np.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
        raise ValueError("alpha must be finite and strictly between 0 and 1")
    m = int(values.size)
    if m == 0:
        return {"method": "Holm", "alpha": alpha_value, "adjusted_p_values": [], "rejected": []}

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    scaled = (m - np.arange(m)) * sorted_values
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(scaled))
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return {
        "method": "Holm",
        "alpha": alpha_value,
        "adjusted_p_values": [float(value) for value in adjusted],
        "rejected": [bool(value <= alpha_value) for value in adjusted],
    }


def benjamini_hochberg(p_values: Any, *, alpha: float = 0.05) -> dict[str, Any]:
    """Benjamini-Hochberg FDR correction, restored to input order."""

    values = _validate_p_values(p_values)
    alpha_value = float(alpha)
    if not np.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
        raise ValueError("alpha must be finite and strictly between 0 and 1")
    m = int(values.size)
    if m == 0:
        return {
            "method": "Benjamini-Hochberg",
            "alpha": alpha_value,
            "adjusted_p_values": [],
            "rejected": [],
        }

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    scaled = sorted_values * m / ranks
    adjusted_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted_sorted = np.minimum(1.0, adjusted_sorted)
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return {
        "method": "Benjamini-Hochberg",
        "alpha": alpha_value,
        "adjusted_p_values": [float(value) for value in adjusted],
        "rejected": [bool(value <= alpha_value) for value in adjusted],
    }
