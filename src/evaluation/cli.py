"""Frozen-paper command-line scoring, collapse, and comparison workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .core import (
    CLASS_NAMES,
    aggregate_seed_folds,
    classification_metrics,
    mean_checkpoint_probabilities,
    predict_labels,
)
from .diagnostics import (
    collapse_from_counts,
    collapse_from_predictions,
    epoch_prediction_collapse_routes,
    triangulate_collapse,
)
from .statistics import (
    benjamini_hochberg,
    holm_adjust,
    paired_seed_differences,
    paired_t_difference,
)


PAPER_CONTRACT = "paper_primary_v1"
PAPER_K = 5
PAPER_FOLDS = tuple(range(5))
PAPER_SEEDS = tuple(range(5))
PAPER_FREE_ARM = "source_only_adabn"
PAPER_ADVERSARIAL_ARMS = ("cdan", "cdan_adabn", "iw_cdan", "iw_cdan_adabn")
PAPER_METRICS = (
    "macro_f1",
    "per_class.airplane.f1",
    "per_class.airplane.precision",
    "per_class.airplane.recall",
    "airplane_pr_auc",
)
PAPER_HEADLINE_METRICS = (
    "macro_f1",
    "per_class.airplane.f1",
    "airplane_pr_auc",
)


def _strict_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict_json_value(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if np.isposinf(number):
            return "Infinity"
        if np.isneginf(number):
            return "-Infinity"
        if np.isnan(number):
            raise ValueError("refusing to write NaN to JSON")
        return number
    return value


def _write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_strict_json_value(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_json_object(path: str | Path, *, name: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"{name} file does not exist: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} JSON must be an object")
    return payload


def _load_npy(path: str | Path, *, name: str) -> np.ndarray:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"{name} file does not exist: {source}")
    try:
        return np.load(source, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"could not load {name} as a non-pickled NumPy array: {exc}") from exc


def _score(args: argparse.Namespace) -> int:
    if not args.arm.strip():
        raise ValueError("--arm must not be empty")
    if args.authorize_target_labels is not True:
        raise ValueError("score requires --authorize-target-labels")
    if not args.authorization_reason.strip():
        raise ValueError("score requires a non-empty --authorization-reason")
    if args.seed not in PAPER_SEEDS:
        raise ValueError(f"paper score seed must be one of {list(PAPER_SEEDS)}")
    if args.fold not in PAPER_FOLDS:
        raise ValueError(f"paper score fold must be one of {list(PAPER_FOLDS)}")

    checkpoint_probs = _load_npy(args.checkpoint_probabilities, name="checkpoint probabilities")
    labels = _load_npy(args.labels, name="labels")
    averaged = mean_checkpoint_probabilities(checkpoint_probs, expected_k=PAPER_K)
    metrics = classification_metrics(labels, averaged, prediction_mode="argmax")
    predicted = predict_labels(averaged, mode="argmax")
    record = {
        "schema_version": 1,
        "analysis_contract": PAPER_CONTRACT,
        "arm": args.arm.strip(),
        "seed": args.seed,
        "fold": args.fold,
        "class_order": list(CLASS_NAMES),
        "checkpoint_ensemble": {
            "n_checkpoints": PAPER_K,
            "operation": "softmax per checkpoint, then fixed-order float64 probability average",
        },
        "target_label_access": {
            "authorized": True,
            "reason": args.authorization_reason.strip(),
            "policy": "final scoring only after training and model selection are frozen",
        },
        "metrics": metrics,
        "lastk_prediction_collapse_route": collapse_from_predictions(predicted),
    }
    _write_json(args.output, record)
    return 0


def _validate_score_record(record: dict[str, Any], *, require_collapse: bool) -> None:
    if record.get("analysis_contract") != PAPER_CONTRACT:
        raise ValueError(f"record must use analysis_contract={PAPER_CONTRACT}")
    if record.get("class_order") != list(CLASS_NAMES):
        raise ValueError("record has the wrong class order")
    access = record.get("target_label_access")
    if not isinstance(access, dict) or access.get("authorized") is not True:
        raise ValueError("record does not document explicit target-label authorization")
    ensemble = record.get("checkpoint_ensemble")
    if not isinstance(ensemble, dict) or ensemble.get("n_checkpoints") != PAPER_K:
        raise ValueError(f"record must contain exactly {PAPER_K} checkpoints")
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("record is missing metrics")
    rule = metrics.get("prediction_rule")
    if not isinstance(rule, dict) or rule.get("mode") != "argmax":
        raise ValueError("paper comparison accepts only argmax primary-scoring records")
    if require_collapse:
        if not isinstance(record.get("collapsed"), bool):
            raise ValueError("record must be enriched by the collapse command before comparison")
        evidence = record.get("collapse_evidence")
        decision = record.get("collapse_adjudication")
        if not isinstance(evidence, dict) or set(evidence) != {
            "last_epoch_prediction_fraction",
            "lastk_prediction_fraction",
            "any_epoch_exact_single_class",
        }:
            raise ValueError("record does not contain the three required collapse routes")
        last_epoch_stored = evidence["last_epoch_prediction_fraction"]
        lastk_stored = evidence["lastk_prediction_fraction"]
        any_epoch_stored = evidence["any_epoch_exact_single_class"]
        if not isinstance(last_epoch_stored, dict) or not isinstance(lastk_stored, dict):
            raise ValueError("record collapse fraction routes must be objects")
        epoch_histograms = record.get("epoch_prediction_histograms")
        recomputed_epoch_routes = epoch_prediction_collapse_routes(epoch_histograms)
        last_epoch_route = recomputed_epoch_routes["last_epoch_prediction_fraction"]
        lastk = collapse_from_counts(
            lastk_stored.get("airplane_count"),
            lastk_stored.get("ship_count"),
        )
        if (
            last_epoch_route != last_epoch_stored
            or recomputed_epoch_routes["any_epoch_exact_single_class"] != any_epoch_stored
            or lastk != lastk_stored
            or lastk["total"] != recomputed_epoch_routes["target_count_per_epoch"]
        ):
            raise ValueError("record contains an inconsistent collapse fraction route")
        if not isinstance(any_epoch_stored, dict) or not isinstance(
            any_epoch_stored.get("collapsed"), bool
        ):
            raise ValueError("record any-epoch collapse route must contain a boolean collapsed flag")
        recomputed = triangulate_collapse(
            [
                last_epoch_route["collapsed"],
                lastk["collapsed"],
                any_epoch_stored["collapsed"],
            ]
        )
        if decision != recomputed:
            raise ValueError("record collapse adjudication does not match its three routes")
        if decision.get("verdict") is not record["collapsed"]:
            raise ValueError("record collapse verdict and top-level collapsed flag disagree")


def _collapse(args: argparse.Namespace) -> int:
    record = _load_json_object(args.score_record, name="score record")
    _validate_score_record(record, require_collapse=False)
    epoch_payload = json.loads(Path(args.epoch_counts).read_text(encoding="utf-8"))
    epoch_counts = epoch_payload.get("epochs") if isinstance(epoch_payload, dict) else epoch_payload
    epoch_routes = epoch_prediction_collapse_routes(epoch_counts)
    last_epoch = epoch_routes["last_epoch_prediction_fraction"]
    stored_lastk = record.get("lastk_prediction_collapse_route")
    if not isinstance(stored_lastk, dict):
        raise ValueError("score record is missing its last-K collapse route")
    lastk = collapse_from_counts(
        stored_lastk.get("airplane_count"),
        stored_lastk.get("ship_count"),
    )
    if lastk != stored_lastk:
        raise ValueError("score record contains an inconsistent last-K collapse route")
    if lastk["total"] != epoch_routes["target_count_per_epoch"]:
        raise ValueError("epoch and last-K collapse routes must cover the same target count")
    any_epoch = epoch_routes["any_epoch_exact_single_class"]
    decision = triangulate_collapse(
        [last_epoch["collapsed"], lastk["collapsed"], any_epoch["collapsed"]]
    )
    if decision["verdict"] is None:
        raise ValueError("three complete collapse routes unexpectedly produced no verdict")

    enriched = dict(record)
    enriched["collapse_evidence"] = {
        "last_epoch_prediction_fraction": last_epoch,
        "lastk_prediction_fraction": lastk,
        "any_epoch_exact_single_class": any_epoch,
    }
    enriched["epoch_prediction_histograms"] = epoch_routes[
        "epoch_prediction_histograms"
    ]
    enriched["collapse_adjudication"] = decision
    enriched["collapsed"] = bool(decision["verdict"])
    _write_json(args.output, enriched)
    return 0


def _load_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"records file does not exist: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise ValueError("records JSON must be a non-empty array or an object with a records array")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("every records entry must be an object")
    return records


def _family_summary(
    comparisons: list[dict[str, Any]],
    indices: list[int],
    family_key: str,
) -> dict[str, Any]:
    return {
        "n_tests": len(indices),
        "raw_p_below_alpha": int(
            sum(comparisons[index]["paired_t"]["p_value_two_sided"] < 0.05 for index in indices)
        ),
        "holm_rejected": int(
            sum(comparisons[index]["multiplicity"][family_key]["holm_rejected"] for index in indices)
        ),
        "bh_fdr_rejected": int(
            sum(comparisons[index]["multiplicity"][family_key]["bh_fdr_rejected"] for index in indices)
        ),
    }


def _compare(args: argparse.Namespace) -> int:
    records = _load_records(args.records)
    required_arms = {PAPER_FREE_ARM, *PAPER_ADVERSARIAL_ARMS}
    paper_records = [record for record in records if record.get("arm") in required_arms]
    if not paper_records:
        raise ValueError("records contain none of the required paper head-to-head arms")
    for record in paper_records:
        _validate_score_record(record, require_collapse=True)

    comparisons: list[dict[str, Any]] = []
    for adversarial_arm in PAPER_ADVERSARIAL_ARMS:
        for metric in PAPER_METRICS:
            aggregated = aggregate_seed_folds(
                paper_records,
                metric,
                expected_folds=PAPER_FOLDS,
            )
            aggregate_a = [record for record in aggregated if record["arm"] == PAPER_FREE_ARM]
            aggregate_b = [record for record in aggregated if record["arm"] == adversarial_arm]
            pairing = paired_seed_differences(
                aggregate_a,
                aggregate_b,
                expected_seeds=PAPER_SEEDS,
            )
            inference = paired_t_difference(pairing["differences"], confidence=0.95)
            comparisons.append(
                {
                    "comparison": f"{PAPER_FREE_ARM} - {adversarial_arm}",
                    "arm_a": PAPER_FREE_ARM,
                    "arm_b": adversarial_arm,
                    "metric": metric,
                    "headline_metric": metric in PAPER_HEADLINE_METRICS,
                    "arm_a_seed_means": aggregate_a,
                    "arm_b_seed_means": aggregate_b,
                    "pairing": pairing,
                    "paired_t": inference,
                    "multiplicity": {},
                }
            )

    all_indices = list(range(len(comparisons)))
    headline_indices = [
        index for index, comparison in enumerate(comparisons) if comparison["headline_metric"]
    ]

    def apply_family(indices: list[int], family_key: str) -> None:
        p_values = [comparisons[index]["paired_t"]["p_value_two_sided"] for index in indices]
        holm = holm_adjust(p_values, alpha=0.05)
        bh = benjamini_hochberg(p_values, alpha=0.05)
        for family_index, comparison_index in enumerate(indices):
            comparisons[comparison_index]["multiplicity"][family_key] = {
                "holm_adjusted_p": holm["adjusted_p_values"][family_index],
                "holm_rejected": holm["rejected"][family_index],
                "bh_fdr_adjusted_p": bh["adjusted_p_values"][family_index],
                "bh_fdr_rejected": bh["rejected"][family_index],
            }

    apply_family(all_indices, "family20")
    apply_family(headline_indices, "headline12")
    if len(all_indices) != 20 or len(headline_indices) != 12:
        raise AssertionError("paper family construction must produce exactly 20 and 12 tests")

    output = {
        "schema_version": 1,
        "analysis_contract": PAPER_CONTRACT,
        "direction": f"{PAPER_FREE_ARM} minus each adversarial arm",
        "comparison_arms": list(PAPER_ADVERSARIAL_ARMS),
        "metrics": list(PAPER_METRICS),
        "headline_metrics": list(PAPER_HEADLINE_METRICS),
        "expected_folds": list(PAPER_FOLDS),
        "expected_seeds": list(PAPER_SEEDS),
        "confidence": 0.95,
        "alpha": 0.05,
        "comparisons": comparisons,
        "family_summary": {
            "headline12": _family_summary(comparisons, headline_indices, "headline12"),
            "family20": _family_summary(comparisons, all_indices, "family20"),
        },
    }
    _write_json(args.output, output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen paper scoring, collapse triangulation, and full-family comparison"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score", help="score one paper arm/seed/fold with fixed K=5 argmax")
    score.add_argument("--checkpoint-probabilities", required=True, help="NumPy [5,N,2] softmax array")
    score.add_argument("--labels", required=True, help="NumPy [N] integer target-label array")
    score.add_argument(
        "--authorize-target-labels",
        action="store_true",
        help="explicitly authorize post-freeze target-label scoring",
    )
    score.add_argument("--authorization-reason", default="")
    score.add_argument("--arm", required=True)
    score.add_argument("--seed", required=True, type=int)
    score.add_argument("--fold", required=True, type=int)
    score.add_argument("--output", required=True)
    score.set_defaults(handler=_score)

    collapse = subparsers.add_parser(
        "collapse",
        help="add the three frozen collapse routes and verdict to one score record",
    )
    collapse.add_argument("--score-record", required=True)
    collapse.add_argument(
        "--epoch-counts",
        required=True,
        help="JSON array (or {epochs:[...]}) of epoch/airplane_count/ship_count objects",
    )
    collapse.add_argument("--output", required=True)
    collapse.set_defaults(handler=_collapse)

    compare = subparsers.add_parser(
        "compare",
        help="compute the frozen four-comparison headline-12 and full-20 families",
    )
    compare.add_argument("--records", required=True, help="JSON array of collapse-enriched score records")
    compare.add_argument("--output", required=True)
    compare.set_defaults(handler=_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError, PermissionError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    raise AssertionError("argparse.error should have exited")


if __name__ == "__main__":
    raise SystemExit(main())
