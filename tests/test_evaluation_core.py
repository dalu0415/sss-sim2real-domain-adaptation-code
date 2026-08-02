from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from evaluation.calibration_metrics import (  # noqa: E402
    brier_score,
    calibration_errors,
    confidence_entropy_summary,
    directed_confusion_rates,
    imbalance_metrics,
)
from evaluation.core import (  # noqa: E402
    aggregate_seed_folds,
    classification_metrics,
    mean_checkpoint_probabilities,
    predict_labels,
)
from evaluation.diagnostics import (  # noqa: E402
    collapse_from_counts,
    epoch_prediction_collapse_routes,
    oracle_best_epochs,
    threshold_sweep,
    triangulate_collapse,
)
from evaluation.domain_metrics import (  # noqa: E402
    bbse_hard_source_soft_target_binary,
    coral_distance,
    proxy_a_distance,
    rbf_mmd2,
)
from evaluation.image_diagnostics import niqe_scores, stratified_classification_metrics  # noqa: E402
from evaluation.statistics import (  # noqa: E402
    benjamini_hochberg,
    holm_adjust,
    paired_seed_differences,
    paired_t_difference,
)


class ProbabilityAndClassificationTests(unittest.TestCase):
    def test_float64_checkpoint_average_preserves_boundary_decision(self) -> None:
        lower = np.float32(0.4999999701976776)
        checkpoint_probs = np.asarray(
            [
                [[lower, np.float32(1.0) - lower], [0.5, 0.5]],
                [[lower, np.float32(1.0) - lower], [0.5, 0.5]],
                [[lower, np.float32(1.0) - lower], [0.5, 0.5]],
                [[lower, np.float32(1.0) - lower], [0.5, 0.5]],
                [[0.5, 0.5], [0.5, 0.5]],
            ],
            dtype=np.float32,
        )
        averaged = mean_checkpoint_probabilities(checkpoint_probs)
        self.assertEqual(averaged.dtype, np.float64)
        self.assertLess(averaged[0, 0], averaged[0, 1])
        np.testing.assert_array_equal(predict_labels(averaged), [1, 0])

    def test_checkpoint_count_and_probability_validation(self) -> None:
        valid = np.full((4, 2, 2), 0.5)
        with self.assertRaisesRegex(ValueError, "exactly 5"):
            mean_checkpoint_probabilities(valid)
        invalid = np.full((5, 2, 2), 0.5)
        invalid[0, 0] = [0.2, 0.2]
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            mean_checkpoint_probabilities(invalid)
        outside = np.full((5, 2, 2), 0.5)
        outside[0, 0] = [-1e-12, 1.0 + 1e-12]
        with self.assertRaisesRegex(ValueError, "outside"):
            mean_checkpoint_probabilities(outside)

    def test_classification_contract_and_average_precision_alias(self) -> None:
        truth = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
        airplane_scores = np.asarray([0.90, 0.80, 0.60, 0.40, 0.55, 0.70, 0.20, 0.10])
        probs = np.column_stack((airplane_scores, 1.0 - airplane_scores))
        result = classification_metrics(truth, probs)
        self.assertEqual(result["confusion_matrix"], [[3, 1], [2, 2]])
        self.assertAlmostEqual(result["macro_f1"], 0.6190476190476191)
        self.assertAlmostEqual(result["per_class"]["airplane"]["f1"], 2.0 / 3.0)
        self.assertAlmostEqual(result["per_class"]["ship"]["f1"], 4.0 / 7.0)
        self.assertAlmostEqual(result["airplane_average_precision"], 0.8541666666666666)
        self.assertEqual(result["airplane_average_precision"], result["airplane_pr_auc"])

    def test_single_true_class_has_undefined_average_precision(self) -> None:
        result = classification_metrics([0, 0], [[0.9, 0.1], [0.8, 0.2]])
        self.assertIsNone(result["airplane_average_precision"])

    def test_argmax_tie_and_explicit_threshold_have_closed_boundary(self) -> None:
        probabilities = np.asarray([[0.5, 0.5], [0.499, 0.501]])
        np.testing.assert_array_equal(predict_labels(probabilities), [0, 1])
        np.testing.assert_array_equal(
            predict_labels(probabilities, mode="airplane_threshold", airplane_threshold=0.5),
            [0, 1],
        )


class AggregationAndStatisticsTests(unittest.TestCase):
    @staticmethod
    def _records() -> list[dict]:
        records = []
        for fold, value in enumerate([1.0, 1.0, 1.0, 1.0, 0.0]):
            records.append(
                {
                    "arm": "A",
                    "seed": 0,
                    "fold": fold,
                    "collapsed": fold == 4,
                    "metrics": {"macro_f1": value},
                }
            )
        return records

    def test_exact_five_folds_and_collapsed_fold_is_included(self) -> None:
        result = aggregate_seed_folds(self._records(), "macro_f1")
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["mean"], 0.8)
        self.assertEqual(result[0]["collapsed_folds"], [4])
        self.assertTrue(result[0]["collapsed_folds_included"])

    def test_missing_duplicate_and_extra_folds_are_rejected(self) -> None:
        records = self._records()
        with self.assertRaisesRegex(ValueError, "missing"):
            aggregate_seed_folds(records[:-1], "macro_f1")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            aggregate_seed_folds(records + [dict(records[2])], "macro_f1")
        extra = records + [
            {"arm": "A", "seed": 0, "fold": 5, "metrics": {"macro_f1": 0.2}}
        ]
        with self.assertRaisesRegex(ValueError, "extra"):
            aggregate_seed_folds(extra, "macro_f1")

    def test_seed_pairing_uses_ids_not_input_order(self) -> None:
        arm_a = {3: 7 / 16, 1: 3 / 16, 4: 9 / 16, 0: 1 / 16, 2: 5 / 16}
        arm_b = {4: 4 / 16, 2: 2 / 16, 0: 0, 3: 3 / 16, 1: 1 / 16}
        result = paired_seed_differences(arm_a, arm_b)
        self.assertEqual(result["seeds"], [0, 1, 2, 3, 4])
        np.testing.assert_allclose(result["differences"], np.arange(1, 6) / 16)

    def test_paired_t_exact_values_and_dynamic_df(self) -> None:
        result = paired_t_difference([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertEqual(result["df"], 4)
        self.assertAlmostEqual(result["sample_sd"], 0.15811388300841897)
        self.assertAlmostEqual(result["t_statistic"], 4.242640687119285)
        self.assertAlmostEqual(result["p_value_two_sided"], 0.01323559956368269)
        np.testing.assert_allclose(
            result["confidence_interval"],
            [0.1036756838522443, 0.4963243161477557],
        )
        three = paired_t_difference([1 / 16, 2 / 16, 3 / 16])
        self.assertEqual(three["df"], 2)
        self.assertAlmostEqual(three["t_critical"], 4.302652729749462)

    def test_paired_t_degenerate_cases(self) -> None:
        zero = paired_t_difference([0.0] * 5)
        self.assertEqual(zero["t_statistic"], 0.0)
        self.assertEqual(zero["p_value_two_sided"], 1.0)
        negative = paired_t_difference([-0.25] * 5)
        self.assertTrue(math.isinf(negative["t_statistic"]))
        self.assertLess(negative["t_statistic"], 0.0)
        self.assertEqual(negative["p_value_two_sided"], 0.0)

    def test_holm_and_bh_restore_unsorted_tied_input(self) -> None:
        p_values = [0.041, 0.008, 0.2, 0.008, 0.029]
        holm = holm_adjust(p_values)
        bh = benjamini_hochberg(p_values)
        np.testing.assert_allclose(holm["adjusted_p_values"], [0.087, 0.04, 0.2, 0.04, 0.087])
        np.testing.assert_allclose(
            bh["adjusted_p_values"],
            [0.05125, 0.02, 0.2, 0.02, 29 / 600],
        )
        self.assertEqual(holm["rejected"], [False, True, False, True, False])
        self.assertEqual(bh["rejected"], [False, True, False, True, True])

    def test_holm_uses_the_complete_declared_family_size(self) -> None:
        headline = holm_adjust([0.01] * 12)
        full = holm_adjust([0.01] * 20)
        np.testing.assert_allclose(headline["adjusted_p_values"], [0.12] * 12)
        np.testing.assert_allclose(full["adjusted_p_values"], [0.20] * 20)


class DiagnosticAndOptionalMetricTests(unittest.TestCase):
    def test_collapse_closed_endpoints_and_invalid_counts(self) -> None:
        expected = {
            (0, 100): True,
            (2, 98): True,
            (3, 97): False,
            (59, 41): False,
            (60, 40): True,
        }
        for counts, collapsed in expected.items():
            with self.subTest(counts=counts):
                self.assertEqual(collapse_from_counts(*counts)["collapsed"], collapsed)
        for counts in [(-1, 101), (0, 0), (1.5, 98.5), (np.nan, 100), (np.inf, 0)]:
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                collapse_from_counts(*counts)

    def test_collapse_triangulation_requires_two_routes(self) -> None:
        cases = [
            ([], (None, False, True)),
            ([True], (None, True, True)),
            ([False], (None, True, True)),
            ([True, True], (True, True, False)),
            ([False, False], (False, True, False)),
            ([True, False], (None, False, True)),
            ([True, True, False], (True, False, True)),
            ([True, False, False], (False, False, True)),
        ]
        for routes, expected in cases:
            with self.subTest(routes=routes):
                result = triangulate_collapse(routes)
                self.assertEqual(
                    (result["verdict"], result["unanimous"], result["disagreement_warning"]),
                    expected,
                )

    def test_epoch_collapse_routes_sort_validate_and_scan_all_epochs(self) -> None:
        routes = epoch_prediction_collapse_routes(
            [
                {"epoch": 2, "airplane_count": 10, "ship_count": 90},
                {"epoch": 1, "airplane_count": 0, "ship_count": 100},
            ]
        )
        self.assertEqual(routes["last_epoch_prediction_fraction"]["epoch"], 2)
        self.assertFalse(routes["last_epoch_prediction_fraction"]["collapsed"])
        self.assertTrue(routes["any_epoch_exact_single_class"]["collapsed"])
        self.assertEqual(routes["any_epoch_exact_single_class"]["collapsed_epochs"], [1])
        with self.assertRaisesRegex(ValueError, "same positive total"):
            epoch_prediction_collapse_routes(
                [
                    {"epoch": 1, "airplane_count": 10, "ship_count": 90},
                    {"epoch": 2, "airplane_count": 10, "ship_count": 89},
                ]
            )
        with self.assertRaisesRegex(ValueError, "duplicate epoch"):
            epoch_prediction_collapse_routes(
                [
                    {"epoch": 1, "airplane_count": 10, "ship_count": 90},
                    {"epoch": 1, "airplane_count": 11, "ship_count": 89},
                ]
            )

    def test_target_label_diagnostics_are_gated_and_read_only(self) -> None:
        truth = [0, 1]
        probs = [[0.8, 0.2], [0.2, 0.8]]
        with self.assertRaises(PermissionError):
            threshold_sweep(truth, probs, [0.5])
        sweep = threshold_sweep(
            truth,
            probs,
            [0.3, 0.5, 0.7],
            authorised=True,
            reason="post-freeze sensitivity analysis",
        )
        self.assertFalse(sweep["selection_performed"])
        with self.assertRaises(PermissionError):
            oracle_best_epochs([{"epoch": 1, "metrics": {"macro_f1": 0.5}}], ["macro_f1"])
        oracle = oracle_best_epochs(
            [
                {"epoch": 2, "metrics": {"macro_f1": 0.7}},
                {"epoch": 1, "metrics": {"macro_f1": 0.7}},
            ],
            ["macro_f1"],
            authorised=True,
            reason="post-freeze optimistic upper bound",
        )
        self.assertEqual(oracle["best_by_metric"]["macro_f1"]["epoch"], 1)
        self.assertFalse(oracle["single_deployable_checkpoint"])

    def test_domain_metrics_validate_and_recover_simple_cases(self) -> None:
        constant = np.ones((3, 2))
        self.assertAlmostEqual(rbf_mmd2(constant, constant, bandwidth=1.0)["mmd2"], 0.0)
        self.assertAlmostEqual(coral_distance(constant, constant)["coral_distance"], 0.0)
        source_1d = np.asarray([[0.0], [1.0]])
        target_1d = np.asarray([[2.0], [3.0]])
        expected_mmd2 = (
            1.5 * math.exp(-0.5)
            - math.exp(-2.0)
            - 0.5 * math.exp(-4.5)
        )
        self.assertAlmostEqual(
            rbf_mmd2(source_1d, target_1d, bandwidth=1.0)["mmd2"],
            expected_mmd2,
        )
        coral = coral_distance(
            np.asarray([[0.0], [1.0], [2.0]]),
            np.asarray([[0.0], [2.0], [4.0]]),
        )
        self.assertAlmostEqual(coral["coral_distance"], 2.25)
        self.assertAlmostEqual(coral["mean_difference_l2"], 1.0)
        with self.assertRaises(ValueError):
            rbf_mmd2(np.ones((1, 2)), constant, bandwidth=1.0)
        for bandwidth in [0.0, np.nan, np.inf]:
            with self.subTest(bandwidth=bandwidth), self.assertRaises(ValueError):
                rbf_mmd2(constant, constant, bandwidth=bandwidth)
        source = np.column_stack((np.arange(10, dtype=float), np.zeros(10)))
        target = source + np.asarray([100.0, 0.0])
        pad = proxy_a_distance(source, target, seed=0)
        self.assertEqual(pad["domain_classifier_error"], 0.0)
        self.assertEqual(pad["proxy_a_distance"], 2.0)
        self.assertEqual(pad["convention"], "unclipped finite-sample estimate 2*(1-2*error)")

    def test_bbse_identity_case_and_singular_rejection(self) -> None:
        source_labels = [0, 0, 0, 0, 1, 1, 1, 1]
        source_probs = [
            [1, 0], [1, 0], [1, 0], [0, 1],
            [1, 0], [0, 1], [0, 1], [0, 1],
        ]
        target_probs = [[0.65, 0.35], [0.65, 0.35]]
        result = bbse_hard_source_soft_target_binary(
            source_labels,
            source_probs,
            target_probs,
        )
        np.testing.assert_allclose(result["estimated_target_prior"], [0.8, 0.2])
        self.assertFalse(
            np.allclose(
                result["estimated_target_prior"],
                result["target_soft_prediction_marginal"],
            )
        )
        clipped = bbse_hard_source_soft_target_binary(
            source_labels,
            source_probs,
            [[1.0, 0.0], [1.0, 0.0]],
        )
        self.assertTrue(clipped["negative_weights_clipped"])
        self.assertEqual(clipped["status"], "negative_weights_clipped")
        np.testing.assert_allclose(clipped["estimated_target_prior"], [1.0, 0.0])
        singular_source_probs = [[1, 0]] * len(source_labels)
        with self.assertRaisesRegex(ValueError, "singular|ill-conditioned"):
            bbse_hard_source_soft_target_binary(
                source_labels,
                singular_source_probs,
                target_probs,
            )

    def test_brier_entropy_missing_denominator_and_strata(self) -> None:
        perfect = brier_score([0, 1], [[1, 0], [0, 1]])
        self.assertEqual(perfect["brier"], 0.0)
        uniform = brier_score([0, 1], [[0.5, 0.5], [0.5, 0.5]])
        self.assertEqual(uniform["brier"], 0.5)
        entropy = confidence_entropy_summary([0, 1], [[1, 0], [0, 1]])
        self.assertEqual(entropy["overall"]["mean_entropy_nats"], 0.0)
        calibration = calibration_errors([0, 1], [[1, 0], [0, 1]], n_bins=2)
        self.assertEqual(calibration["ece"], 0.0)
        self.assertEqual(calibration["classwise_calibration_error"], 0.0)
        imbalance = imbalance_metrics([0, 1], [0, 1])
        self.assertEqual(imbalance["balanced_accuracy"], 1.0)
        self.assertEqual(imbalance["gmean_recall"], 1.0)
        missing = imbalance_metrics([1, 1], [1, 0])
        self.assertIsNone(missing["balanced_accuracy"])
        directed = directed_confusion_rates([1, 1], [1, 0])
        self.assertIsNone(directed["airplane_to_ship_rate"])

        strata = stratified_classification_metrics(
            [0, 0, 1, 1],
            [0, 1, 1, 0],
            [5.0, 5.0, None, 5.0],
            n_bins=3,
        )
        self.assertEqual(strata["status"], "degenerate_equal_covariate")
        self.assertEqual(strata["effective_bins"], 1)
        self.assertEqual(strata["excluded_nonfinite"], 1)
        with self.assertRaisesRegex(ValueError, "same length"):
            stratified_classification_metrics([0, 1], [0, 1], [1.0])

    def test_niqe_cache_is_explicit_and_reports_per_image_misses(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = Path(temp_dir) / "niqe.json"
            cache.write_text(
                '{"impl":"fixture","niqe_by_path":{"image-a":4.5}}',
                encoding="utf-8",
            )
            result = niqe_scores(
                ["image-a", "image-b"],
                backend="cache",
                cache_path=cache,
            )
        self.assertEqual(result["scores"], [4.5, None])
        self.assertEqual(result["failures"], [{"index": 1, "error": "cache_miss"}])


class LastKIntegrationTests(unittest.TestCase):
    @staticmethod
    def _import_lastk_module():
        import _lastk_eval
        return _lastk_eval

    def test_checkpoint_selector_uses_numeric_epoch_order(self) -> None:
        _lastk_eval = self._import_lastk_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for epoch in [1, 2, 9, 10, 11, 12]:
                (root / f"model_epoch{epoch}.pth").touch()
            paths, epochs = _lastk_eval.select_lastk_ckpts(root, "model", k=5)
        self.assertEqual(epochs, [2, 9, 10, 11, 12])
        self.assertEqual([Path(path).name for path in paths], [f"model_epoch{e}.pth" for e in epochs])

    def test_checkpoint_selector_rejects_invalid_k_and_duplicate_epochs(self) -> None:
        _lastk_eval = self._import_lastk_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "model_epoch1.pth").touch()
            for invalid_k in [0, -1, 1.5, True]:
                with self.subTest(k=invalid_k), self.assertRaisesRegex(ValueError, "positive integer"):
                    _lastk_eval.select_lastk_ckpts(root, "model", k=invalid_k)
            (root / "model_epoch001.pth").touch()
            with self.assertRaisesRegex(ValueError, "duplicate parsed epoch"):
                _lastk_eval.select_lastk_ckpts(root, "model", k=1)

    def test_lastk_shared_kernel_promotes_to_float64_before_accumulation(self) -> None:
        _lastk_eval = self._import_lastk_module()
        lower = np.float32(0.4999999701976776)
        arrays = {
            str(index): np.asarray([[lower, np.float32(1.0) - lower]], dtype=np.float32)
            for index in range(4)
        }
        arrays["4"] = np.asarray([[0.5, 0.5]], dtype=np.float32)
        original_loader = _lastk_eval.load_model_for_eval
        original_eval = _lastk_eval.eval_probs_unlabeled
        _lastk_eval.load_model_for_eval = lambda model_factory, path, device: path
        _lastk_eval.eval_probs_unlabeled = (
            lambda model, loader, device, forward_logits: arrays[model]
        )
        try:
            predicted, averaged, per_checkpoint = _lastk_eval.lastk_prob_average_unlabeled(
                [str(index) for index in range(5)],
                eval_loader=None,
                device=None,
                model_factory=None,
                forward_logits=None,
            )
        finally:
            _lastk_eval.load_model_for_eval = original_loader
            _lastk_eval.eval_probs_unlabeled = original_eval
        self.assertEqual(averaged.dtype, np.float64)
        self.assertTrue(all(array.dtype == np.float64 for array in per_checkpoint))
        self.assertEqual(predicted, [1])


if __name__ == "__main__":
    unittest.main()
