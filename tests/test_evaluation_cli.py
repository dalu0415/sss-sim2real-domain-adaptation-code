from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRY_POINT = REPO_ROOT / "src" / "evaluate.py"
PAPER_ARMS = (
    "source_only_adabn",
    "cdan",
    "cdan_adabn",
    "iw_cdan",
    "iw_cdan_adabn",
)


class EvaluationCliTests(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(ENTRY_POINT), *args],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    @staticmethod
    def _write_arrays(root: Path, *, k: int = 5) -> tuple[Path, Path]:
        checkpoint_probs = np.asarray(
            [
                [[0.8, 0.2], [0.3, 0.7]],
                [[0.7, 0.3], [0.2, 0.8]],
                [[0.9, 0.1], [0.4, 0.6]],
                [[0.6, 0.4], [0.1, 0.9]],
                [[0.8, 0.2], [0.2, 0.8]],
            ][:k],
            dtype=np.float32,
        )
        labels = np.asarray([0, 1], dtype=np.int64)
        probs_path = root / f"checkpoint_probs_k{k}.npy"
        labels_path = root / "labels.npy"
        np.save(probs_path, checkpoint_probs)
        np.save(labels_path, labels)
        return probs_path, labels_path

    def _score(self, root: Path, *, k: int = 5, output_name: str = "score.json") -> subprocess.CompletedProcess[str]:
        probs_path, labels_path = self._write_arrays(root, k=k)
        return self._run(
            "score",
            "--checkpoint-probabilities",
            str(probs_path),
            "--labels",
            str(labels_path),
            "--authorize-target-labels",
            "--authorization-reason",
            "post-freeze final scoring",
            "--arm",
            "source_only_adabn",
            "--seed",
            "0",
            "--fold",
            "0",
            "--output",
            str(root / output_name),
        )

    def test_score_round_trip_is_deterministic_and_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = self._score(root, output_name="first.json")
            second = self._score(root, output_name="second.json")
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            first_path = root / "first.json"
            second_path = root / "second.json"
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            payload = json.loads(first_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["analysis_contract"], "paper_primary_v1")
            self.assertEqual(payload["metrics"]["prediction_rule"]["mode"], "argmax")
            self.assertEqual(payload["checkpoint_ensemble"]["n_checkpoints"], 5)

            help_result = self._run("score", "--help")
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            self.assertNotIn("expected-k", help_result.stdout)
            self.assertNotIn("prediction-mode", help_result.stdout)
            self.assertNotIn("airplane-threshold", help_result.stdout)

    def test_score_rejects_incomplete_checkpoint_ensemble(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = self._score(root, k=4)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exactly 5 checkpoints", result.stderr)

    def test_score_requires_explicit_target_label_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            probs_path, labels_path = self._write_arrays(root)
            result = self._run(
                "score",
                "--checkpoint-probabilities",
                str(probs_path),
                "--labels",
                str(labels_path),
                "--arm",
                "source_only_adabn",
                "--seed",
                "0",
                "--fold",
                "0",
                "--output",
                str(root / "out.json"),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--authorize-target-labels", result.stderr)

    def test_score_rejects_nonpaper_seed_and_fold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            probs_path, labels_path = self._write_arrays(root)
            common = [
                "score",
                "--checkpoint-probabilities",
                str(probs_path),
                "--labels",
                str(labels_path),
                "--authorize-target-labels",
                "--authorization-reason",
                "post-freeze final scoring",
                "--arm",
                "source_only_adabn",
                "--output",
                str(root / "out.json"),
            ]
            bad_seed = self._run(*common, "--seed", "5", "--fold", "0")
            bad_fold = self._run(*common, "--seed", "0", "--fold", "5")
            self.assertNotEqual(bad_seed.returncode, 0)
            self.assertIn("seed must be one of", bad_seed.stderr)
            self.assertNotEqual(bad_fold.returncode, 0)
            self.assertIn("fold must be one of", bad_fold.stderr)

    def test_score_collapse_compare_full_paper_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            score_result = self._score(root)
            self.assertEqual(score_result.returncode, 0, score_result.stderr)

            healthy_path = root / "healthy.json"
            collapsed_path = root / "collapsed.json"
            healthy_epochs = root / "healthy_epochs.json"
            collapsed_epochs = root / "collapsed_epochs.json"
            mismatched_epochs = root / "mismatched_epochs.json"
            healthy_epochs.write_text(
                json.dumps(
                    {
                        "epochs": [
                            {"epoch": 1, "airplane_count": 1, "ship_count": 1},
                            {"epoch": 2, "airplane_count": 1, "ship_count": 1},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            collapsed_epochs.write_text(
                json.dumps(
                    {
                        "epochs": [
                            {"epoch": 1, "airplane_count": 0, "ship_count": 2},
                            {"epoch": 2, "airplane_count": 0, "ship_count": 2},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            mismatched_epochs.write_text(
                json.dumps(
                    {
                        "epochs": [
                            {"epoch": 1, "airplane_count": 1, "ship_count": 2},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            mismatched_result = self._run(
                "collapse",
                "--score-record",
                str(root / "score.json"),
                "--epoch-counts",
                str(mismatched_epochs),
                "--output",
                str(root / "mismatched.json"),
            )
            self.assertNotEqual(mismatched_result.returncode, 0)
            self.assertIn("same target count", mismatched_result.stderr)
            healthy_result = self._run(
                "collapse",
                "--score-record",
                str(root / "score.json"),
                "--epoch-counts",
                str(healthy_epochs),
                "--output",
                str(healthy_path),
            )
            collapsed_result = self._run(
                "collapse",
                "--score-record",
                str(root / "score.json"),
                "--epoch-counts",
                str(collapsed_epochs),
                "--output",
                str(collapsed_path),
            )
            self.assertEqual(healthy_result.returncode, 0, healthy_result.stderr)
            self.assertEqual(collapsed_result.returncode, 0, collapsed_result.stderr)
            healthy = json.loads(healthy_path.read_text(encoding="utf-8"))
            collapsed = json.loads(collapsed_path.read_text(encoding="utf-8"))
            self.assertFalse(healthy["collapsed"])
            self.assertTrue(collapsed["collapsed"])
            self.assertEqual(len(healthy["collapse_evidence"]), 3)
            self.assertEqual(len(healthy["epoch_prediction_histograms"]), 2)

            records = []
            for arm in PAPER_ARMS:
                for seed in range(5):
                    for fold in range(5):
                        use_collapsed = arm == "source_only_adabn" and seed == 4 and fold == 4
                        record = copy.deepcopy(collapsed if use_collapsed else healthy)
                        record["arm"] = arm
                        record["seed"] = seed
                        record["fold"] = fold
                        value = 0.1 * (seed + 1) if arm == "source_only_adabn" else 0.0
                        record["metrics"]["macro_f1"] = value
                        record["metrics"]["per_class"]["airplane"]["f1"] = value
                        record["metrics"]["per_class"]["airplane"]["precision"] = value
                        record["metrics"]["per_class"]["airplane"]["recall"] = value
                        record["metrics"]["airplane_pr_auc"] = value
                        records.append(record)

            records_path = root / "records.json"
            output_path = root / "comparison.json"
            records_path.write_text(json.dumps({"records": records}), encoding="utf-8")
            compare_result = self._run(
                "compare",
                "--records",
                str(records_path),
                "--output",
                str(output_path),
            )
            self.assertEqual(compare_result.returncode, 0, compare_result.stderr)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["comparisons"]), 20)
            self.assertEqual(payload["family_summary"]["headline12"]["n_tests"], 12)
            self.assertEqual(payload["family_summary"]["family20"]["n_tests"], 20)

            headline = next(
                comparison
                for comparison in payload["comparisons"]
                if comparison["arm_b"] == "cdan" and comparison["metric"] == "macro_f1"
            )
            np.testing.assert_allclose(
                headline["pairing"]["differences"],
                [0.1, 0.2, 0.3, 0.4, 0.5],
            )
            raw_p = headline["paired_t"]["p_value_two_sided"]
            self.assertAlmostEqual(
                headline["multiplicity"]["headline12"]["holm_adjusted_p"],
                raw_p * 12,
            )
            self.assertAlmostEqual(
                headline["multiplicity"]["family20"]["holm_adjusted_p"],
                raw_p * 20,
            )
            seed_four = next(
                record for record in headline["arm_a_seed_means"] if record["seed"] == 4
            )
            self.assertEqual(seed_four["collapsed_folds"], [4])
            self.assertTrue(seed_four["collapsed_folds_included"])

            tampered = copy.deepcopy(records)
            tampered[0]["metrics"]["prediction_rule"]["mode"] = "airplane_threshold"
            tampered_path = root / "tampered.json"
            tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
            rejected = self._run(
                "compare",
                "--records",
                str(tampered_path),
                "--output",
                str(root / "rejected.json"),
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("argmax", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
