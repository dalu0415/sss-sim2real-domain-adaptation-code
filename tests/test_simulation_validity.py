from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = REPO_ROOT / "evidence" / "simulation_validity"
ANALYZER_PATH = REPO_ROOT / "src" / "simulation_validity" / "analyze_paired_delta.py"
RUNNER_PATH = REPO_ROOT / "src" / "simulation_validity" / "e2_ablation_runner.py"


def load_analyzer():
    spec = importlib.util.spec_from_file_location("simulation_validity_analysis", ANALYZER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load analyzer module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics_from_confusion_matrix(matrix):
    aa, a_to_s = matrix[0]
    s_to_a, ss = matrix[1]

    def safe_div(num, den):
        return float(num / den) if den else 0.0

    precision_airplane = safe_div(aa, aa + s_to_a)
    recall_airplane = safe_div(aa, aa + a_to_s)
    precision_ship = safe_div(ss, ss + a_to_s)
    recall_ship = safe_div(ss, ss + s_to_a)
    f1_airplane = safe_div(
        2.0 * precision_airplane * recall_airplane,
        precision_airplane + recall_airplane,
    )
    f1_ship = safe_div(
        2.0 * precision_ship * recall_ship,
        precision_ship + recall_ship,
    )
    return {
        "precision_airplane": precision_airplane,
        "recall_airplane": recall_airplane,
        "f1_airplane": f1_airplane,
        "precision_ship": precision_ship,
        "recall_ship": recall_ship,
        "f1_ship": f1_ship,
        "macro_f1": (f1_airplane + f1_ship) / 2.0,
        "balanced_accuracy": (recall_airplane + recall_ship) / 2.0,
        "aircraft_recall": recall_airplane,
        "accuracy": safe_div(aa + ss, aa + a_to_s + s_to_a + ss),
    }


class EvidenceIntegrityTests(unittest.TestCase):
    def test_all_72_cells_are_complete_and_recompute_exactly(self) -> None:
        full = sorted((EVIDENCE_ROOT / "per_run").glob("*.json"))
        subset = sorted((EVIDENCE_ROOT / "per_run_structure_subset").glob("*.json"))
        self.assertEqual(len(full), 48)
        self.assertEqual(len(subset), 24)

        signatures = set()
        for path in full + subset:
            record = json.loads(path.read_text(encoding="utf-8"))
            signatures.add(record["R_signature_md5"])
            recalculated = metrics_from_confusion_matrix(record["confmat_full_R"])
            for metric, expected in recalculated.items():
                self.assertEqual(record["metrics_full_R"][metric], expected, (path, metric))
            self.assertTrue(record["convergence"]["loss_decreased"], path)
            self.assertFalse(record["nan_seen"], path)
            self.assertFalse(record["convergence"]["nan_seen"], path)
        self.assertEqual(signatures, {"b4f9077323250ff506ac6580cf82d022"})

    def test_structure_subset_and_format_protocol_metadata(self) -> None:
        manifest = json.loads(
            (EVIDENCE_ROOT / "structure_subset_ship_fnames.json").read_text(
                encoding="utf-8"
            )
        )
        changed = manifest["diff_ship_fnames"]
        unchanged = manifest["same_ship_fnames"]
        self.assertEqual((manifest["n_diff"], manifest["n_same"]), (263, 137))
        self.assertEqual(len(changed), len(set(changed)))
        self.assertEqual(len(unchanged), len(set(unchanged)))
        self.assertFalse(set(changed).intersection(unchanged))
        subset_hash = hashlib.sha256()
        for name in sorted(changed):
            encoded = name.encode("utf-8")
            subset_hash.update(len(encoded).to_bytes(4, "big"))
            subset_hash.update(encoded)
        self.assertEqual(
            manifest["structure_subset_signature_sha256"],
            subset_hash.hexdigest(),
        )

        for arm in range(4):
            probe = json.loads(
                (EVIDENCE_ROOT / ("format_probe_arm%d.json" % arm)).read_text(
                    encoding="utf-8"
                )
            )
            for cls in ("airplane", "ship"):
                values = probe[cls]
                self.assertEqual(values["n_resamples"], 20)
                self.assertEqual(values["folds_per_resample"], 5)
                self.assertEqual(values["n_auc_values"], 100)
                self.assertIn("not a standard error", values["auc_std_note"])

    def test_packaged_text_contains_no_author_machine_path(self) -> None:
        slash = chr(92)
        forbidden = (
            "E:" + slash,
            "C:" + slash + "Users" + slash,
            "claude" + "-gpt-" + "cowork",
        )
        for path in REPO_ROOT.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {
                ".py", ".json", ".md", ".yaml", ".csv", ".txt"
            }:
                continue
            text = path.read_text(encoding="utf-8")
            for marker in forbidden:
                self.assertNotIn(marker, text, path)

    def test_runner_contracts_do_not_use_optimizable_assert_statements(self) -> None:
        tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
        assert_lines = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assert)]
        self.assertEqual(assert_lines, [])

    def test_evidence_sha256_manifest(self) -> None:
        manifest_path = EVIDENCE_ROOT / "SHA256SUMS.txt"
        entries = {}
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            digest, relative = line.split("  ", 1)
            self.assertEqual(len(digest), 64)
            int(digest, 16)
            self.assertNotIn(relative, entries)
            entries[relative] = digest

        evidence_files = {
            path.relative_to(EVIDENCE_ROOT).as_posix()
            for path in EVIDENCE_ROOT.rglob("*")
            if path.is_file() and path != manifest_path
        }
        self.assertEqual(set(entries), evidence_files)
        for relative, expected in entries.items():
            actual = hashlib.sha256(
                (EVIDENCE_ROOT / relative).read_bytes()
            ).hexdigest()
            self.assertEqual(actual, expected, relative)


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.analysis = load_analyzer()

    def load_strict_inputs(self):
        table, missing, _ = self.analysis.load_per_run(
            str(EVIDENCE_ROOT / "per_run")
        )
        self.assertEqual(missing, [])
        subset = self.analysis.load_per_run_loose(
            str(EVIDENCE_ROOT / "per_run_structure_subset")
        )
        probes = self.analysis.load_format_probe(str(EVIDENCE_ROOT))
        manifest = json.loads(
            (EVIDENCE_ROOT / "structure_subset_ship_fnames.json").read_text(
                encoding="utf-8"
            )
        )
        return table, subset, probes, manifest

    def test_frozen_arm0_to_arm2_result(self) -> None:
        table, missing, present = self.analysis.load_per_run(
            str(EVIDENCE_ROOT / "per_run")
        )
        self.assertEqual(len(present), 48)
        self.assertEqual(missing, [])
        result = self.analysis.analyze_one_metric(
            table, 0, 2, "macro_f1", self.analysis.SEEDS
        )
        self.assertTrue(math.isclose(result["delta_mean"], 0.193771039413280))
        self.assertTrue(
            math.isclose(result["ci_t_primary"]["lo"], 0.123311926195007)
        )
        self.assertTrue(
            math.isclose(result["ci_t_primary"]["hi"], 0.264230152631554)
        )

    def test_cli_writes_distinct_text_and_json_and_rejects_partial_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            report = tmp_path / "report.txt"
            completed = subprocess.run(
                [sys.executable, str(ANALYZER_PATH), "--out", str(report)],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            json_report = tmp_path / "report.txt.json"
            self.assertTrue(report.read_text(encoding="utf-8").startswith("# "))
            parsed = json.loads(json_report.read_text(encoding="utf-8"))
            comparison = parsed["comparisons"]["0->2"]["primary"][0]
            self.assertEqual(comparison["metric"], "macro_f1")
            self.assertEqual(comparison["n_pairs"], 12)

            partial = tmp_path / "partial"
            partial.mkdir()
            failed = subprocess.run(
                [
                    sys.executable,
                    str(ANALYZER_PATH),
                    "--per_run",
                    str(partial),
                    "--out",
                    str(tmp_path / "partial.md"),
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("full per-run directory", failed.stderr)

    def test_strict_validation_rejects_internal_id_and_metric_corruption(self) -> None:
        table, subset, probes, manifest = self.load_strict_inputs()
        bad_id = copy.deepcopy(table)
        bad_id[(0, 0)]["arm"] = 3
        with self.assertRaisesRegex(RuntimeError, "protocol-field mismatches"):
            self.analysis.validate_strict_evidence(bad_id, subset, probes, manifest)

        bad_metric = copy.deepcopy(table)
        bad_metric[(0, 0)]["metrics_full_R"]["macro_f1"] += 0.01
        with self.assertRaisesRegex(RuntimeError, "does not match confmat_full_R"):
            self.analysis.validate_strict_evidence(bad_metric, subset, probes, manifest)

        bad_environment = copy.deepcopy(table)
        bad_environment[(0, 0)]["torch_version"] = "DIFFERENT-TORCH-BUILD"
        with self.assertRaisesRegex(RuntimeError, "torch_version differs"):
            self.analysis.validate_strict_evidence(
                bad_environment, subset, probes, manifest
            )

    def test_strict_validation_checks_new_environment_signatures(self) -> None:
        table, subset, probes, manifest = self.load_strict_inputs()
        metadata = {
            "python_version": "3.11.15",
            "torch_version": "2.9.1+cu128",
            "torchvision_version": "0.24.1+cu128",
            "numpy_version": "2.2.6",
            "opencv_version": "4.12.0",
            "cuda_version": "12.8",
            "cudnn_version": 91002,
        }
        signature = self.analysis._canonical_json_sha256(metadata)
        for record in list(table.values()) + list(subset.values()):
            record["software_environment"] = copy.deepcopy(metadata)
            record["software_environment_signature_sha256"] = signature
        self.analysis.validate_strict_evidence(table, subset, probes, manifest)

        table[(0, 0)]["software_environment"]["numpy_version"] = "DIFFERENT"
        with self.assertRaisesRegex(RuntimeError, "does not match its metadata"):
            self.analysis.validate_strict_evidence(table, subset, probes, manifest)

    def test_strict_validation_rejects_failed_or_incomplete_format_probe(self) -> None:
        table, subset, probes, manifest = self.load_strict_inputs()
        bad_probe = copy.deepcopy(probes)
        bad_probe[2]["_error"] = "synthetic failure"
        with self.assertRaisesRegex(RuntimeError, "successful JSON object"):
            self.analysis.validate_strict_evidence(table, subset, bad_probe, manifest)

        bad_probe = copy.deepcopy(probes)
        del bad_probe[2]["ship"]["n_auc_values"]
        with self.assertRaisesRegex(RuntimeError, "protocol mismatches"):
            self.analysis.validate_strict_evidence(table, subset, bad_probe, manifest)

    def test_default_recomputed_output_is_outside_frozen_evidence(self) -> None:
        default_report = Path(self.analysis.DEFAULT_REPORT).resolve()
        self.assertNotEqual(default_report.parent, EVIDENCE_ROOT.resolve())
        self.analysis.validate_evidence_sha256_manifest(
            str(EVIDENCE_ROOT), str(EVIDENCE_ROOT / "SHA256SUMS.txt")
        )


if __name__ == "__main__":
    unittest.main()
