# -*- coding: utf-8 -*-
"""Recompute the frozen seed-matched simulation-validity analysis.

Each arm contributes one score for each of the same 12 random seeds. For every
pre-specified comparison, the analysis forms 12 matched differences and reports
a two-sided 95% paired-t interval. A paired bootstrap over those same 12
differences is retained only as a cross-check; the paired-t interval is the
fixed decision rule. An interval that includes zero is reported as
inconclusive, not as proof of no effect or proof of low statistical power.

Strict mode requires all 48 full-data cells, all 24 structure-subset cells,
and one shared non-empty target-set fingerprint. This prevents a partial run
from silently producing a scientific verdict.
"""
import os
import json
import argparse
import hashlib
import math
import re

import numpy as np

# ---------------------------------------------------------------------------
# Included evidence paths + frozen analysis constants.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
EVIDENCE_ROOT = os.path.join(REPO_ROOT, "evidence", "simulation_validity")
PER_RUN_DIR = os.path.join(EVIDENCE_ROOT, "per_run")
PER_RUN_268_DIR = os.path.join(EVIDENCE_ROOT, "per_run_structure_subset")
STRUCTURE_MANIFEST = os.path.join(EVIDENCE_ROOT, "structure_subset_ship_fnames.json")
EVIDENCE_SHA256_MANIFEST = os.path.join(EVIDENCE_ROOT, "SHA256SUMS.txt")
DEFAULT_REPORT = os.path.join(REPO_ROOT, "paired_delta_report_recomputed.md")

SEEDS = list(range(12))            # 12 配对 seed (写死, 与 runner 一致)
ARMS = [0, 1, 2, 3]
N_PAIRS = 12                       # 配对样本数 (固定 12)
CEILING_MACRO_F1 = 0.890          # Rounded real-to-real five-fold OOF reference
N_BOOT = 10000                    # 配对 bootstrap 次数 (交叉核)
BOOT_SEED = 12345                 # bootstrap RNG seed (写死)

# t(11, 0.975) 双侧 95% 临界值 (df=11)。写死, 不依赖运行时库可用性。
# scipy.stats.t.ppf(0.975, 11) = 2.200985160082949
T_CRIT_DF11 = 2.200985160082949

# 指标全集 (与 runner SCALAR_METRIC_KEYS 一致)
SCALAR_METRIC_KEYS = [
    "precision_airplane", "recall_airplane", "f1_airplane",
    "precision_ship", "recall_ship", "f1_ship",
    "macro_f1", "balanced_accuracy", "aircraft_recall", "accuracy",
]

ARM_DATASET_IDS = {
    0: "arm0_native_baseline",
    1: "arm1_low_level_realism",
    2: "arm2_selected_source",
    3: "arm3_internal_structure",
}
TARGET_DATASET_ID = "KLSG-II/canonical_evaluation"
FORMAT_TARGET_DATASET_ID = "KLSG-II/raw_format_probe"
HEX32_RE = re.compile(r"^[0-9a-fA-F]{32}$")
HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")

# Six frozen comparison blocks. Primary and secondary roles remain distinct.
COMPARISONS = [
    {"name": "0->1", "x": 0, "y": 1, "desc": "low-level treatment increment (both classes)",
     "primary": ["macro_f1"], "secondary": ["f1_ship", "recall_ship", "f1_airplane", "recall_airplane", "balanced_accuracy"],
     "use_structure_subset": False},
    {"name": "1->2", "x": 1, "y": 2, "desc": "ship morphology increment",
     "primary": ["f1_ship", "recall_ship"], "secondary": ["macro_f1", "balanced_accuracy"],
     "use_structure_subset": False},
    {"name": "2->3", "x": 2, "y": 3, "desc": "internal-structure increment over all 400 ships",
     "primary": ["f1_ship", "recall_ship"], "secondary": ["macro_f1", "balanced_accuracy"],
     "use_structure_subset": False},
    {"name": "2->3 (structure subset)", "x": 2, "y": 3, "desc": "internal-structure increment over 263 changed ships",
     "primary": ["f1_ship", "recall_ship"], "secondary": ["macro_f1", "balanced_accuracy"],
     "use_structure_subset": True},
    {"name": "0->3", "x": 0, "y": 3, "desc": "all-stage endpoint relative to arm0",
     "primary": ["macro_f1"], "secondary": ["f1_ship", "recall_ship", "f1_airplane", "recall_airplane", "balanced_accuracy"],
     "use_structure_subset": False},
    {"name": "0->2", "x": 0, "y": 2, "desc": "selected arm2 source relative to arm0",
     "primary": ["macro_f1"], "secondary": ["f1_ship", "recall_ship", "f1_airplane", "recall_airplane", "balanced_accuracy"],
     "use_structure_subset": False},
]


# ===========================================================================
# 读 per_run JSON: {(arm,seed): record}。缺 cell 记缺。
# ===========================================================================
def load_per_run(per_run_dir):
    table = {}
    missing = []
    present = []
    if not os.path.isdir(per_run_dir):
        return table, missing, present
    for arm in ARMS:
        for seed in SEEDS:
            p = os.path.join(per_run_dir, "%d_seed%d.json" % (arm, seed))
            if os.path.isfile(p):
                with open(p, "r", encoding="utf-8") as f:
                    table[(arm, seed)] = json.load(f)
                present.append((arm, seed))
            else:
                missing.append((arm, seed))
    return table, missing, present


def load_per_run_loose(per_run_dir):
    """读任意 {arm}_seed{seed}.json (含冒烟 throwaway seed 如 9999), 不限 SEEDS/ARMS。验流程用。"""
    table = {}
    if not os.path.isdir(per_run_dir):
        return table
    for fn in os.listdir(per_run_dir):
        if not fn.endswith(".json"):
            continue
        if "_seed" not in fn:
            continue
        try:
            arm = int(fn.split("_seed")[0])
            seed = int(fn.split("_seed")[1].replace(".json", ""))
        except Exception:
            continue
        with open(os.path.join(per_run_dir, fn), "r", encoding="utf-8") as f:
            table[(arm, seed)] = json.load(f)
    return table


def _require(condition, source, message):
    if not condition:
        raise RuntimeError("%s: %s" % (source, message))


def _finite_number(value):
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
    )


def _canonical_json_sha256(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _safe_div(num, den):
    return float(num / den) if den else 0.0


def metrics_from_confusion_matrix(matrix):
    aa, a_to_s = matrix[0]
    s_to_a, ss = matrix[1]
    precision_airplane = _safe_div(aa, aa + s_to_a)
    recall_airplane = _safe_div(aa, aa + a_to_s)
    precision_ship = _safe_div(ss, ss + a_to_s)
    recall_ship = _safe_div(ss, ss + s_to_a)
    f1_airplane = _safe_div(
        2.0 * precision_airplane * recall_airplane,
        precision_airplane + recall_airplane,
    )
    f1_ship = _safe_div(
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
        "accuracy": _safe_div(aa + ss, aa + a_to_s + s_to_a + ss),
    }


def _validate_record(record, arm, seed, subset, source):
    _require(isinstance(record, dict), source, "record must be a JSON object")
    expected = {
        "arm": arm,
        "seed": seed,
        "smoke": False,
        "epochs": 40,
        "train_subset": subset,
        "arm_dir": ARM_DATASET_IDS[arm],
        "R_dir": TARGET_DATASET_ID,
        "R_n": 553,
        "ceiling_macro_f1": CEILING_MACRO_F1,
        "ckpt_avg_k": 5,
        "label_map": {"airplane": 0, "ship": 1},
        "n_structure_diff_count": 263,
        "n_structure_same_count": 137,
    }
    if subset == "full":
        expected.update({
            "n_train": 490,
            "n_train_airplane": 90,
            "n_train_ship": 400,
            "R_n_airplane": 66,
            "R_n_ship": 487,
        })
    else:
        expected.update({
            "n_train": 353,
            "n_train_airplane": 90,
            "n_train_ship": 263,
            "n_structure_ship_used": 263,
        })
    mismatches = {
        key: (record.get(key), value)
        for key, value in expected.items()
        if record.get(key) != value
    }
    _require(not mismatches, source, "protocol-field mismatches=%s" % mismatches)

    target_signature = record.get("R_signature_md5")
    _require(
        isinstance(target_signature, str) and HEX32_RE.fullmatch(target_signature),
        source,
        "R_signature_md5 must be a non-empty 32-digit hexadecimal digest",
    )
    for field in (
        "arm_input_signature_sha256",
        "structure_subset_signature_sha256",
        "protocol_signature_sha256",
        "runner_source_sha256",
        "software_environment_signature_sha256",
    ):
        value = record.get(field)
        if value is not None:
            _require(
                isinstance(value, str) and HEX64_RE.fullmatch(value),
                source,
                "%s must be a 64-digit hexadecimal digest when present" % field,
            )
    software_environment = record.get("software_environment")
    software_environment_signature = record.get(
        "software_environment_signature_sha256"
    )
    if software_environment is not None or software_environment_signature is not None:
        _require(isinstance(software_environment, dict), source,
                 "software_environment must be an object when its signature is present")
        _require(
            software_environment_signature == _canonical_json_sha256(software_environment),
            source,
            "software_environment_signature_sha256 does not match its metadata",
        )
    for field in ("torch_version", "python_version"):
        value = record.get(field)
        _require(isinstance(value, str) and value, source,
                 "%s must be a non-empty string" % field)
    cuda_version = record.get("cuda_version")
    _require(cuda_version is None or (isinstance(cuda_version, str) and cuda_version),
             source, "cuda_version must be null or a non-empty string")

    matrix = record.get("confmat_full_R")
    matrix_ok = (
        isinstance(matrix, list)
        and len(matrix) == 2
        and all(isinstance(row, list) and len(row) == 2 for row in matrix)
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for row in matrix
            for value in row
        )
    )
    _require(matrix_ok, source, "confmat_full_R must be a non-negative 2x2 integer matrix")
    _require(sum(matrix[0]) == 66, source, "airplane confusion-matrix row must sum to 66")
    _require(sum(matrix[1]) == 487, source, "ship confusion-matrix row must sum to 487")

    metrics = record.get("metrics_full_R")
    _require(isinstance(metrics, dict), source, "metrics_full_R must be an object")
    recomputed = metrics_from_confusion_matrix(matrix)
    for key in SCALAR_METRIC_KEYS:
        value = metrics.get(key)
        _require(_finite_number(value), source, "%s must be finite" % key)
        _require(0.0 <= float(value) <= 1.0, source, "%s must lie in [0, 1]" % key)
        _require(
            math.isclose(float(value), recomputed[key], rel_tol=1e-12, abs_tol=1e-12),
            source,
            "%s does not match confmat_full_R" % key,
        )
    gap = record.get("gap_to_ceiling_macro_f1")
    _require(_finite_number(gap), source, "gap_to_ceiling_macro_f1 must be finite")
    _require(
        math.isclose(
            float(gap),
            CEILING_MACRO_F1 - float(metrics["macro_f1"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        source,
        "gap_to_ceiling_macro_f1 is inconsistent with macro_f1",
    )

    losses = record.get("train_epoch_losses")
    _require(
        isinstance(losses, list) and len(losses) == 40 and all(_finite_number(x) for x in losses),
        source,
        "train_epoch_losses must contain 40 finite values",
    )
    first_last = record.get("train_first_last_loss")
    _require(
        isinstance(first_last, list)
        and len(first_last) == 2
        and all(_finite_number(x) for x in first_last)
        and math.isclose(float(first_last[0]), float(losses[0]), rel_tol=1e-12, abs_tol=1e-12)
        and math.isclose(float(first_last[1]), float(losses[-1]), rel_tol=1e-12, abs_tol=1e-12),
        source,
        "train_first_last_loss must match train_epoch_losses endpoints",
    )
    convergence = record.get("convergence")
    _require(isinstance(convergence, dict), source, "convergence must be an object")
    _require(record.get("nan_seen") is False, source, "nan_seen must be false")
    _require(convergence.get("nan_seen") is False, source, "convergence.nan_seen must be false")
    _require(convergence.get("loss_decreased") is True, source, "training loss did not decrease")


def _validate_structure_manifest(manifest, source):
    _require(isinstance(manifest, dict), source, "manifest must be a JSON object")
    changed = manifest.get("diff_ship_fnames")
    unchanged = manifest.get("same_ship_fnames")
    _require(isinstance(changed, list) and isinstance(unchanged, list), source,
             "diff_ship_fnames and same_ship_fnames must be lists")
    _require(manifest.get("n_diff") == 263 and len(changed) == 263, source,
             "changed-ship count must be 263")
    _require(manifest.get("n_same") == 137 and len(unchanged) == 137, source,
             "unchanged-ship count must be 137")
    _require(len(changed) == len(set(changed)), source, "changed filenames must be unique")
    _require(len(unchanged) == len(set(unchanged)), source, "unchanged filenames must be unique")
    _require(not set(changed).intersection(unchanged), source,
             "changed and unchanged filename sets must be disjoint")
    _require(len(set(changed).union(unchanged)) == 400, source,
             "changed and unchanged filename sets must cover 400 ships")
    digest = hashlib.sha256()
    for name in sorted(changed):
        _require(isinstance(name, str) and name, source, "ship filenames must be non-empty strings")
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    _require(
        manifest.get("structure_subset_signature_sha256") == digest.hexdigest(),
        source,
        "structure_subset_signature_sha256 does not match diff_ship_fnames",
    )
    return digest.hexdigest()


def _validate_format_probes(fmt_traj):
    _require(set(fmt_traj) == set(ARMS), "format probes", "exactly arms 0,1,2,3 are required")
    expected_counts = {
        "airplane": (90, 66, 66),
        "ship": (400, 487, 400),
    }
    for arm in ARMS:
        probe = fmt_traj[arm]
        source = "format_probe_arm%d.json" % arm
        _require(isinstance(probe, dict) and "_error" not in probe, source,
                 "probe must be a successful JSON object")
        _require(probe.get("arm") == arm, source, "internal arm does not match filename")
        _require(probe.get("arm_dir") == ARM_DATASET_IDS[arm], source,
                 "arm_dir does not match the frozen arm")
        _require(probe.get("klsg_dir") == FORMAT_TARGET_DATASET_ID, source,
                 "klsg_dir does not match the frozen probe target")
        for field in ("arm_input_signature_sha256", "klsg_input_signature_sha256"):
            value = probe.get(field)
            if value is not None:
                _require(isinstance(value, str) and HEX64_RE.fullmatch(value), source,
                         "%s must be a 64-digit hexadecimal digest when present" % field)
        software_environment = probe.get("software_environment")
        software_environment_signature = probe.get(
            "software_environment_signature_sha256"
        )
        if software_environment is not None or software_environment_signature is not None:
            _require(isinstance(software_environment, dict), source,
                     "software_environment must be an object when its signature is present")
            _require(
                software_environment_signature
                == _canonical_json_sha256(software_environment),
                source,
                "software_environment_signature_sha256 does not match its metadata",
            )
        runner_source_signature = probe.get("runner_source_sha256")
        if runner_source_signature is not None:
            _require(
                isinstance(runner_source_signature, str)
                and HEX64_RE.fullmatch(runner_source_signature),
                source,
                "runner_source_sha256 must be a 64-digit hexadecimal digest when present",
            )
        for cls, (n_synth, n_real, n_balanced) in expected_counts.items():
            values = probe.get(cls)
            _require(isinstance(values, dict), source, "%s probe must be an object" % cls)
            protocol = {
                "n_synthetic": n_synth,
                "n_real": n_real,
                "n_balanced_per_domain": n_balanced,
                "n_resamples": 20,
                "folds_per_resample": 5,
                "n_auc_values": 100,
            }
            mismatches = {
                key: (values.get(key), expected)
                for key, expected in protocol.items()
                if values.get(key) != expected
            }
            _require(not mismatches, source, "%s protocol mismatches=%s" % (cls, mismatches))
            for key in ("auc_mean", "auc_std"):
                _require(_finite_number(values.get(key)), source, "%s.%s must be finite" % (cls, key))
            _require(0.0 <= float(values["auc_mean"]) <= 1.0, source,
                     "%s.auc_mean must lie in [0, 1]" % cls)
            _require(float(values["auc_std"]) >= 0.0, source,
                     "%s.auc_std must be non-negative" % cls)
            _require("not a standard error" in values.get("auc_std_note", ""), source,
                     "%s.auc_std_note must state its descriptive scope" % cls)
    for field in ("software_environment_signature_sha256", "runner_source_sha256"):
        values = [fmt_traj[arm].get(field) for arm in ARMS]
        if any(value is not None for value in values):
            _require(all(value is not None for value in values), "format probes",
                     "%s must be present in all four probes or none" % field)
            _require(len(set(values)) == 1, "format probes",
                     "%s differs across format probes" % field)


def validate_strict_evidence(table, table_subset, fmt_traj, structure_manifest):
    subset_signature = _validate_structure_manifest(
        structure_manifest, "structure_subset_ship_fnames.json"
    )
    for arm in ARMS:
        for seed in SEEDS:
            _validate_record(table[(arm, seed)], arm, seed, "full",
                             "per_run/%d_seed%d.json" % (arm, seed))
    for arm in (2, 3):
        for seed in SEEDS:
            record = table_subset[(arm, seed)]
            source = "per_run_structure_subset/%d_seed%d.json" % (arm, seed)
            _validate_record(record, arm, seed, "structure_change_subset", source)
            value = record.get("structure_subset_signature_sha256")
            if value is not None:
                _require(value == subset_signature, source,
                         "structure-subset signature does not match the manifest")

    records = list(table.values()) + list(table_subset.values())
    target_signatures = {record["R_signature_md5"] for record in records}
    _require(len(target_signatures) == 1, "strict evidence",
             "all 72 cells must share one target input fingerprint")
    for field in (
        "protocol_signature_sha256",
        "runner_source_sha256",
        "software_environment_signature_sha256",
    ):
        values = [record.get(field) for record in records]
        if any(value is not None for value in values):
            _require(all(value is not None for value in values), "strict evidence",
                     "%s must be present in all 72 cells or none" % field)
            _require(len(set(values)) == 1, "strict evidence",
                     "%s must be common to all 72 cells" % field)

    for field in ("torch_version", "cuda_version", "python_version"):
        values = {record[field] for record in records}
        _require(len(values) == 1, "strict evidence",
                 "%s differs across the 72 cells: %s"
                 % (field, sorted(str(value) for value in values)))

    subset_values = [
        table_subset[(arm, seed)].get("structure_subset_signature_sha256")
        for arm in (2, 3)
        for seed in SEEDS
    ]
    if any(value is not None for value in subset_values):
        _require(all(value is not None for value in subset_values), "strict evidence",
                 "structure-subset SHA-256 must be present in all 24 subset cells or none")
        _require(set(subset_values) == {subset_signature}, "strict evidence",
                 "structure-subset SHA-256 differs from the filename manifest")

    _validate_format_probes(fmt_traj)
    record_runner_sources = [record.get("runner_source_sha256") for record in records]
    probe_runner_sources = [fmt_traj[arm].get("runner_source_sha256") for arm in ARMS]
    all_runner_sources = record_runner_sources + probe_runner_sources
    if any(value is not None for value in all_runner_sources):
        _require(all(value is not None for value in all_runner_sources), "strict evidence",
                 "runner_source_sha256 must cover all cells and probes or none")
        _require(len(set(all_runner_sources)) == 1, "strict evidence",
                 "runner_source_sha256 differs between cells and probes")
    for arm in ARMS:
        records_for_arm = [table[(arm, seed)] for seed in SEEDS]
        if arm in (2, 3):
            records_for_arm += [table_subset[(arm, seed)] for seed in SEEDS]
        source_values = [record.get("arm_input_signature_sha256") for record in records_for_arm]
        if any(value is not None for value in source_values):
            _require(all(value is not None for value in source_values), "strict evidence",
                     "arm%d source SHA-256 must be present in every related cell or none" % arm)
            _require(len(set(source_values)) == 1, "strict evidence",
                     "arm%d source SHA-256 is inconsistent across cells" % arm)
            probe_value = fmt_traj[arm].get("arm_input_signature_sha256")
            _require(probe_value == source_values[0], "strict evidence",
                     "arm%d format-probe and cell source SHA-256 differ" % arm)

    klsg_values = [fmt_traj[arm].get("klsg_input_signature_sha256") for arm in ARMS]
    if any(value is not None for value in klsg_values):
        _require(all(value is not None for value in klsg_values), "strict evidence",
                 "KLSG-II SHA-256 must be present in all four format probes or none")
        _require(len(set(klsg_values)) == 1, "strict evidence",
                 "KLSG-II SHA-256 differs across format probes")


def validate_evidence_sha256_manifest(evidence_root, manifest_path):
    source = os.path.basename(manifest_path)
    entries = {}
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle.read().splitlines(), 1):
            parts = line.split("  ", 1)
            _require(len(parts) == 2, source, "invalid line %d" % line_no)
            digest, relative = parts
            _require(HEX64_RE.fullmatch(digest) is not None, source,
                     "invalid SHA-256 on line %d" % line_no)
            _require(relative not in entries, source, "duplicate path %s" % relative)
            entries[relative] = digest.lower()
    actual_files = set()
    for root, _, files in os.walk(evidence_root):
        for filename in files:
            path = os.path.join(root, filename)
            if os.path.abspath(path) == os.path.abspath(manifest_path):
                continue
            actual_files.add(os.path.relpath(path, evidence_root).replace("\\", "/"))
    _require(set(entries) == actual_files, source,
             "manifest roster differs from the evidence directory")
    root_abs = os.path.abspath(evidence_root)
    for relative, expected in entries.items():
        path = os.path.abspath(os.path.join(evidence_root, relative.replace("/", os.sep)))
        _require(os.path.commonpath([root_abs, path]) == root_abs, source,
                 "path escapes evidence root: %s" % relative)
        with open(path, "rb") as handle:
            actual = hashlib.sha256(handle.read()).hexdigest()
        _require(actual == expected, source, "SHA-256 mismatch for %s" % relative)


def _atomic_write_text(path, text):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    temporary = "%s.tmp-%d" % (path, os.getpid())
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(temporary, path)


def _atomic_write_json(path, payload):
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


# ===========================================================================
# Frozen paired-t rule with a paired-bootstrap cross-check.
# ===========================================================================
def paired_t_ci(d):
    """配对 t 95%CI (主)。d = 12 个同 seed 配对差 (np.array)。
    返回 mean, se(ddof=1)/sqrt(n), ci_lo, ci_hi, std(ddof=1)。"""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    mean = float(np.mean(d))
    std = float(np.std(d, ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 0 else float("nan")
    # df = n-1; 仅当 n==12 用写死的 T_CRIT_DF11, 否则尝试 scipy (冒烟/不全时), 失败回退到该值并标注。
    if n == 12:
        tcrit = T_CRIT_DF11
        tcrit_src = "hardcoded_t(11,0.975)=%.6f" % T_CRIT_DF11
    else:
        try:
            from scipy.stats import t as _t
            tcrit = float(_t.ppf(0.975, max(1, n - 1)))
            tcrit_src = "scipy_t(%d,0.975)=%.6f" % (n - 1, tcrit)
        except Exception:
            tcrit = T_CRIT_DF11
            tcrit_src = "FALLBACK_t(11)_n!=12_APPROX"
    lo = mean - tcrit * se
    hi = mean + tcrit * se
    return {"mean": mean, "std_ddof1": std, "se": float(se),
            "ci_lo": float(lo), "ci_hi": float(hi), "tcrit": float(tcrit),
            "tcrit_src": tcrit_src, "n": int(n)}


def paired_bootstrap_ci(d, n_boot, rng):
    """交叉核: 对 12 个配对差 d_i 有放回重抽 n_boot 次, 取每次 mean(d*) 的 2.5/97.5 分位。
    ★单一方差源 = 这 12 个 d_i (不嵌套测试集 bootstrap)。"""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    if n == 0:
        return {"ci_lo": float("nan"), "ci_hi": float("nan"), "n_boot": int(n_boot)}
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)     # 有放回重抽配对差
        means[b] = np.mean(d[idx])
    lo = float(np.percentile(means, 2.5))
    hi = float(np.percentile(means, 97.5))
    return {"ci_lo": lo, "ci_hi": hi, "n_boot": int(n_boot),
            "boot_mean": float(np.mean(means))}


def decide(ci_t):
    """Mechanical rule: effective iff the primary paired-t CI excludes zero."""
    lo, hi = ci_t["ci_lo"], ci_t["ci_hi"]
    excludes_zero = (lo > 0.0) or (hi < 0.0)
    if excludes_zero:
        direction = "improve(+)" if ci_t["mean"] > 0 else "degrade(-)"
        return {"verdict": "EFFECTIVE", "direction": direction, "excludes_zero": True}
    return {
        "verdict": "INCONCLUSIVE(95% paired-t CI includes zero)",
        "direction": "n/a",
        "excludes_zero": False,
    }


def ci_agree(ci_t, ci_boot):
    """两 CI 一致性核 (是否同含/同不含 0; 端点是否量级相近)。仅作交叉核报告, 不改判定。"""
    t_excl = (ci_t["ci_lo"] > 0) or (ci_t["ci_hi"] < 0)
    b_excl = (ci_boot["ci_lo"] > 0) or (ci_boot["ci_hi"] < 0)
    same_sign_decision = (t_excl == b_excl)
    # 端点相对差 (相对于 t-CI 半宽)
    half = max(1e-9, (ci_t["ci_hi"] - ci_t["ci_lo"]) / 2.0)
    lo_gap = abs(ci_t["ci_lo"] - ci_boot["ci_lo"]) / half
    hi_gap = abs(ci_t["ci_hi"] - ci_boot["ci_hi"]) / half
    return {"same_decision": bool(same_sign_decision),
            "endpoint_rel_gap_lo": float(lo_gap), "endpoint_rel_gap_hi": float(hi_gap),
            "consistent": bool(same_sign_decision and lo_gap < 0.5 and hi_gap < 0.5)}


def analyze_one_metric(table, armX, armY, metric, seeds):
    """对某比较某指标算配对差 + 两 CI + 判定。要求两臂在所有 seed 都有 cell。"""
    d = []
    used_seeds = []
    for s in seeds:
        if (armX, s) not in table or (armY, s) not in table:
            continue
        mx = table[(armX, s)]["metrics_full_R"][metric]
        my = table[(armY, s)]["metrics_full_R"][metric]
        d.append(my - mx)
        used_seeds.append(s)
    if len(d) < 2:
        return {"metric": metric, "n_pairs": len(d), "insufficient": True, "used_seeds": used_seeds}
    d = np.array(d, dtype=np.float64)
    ci_t = paired_t_ci(d)
    rng = np.random.RandomState(BOOT_SEED)
    ci_boot = paired_bootstrap_ci(d, N_BOOT, rng)
    verdict = decide(ci_t)
    agree = ci_agree(ci_t, ci_boot)
    abs_delta_over_se = abs(ci_t["mean"]) / ci_t["se"] if ci_t["se"] > 1e-12 else float("inf")
    return {
        "metric": metric, "n_pairs": int(len(d)), "used_seeds": used_seeds,
        "paired_diffs": [float(x) for x in d],
        "delta_mean": ci_t["mean"], "se": ci_t["se"], "std_ddof1": ci_t["std_ddof1"],
        "abs_delta_over_se": float(abs_delta_over_se),
        "ci_t_primary": {"lo": ci_t["ci_lo"], "hi": ci_t["ci_hi"], "tcrit_src": ci_t["tcrit_src"]},
        "ci_bootstrap_crosscheck": {"lo": ci_boot["ci_lo"], "hi": ci_boot["ci_hi"], "n_boot": ci_boot["n_boot"]},
        "ci_agreement": agree,
        "verdict": verdict["verdict"], "direction": verdict["direction"],
        "insufficient": False,
    }


# ===========================================================================
# 天花板 gap 轨迹 + format-probe 轨迹
# ===========================================================================
def ceiling_gap_trajectory(table, seeds):
    """每臂 macro-F1 跨 seed 均值 + 到 ceiling 的 gap。"""
    traj = {}
    for arm in ARMS:
        vals = [table[(arm, s)]["metrics_full_R"]["macro_f1"] for s in seeds if (arm, s) in table]
        if not vals:
            continue
        mean = float(np.mean(vals))
        traj[arm] = {"macro_f1_mean": mean, "macro_f1_std": float(np.std(vals, ddof=0)),
                     "gap_to_ceiling": float(CEILING_MACRO_F1 - mean), "n_seed": len(vals)}
    return traj


def load_format_probe(format_probe_dir):
    traj = {}
    for arm in ARMS:
        p = os.path.join(format_probe_dir, "format_probe_arm%d.json" % arm)
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                fp = json.load(f)
            traj[arm] = fp
    return traj


# ===========================================================================
# 报告
# ===========================================================================
def fmt_metric_block(res):
    if res.get("insufficient"):
        return "  - **%s**: 配对数不足 (n=%d), 跳过。" % (res["metric"], res["n_pairs"])
    ci = res["ci_t_primary"]
    cb = res["ci_bootstrap_crosscheck"]
    ag = res["ci_agreement"]
    lines = []
    lines.append("  - **%s**: Δ=%.4f  SE=%.4f  |Δ|/SE=%.2f  (n=%d)"
                 % (res["metric"], res["delta_mean"], res["se"], res["abs_delta_over_se"], res["n_pairs"]))
    lines.append("    - 主CI(配对t): [%.4f, %.4f]  (%s)" % (ci["lo"], ci["hi"], ci["tcrit_src"]))
    lines.append("    - 交叉核(配对bootstrap B=%d): [%.4f, %.4f]  一致=%s (endpoint_rel_gap lo=%.2f hi=%.2f)"
                 % (cb["n_boot"], cb["lo"], cb["hi"], ag["consistent"], ag["endpoint_rel_gap_lo"], ag["endpoint_rel_gap_hi"]))
    lines.append("    - **判定: %s** (%s)" % (res["verdict"], res["direction"]))
    if not ag["same_decision"]:
        lines.append("    - ⚠ 两CI判定不一致 (主t与bootstrap对'是否含0'不同) -> 标注复核, 仍以主t为准。")
    return "\n".join(lines)


def build_report(table, table268, missing, present, fmt_traj, ceil_traj, per_run_dir, smoke_mode):
    L = []
    L.append("# Simulation-validity E2 paired-difference report")
    L.append("")
    L.append("- per_run 目录: `%s`" % per_run_dir)
    L.append("- 配对结构: 同一组 12 seed (SEEDS=%s) 跑全 4 臂; 每 (臂,seed) 在固定全 R 测试集算一点。" % SEEDS)
    L.append("- 主 CI = 配对 t (mean(d) ± t(11,0.975)·std(d,ddof=1)/√12); 交叉核 = 对 12 个 d_i 配对 bootstrap (B=%d)。" % N_BOOT)
    L.append("- 固定判定规则: primary 95% paired-t CI 排除 0 时标为 EFFECTIVE；包含 0 时只标为 INCONCLUSIVE。")
    L.append("- 未作多重比较校正；secondary 指标只能作为描述性结果。")
    L.append("- ceiling macro-F1 = %.3f，来自单独的 KLSG-II 5-fold OOF 监督参照。" % CEILING_MACRO_F1)
    if smoke_mode:
        L.append("")
        L.append("> ⚠ **SMOKE/不全模式**: per_run 不是完整 48 cell (或含 throwaway seed)。本报告仅验【流程跑通】, 数字无科学意义。")
    L.append("")

    # cell 完整性
    L.append("## Cell 完整性")
    L.append("")
    L.append("- present=%d  missing=%d (期望 48 = 4臂×12seed)" % (len(present), len(missing)))
    if missing:
        L.append("- 缺失 cell: %s" % missing[:60])
    L.append("")

    # Target input image-byte signature (not a post-preprocessing tensor hash).
    L.append("## Target image-byte fingerprint consistency")
    L.append("")
    sigs = set(rec.get("R_signature_md5") for rec in table.values())
    L.append("- R_signature_md5 唯一集合 (应只 1 个): %s" % list(sigs))
    if len(sigs) == 1:
        L.append("- 全部 full cell 使用同一组目标输入图片字节。")
    elif len(sigs) > 1:
        L.append("- ⚠ R 指纹不唯一! 不同 cell 用了不同 R -> 配对失效, 必须排查。")
    L.append("")

    # Six registered blocks: five full-data comparisons and one subset comparison.
    L.append("## 6 个预先指定的比较块 × 配对 Δ")
    L.append("")
    all_results = {}
    for comp in COMPARISONS:
        name = comp["name"]
        L.append("### %s  (%s)" % (name, comp["desc"]))
        L.append("")
        # Only the dedicated 2->3 block uses the structure-change subset.
        if comp["use_structure_subset"]:
            src = table268
            srcname = "per_run_structure_subset/ (训练=90 airplane + 263 张实际结构差异 ship)"
            L.append("- **数据源: %s**" % srcname)
            L.append("- _注: 原内部代号为 268-subset，冻结清单实际为 263 张差异船和 137 张 no-op 船。_")
        else:
            src = table
            srcname = "per_run/ (全 490 张)"
            L.append("- 数据源: %s" % srcname)
        L.append("")
        comp_res = {"primary": [], "secondary": []}
        L.append("- 主判指标:")
        for metric in comp["primary"]:
            r = analyze_one_metric(src, comp["x"], comp["y"], metric, SEEDS)
            comp_res["primary"].append(r)
            L.append(fmt_metric_block(r))
        L.append("- 照报指标:")
        for metric in comp["secondary"]:
            r = analyze_one_metric(src, comp["x"], comp["y"], metric, SEEDS)
            comp_res["secondary"].append(r)
            L.append(fmt_metric_block(r))
        L.append("")
        all_results[name] = comp_res

    # 天花板 gap 轨迹
    L.append("## 天花板 gap 轨迹 (每臂 macro-F1 -> ceiling=%.3f)" % CEILING_MACRO_F1)
    L.append("")
    L.append("| arm | macro_f1 mean | std | gap_to_ceiling | n_seed |")
    L.append("|---|---|---|---|---|")
    for arm in ARMS:
        if arm in ceil_traj:
            t = ceil_traj[arm]
            L.append("| %d | %.4f | %.4f | %.4f | %d |"
                     % (arm, t["macro_f1_mean"], t["macro_f1_std"], t["gap_to_ceiling"], t["n_seed"]))
    L.append("")

    # format-probe 轨迹
    L.append("## format-probe 轨迹 (每臂合成 vs KLSG 低级可分性 AUC; DIRECTION-ONLY)")
    L.append("")
    L.append("_Arm2 and arm3 are interpreted directionally; their AUC values are not evaluated against a new pass/fail threshold._")
    L.append("")
    L.append("| arm | ship AUC | airplane AUC |")
    L.append("|---|---|---|")
    for arm in ARMS:
        if arm in fmt_traj:
            fp = fmt_traj[arm]
            sa = fp.get("ship", {}).get("auc_mean", float("nan"))
            aa = fp.get("airplane", {}).get("auc_mean", float("nan"))
            L.append("| %d | %.4f | %.4f |" % (arm, sa, aa))
        else:
            L.append("| %d | (缺 format_probe_arm%d.json) | |" % (arm, arm))
    L.append("")

    L.append("---")
    L.append("_The code reports the frozen numerical rule only. Interpretation, "
             "target-aware selection limits, and the distinction between primary "
             "and secondary metrics are documented in src/simulation_validity/README.md._")
    return "\n".join(L) + "\n", all_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_run", type=str, default=PER_RUN_DIR, help="per_run 目录 (默认全量)。")
    ap.add_argument(
        "--per_run_structure_subset", "--per_run_268",
        dest="per_run_structure_subset",
        type=str,
        default=PER_RUN_268_DIR,
        help="arm2/arm3 structure-change subset per-run directory.",
    )
    ap.add_argument(
        "--format_probe_dir",
        type=str,
        default=EVIDENCE_ROOT,
        help="Directory containing format_probe_arm{0,1,2,3}.json.",
    )
    ap.add_argument(
        "--structure_manifest",
        type=str,
        default=STRUCTURE_MANIFEST,
        help="Path to structure_subset_ship_fnames.json.",
    )
    ap.add_argument("--out", type=str, default=DEFAULT_REPORT, help="报告输出路径。")
    ap.add_argument("--loose", action="store_true",
                    help="loose 模式: 读任意 {arm}_seed{seed}.json (含 throwaway), 仅验流程跑通。")
    args = ap.parse_args()

    if args.loose:
        # 冒烟验流程: 直接读所有 json, 不做严格配对 (可能只 1 个 throwaway cell)。
        table = load_per_run_loose(args.per_run)
        print("[loose] 读到 %d 个 cell: %s" % (len(table), list(table.keys())))
        # 用 loose table 的可用 seed 演示一次配对算法 (若同一臂≥2 seed) -- 仅证明函数能跑。
        report_lines = ["# E2 配对 Δ 报告 (LOOSE/SMOKE 流程验证)", "",
                        "读到 cell: %s" % list(table.keys()),
                        "", "本模式仅验 analyze 能吃 JSON 出报告; 配对统计需完整 48 cell, 此处不产科学数字。"]
        # 演示统计函数: 对人造两点配对差跑一次 (证明 paired_t_ci/bootstrap 不崩)
        demo_d = np.array([0.01, -0.02])
        ci_t = paired_t_ci(demo_d)
        rng = np.random.RandomState(BOOT_SEED)
        ci_b = paired_bootstrap_ci(demo_d, 1000, rng)
        report_lines += ["", "## 统计函数自检 (人造配对差 [0.01,-0.02])",
                         "- 配对t CI: [%.4f, %.4f] (%s)" % (ci_t["ci_lo"], ci_t["ci_hi"], ci_t["tcrit_src"]),
                         "- bootstrap CI: [%.4f, %.4f]" % (ci_b["ci_lo"], ci_b["ci_hi"]),
                         "- 判定函数: %s" % decide(ci_t)["verdict"]]
        _atomic_write_text(args.out, "\n".join(report_lines) + "\n")
        print("[loose] 报告 ->", args.out)
        return

    expected_full_files = {"%d_seed%d.json" % (arm, seed) for arm in ARMS for seed in SEEDS}
    expected_subset_files = {
        "%d_seed%d.json" % (arm, seed) for arm in (2, 3) for seed in SEEDS
    }
    for directory, expected_files, label in (
        (args.per_run, expected_full_files, "full per-run directory"),
        (args.per_run_structure_subset, expected_subset_files, "structure-subset directory"),
    ):
        if not os.path.isdir(directory):
            raise RuntimeError("%s does not exist: %s" % (label, directory))
        actual_files = {name for name in os.listdir(directory) if name.endswith(".json")}
        if actual_files != expected_files:
            raise RuntimeError(
                "%s must contain the exact frozen JSON roster; missing=%s unexpected=%s"
                % (label, sorted(expected_files - actual_files), sorted(actual_files - expected_files))
            )

    bundled_inputs = (
        os.path.abspath(args.per_run) == os.path.abspath(PER_RUN_DIR)
        and os.path.abspath(args.per_run_structure_subset) == os.path.abspath(PER_RUN_268_DIR)
        and os.path.abspath(args.format_probe_dir) == os.path.abspath(EVIDENCE_ROOT)
        and os.path.abspath(args.structure_manifest) == os.path.abspath(STRUCTURE_MANIFEST)
    )
    if bundled_inputs:
        validate_evidence_sha256_manifest(EVIDENCE_ROOT, EVIDENCE_SHA256_MANIFEST)

    table, missing, present = load_per_run(args.per_run)
    if missing or len(table) != 48:
        raise RuntimeError(
            "Strict analysis requires 48 full cells; present=%d missing=%s"
            % (len(table), missing)
        )

    table268 = load_per_run_loose(args.per_run_structure_subset)
    expected_subset = {(arm, seed) for arm in (2, 3) for seed in SEEDS}
    missing_subset = sorted(expected_subset.difference(table268))
    unexpected_subset = sorted(set(table268).difference(expected_subset))
    if missing_subset or unexpected_subset:
        raise RuntimeError(
            "Strict analysis requires exactly 24 arm2/arm3 structure-subset cells; "
            "missing=%s unexpected=%s" % (missing_subset, unexpected_subset)
        )

    fmt_traj = load_format_probe(args.format_probe_dir)
    if set(fmt_traj) != set(ARMS):
        raise RuntimeError(
            "Strict analysis requires four format-probe JSON files; found arms=%s"
            % sorted(fmt_traj)
        )
    with open(args.structure_manifest, "r", encoding="utf-8") as handle:
        structure_manifest = json.load(handle)
    validate_strict_evidence(table, table268, fmt_traj, structure_manifest)
    ceil_traj = ceiling_gap_trajectory(table, SEEDS)
    smoke_mode = False
    try:
        display_per_run = os.path.relpath(
            os.path.abspath(args.per_run), REPO_ROOT
        ).replace("\\", "/")
        if display_per_run.startswith("../"):
            display_per_run = "<user-supplied per-run directory>"
    except ValueError:
        display_per_run = "<user-supplied per-run directory>"

    report, all_results = build_report(table, table268, missing, present,
                                       fmt_traj, ceil_traj, display_per_run, smoke_mode)
    _atomic_write_text(args.out, report)
    # 同时落 JSON (机器可读, 含全部配对差/CI/判定)
    stem, suffix = os.path.splitext(args.out)
    json_out = stem + ".json" if suffix.lower() == ".md" else args.out + ".json"
    _atomic_write_json(
        json_out,
        {"comparisons": all_results, "ceiling_trajectory": ceil_traj,
         "n_present": len(present), "n_missing": len(missing),
         "smoke_mode": smoke_mode},
    )
    print("[analyze] 报告 ->", args.out)
    print("[analyze] JSON ->", json_out)
    print("[analyze] present=%d missing=%d smoke_mode=%s" % (len(present), len(missing), smoke_mode))


if __name__ == "__main__":
    main()
