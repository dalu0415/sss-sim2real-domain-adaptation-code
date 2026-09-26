"""Separate authorized scoring, strict last5 probability averaging, and 5fold/5seed summary."""
import argparse
import datetime
import math
import statistics
import time
from pathlib import Path

from common import (ROOT, SRC, OUT, CLASSES, EPOCHS, IDS, canonical_hash, check_environment,
                    ensure_output, read_json, require, run_root, sha, write_json)
import numpy as np
import torch
from torch.utils.data import DataLoader
import _klsg_blind
from data import BlindTargetDataset, SourceDataset, source_rows, eval_transform
from model import ASDAResNet18, bn_counts
from artifacts import (complete_training, complete_evaluation, validate_checkpoint,
                       validate_prediction, average_five_predictions, checkpoint_paths)
from scoring import adjudication_metrics, paired_delta, T_CRIT

METRICS = ["macro_f1", "airplane_f1", "airplane_pr_auc", "airplane_p", "airplane_r",
           "ship_f1", "ship_p", "ship_r"]


def target_label_alignment(auth, fold, audit_dir):
    """Use existing authorization gate; map returned paths to immutable manifest ID order."""
    require(isinstance(auth, _klsg_blind.KlsgLabelAuth), "Explicit label authorization required")
    audit_path = ensure_output(audit_dir) / "_LABEL_ACCESS_LOG.txt"
    old = _klsg_blind._LABEL_ACCESS_LOG_REL
    try:
        # Single-process evaluator: redirect only this helper instance's log, not old source/files.
        _klsg_blind._LABEL_ACCESS_LOG_REL = audit_path.relative_to(ROOT).parts
        labeled = _klsg_blind.read_klsg_labeled(auth, ROOT, fold, split=None)
    finally:
        _klsg_blind._LABEL_ACCESS_LOG_REL = old
    labels = {}
    for path, label in labeled:
        key = str(Path(path).resolve().relative_to(ROOT.resolve())).replace("\\", "/")
        require(key not in labels, "Duplicate labeled target path")
        labels[key] = int(label)
    blinded = _klsg_blind.read_klsg_unlabeled(ROOT)
    keys = [str(Path(path).resolve().relative_to(ROOT.resolve())).replace("\\", "/") for path, _ in blinded]
    require(set(keys) == set(labels) and len(keys) == 553, "Label/path identity mismatch")
    return np.asarray([labels[key] for key in keys], dtype=np.int64), keys


@torch.no_grad()
def collect_target(model, loader, device):
    model.eval()
    before = bn_counts(model)
    ids, probs, features = [], [], []
    for x, batch_ids in loader:
        f, z = model(x.to(device))
        ids.extend(batch_ids.tolist())
        probs.append(z.softmax(1).cpu().numpy().astype(np.float64))
        features.append(f.cpu().numpy())
    require(bn_counts(model) == before, "Evaluation updated BN buffers")
    probs = np.concatenate(probs)
    validate_prediction(ids, CLASSES, probs)
    return np.asarray(ids), probs, np.concatenate(features)


@torch.no_grad()
def collect_source(model, loader, device):
    model.eval()
    ids, ys, probs, features = [], [], [], []
    for x, y, batch_ids in loader:
        f, z = model(x.to(device))
        ids.extend(batch_ids)
        ys.extend(y.tolist())
        probs.append(z.softmax(1).cpu().numpy().astype(np.float64))
        features.append(f.cpu().numpy())
    return ids, np.asarray(ys), np.concatenate(probs), np.concatenate(features)


def evaluate(seed, fold, authorize=False, reason=None, restart_incomplete=False):
    check_environment()
    auth = _klsg_blind.klsg_label_auth_from_cli(authorize, reason, "ASDA.evaluate")
    require(auth is not None, "Use --authorize-klsg-labels and --reason for final scoring")
    attempt, config, _ = complete_training(run_root(seed, fold))
    directory = attempt / "evaluation"
    if (directory / "EVALUATION_COMPLETE.json").exists():
        complete_evaluation(attempt, config)
        print(f"Already evaluated and validated: {directory}")
        return
    if directory.exists():
        require(restart_incomplete, "Incomplete evaluation retained; --restart-incomplete needed")
        archive = attempt / ("evaluation_interrupted_" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S%f"))
        require(directory.resolve().parent == attempt.resolve() and archive.resolve().parent == attempt.resolve(), "Invalid archive path")
        directory.rename(archive)
    directory.mkdir()
    start = time.perf_counter()
    target_loader = DataLoader(BlindTargetDataset(training=False), batch_size=64, shuffle=False, num_workers=0)
    source_loader = DataLoader(SourceDataset(source_rows(fold, "test"), eval_transform()), batch_size=64,
                               shuffle=False, num_workers=0)
    model = ASDAResNet18(pretrained=False).cuda()
    predictions, source_probabilities, timing = [], [], []
    source_ids = source_y = None
    for epoch, path in zip(EPOCHS, checkpoint_paths(attempt)):
        load_start = time.perf_counter()
        state = torch.load(path, map_location="cpu", weights_only=True)
        validate_checkpoint(state, config, epoch)
        model.load_state_dict(state["model_state_dict"], strict=True)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_start
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        ids, probabilities, target_features = collect_target(model, target_loader, "cuda")
        torch.cuda.synchronize()
        target_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        source_ids_now, source_y_now, sp, source_features = collect_source(model, source_loader, "cuda")
        torch.cuda.synchronize()
        source_seconds = time.perf_counter() - t0
        if source_ids is None:
            source_ids, source_y = source_ids_now, source_y_now
        else:
            require(source_ids_now == source_ids and np.array_equal(source_y_now, source_y), "Source IDs changed between checkpoints")
        item = {"epoch": epoch, "ids": ids, "classes": CLASSES, "probs": probabilities}
        predictions.append(item)
        source_probabilities.append(sp)
        np.savez(directory / f"epoch{epoch:03d}.npz", **item,
                 config_sha256=canonical_hash(config), checkpoint_sha256=sha(path))
        timing.append({"epoch": epoch, "checkpoint_load_seconds": load_seconds,
                       "target553_seconds_with_transfer_and_loader": target_seconds,
                       "source98_seconds_with_transfer_and_loader": source_seconds,
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
    mean = average_five_predictions(predictions)
    np.savez(directory / "predictions.npz", ids=IDS, classes=CLASSES, probs=mean)
    # This is the first access to target truth in this process; no labels reached inference.
    target_y, target_paths = target_label_alignment(auth, fold, directory)
    metrics = {"seed": seed, "fold": fold, "config_sha256": canonical_hash(config),
               "adjudication": adjudication_metrics(target_y, mean.argmax(1), mean),
               "identity": "Supplementary ASDA-SEL-01/A; all553 transductive target; fixed last5"}
    write_json(directory / "metrics.json", metrics)
    bundle = directory / "bundle"
    bundle.mkdir()
    for name, value in {"y_true": target_y, "y_pred": mean.argmax(1), "probs": mean,
                        "feats_target": target_features, "feats_source": source_features,
                        "src_y_true": source_y, "src_probs": np.stack(source_probabilities).mean(0),
                        "target_ids": np.asarray(IDS)}.items():
        np.save(bundle / f"{name}.npy", value)
    (bundle / "tgt_paths.txt").write_text("\n".join(target_paths) + "\n", encoding="utf-8")
    write_json(bundle / "meta.json", {"arch": "cdan", "classes": CLASSES, "seed": seed, "fold": fold,
               "target_ids": IDS, "source_ids": source_ids, "last_k": 5, "epochs": list(EPOCHS),
               "manifest_sha256": config["inputs"]["klsg_unlabeled_manifest.txt"],
               "config_sha256": canonical_hash(config), "features": "last epoch40; probabilities last5 average",
               "authorization": {"reason": reason,
                                 "audit": (directory / "_LABEL_ACCESS_LOG.txt").relative_to(ROOT).as_posix()}})
    write_json(directory / "evaluation_cost.json", {"checkpoints": timing,
               "target_G_C_image_forwards": 553 * 5, "source_diagnostic_G_C_image_forwards": 98 * 5,
               "total_seconds_including_loading_scoring_saving": time.perf_counter() - start,
               "scope": "single-checkpoint target times listed; last5 sums all5; D/RA/AdaBN absent"})
    artifacts = {str(path.relative_to(directory)).replace("\\", "/"): sha(path)
                 for path in directory.rglob("*") if path.is_file()}
    write_json(directory / "EVALUATION_COMPLETE.json", {"config_sha256": canonical_hash(config),
               "artifact_sha256": artifacts})
    complete_evaluation(attempt, config)
    print(f"Validated evaluation: {directory}", flush=True)


def metric_values(adjudication):
    a, s = adjudication["per_class"]["airplane"], adjudication["per_class"]["ship"]
    values = {"macro_f1": adjudication["macro_f1"], "airplane_f1": a["f1"],
              "airplane_pr_auc": adjudication["airplane_pr_auc"], "airplane_p": a["precision"],
              "airplane_r": a["recall"], "ship_f1": s["f1"], "ship_p": s["precision"], "ship_r": s["recall"]}
    require(all(isinstance(v, (int, float)) and math.isfinite(v) for v in values.values()), "Missing/nonfinite metric")
    return values


def seed_summary(records):
    keys = [(r["seed"], r["fold"]) for r in records]
    require(len(keys) == 25 and len(set(keys)) == 25 and set(keys) == {(s, f) for s in range(5) for f in range(5)},
            "Exactly all25 unique seed/fold records are required")
    points = []
    for seed in range(5):
        rows = [metric_values(r["adjudication"]) for r in records if r["seed"] == seed]
        points.append({"seed": seed, "n_folds": 5, "values": {k: statistics.fmean(r[k] for r in rows) for k in METRICS}})
    intervals = {}
    for metric in METRICS:
        values = [p["values"][metric] for p in points]
        mean, sd = statistics.fmean(values), statistics.stdev(values)
        intervals[metric] = {"mean": mean, "sd": sd, "ci_lo": mean - T_CRIT * sd / math.sqrt(5),
                             "ci_hi": mean + T_CRIT * sd / math.sqrt(5), "n": 5, "df": 4, "t_crit": T_CRIT}
    return {"arm_id": "ASDA", "scope": "all", "per_seed_points": points, "across_seed": intervals}


def aggregate():
    records, identities = [], []
    implementation_identity = None
    for seed in range(5):
        for fold in range(5):
            attempt, config, marker = complete_training(run_root(seed, fold))
            identity = {key: config[key] for key in ("implementation_sha256", "initialization", "ra_source_sha256")}
            if implementation_identity is None:
                implementation_identity = identity
            require(identity == implementation_identity, "Cannot aggregate runs produced by different implementations/initializers")
            records.append(complete_evaluation(attempt, config))
            identities.append({"seed": seed, "fold": fold, "attempt": attempt.relative_to(ROOT).as_posix(),
                               "config_sha256": marker["config_sha256"]})
    candidate = seed_summary(records)
    # Seed-level values of #1 source-only and #2 source-only+AdaBN, the two references of the paired ASDA
    # contrasts, copied from the frozen BN-focused aggregate; provenance in the file itself.
    reference_path = ROOT / "evidence/asda/reference_arms.json"
    reference = read_json(reference_path)
    comparisons = {}
    for arm_id in ("#1", "#2"):
        arms = [arm for arm in reference["arms"] if arm["arm_id"] == arm_id and arm["scope"] == "all"]
        require(len(arms) == 1, "Missing/duplicate reference arm")
        points = arms[0]["per_seed_points"]
        require(len(points) == 5 and {p["seed"] for p in points} == set(range(5))
                and all(p["n_folds"] == 5 for p in points), "Reference arm incomplete")
        comparisons[arm_id] = [paired_delta(candidate, arms[0], metric) for metric in METRICS[:5]]
    output = OUT / "summary"
    output.mkdir(exist_ok=False)  # Do not overwrite a previously reviewed result.
    write_json(output / "summary.json", {"candidate": candidate, "runs": identities,
               "paired_deltas_ASDA_minus_reference": comparisons, "reference_sha256": sha(reference_path),
               "inference_unit": "5 seed means, each mean contains5 folds; not25 independent samples",
               "status": "Supplementary exploratory comparisons; no inherited confirmatory Holm family/p-values"})
    print(f"Complete25-run summary: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--seed", type=int, choices=range(5))
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--authorize-klsg-labels", action="store_true")
    parser.add_argument("--reason")
    parser.add_argument("--restart-incomplete", action="store_true")
    args = parser.parse_args()
    check_environment()
    if args.aggregate:
        aggregate()
    else:
        require(args.seed is not None and args.fold is not None, "seed/fold required")
        evaluate(args.seed, args.fold, args.authorize_klsg_labels, args.reason, args.restart_incomplete)


if __name__ == "__main__":
    main()
