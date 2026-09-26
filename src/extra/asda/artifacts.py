"""Strict epoch/step/config/ID/class and byte-integrity gates; no permissive last-K."""
from pathlib import Path

from common import (CLASSES, EPOCHS, IDS, PROTOCOL, canonical_hash, input_hashes,
                    read_json, require, sha, source_hashes, write_json)
import numpy as np
import torch


def checkpoint_payload(model, discriminator, config, epoch, global_step):
    return {"schema": "asda_checkpoint_v1", "epoch": epoch, "global_step": global_step,
            "seed": config["seed"], "fold": config["fold"], "config_sha256": canonical_hash(config),
            "target_ids": IDS, "classes": CLASSES, "manifest_sha256": config["inputs"]["klsg_unlabeled_manifest.txt"],
            "model_state_dict": model.state_dict(), "ad_net_state_dict": discriminator.state_dict()}


def validate_checkpoint(state, config, epoch):
    require(epoch in EPOCHS, "Only fixed epochs36..40 can enter final evaluation")
    require(state.get("schema") == "asda_checkpoint_v1", "Invalid checkpoint schema")
    require(state.get("epoch") == epoch and state.get("global_step") == epoch * 6, "Wrong epoch/global_step")
    require(state.get("seed") == config["seed"] and state.get("fold") == config["fold"], "Wrong seed/fold")
    require(state.get("config_sha256") == canonical_hash(config), "Wrong checkpoint configuration")
    require(state.get("target_ids") == IDS and state.get("classes") == CLASSES, "Wrong target ID or class axis")
    require(state.get("manifest_sha256") == config["inputs"]["klsg_unlabeled_manifest.txt"], "Wrong manifest identity")
    require("model_state_dict" in state and "ad_net_state_dict" in state, "Missing model/discriminator state")
    counters = [int(value.item()) for key, value in state["model_state_dict"].items()
                if key.endswith("num_batches_tracked")]
    require(len(counters) == 20 and all(value == epoch * 6 for value in counters), "Wrong saved BN update count")


def checkpoint_paths(attempt):
    paths = [Path(attempt) / f"asda_epoch{epoch:03d}.pth" for epoch in EPOCHS]
    require(all(p.is_file() for p in paths), "All five checkpoints36..40 are required; missing checkpoint")
    require(set(Path(attempt).glob("asda_epoch*.pth")) == set(paths), "Unexpected checkpoint set")
    return paths


def validate_five_checkpoints(attempt, config, strict_load=False):
    from model import ASDAResNet18, Discriminator
    model = ASDAResNet18(pretrained=False) if strict_load else None
    discriminator = Discriminator() if strict_load else None
    for epoch, path in zip(EPOCHS, checkpoint_paths(attempt)):
        state = torch.load(path, map_location="cpu", weights_only=True)
        validate_checkpoint(state, config, epoch)
        require(all(torch.isfinite(v).all().item() for key in ("model_state_dict", "ad_net_state_dict")
                    for v in state[key].values()), "Nonfinite checkpoint tensor")
        if strict_load:
            model.load_state_dict(state["model_state_dict"], strict=True)
            discriminator.load_state_dict(state["ad_net_state_dict"], strict=True)
    return {p.name: sha(p) for p in checkpoint_paths(attempt)}


def complete_training(root, current_implementation=False):
    """Return validated complete attempt or raise. Does not silently forgive a stale marker."""
    root = Path(root)
    marker = read_json(root / "TRAINING_COMPLETE.json")
    require(marker.get("status") == "complete" and marker.get("global_step") == 240, "Invalid completion marker")
    attempt = (root / marker["attempt"]).resolve()
    require(attempt.parent == root.resolve(), "Invalid attempt path")
    config = read_json(attempt / "config.json")
    require(root.name == f"asda_s{config['seed']}_f{config['fold']}", "Directory seed/fold differs from configuration")
    require(config["protocol"] == PROTOCOL, "Protocol differs from the fixed protocol")
    require(config["target_ids"] == IDS, "Wrong configured target IDs")
    require(marker["config_sha256"] == canonical_hash(config), "Configuration changed after completion")
    require(config["inputs"] == input_hashes(), "Input manifests/splits changed")
    if current_implementation:
        require(config["implementation_sha256"] == source_hashes(), "Implementation changed; cannot skip completed run")
    require(set(marker["checkpoint_sha256"]) == {p.name for p in checkpoint_paths(attempt)}, "Wrong checkpoint marker set")
    for filename, digest in marker["artifact_sha256"].items():
        path = (attempt / filename).resolve()
        require(path.is_relative_to(attempt) and sha(path) == digest, f"Artifact integrity failure: {filename}")
    require(validate_five_checkpoints(attempt, config) == marker["checkpoint_sha256"], "Checkpoint integrity failure")
    return attempt, config, marker


def validate_prediction(ids, classes, probs):
    require(np.array_equal(np.asarray(ids), np.asarray(IDS)), "Prediction IDs differ in count/order/identity")
    require(list(classes) == CLASSES, "Prediction class axis differs")
    probs = np.asarray(probs)
    require(probs.shape == (553, 2), "Prediction shape must be553x2")
    require(np.isfinite(probs).all() and (probs >= 0).all() and (probs <= 1).all(), "Invalid probabilities")
    require(np.allclose(probs.sum(1), 1, atol=1e-6, rtol=0), "Probabilities not normalized")


def average_five_predictions(predictions):
    require(len(predictions) == 5, "Exactly five prediction sets required")
    require([p["epoch"] for p in predictions] == list(EPOCHS), "Wrong prediction epoch set/order")
    for prediction in predictions:
        validate_prediction(prediction["ids"], prediction["classes"], prediction["probs"])
    return np.stack([p["probs"].astype(np.float64) for p in predictions]).mean(0)


def write_training_complete(root, attempt, config, cost):
    checkpoints = validate_five_checkpoints(attempt, config, strict_load=True)
    write_json(attempt / "cost.json", cost)
    artifacts = {name: sha(attempt / name) for name in ("config.json", "steps.jsonl", "cost.json")}
    marker = {"status": "complete", "attempt": attempt.name, "global_step": 240,
              "config_sha256": canonical_hash(config), "checkpoint_sha256": checkpoints,
              "artifact_sha256": artifacts}
    write_json(root / "TRAINING_COMPLETE.json", marker)


def complete_evaluation(attempt, config):
    directory = Path(attempt) / "evaluation"
    marker = read_json(directory / "EVALUATION_COMPLETE.json")
    require(marker["config_sha256"] == canonical_hash(config), "Evaluation configuration changed")
    for name, digest in marker["artifact_sha256"].items():
        path = (directory / name).resolve()
        require(path.is_relative_to(directory.resolve()) and sha(path) == digest, "Evaluation byte-integrity failure")
    data = np.load(directory / "predictions.npz", allow_pickle=False)
    validate_prediction(data["ids"], data["classes"], data["probs"])
    predictions = []
    for epoch in EPOCHS:
        pred = np.load(directory / f"epoch{epoch:03d}.npz", allow_pickle=False)
        require(str(pred["config_sha256"]) == canonical_hash(config), "Wrong per-epoch prediction configuration")
        require(str(pred["checkpoint_sha256"]) == sha(Path(attempt) / f"asda_epoch{epoch:03d}.pth"),
                "Prediction came from a different checkpoint")
        predictions.append({"epoch": int(pred["epoch"]), "ids": pred["ids"],
                            "classes": pred["classes"], "probs": pred["probs"]})
    require(np.array_equal(average_five_predictions(predictions), data["probs"]), "Incorrect probability mean")
    metrics = read_json(directory / "metrics.json")
    require(metrics["seed"] == config["seed"] and metrics["fold"] == config["fold"], "Wrong evaluation seed/fold")
    require(metrics["config_sha256"] == canonical_hash(config), "Wrong metrics configuration")
    return metrics
