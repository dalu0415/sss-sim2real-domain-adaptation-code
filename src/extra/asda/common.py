"""Frozen ASDA-SEL-01/A protocol and local artifact identities."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import hashlib
import json
import platform
import random
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torchvision

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
OUT = ROOT / "results/extra/asda"
# ImageNet initialisation used by the archived runs: torchvision ResNet18_Weights.IMAGENET1K_V1.
INIT_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
INIT_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
# Versions recorded for the archived runs; other versions are reported, not refused.
RECORDED_VERSIONS = {"python": "3.11.15", "torch": "2.9.1+cu128", "torchvision": "0.24.1+cu128"}
sys.path.insert(0, str(SRC))
CLASSES = ["airplane", "ship"]
EPOCHS = tuple(range(36, 41))
IDS = list(range(553))
PROTOCOL = {
    "id": "ASDA-SEL-01/A", "classes": CLASSES,
    "backbone": "ResNet18 ImageNet1K_V1; all parameters trainable; feature_layers/fc",
    "discriminator": "1024-1024-1024-1; ReLU/dropout0.5; Xavier normal/bias0; logits",
    "batch_source": 64, "batch_target": 64, "joint_images": 320,
    "views": 3, "ra_ops": 3, "ra_magnitude_index": 2, "ra_bins": 31,
    "ra_pool": "torchvision full 14", "ra_interpolation": "NEAREST", "ra_fill": 0,
    "ra_input": "CPU uint8 round(255*clamp(float base)), ties-to-even; independent clones",
    "base": "PIL crop(.8,1)/hflip.5/rotate15/jitter(.15,.15), ToTensor, speckle.1",
    "gate": "2 * matches(original argmax, 3 augmented argmax) > 3",
    "entropy": "last augmented view, reliable +H/unreliable -H; mean over original targets",
    "domain": "p.detach outer f; source1/target0; source mean + target mean",
    "grl": 1.0, "lambda_domain": 1.0, "lambda_entropy": 1.0,
    "bn": "one joint forward; ordinary shared BN eps1e-5 momentum.1 affine/track true",
    "source_ce": "effective number beta.999 counts72/320 mean-normalized weights; weighted mean",
    "lr": [0.001, 0.01, 0.01], "momentum": 0.9, "nesterov": True, "weight_decay": 0.0005,
    "scheduler": "after optimizer; (1+10*completed_steps/240)^-.75",
    "epochs": 40, "steps_per_epoch": 6, "total_steps": 240,
    "workers": 0, "shuffle": True, "drop_last": True,
    "target_iterator": "8 full batches, persists across source epochs; renew only on exhaustion",
    "seeds": list(range(5)), "folds": list(range(5)), "saved_epochs": list(EPOCHS),
    "evaluation": "each checkpoint eval, all553 IDs, arithmetic mean of 5 softmax probabilities",
    "aggregation": "score each fold; average 5 folds per seed; pair 5 seeds; finite collapse retained",
    "amp": False, "restart": "complete run only; interrupted attempts preserved; restart from seed",
    "provenance": "P: ASDA Algorithm1 last-view interpretation; C: torchvision RA/GLS D; J: common protocol; E: explicit adaptation",
    "limits": "Algorithm1 last differs from prose consistent-version wording; index2/31 not proven author M2; joint BN not verified author default",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_json(path, obj):
    with open(path, "a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")


def init_path():
    """Download the torchvision ImageNet weights if needed and verify them against the recorded digest."""
    path = Path(torch.hub.get_dir()) / "checkpoints" / INIT_URL.rsplit("/", 1)[-1]
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(INIT_URL, str(path))
    require(sha(path) == INIT_SHA256, "ImageNet initialisation differs from the recorded weights")
    return path


def init_state_dict():
    return torch.load(init_path(), map_location="cpu", weights_only=True)


def check_environment():
    found = {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__}
    for name, version in found.items():
        if version != RECORDED_VERSIONS[name]:
            warnings.warn(f"{name} {version} differs from the recorded {RECORDED_VERSIONS[name]}; "
                          "results need not be bit-identical to the archived runs")
    require(os.environ["CUBLAS_WORKSPACE_CONFIG"] in (":4096:8", ":16:8"), "Invalid deterministic cuBLAS workspace")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def driver_version():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment():
    driver = driver_version()
    return {"python": sys.version, "torch": torch.__version__,
            "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0),
            "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "cpu": platform.processor(), "platform": platform.platform(), "driver": driver,
            "cublas_workspace": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(), "amp": False,
            "torch_threads": torch.get_num_threads()}


def source_hashes():
    files = list(Path(__file__).parent.glob("*.py"))
    files += [SRC / "_klsg_blind.py"]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in sorted(files)}


def input_hashes():
    return {name: sha(ROOT / "data/split" / name) for name in
            ("splits.csv", "splits_manifest.json", "klsg_unlabeled_manifest.txt")}


def make_config(seed, fold, source_ids):
    import torchvision.transforms.autoaugment as autoaugment
    require(seed in range(5) and fold in range(5), "seed/fold must be 0..4")
    return {"schema": "asda_config_v1", "seed": seed, "fold": fold,
            "protocol": PROTOCOL, "inputs": input_hashes(), "target_ids": IDS,
            "source_train_ids": source_ids, "implementation_sha256": source_hashes(),
            "initialization": {"weights": INIT_URL.rsplit("/", 1)[-1], "sha256": sha(init_path())},
            "ra_source_sha256": sha(autoaugment.__file__), "environment": environment()}


def run_root(seed, fold):
    require(seed in range(5) and fold in range(5), "seed/fold must be 0..4")
    return OUT / "runs" / f"asda_s{seed}_f{fold}"


def ensure_output(path):
    path = Path(path).resolve()
    require(path.is_relative_to(OUT.resolve()), "ASDA output must stay in results/extra/asda")
    return path
