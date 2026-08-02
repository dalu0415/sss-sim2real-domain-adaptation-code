# -*- coding: utf-8 -*-
"""Run the frozen four-arm simulation-validity experiment.

The protocol trains a source-only ImageNet-pretrained ResNet-18 for each of
four semi-synthetic source variants and 12 matched random seeds, then scores
every run on the same 553-image KLSG-II target set. Training uses Adam
(lr=1e-4, weight decay=1e-4), class-balanced cross-entropy, 40 fixed epochs,
and the mean softmax probability from the final five checkpoints. Target
labels are used only for the frozen final evaluation, never for validation,
early stopping, checkpoint selection, or hyperparameter tuning.

The same file also implements the low-level format probe: 32 normalized
histogram bins plus an 8x8 grayscale thumbnail, evaluated separately by class
with 20 repeated balanced resamples and five-fold stratified CV per
resample. This yields 100 correlated fold-level AUC values; it is not a
single 100-fold partition.

Before running, replace every PATH_TO_* value in the user configuration block
below. See README.md in this directory for the exact dataset roles, evidence
scope, and limitations.
"""
import os
import time
import json
import random
import hashlib
import argparse
import platform

import numpy as np

# ---------------------------------------------------------------------------
# Deterministic import state; each training seed resets all three RNG families.
# ---------------------------------------------------------------------------
_IMPORT_SEED = 42
random.seed(_IMPORT_SEED)
np.random.seed(_IMPORT_SEED)

import torch  # noqa: E402
torch.manual_seed(_IMPORT_SEED)
torch.cuda.manual_seed_all(_IMPORT_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchvision  # noqa: E402
import cv2  # noqa: E402

# ---------------------------------------------------------------------------
# User configuration. Replace every PATH_TO_* placeholder before running.
# Each dataset root must contain airplane/ and ship/ subdirectories.
# ---------------------------------------------------------------------------
OUT_ROOT = r"PATH_TO_SIMULATION_VALIDITY_OUTPUT"
PER_RUN_DIR = os.path.join(OUT_ROOT, "per_run")
LOGS_DIR = os.path.join(OUT_ROOT, "logs")
SMOKE_DIR = os.path.join(OUT_ROOT, "_smoke_throwaway")

ARM_DIRS = {
    0: r"PATH_TO_ARM0_NATIVE_DATASET",
    1: r"PATH_TO_ARM1_LOW_LEVEL_REALISM_DATASET",
    2: r"PATH_TO_ARM2_SELECTED_DATASET",
    3: r"PATH_TO_ARM3_INTERNAL_STRUCTURE_DATASET",
}
ARM_DATASET_IDS = {
    0: "arm0_native_baseline",
    1: "arm1_low_level_realism",
    2: "arm2_selected_source",
    3: "arm3_internal_structure",
}
# Canonical grayscale 224x224 KLSG-II evaluation set used by all 48 cells.
R_DIR = r"PATH_TO_CANONICAL_KLSG_EVALUATION_ROOT"
R_DATASET_ID = "KLSG-II/canonical_evaluation"
# Optional guard against accidentally using the pre-resized arm0 copy.
FORBIDDEN_ARM0 = r"PATH_TO_PRE_RESIZED_ARM0_COPY"

# Frozen label mapping: airplane=0, ship=1.
CLASSES = ["airplane", "ship"]
AIRCRAFT_LABEL = 0

# 上游契约 (量级断言): 4 臂各类期望张数 + R 各类期望张数。
EXPECTED_ARM = {"airplane": 90, "ship": 400}
EXPECTED_R = {"airplane": 66, "ship": 487}

# Frozen ImageNet normalization constants.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TARGET_SIZE = 224

# Frozen training hyperparameters.
LR = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
CKPT_AVG_K = 5
EVAL_BATCH = 64

# The same 12 pre-specified seeds are used for all four arms.
SEEDS_FULL = list(range(12))   # [0,1,...,11]
ARMS_FULL = [0, 1, 2, 3]

# Rounded KLSG-II real-to-real five-fold OOF macro-F1 reference.
CEILING_MACRO_F1 = 0.890

# 运行参数 (smoke 覆盖)
EPOCHS = 40
SUBSET_N = None     # 每类截张数 (None=全量); smoke=12
_SMOKE = False
_t_start = time.time()

# 当前 cell 的日志文件句柄 (per-cell 重定向 train log)
_cur_logf = None
PROTOCOL_SCHEMA_VERSION = 1


def _clog(msg):
    """Write the current cell's diagnostic log and echo it to stdout."""
    line = "%s | elapsed=%.1fs | %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), time.time() - _t_start, msg)
    print("[LOG]", line)
    if _cur_logf is not None:
        _cur_logf.write(line + "\n")
        _cur_logf.flush()


def _atomic_write_text(path, text):
    """Write a complete UTF-8 file, then atomically replace the destination."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    temporary = "%s.tmp-%d" % (path, os.getpid())
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(temporary, path)


def _atomic_write_json(path, payload):
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _canonical_json_sha256(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def compute_protocol_signature():
    """Fingerprint the frozen settings that determine one training cell."""
    payload = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "model": "torchvision.resnet18/IMAGENET1K_V1/2-class-fc",
        "optimizer": "Adam",
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH,
        "epochs": EPOCHS,
        "checkpoint_average_k": min(CKPT_AVG_K, EPOCHS),
        "loss": "inverse-frequency-class-weighted-cross-entropy",
        "prediction": "mean-softmax-then-argmax",
        "target_size": TARGET_SIZE,
        "resize_interpolation": "cv2.INTER_LINEAR",
        "input_scaling": "uint8/255",
        "channel_mapping": "grayscale-repeat-3",
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
        "label_map": {"airplane": 0, "ship": 1},
        "deterministic_cudnn": True,
    }
    return _canonical_json_sha256(payload)


def training_software_environment(device):
    """Return the runtime identity that must stay fixed across cached cells."""
    is_cuda = device.type == "cuda" and torch.cuda.is_available()
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_type": device.type,
        "cuda_device_name": torch.cuda.get_device_name(device) if is_cuda else None,
    }


def format_software_environment():
    """Return the runtime identity for the low-level format probe."""
    import sklearn

    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "scikit_learn_version": sklearn.__version__,
    }


# ===========================================================================
# Reset Python, NumPy, and PyTorch RNGs for each training seed.
# ===========================================================================
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _list_png(d):
    return sorted([f for f in os.listdir(d) if f.lower().endswith(".png")])


def _md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_image_tree_signature(root):
    """SHA-256 over class-relative PNG names and bytes."""
    h = hashlib.sha256()
    count = 0
    for cls in CLASSES:
        class_dir = os.path.join(root, cls)
        for name in _list_png(class_dir):
            rel = ("%s/%s" % (cls, name)).encode("utf-8")
            h.update(len(rel).to_bytes(4, "big"))
            h.update(rel)
            image_path = os.path.join(class_dir, name)
            h.update(os.path.getsize(image_path).to_bytes(8, "big"))
            with open(image_path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    h.update(chunk)
            count += 1
    return h.hexdigest(), count


def compute_name_list_signature(names):
    """SHA-256 over a sorted filename set used to define a source subset."""
    h = hashlib.sha256()
    for name in sorted(names):
        encoded = name.encode("utf-8")
        h.update(len(encoded).to_bytes(4, "big"))
        h.update(encoded)
    return h.hexdigest()


def _require_configured_path(label, path):
    if not path or str(path).startswith("PATH_TO_"):
        raise RuntimeError(
            "%s is not configured. Replace its PATH_TO_* placeholder first." % label
        )


# ===========================================================================
# Frozen preprocessing: gray uint8 -> resize224 INTER_LINEAR
#   -> /255 (在 Normalize 之前) -> 灰度复制 3 通道 -> ImageNet 逐通道 Normalize -> CHW。
# ===========================================================================
def preprocess_one(arr_gray_uint8):
    rs = cv2.resize(arr_gray_uint8, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
    x = rs.astype(np.float32) / 255.0
    x3 = np.repeat(x[:, :, None], 3, axis=2)
    x3 = (x3 - IMAGENET_MEAN[None, None, :]) / IMAGENET_STD[None, None, :]
    return np.transpose(x3, (2, 0, 1)).astype(np.float32)


def load_image_set(spec_list):
    """spec_list: [(abs_dir, fname, label), ...] -> X[N,3,224,224] f32, y[N] i64 (airplane=0/ship=1)。"""
    n = len(spec_list)
    X = np.zeros((n, 3, TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
    y = np.zeros((n,), dtype=np.int64)
    for i, (d, fname, label) in enumerate(spec_list):
        arr = cv2.imread(os.path.join(d, fname), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise RuntimeError("无法读取图: %s/%s" % (d, fname))   # 不静默吞错
        X[i] = preprocess_one(arr)
        y[i] = label
    return X, y


def build_arm_spec(arm, ship_whitelist=None):
    """构造某臂训练集 spec [(dir,fname,label),...], airplane:0/ship:1。smoke 下每类截 SUBSET_N。
    ship_whitelist: 若给 (set of ship fname), ship 类只保留白名单内的图 (268-subset 用; airplane 不动)。
    同时返回每张图的 (cls, fname) 顺序键。"""
    root = ARM_DIRS[arm]
    spec = []
    keys = []   # (cls, fname) 与 spec 行一一对应
    for label, cls in enumerate(CLASSES):
        d = os.path.join(root, cls)
        files = _list_png(d)
        if cls == "ship" and ship_whitelist is not None:
            files = [f for f in files if f in ship_whitelist]
        if SUBSET_N is not None:
            files = files[:SUBSET_N]
        for fn in files:
            spec.append((d, fn, label))
            keys.append((cls, fn))
    return spec, keys


def build_R_spec():
    """测试域 R = canonical R 全部 (两类)。smoke 下也截 SUBSET_N (仅冒烟用)。"""
    spec = []
    for label, cls in enumerate(CLASSES):
        d = os.path.join(R_DIR, cls)
        files = _list_png(d)
        if SUBSET_N is not None:
            files = files[:SUBSET_N]
        for fn in files:
            spec.append((d, fn, label))
    return spec


# ===========================================================================
# Input contract and native-arm guard:
#   (a) arm0 源 = 原生 manmade, 绝不是 canonical/S 预 resize 副本。
#   (b) 4 臂 + R 目录齐、张数符 (smoke 跳量级断言)。
#   (c) 4 臂共用同一 R 张量 -> 用 R 文件 md5 集合做"逐字节同"断言 (4 臂跑同一份 R, 这里断言 R 源唯一且稳定)。
# ===========================================================================
def assert_arm0_is_native():
    a0 = os.path.normcase(os.path.normpath(ARM_DIRS[0]))
    if FORBIDDEN_ARM0 and not FORBIDDEN_ARM0.startswith("PATH_TO_"):
        forbidden = os.path.normcase(os.path.normpath(FORBIDDEN_ARM0))
        if a0 == forbidden:
            raise RuntimeError(
                "arm0 points to the forbidden pre-resized copy (%s); use the "
                "native variable-resolution arm0 dataset." % FORBIDDEN_ARM0
            )
    # 原生 manmade 应为变尺寸 (非全 224); 抽样确认至少一张非 224 (canonical/S 全是 224)。
    d = os.path.join(ARM_DIRS[0], "ship")
    files = _list_png(d)[:10]
    sizes = set()
    for f in files:
        arr = cv2.imread(os.path.join(d, f), cv2.IMREAD_GRAYSCALE)
        if arr is not None:
            sizes.add(arr.shape[:2])
    non224 = [s for s in sizes if s != (TARGET_SIZE, TARGET_SIZE)]
    if not non224:
        raise RuntimeError(
            "arm0 ship sample is entirely 224x224; this may be the pre-resized "
            "copy rather than the native variable-resolution source; sizes=%s" % sizes
        )
    print("[contract] arm0 contains native non-224 samples: %s" % list(sizes)[:5])


def check_upstream_contract(arms_to_check):
    from PIL import Image  # 仅用于格式抽检
    # 4 臂 (要跑的) 训练集
    for arm in arms_to_check:
        root = ARM_DIRS[arm]
        for cls in CLASSES:
            d = os.path.join(root, cls)
            if not os.path.isdir(d):
                raise RuntimeError("上游契约失败: 缺臂%d 子目录 %s" % (arm, d))
            n = len([f for f in os.listdir(d) if f.lower().endswith(".png")])
            if not _SMOKE:
                exp = EXPECTED_ARM[cls]
                if n != exp:
                    raise RuntimeError(
                        "上游契约失败: arm%d %s 张数=%d, 期望 %d"
                        % (arm, cls, n, exp)
                    )
    # R 测试域
    for cls in CLASSES:
        d = os.path.join(R_DIR, cls)
        if not os.path.isdir(d):
            raise RuntimeError("上游契约失败: 缺 R 子目录 %s" % d)
        files = _list_png(d)
        n = len(files)
        if not _SMOKE:
            if n != EXPECTED_R[cls]:
                raise RuntimeError(
                    "上游契约失败: R %s 张数=%d, 期望 %d"
                    % (cls, n, EXPECTED_R[cls])
                )
        if not files:
            raise RuntimeError("上游契约失败: R %s 没有 PNG 文件" % cls)
        # 格式抽检: R 应 L / 224 / PNG (canonical)
        for i in sorted(set([0, n // 2, n - 1])):
            with Image.open(os.path.join(d, files[i])) as im:
                if im.mode != "L":
                    raise RuntimeError(
                        "上游契约失败: R %s mode=%s != L" % (files[i], im.mode)
                    )
                if im.size != (TARGET_SIZE, TARGET_SIZE):
                    raise RuntimeError("上游契约失败: R %s size!=224" % files[i])
    print("[upstream] 要跑的臂 %s + R 目录齐、张数符 (smoke=%s)。" % (arms_to_check, _SMOKE))


def compute_R_signature():
    """Target image-byte fingerprint over sorted class/name/file-MD5 entries.

    Every cell records this value so strict analysis can verify that all arms
    used the same target input files. It is not a post-preprocessing tensor
    hash.
    """
    h = hashlib.md5()
    parts = []
    for label, cls in enumerate(CLASSES):
        d = os.path.join(R_DIR, cls)
        for fn in _list_png(d):
            m = _md5_file(os.path.join(d, fn))
            parts.append("%s/%s:%s" % (cls, fn, m))
    h.update("|".join(parts).encode("utf-8"))
    return h.hexdigest(), len(parts)


# ===========================================================================
# Structure-change subset used only for the arm2-to-arm3 comparison.
# It is detected from byte differences rather than a hard-coded count. The
# frozen experiment contains 263 changed and 137 unchanged ship images.
# ===========================================================================
def assert_cross_arm_control_contract():
    """Fail closed on the filename and byte-equality control variables."""
    airplane_sets = {
        arm: set(_list_png(os.path.join(ARM_DIRS[arm], "airplane")))
        for arm in (1, 2, 3)
    }
    reference_airplanes = airplane_sets[1]
    for arm in (2, 3):
        if airplane_sets[arm] != reference_airplanes:
            raise RuntimeError(
                "arm1/arm%d airplane filename sets differ; missing=%s unexpected=%s"
                % (
                    arm,
                    sorted(reference_airplanes - airplane_sets[arm]),
                    sorted(airplane_sets[arm] - reference_airplanes),
                )
            )
    for name in sorted(reference_airplanes):
        digests = {
            _sha256_file(os.path.join(ARM_DIRS[arm], "airplane", name))
            for arm in (1, 2, 3)
        }
        if len(digests) != 1:
            raise RuntimeError(
                "arm1/arm2/arm3 airplane control differs byte-for-byte: %s" % name
            )

    arm2_ships = set(_list_png(os.path.join(ARM_DIRS[2], "ship")))
    arm3_ships = set(_list_png(os.path.join(ARM_DIRS[3], "ship")))
    if arm2_ships != arm3_ships:
        raise RuntimeError(
            "arm2/arm3 ship filename sets differ; missing_in_arm2=%s missing_in_arm3=%s"
            % (sorted(arm3_ships - arm2_ships), sorted(arm2_ships - arm3_ships))
        )


def compute_structure_subset_ship_fnames():
    a2 = os.path.join(ARM_DIRS[2], "ship")
    a3 = os.path.join(ARM_DIRS[3], "ship")
    fs2 = _list_png(a2)
    fs3 = _list_png(a3)
    if fs2 != fs3:
        raise RuntimeError("arm2/arm3 ship filename rosters must match exactly")
    diff = []
    same = []
    for fn in fs3:
        p2 = os.path.join(a2, fn)
        p3 = os.path.join(a3, fn)
        if _sha256_file(p2) != _sha256_file(p3):
            diff.append(fn)
        else:
            same.append(fn)
    return diff, same


# ===========================================================================
# Frozen network, training, prediction, and metric implementation.
# ===========================================================================
def build_classifier():
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, 2)
    return model


def reinit_fc(model, seed):
    model.fc = nn.Linear(512, 2)
    return model


def class_weights(y_train):
    n_total = len(y_train)
    w = np.zeros((len(CLASSES),), dtype=np.float32)
    for c in range(len(CLASSES)):
        n_c = int(np.sum(y_train == c))
        if n_c == 0:
            raise RuntimeError("class_weights: 训练集类 %d 0 张" % c)
        w[c] = n_total / (2.0 * n_c)
    return w


def train_model(X_train, y_train, device, seed, keep_last_k=CKPT_AVG_K):
    set_all_seeds(seed)
    model = build_classifier()
    reinit_fc(model, seed)
    model.to(device)
    model.train()

    w = class_weights(y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(w).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    n = X_train.shape[0]
    Xt = torch.from_numpy(X_train)
    yt = torch.from_numpy(y_train)

    ckpts = []
    epoch_losses = []
    nan_seen = False
    rng = np.random.RandomState(seed)
    for epoch in range(EPOCHS):
        perm = rng.permutation(n)
        model.train()
        epoch_loss_sum = 0.0
        epoch_count = 0
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            xb = Xt[idx].to(device)
            yb = yt[idx].to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            if torch.isnan(loss).any():
                nan_seen = True
            loss.backward()
            optimizer.step()
            epoch_loss_sum += float(loss.detach().cpu()) * len(idx)
            epoch_count += len(idx)
        mean_loss = epoch_loss_sum / max(1, epoch_count)
        epoch_losses.append(float(mean_loss))
        ckpts.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    k = min(keep_last_k, EPOCHS)
    last_k = ckpts[-k:]
    train_info = {
        "first_loss": float(epoch_losses[0]),
        "last_loss": float(epoch_losses[-1]),
        "epoch_losses": epoch_losses,
        "nan_seen": bool(nan_seen),
        "k_used": int(k),
        "epochs": int(EPOCHS),
    }
    return model, last_k, train_info


def _forward_proba(model, X, device):
    model.eval()
    n = X.shape[0]
    probs = np.zeros((n, 2), dtype=np.float64)
    Xt = torch.from_numpy(X)
    with torch.no_grad():
        for start in range(0, n, EVAL_BATCH):
            xb = Xt[start:start + EVAL_BATCH].to(device)
            logits = model(xb)
            p = F.softmax(logits, dim=1)
            probs[start:start + p.shape[0]] = p.detach().cpu().numpy().astype(np.float64)
    return probs


def predict_proba_ckptavg(last_k_states, X_test, device):
    model = build_classifier()
    model.to(device)
    acc = None
    for state in last_k_states:
        model.load_state_dict(state)
        p = _forward_proba(model, X_test, device)
        acc = p if acc is None else (acc + p)
    return acc / float(len(last_k_states))


def confusion_matrix_2(y_true, y_pred):
    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def metrics_from_proba(proba, y_true):
    """per-class P/R/F1 + macro-F1 + balanced acc + aircraft recall + accuracy + 混淆矩阵。
    label airplane=0 / ship=1。"""
    y_pred = np.argmax(proba, axis=1).astype(np.int64)
    cm = confusion_matrix_2(y_true, y_pred)
    out = {}
    recalls = []
    for c in range(2):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        out["precision_%s" % CLASSES[c]] = float(prec)
        out["recall_%s" % CLASSES[c]] = float(rec)
        out["f1_%s" % CLASSES[c]] = float(f1)
        recalls.append(rec)
    out["macro_f1"] = float(np.mean([out["f1_%s" % CLASSES[c]] for c in range(2)]))
    out["balanced_accuracy"] = float(np.mean(recalls))
    out["aircraft_recall"] = float(out["recall_%s" % CLASSES[AIRCRAFT_LABEL]])
    out["accuracy"] = float(np.trace(cm) / cm.sum()) if cm.sum() > 0 else 0.0
    out["_confmat"] = cm.tolist()
    return out


# 标量指标键 (per_run JSON 落盘 + analyze 配对 Δ 用; 不含 _confmat)
SCALAR_METRIC_KEYS = [
    "precision_airplane", "recall_airplane", "f1_airplane",
    "precision_ship", "recall_ship", "f1_ship",
    "macro_f1", "balanced_accuracy", "aircraft_recall", "accuracy",
]


# ===========================================================================
# format-probe (每臂级、与 cell 无关): 该臂合成 490 vs 真实 KLSG 的低级统计可分性 AUC。
#   Final protocol: 32-bin histogram + 8x8 thumbnail = 96D; one probe per
#   object class; balanced downsampling; 20 resamples x 5-fold CV; liblinear
#   logistic regression with C=1.0 and seed 42. Later arms are interpreted
#   directionally rather than through a new pass/fail threshold.
# ===========================================================================
KLSG_RAW = r"PATH_TO_RAW_KLSG_ROOT"
KLSG_RAW_DATASET_ID = "KLSG-II/raw_format_probe"
FMT_SEED = 42
FMT_R_RESAMPLE = 20


def _fmt_lowlevel_feature(arr_gray_uint8):
    rs = cv2.resize(arr_gray_uint8, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
    hist, _ = np.histogram(rs.ravel(), bins=32, range=(0, 256))
    hist = hist.astype(np.float64)
    s = hist.sum()
    if s > 0:
        hist = hist / s
    thumb = cv2.resize(rs, (8, 8), interpolation=cv2.INTER_LINEAR).astype(np.float64).ravel() / 255.0
    return np.concatenate([hist, thumb])


def _fmt_load_feats(root, cls):
    d = os.path.join(root, cls)
    if not os.path.isdir(d):
        return np.zeros((0, 96))
    fs = _list_png(d)
    X = []
    for f in fs:
        arr = cv2.imread(os.path.join(d, f), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            continue
        X.append(_fmt_lowlevel_feature(arr))
    return np.array(X) if X else np.zeros((0, 96))


def _fmt_probe(Xs, Xr):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(FMT_SEED)
    n_min = min(len(Xs), len(Xr))
    aucs = []
    if n_min < 2:
        return {
            "auc_mean": float("nan"),
            "auc_std": float("nan"),
            "n_synthetic": int(len(Xs)),
            "n_real": int(len(Xr)),
            "n_balanced_per_domain": int(n_min),
            "n_resamples": FMT_R_RESAMPLE,
            "folds_per_resample": 0,
            "n_auc_values": 0,
            "auc_std_note": "No AUC values were produced.",
        }
    for _ in range(FMT_R_RESAMPLE):
        si = rng.choice(len(Xs), n_min, replace=False)
        ri = rng.choice(len(Xr), n_min, replace=False)
        X = np.vstack([Xs[si], Xr[ri]])
        y = np.array([0] * n_min + [1] * n_min)
        ns = min(5, n_min)
        if ns < 2:
            continue
        skf = StratifiedKFold(n_splits=ns, shuffle=True, random_state=FMT_SEED)
        for tr, te in skf.split(X, y):
            clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000)
            clf.fit(X[tr], y[tr])
            if len(np.unique(y[te])) < 2:
                continue
            aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    a = np.array(aucs)
    return {
        "auc_mean": float(a.mean()),
        "auc_std": float(a.std()),
        "n_synthetic": int(len(Xs)),
        "n_real": int(len(Xr)),
        "n_balanced_per_domain": int(n_min),
        "n_resamples": FMT_R_RESAMPLE,
        "folds_per_resample": min(5, n_min),
        "n_auc_values": int(len(a)),
        "auc_std_note": "Descriptive SD across correlated fold-level AUC values; not a standard error.",
    }


def compute_format_probe_for_arm(arm, arm_signature=None):
    """Compute one arm's per-class format probe and overwrite its JSON output.

    The original internal runner reused an existing JSON without checking
    whether its input images had changed. The public snapshot recomputes the
    probe so a stale cache cannot be mistaken for fresh evidence.
    """
    cache = os.path.join(OUT_ROOT, "format_probe_arm%d.json" % arm)
    software_environment = format_software_environment()
    out = {"arm": arm, "arm_dir": ARM_DATASET_IDS[arm],
           "klsg_dir": KLSG_RAW_DATASET_ID,
           "software_environment": software_environment,
           "software_environment_signature_sha256": _canonical_json_sha256(
               software_environment
           ),
           "runner_source_sha256": _sha256_file(os.path.abspath(__file__)),
           "_note": (
               "20 repeated balanced resamples x 5-fold stratified CV = "
               "100 correlated fold-level AUC values per class. Direction-only "
               "for the later shape/structure arms."
           )}
    if not os.path.isdir(KLSG_RAW):
        raise RuntimeError("Configured KLSG_RAW directory does not exist.")
    for cls in CLASSES:
        real_class_dir = os.path.join(KLSG_RAW, cls)
        n_real = len(_list_png(real_class_dir)) if os.path.isdir(real_class_dir) else 0
        if n_real != EXPECTED_R[cls]:
            raise RuntimeError(
                "KLSG_RAW/%s has %d PNGs; expected %d"
                % (cls, n_real, EXPECTED_R[cls])
            )

    if arm_signature is None:
        arm_signature, arm_count = compute_image_tree_signature(ARM_DIRS[arm])
        expected_arm_count = sum(EXPECTED_ARM.values())
        if arm_count != expected_arm_count:
            raise RuntimeError(
                "arm%d fingerprint covered %d PNGs; expected %d"
                % (arm, arm_count, expected_arm_count)
            )
    klsg_signature, klsg_count = compute_image_tree_signature(KLSG_RAW)
    expected_klsg_count = sum(EXPECTED_R.values())
    if klsg_count != expected_klsg_count:
        raise RuntimeError(
            "KLSG_RAW fingerprint covered %d PNGs; expected %d"
            % (klsg_count, expected_klsg_count)
        )
    out["arm_input_signature_sha256"] = arm_signature
    out["klsg_input_signature_sha256"] = klsg_signature

    for cls in CLASSES:
        Xs = _fmt_load_feats(ARM_DIRS[arm], cls)
        Xr = _fmt_load_feats(KLSG_RAW, cls)
        out[cls] = _fmt_probe(Xs, Xr)
    _atomic_write_json(cache, out)
    return out


# ===========================================================================
# 收敛 sanity (per_run 落盘): loss 末<首、无 NaN、末段 plateau (末 5 epoch loss 标准差 / 均值)。
# ===========================================================================
def convergence_sanity(train_info):
    losses = train_info["epoch_losses"]
    first, last = losses[0], losses[-1]
    tail = losses[-min(5, len(losses)):]
    tail_mean = float(np.mean(tail))
    tail_std = float(np.std(tail, ddof=0))
    plateau_ratio = (tail_std / tail_mean) if tail_mean > 1e-9 else float("nan")
    return {
        "loss_first": float(first),
        "loss_last": float(last),
        "loss_decreased": bool(last < first),
        "nan_seen": bool(train_info["nan_seen"]),
        "tail_mean_loss": tail_mean,
        "tail_std_loss": tail_std,
        "plateau_ratio": plateau_ratio,           # 越小越收敛 (末段 loss 相对波动)
        "plateau_ok": bool(plateau_ratio < 0.15) if plateau_ratio == plateau_ratio else False,
    }


# ===========================================================================
# 单 cell: 训 arm/seed -> 测固定全 R -> 全指标 -> per_run JSON 落盘。可续跑 (已存在则跳过)。
# ===========================================================================
def _validate_existing_record(path, expected):
    with open(path, "r", encoding="utf-8") as handle:
        record = json.load(handle)
    mismatches = {
        key: (record.get(key), value)
        for key, value in expected.items()
        if record.get(key) != value
    }
    environment = record.get("software_environment")
    environment_signature = record.get("software_environment_signature_sha256")
    if (
            environment_signature is not None
            and (
                not isinstance(environment, dict)
                or _canonical_json_sha256(environment) != environment_signature
            )
    ):
        mismatches["software_environment_integrity"] = (
            environment_signature,
            "SHA-256 of software_environment",
        )
    if mismatches:
        raise RuntimeError(
            "Refusing stale/incompatible cached cell %s; mismatches=%s. "
            "Review the inputs, then pass --force to recompute." % (path, mismatches)
        )


def run_cell(
        arm, seed, device, Xr, yr, R_sig, R_n, arm_signature,
        structure_info, protocol_signature, runner_source_signature,
        software_environment, software_environment_signature,
        per_run_dir, log_dir, force=False):
    global _cur_logf
    out_json = os.path.join(per_run_dir, "%d_seed%d.json" % (arm, seed))
    if os.path.isfile(out_json) and not force:
        _validate_existing_record(
            out_json,
            {
                "arm": arm,
                "seed": seed,
                "epochs": EPOCHS,
                "train_subset": "full",
                "arm_input_signature_sha256": arm_signature,
                "R_signature_md5": R_sig,
                "n_structure_diff_count": structure_info["n_diff"],
                "n_structure_same_count": structure_info["n_same"],
                "protocol_signature_sha256": protocol_signature,
                "runner_source_sha256": runner_source_signature,
                "software_environment": software_environment,
                "software_environment_signature_sha256": software_environment_signature,
            },
        )
        print("[skip verified] cell arm%d seed%d -> %s" % (arm, seed, out_json))
        return "skipped"

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "%d_seed%d.log" % (arm, seed))
    _cur_logf = open(log_path, "a", encoding="utf-8")
    t0 = time.time()
    try:
        _clog("=== cell arm%d seed%d START (epochs=%d, subset=%s) ===" % (arm, seed, EPOCHS, SUBSET_N))
        spec, keys = build_arm_spec(arm)
        Xs, ys = load_image_set(spec)
        _clog("loaded arm%d train: N=%d (air=%d ship=%d)" % (arm, len(ys), int((ys == 0).sum()), int((ys == 1).sum())))

        _, last_k, tinfo = train_model(Xs, ys, device, seed=seed)
        _clog("train done loss %.4f->%.4f nan=%s k=%d" % (tinfo["first_loss"], tinfo["last_loss"], tinfo["nan_seen"], tinfo["k_used"]))

        # 全 R 测试集指标 (主)
        proba_R = predict_proba_ckptavg(last_k, Xr, device)
        m_full = metrics_from_proba(proba_R, yr)
        _clog("eval full-R: macroF1=%.4f bal_acc=%.4f ship_f1=%.4f ship_rec=%.4f air_rec=%.4f"
              % (m_full["macro_f1"], m_full["balanced_accuracy"], m_full["f1_ship"],
                 m_full["recall_ship"], m_full["aircraft_recall"]))

        conv = convergence_sanity(tinfo)

        rec = {
            "arm": arm,
            "seed": seed,
            "smoke": _SMOKE,
            "epochs": EPOCHS,
            "subset_n": SUBSET_N,
            "n_train": int(len(ys)),
            "n_train_airplane": int((ys == 0).sum()),
            "n_train_ship": int((ys == 1).sum()),
            "arm_dir": ARM_DATASET_IDS[arm],
            "arm_input_signature_sha256": arm_signature,
            "R_dir": R_DATASET_ID,
            "R_n": int(R_n),
            "R_n_airplane": int((yr == 0).sum()),
            "R_n_ship": int((yr == 1).sum()),
            "R_signature_md5": R_sig,                     # 4 臂共用同一 R 张量的逐字节证据 (analyze 断言全同)
            "ceiling_macro_f1": CEILING_MACRO_F1,         # rounded real-to-real reference
            "metrics_full_R": {k: m_full[k] for k in SCALAR_METRIC_KEYS},
            "confmat_full_R": m_full["_confmat"],
            "gap_to_ceiling_macro_f1": float(CEILING_MACRO_F1 - m_full["macro_f1"]),
            "convergence": conv,
            "train_first_last_loss": [tinfo["first_loss"], tinfo["last_loss"]],
            "train_epoch_losses": tinfo["epoch_losses"],
            "nan_seen": tinfo["nan_seen"],
            "n_structure_diff_count": structure_info["n_diff"],
            "n_structure_same_count": structure_info["n_same"],
            "protocol_signature_sha256": protocol_signature,
            "runner_source_sha256": runner_source_signature,
            "software_environment": software_environment,
            "software_environment_signature_sha256": software_environment_signature,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "python_version": platform.python_version(),
            "elapsed_sec": float(time.time() - t0),
            "ckpt_avg_k": min(CKPT_AVG_K, EPOCHS),
            "label_map": {"airplane": 0, "ship": 1},
            "_protocol_note": (
                "Full-source cell. Statistical interpretation is performed "
                "by analyze_paired_delta.py using matched seeds."
            ),
        }

        # The structure subset is a separate training-set comparison, not a
        # target-test subset. Full cells always use all 490 source images.
        rec["train_subset"] = "full"

        _atomic_write_json(out_json, rec)
        _clog("=== cell arm%d seed%d DONE (%.1fs) -> %s ===" % (arm, seed, time.time() - t0, out_json))
        return "ran"
    finally:
        if _cur_logf is not None:
            _cur_logf.close()
            _cur_logf = None


# ===========================================================================
# Structure-subset cell: train on all 90 airplanes plus only those ship images
# whose arm2/arm3 bytes differ, then evaluate on the same full target set.
# ===========================================================================
def run_cell_structure_subset(
        arm, seed, device, Xr, yr, R_sig, R_n, arm_signature,
        ship_whitelist, subset_signature, structure_info, per_run_dir,
        log_dir, protocol_signature, runner_source_signature,
        software_environment, software_environment_signature, force=False):
    global _cur_logf
    if arm not in (2, 3):
        raise RuntimeError("Structure-subset cells are defined only for arm2/arm3.")
    out_json = os.path.join(per_run_dir, "%d_seed%d.json" % (arm, seed))
    if os.path.isfile(out_json) and not force:
        _validate_existing_record(
            out_json,
            {
                "arm": arm,
                "seed": seed,
                "epochs": EPOCHS,
                "train_subset": "structure_change_subset",
                "arm_input_signature_sha256": arm_signature,
                "structure_subset_signature_sha256": subset_signature,
                "R_signature_md5": R_sig,
                "n_structure_diff_count": structure_info["n_diff"],
                "n_structure_same_count": structure_info["n_same"],
                "protocol_signature_sha256": protocol_signature,
                "runner_source_sha256": runner_source_signature,
                "software_environment": software_environment,
                "software_environment_signature_sha256": software_environment_signature,
            },
        )
        print("[skip verified] structure-subset arm%d seed%d -> %s" % (arm, seed, out_json))
        return "skipped"

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "%d_seed%d.log" % (arm, seed))
    _cur_logf = open(log_path, "a", encoding="utf-8")
    t0 = time.time()
    try:
        _clog("=== structure-subset arm%d seed%d START (epochs=%d, n_ship_whitelist=%d) ==="
              % (arm, seed, EPOCHS, len(ship_whitelist)))
        spec, keys = build_arm_spec(arm, ship_whitelist=ship_whitelist)
        Xs, ys = load_image_set(spec)
        _clog("loaded arm%d structure-subset train: N=%d (air=%d ship=%d)"
              % (arm, len(ys), int((ys == 0).sum()), int((ys == 1).sum())))

        _, last_k, tinfo = train_model(Xs, ys, device, seed=seed)
        proba_R = predict_proba_ckptavg(last_k, Xr, device)
        m_full = metrics_from_proba(proba_R, yr)
        _clog("structure-subset eval full-R: macroF1=%.4f ship_f1=%.4f ship_rec=%.4f"
              % (m_full["macro_f1"], m_full["f1_ship"], m_full["recall_ship"]))
        conv = convergence_sanity(tinfo)

        rec = {
            "arm": arm, "seed": seed, "smoke": _SMOKE, "epochs": EPOCHS,
            "train_subset": "structure_change_subset",
            "n_train": int(len(ys)),
            "n_train_airplane": int((ys == 0).sum()),
            "n_train_ship": int((ys == 1).sum()),
            "n_structure_ship_used": len(ship_whitelist),
            "arm_dir": ARM_DATASET_IDS[arm],
            "arm_input_signature_sha256": arm_signature,
            "structure_subset_signature_sha256": subset_signature,
            "R_dir": R_DATASET_ID, "R_n": int(R_n),
            "R_signature_md5": R_sig,
            "ceiling_macro_f1": CEILING_MACRO_F1,
            "metrics_full_R": {k: m_full[k] for k in SCALAR_METRIC_KEYS},
            "confmat_full_R": m_full["_confmat"],
            "gap_to_ceiling_macro_f1": float(CEILING_MACRO_F1 - m_full["macro_f1"]),
            "convergence": conv,
            "train_first_last_loss": [tinfo["first_loss"], tinfo["last_loss"]],
            "train_epoch_losses": tinfo["epoch_losses"],
            "nan_seen": tinfo["nan_seen"],
            "n_structure_diff_count": structure_info["n_diff"],
            "n_structure_same_count": structure_info["n_same"],
            "protocol_signature_sha256": protocol_signature,
            "runner_source_sha256": runner_source_signature,
            "software_environment": software_environment,
            "software_environment_signature_sha256": software_environment_signature,
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "device": str(device), "python_version": platform.python_version(),
            "elapsed_sec": float(time.time() - t0),
            "ckpt_avg_k": min(CKPT_AVG_K, EPOCHS),
            "label_map": {"airplane": 0, "ship": 1},
            "_protocol_note": (
                "Structure-change subset cell for arm2-to-arm3 only; an "
                "interval including zero does not prove zero effect."
            ),
        }
        _atomic_write_json(out_json, rec)
        _clog("=== structure-subset arm%d seed%d DONE (%.1fs) -> %s ==="
              % (arm, seed, time.time() - t0, out_json))
        return "ran"
    finally:
        if _cur_logf is not None:
            _cur_logf.close()
            _cur_logf = None


# ===========================================================================
# 每臂 summary_{arm}.md: 逐 seed 指标 + 均值±std + 收敛 sanity。
# ===========================================================================
def write_arm_summary(arm, per_run_dir, out_dir):
    rows = []
    for seed in SEEDS_FULL:
        p = os.path.join(per_run_dir, "%d_seed%d.json" % (arm, seed))
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                rows.append(json.load(f))
    if not rows:
        return None
    keys_report = ["macro_f1", "balanced_accuracy", "f1_ship", "recall_ship", "f1_airplane",
                   "recall_airplane", "aircraft_recall", "accuracy"]
    lines = []
    lines.append("# E2 ablation summary -- arm%d" % arm)
    lines.append("")
    lines.append("- arm dataset ID: %s" % ARM_DATASET_IDS[arm])
    lines.append("- n_seed present: %d / %d  (seeds=%s)" % (len(rows), len(SEEDS_FULL), [r["seed"] for r in rows]))
    lines.append("- separate real-to-real reference macro-F1: %.3f" % CEILING_MACRO_F1)
    lines.append("- torch=%s cuda=%s" % (rows[0]["torch_version"], rows[0]["cuda_version"]))
    lines.append("")
    lines.append("## 逐 seed 指标 (full-R 测试集)")
    lines.append("")
    header = "| seed | " + " | ".join(keys_report) + " | loss↓ | plateau_ok | nan |"
    sep = "|" + "---|" * (len(keys_report) + 4)
    lines.append(header)
    lines.append(sep)
    for r in sorted(rows, key=lambda x: x["seed"]):
        m = r["metrics_full_R"]
        c = r["convergence"]
        cells = ["%.4f" % m[k] for k in keys_report]
        lines.append("| %d | %s | %s | %s | %s |" % (
            r["seed"], " | ".join(cells), str(c["loss_decreased"]), str(c["plateau_ok"]), str(c["nan_seen"])))
    lines.append("")
    lines.append("## 均值 ± std (跨 %d seed)" % len(rows))
    lines.append("")
    lines.append("| metric | mean | std |")
    lines.append("|---|---|---|")
    for k in keys_report:
        vals = np.array([r["metrics_full_R"][k] for r in rows], dtype=np.float64)
        lines.append("| %s | %.4f | %.4f |" % (k, vals.mean(), vals.std(ddof=0)))
    lines.append("")
    lines.append("## 收敛 sanity")
    n_dec = sum(1 for r in rows if r["convergence"]["loss_decreased"])
    n_nan = sum(1 for r in rows if r["convergence"]["nan_seen"])
    n_plat = sum(1 for r in rows if r["convergence"]["plateau_ok"])
    lines.append("- loss 末<首: %d/%d ; 无 NaN: %d/%d ; plateau_ok: %d/%d"
                 % (n_dec, len(rows), len(rows) - n_nan, len(rows), n_plat, len(rows)))
    lines.append("")
    lines.append("_注: 本 summary 只汇总, 不做配对 Δ 判定 (判定在 analyze_paired_delta.py)。_")
    out_md = os.path.join(out_dir, "summary_arm%d.md" % arm)
    _atomic_write_text(out_md, "\n".join(lines) + "\n")
    print("[summary] arm%d -> %s" % (arm, out_md))
    return out_md


def _parse_int_selection(raw, allowed, label):
    if raw is None:
        return list(allowed)
    try:
        selected = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise RuntimeError("%s must be a comma-separated integer list" % label) from exc
    if not selected:
        raise RuntimeError("%s cannot be empty" % label)
    if len(selected) != len(set(selected)):
        raise RuntimeError("%s contains duplicate values: %s" % (label, selected))
    unexpected = sorted(set(selected) - set(allowed))
    if unexpected:
        raise RuntimeError("%s contains unsupported values: %s" % (label, unexpected))
    return selected


# ===========================================================================
# 入口
# ===========================================================================
def main():
    global _SMOKE, EPOCHS, SUBSET_N
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Non-evidentiary smoke test: arm1 seed=9999, EPOCHS=2, 12 images per class.")
    ap.add_argument(
        "--arms",
        type=str,
        default=None,
        help=(
            "Train only the selected full-data arms (comma-separated). Normal "
            "training still validates and fingerprints the complete four-arm "
            "frozen input contract."
        ),
    )
    ap.add_argument("--seeds", type=str, default=None, help="只跑指定 seed (逗号分隔)。")
    ap.add_argument("--force", action="store_true", help="覆盖已存在 per_run JSON (默认续跑跳过)。")
    ap.add_argument("--no_format_probe", action="store_true", help="跳过 format-probe (调试用)。")
    ap.add_argument(
        "--format_probe_only",
        action="store_true",
        help="Compute format probes for --arms and exit without training.",
    )
    ap.add_argument(
        "--structure_subset", "--subset268",
        dest="structure_subset",
        action="store_true",
        help=(
            "Also run arm2/arm3 on the dynamically detected structure-change "
            "ship subset (263 ships in the frozen experiment)."
        ),
    )
    args = ap.parse_args()
    _SMOKE = args.smoke

    if _SMOKE:
        EPOCHS = 2
        SUBSET_N = 12
        arms_to_run = [1]
        seeds_to_run = [9999]               # throwaway 非实验 seed
        per_run_dir = SMOKE_DIR
        log_dir = os.path.join(SMOKE_DIR, "logs")
        out_dir = SMOKE_DIR
    else:
        EPOCHS = 40
        SUBSET_N = None
        arms_to_run = _parse_int_selection(args.arms, ARMS_FULL, "--arms")
        seeds_to_run = _parse_int_selection(args.seeds, SEEDS_FULL, "--seeds")
        per_run_dir = PER_RUN_DIR
        log_dir = LOGS_DIR
        out_dir = OUT_ROOT

    if args.structure_subset and args.format_probe_only:
        raise RuntimeError("--structure_subset and --format_probe_only cannot be combined.")
    if args.structure_subset and _SMOKE:
        raise RuntimeError("--structure_subset and --smoke cannot be combined.")
    if args.structure_subset and not _SMOKE and not {2, 3}.issubset(arms_to_run):
        raise RuntimeError(
            "--structure_subset is a paired arm2/arm3 comparison; --arms must include both 2 and 3."
        )

    protocol_signature = compute_protocol_signature()
    runner_source_signature = _sha256_file(os.path.abspath(__file__))

    if args.format_probe_only:
        if _SMOKE:
            raise RuntimeError("--format_probe_only and --smoke cannot be combined.")
        if args.no_format_probe:
            raise RuntimeError("--format_probe_only and --no_format_probe conflict.")
        _require_configured_path("OUT_ROOT", OUT_ROOT)
        _require_configured_path("KLSG_RAW", KLSG_RAW)
        for arm in arms_to_run:
            _require_configured_path("ARM_DIRS[%d]" % arm, ARM_DIRS[arm])
            for cls in CLASSES:
                class_dir = os.path.join(ARM_DIRS[arm], cls)
                if not os.path.isdir(class_dir):
                    raise RuntimeError("Missing arm%d/%s directory: %s" % (arm, cls, class_dir))
                n_images = len(_list_png(class_dir))
                if n_images != EXPECTED_ARM[cls]:
                    raise RuntimeError(
                        "arm%d/%s has %d PNGs; expected %d"
                        % (arm, cls, n_images, EXPECTED_ARM[cls])
                    )
        os.makedirs(OUT_ROOT, exist_ok=True)
        for arm in arms_to_run:
            fp = compute_format_probe_for_arm(arm)
            print(
                "[format-probe] arm%d: ship AUC=%.4f airplane AUC=%.4f"
                % (
                    arm,
                    fp["ship"]["auc_mean"],
                    fp["airplane"]["auc_mean"],
                )
            )
        return

    _require_configured_path("OUT_ROOT", OUT_ROOT)
    _require_configured_path("R_DIR", R_DIR)
    for arm in ARMS_FULL:
        _require_configured_path("ARM_DIRS[%d]" % arm, ARM_DIRS[arm])
    if not args.no_format_probe and not _SMOKE:
        _require_configured_path("KLSG_RAW", KLSG_RAW)

    print("=" * 72)
    print("e2_ablation_runner  smoke=%s  arms=%s  seeds=%s  EPOCHS=%d  subset=%s"
          % (_SMOKE, arms_to_run, seeds_to_run, EPOCHS, SUBSET_N))
    print("=" * 72)

    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(per_run_dir, exist_ok=True)

    # Guard against accidentally substituting the pre-resized arm0 copy.
    assert_arm0_is_native()
    # --arms selects training cells, while normal mode always verifies the
    # complete frozen four-arm input contract used to define those cells.
    check_upstream_contract(ARMS_FULL)
    assert_cross_arm_control_contract()
    arm_signatures = {}
    for arm in ARMS_FULL:
        signature, n_files = compute_image_tree_signature(ARM_DIRS[arm])
        expected_n = sum(EXPECTED_ARM.values())
        if n_files != expected_n:
            raise RuntimeError(
                "arm%d signature covered %d PNGs; expected %d"
                % (arm, n_files, expected_n)
            )
        arm_signatures[arm] = signature
        print("[arm%d] input SHA-256=%s (over %d files)" % (arm, signature, n_files))

    # ---- R 张量指纹 (4 臂共用同一份 R 的逐字节证据) ----
    R_sig, R_nfiles = compute_R_signature()
    print("[R] signature md5=%s (over %d files)" % (R_sig, R_nfiles))

    # Structure-change subset: arm2/arm3 ship files that differ byte-for-byte.
    diff_structure, same_structure = ([], [])
    structure_subset_signature = None
    if not _SMOKE:
        diff_structure, same_structure = compute_structure_subset_ship_fnames()
        structure_subset_signature = compute_name_list_signature(diff_structure)
        print(
            "[structure-subset] arm2 vs arm3 ship: changed=%d unchanged=%d SHA-256=%s"
            % (len(diff_structure), len(same_structure), structure_subset_signature)
        )
    structure_info = {
        "n_diff": len(diff_structure),
        "n_same": len(same_structure),
    }
    # Save the dynamically detected structure-change filenames.
    if not _SMOKE:
        _atomic_write_json(
            os.path.join(OUT_ROOT, "structure_subset_ship_fnames.json"),
            {"n_diff": len(diff_structure), "n_same": len(same_structure),
             "structure_subset_signature_sha256": structure_subset_signature,
             "diff_ship_fnames": diff_structure,
             "same_ship_fnames": same_structure,
             "_note": (
                 "Ship files whose arm2 and arm3 bytes differ; used "
                 "only for the arm2-to-arm3 structure comparison."
             )},
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    software_environment = training_software_environment(device)
    software_environment_signature = _canonical_json_sha256(software_environment)
    print("[device]", device, "torch", torch.__version__, "cuda", torch.version.cuda)
    print("[environment] SHA-256=%s" % software_environment_signature)

    # ---- 固定全 R 测试张量 (4 臂所有 cell 共用同一份, 只 load 一次) ----
    Xr, yr = load_image_set(build_R_spec())
    print("[R] loaded test tensor: N=%d (air=%d ship=%d)" % (len(yr), int((yr == 0).sum()), int((yr == 1).sum())))

    # ---- format-probe (每臂一次; smoke 跳) ----
    if not _SMOKE and not args.no_format_probe:
        for arm in arms_to_run:
            fp = compute_format_probe_for_arm(arm, arm_signatures[arm])
            print("[format-probe] arm%d: ship AUC=%.4f airplane AUC=%.4f"
                  % (arm, fp["ship"]["auc_mean"], fp["airplane"]["auc_mean"]))

    # ---- 主循环: (arm, seed) cell ----
    n_ran = n_skip = 0
    for arm in arms_to_run:
        for seed in seeds_to_run:
            st = run_cell(arm, seed, device, Xr, yr, R_sig, R_nfiles,
                          arm_signatures[arm], structure_info,
                          protocol_signature, runner_source_signature,
                          software_environment, software_environment_signature,
                          per_run_dir, os.path.join(log_dir, "arm%d" % arm) if not _SMOKE else log_dir,
                          force=args.force)
            if st == "ran":
                n_ran += 1
            elif st == "skipped":
                n_skip += 1
        # 每臂跑完写 summary (仅全量)
        if not _SMOKE and set(seeds_to_run) == set(SEEDS_FULL):
            write_arm_summary(arm, per_run_dir, out_dir)
        elif not _SMOKE:
            print("[summary] arm%d not rewritten after a partial seed selection." % arm)

    # ---- Optional arm2/arm3 structure-change subset cells ----
    n_ran268 = n_skip268 = 0
    if args.structure_subset and not _SMOKE:
        per_run_268 = os.path.join(OUT_ROOT, "per_run_structure_subset")
        log_268 = os.path.join(LOGS_DIR, "structure_subset")
        os.makedirs(per_run_268, exist_ok=True)
        ship_whitelist = set(diff_structure)
        if not ship_whitelist:
            raise RuntimeError(
                "Structure-change subset is empty; verify the arm2/arm3 paths."
            )
        arms268 = [2, 3]
        print(
            "[structure-subset] arms=%s seeds=%s n_ship=%d"
            % (arms268, seeds_to_run, len(ship_whitelist))
        )
        for arm in arms268:
            for seed in seeds_to_run:
                st = run_cell_structure_subset(
                    arm, seed, device, Xr, yr, R_sig, R_nfiles,
                    arm_signatures[arm], ship_whitelist,
                    structure_subset_signature, structure_info, per_run_268,
                    os.path.join(log_268, "arm%d" % arm),
                    protocol_signature, runner_source_signature,
                    software_environment, software_environment_signature,
                    force=args.force)
                if st == "ran":
                    n_ran268 += 1
                elif st == "skipped":
                    n_skip268 += 1

    print("\n" + "=" * 72)
    print("e2_ablation_runner DONE. full: ran=%d skipped=%d | structure-subset: ran=%d skipped=%d (smoke=%s)"
          % (n_ran, n_skip, n_ran268, n_skip268, _SMOKE))
    print("  per_run dir:", per_run_dir)
    print("=" * 72)


if __name__ == "__main__":
    main()
