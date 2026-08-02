#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_source_only_sgd.py -- source-only **SGD 底座版**
[SSS sim->real 主实验 B]

================================  这是什么 / 跟 Adam 诊断版 full.py 的区别 ================================
本文件由 Adam **诊断版** `train_source_only_full.py` 复制改造而来（full.py **保持不动**，留作
"固定 lr 看过拟合"的诊断对照基准）。本版 = **run2 CDAN 的同 regime source-only 基线**：把优化器底座
换成 SGD+momentum 并加 DANN/CDAN 标准的 inverse-decay lr 调度，使 source-only 与后续对抗训练处在同一
优化 regime（公平对照）。其余配方（整网微调 / 分层 lr 结构 / CB-WCE β0.999 / 增强 / default BN /
数据加载 / 指标 / SEED / 产物落盘方式）与 full.py 完全一致，仅下列优化器相关项改动：

  1. 整网微调 + 分层 lr：ResNet18+ImageNet，fc 换 2 类；**全解冻（backbone 不冻）**；
     backbone lr = fc lr 的 1/10。default BN。（结构同 full.py；lr 量级见第 4 条）
  2. CB-WCE 损失：Class-Balanced 加权 CE（Cui et al. 2019 effective-number，β=0.999），
     权重由每个源训练折的类频（airplane 72 / ship 320）按 w_c=(1-β)/(1-β^n_c) 算、
     归一化（mean=1）→ airplane:ship ≈ 3.9:1。作用于分类损失（train 与 val 同一套损失，便于背离判读）。
  3. 在线数据增强（**只施于 arm2 训练输入**；val/eval 一律不增强）：几何 RandomResizedCrop(224,0.8~1.0)
     + HFlip(0.5) + 旋转 ±15°（垂直翻转 OFF）；光度 ColorJitter(0.15,0.15)；斑点乘性噪声 σ=0.1。
  4. **优化器底座（本版核心改动 vs full.py 的 Adam 固定 lr）**：
     - SGD + momentum=0.9 + nesterov=True；分层 lr fc=1e-2 / backbone=1e-3（保持 1/10）。
     - weight_decay = 5e-4（论文正式配置）。
     - **inverse-decay lr 调度**（DANN/CDAN 标准）：lr = base_lr·(1 + α·p)^(−β)，α=10、β=0.75，
       p = global_step / total_steps **归一化到 [0,1]**（务必用归一化 p，不是迭代号原值；本任务步数
       少 ~600，用原值几乎不衰减）。每个 optimizer.step() 后 scheduler.step()；末端 p≈1 时 lr ≈
       base_lr·11^(−0.75) ≈ base_lr·0.166（训练结束有 sanity 断言核验）。

★★ 本版定位 = **run2 CDAN 的同 regime source-only 基线**（性能锚点，非过拟合诊断） ★★
  - train = arm2/所选fold/train（施增强）；**val = 同一fold/test（源留出折、不增强）** —— 合法，不碰真实域标签。
  - 每个 epoch 同时记并打印 train_loss / val_loss / train_acc / val_acc，并在 KLSG 固定子集上记录无标签 churn。
  - 正式配置 epoch=40；带 inverse-decay 调度（与 run2 对抗训练同 regime）。末 K 个 epoch 存 checkpoint。
  - ⚠ 本版 lr 末端衰减到约 0.166×，会自然压平 train/val 背离；不能仅凭曲线变平判断
    模型没有过拟合。

合法性护栏（plan §3.2.3 / §3.2.4）：源留出折带**源标签**、合法可做收敛/过拟合信号；**绝不**用真实域(KLSG)
表现早停/选模。本版 source-only 不使用目标域更新模型；KLSG 仅以无标签固定子集计算 churn 健康诊断，
**不读取目标标签，也不把该诊断用于选模或调参**。

指标口径（plan §3.2.5）：macro-F1 = sklearn f1_score(average='macro')；airplane/ship 各自 P/R/F1 对称报；混淆矩阵。
==========================================================================
"""

import os
# [determinism] CUBLAS_WORKSPACE_CONFIG 必须在 import torch 之前设置(cuBLAS GEMM 决定论); 用 setdefault 让 driver 级 export 优先、幂等。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import csv
import sys
import json
import argparse
import datetime
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# [determinism] 进程全局 cuDNN/算法决定论(放模块顶: 覆盖 selftest/--unittest 等所有路径)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)   # 硬模式 warn_only=False
from PIL import Image
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support, confusion_matrix, accuracy_score,
)

# [红线护栏] KLSG 目标域盲化入口（单点维护，绝不读真实标签）；src/ 在 parents[1]
#   ★ run1 首次接 KLSG：仅供 churn（固定 100 子集 label-free argmax 翻转率）；source-only 训练流绝不接 KLSG。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _klsg_blind  # read_klsg_churn_subset / UnlabeledImageDataset / collate_unlabeled / FORBIDDEN_LABEL
import _run_safety

# ============================================================ 配置
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLITS_CSV = PROJECT_ROOT / "data" / "split" / "splits.csv"            # 源 arm2（train + 源留出折 val）
RESULTS_DIR = PROJECT_ROOT / "results" / "run1_s4_baseline_ceiling"

CLASSES = ["airplane", "ship"]            # 索引: airplane=0, ship=1
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

# --- SGD 底座版超参（run2 同 regime；RTX 5090，算力充沛）---
SEED = 0
EPOCHS_DEFAULT = 40                        # 论文正式配置 N=40
BATCH_SIZE = 64
FC_LR = 1e-2                               # 新层（fc）—— SGD 量级（vs full.py Adam 的 1e-3）
BACKBONE_LR = 1e-3                         # backbone = fc 的 1/10（分层 lr，保持比例）
WEIGHT_DECAY = 5e-4                        # 论文正式配置
LR_DECAY_ALPHA = 10.0                      # inverse-decay α：lr = base_lr·(1+α·p)^(−β)
LR_DECAY_BETA = 0.75                       # inverse-decay β（DANN/CDAN 标准 0.75）
CB_BETA = 0.999                            # CB-WCE effective-number β（论文标准默认、不调）
SPECKLE_SIGMA = 0.1                        # 乘性 speckle 噪声标准差
SAVE_LAST_K = 5                            # 存末 K 个 epoch 的 checkpoint


# ============================================================ 数据
def read_split_rows(csv_path, domain=None, fold=None, split=None):
    """读 csv，按 (domain,fold,split) 过滤，返回 [(abspath, label_idx), ...]。
    只允许源域 arm2；KLSG 真标签只能经 _klsg_blind.read_klsg_labeled 授权核。"""
    if domain != "arm2":
        raise RuntimeError(
            f"RED LINE: read_split_rows 只允许源域 'arm2'，实得 {domain!r}；"
            "KLSG 真标签只能经 _klsg_blind.read_klsg_labeled 授权核读")
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if domain is not None and r["domain"] != domain:
                continue
            if fold is not None and int(r["fold"]) != int(fold):
                continue
            if split is not None and r["split"] != split:
                continue
            abspath = (PROJECT_ROOT / r["filepath"]).resolve()
            rows.append((str(abspath), LABEL2IDX[r["label"]]))
    return rows


class ImageListDataset(Dataset):
    """从 (abspath, label_idx) 列表建数据集；小数据集——预载 PIL(RGB) 进内存。
    transform 在 __getitem__ 内按调用施加 -> 训练增强每个 epoch 都重新随机（在线增强）。"""
    def __init__(self, items, transform):
        self.transform = transform
        self.labels = [lab for _, lab in items]
        self.images = []
        for path, _ in items:
            with Image.open(path) as im:
                self.images.append(im.convert("RGB").copy())   # 灰度声呐图 -> 3 通道
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        return self.transform(self.images[i]), self.labels[i]


class SpeckleNoise:
    """乘性 speckle 噪声：x <- x * (1 + ε), ε~N(0, σ²)，作用于 [0,1] tensor，clamp 回 [0,1]。
    在 ToTensor 之后、Normalize 之前施加（噪声应作用于物理像素强度，不是归一化后的值）。"""
    def __init__(self, sigma=0.1):
        self.sigma = float(sigma)
    def __call__(self, x):
        if self.sigma <= 0:
            return x
        noise = torch.randn_like(x) * self.sigma
        return (x * (1.0 + noise)).clamp_(0.0, 1.0)
    def __repr__(self):
        return f"SpeckleNoise(sigma={self.sigma})"


def build_train_transform():
    """arm2 训练增强（在线）：几何 + 光度 + speckle。垂直翻转 OFF。"""
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),       # 保目标在框（不激进裁）
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),                          # ±15°，fill=0（黑底，贴声呐背景）
        T.ColorJitter(brightness=0.15, contrast=0.15),         # ±15% 光度抖动
        T.ToTensor(),
        SpeckleNoise(sigma=SPECKLE_SIGMA),                     # 乘性斑点噪声
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_eval_transform():
    """验证与评估：**不增强**。resize + ImageNet 归一化。"""
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ============================================================ CB-WCE
def cb_wce_weights(class_counts, beta=CB_BETA, num_classes=len(CLASSES), device=None):
    """Class-Balanced 权重（Cui et al. 2019, effective number）：w_c = (1-β)/(1-β^{n_c})，
    再归一化使 mean=1（sum=num_classes），比值不变。class_counts 按类索引排列。"""
    eff_num = [1.0 - (beta ** n) for n in class_counts]
    w = [(1.0 - beta) / en for en in eff_num]
    s = sum(w)
    w = [wi / s * num_classes for wi in w]                     # 归一化 mean=1
    return torch.tensor(w, dtype=torch.float32, device=device)


# ============================================================ 模型 / 优化器
def build_model(pretrained=True):
    """ResNet18；训练默认载入ImageNet权重，评估可先建空骨架再严格载入checkpoint。"""
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)  # 未缓存时由torchvision下载；评估传False。
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(CLASSES))
    for p in model.parameters():
        p.requires_grad = True                                 # 全解冻（含 BN gamma/beta）
    return model


def make_optimizer(model):
    """分层 lr 的 **SGD（+momentum +nesterov）**：group 0 = backbone (lr=BACKBONE_LR)，
    group 1 = fc (lr=FC_LR=10×backbone)；momentum=0.9、nesterov=True、weight_decay=WEIGHT_DECAY。
    param group 顺序固定 [backbone, fc]（scheduler.base_lrs 依此对齐）。
    返回 (optimizer, n_backbone_params, n_fc_params)。"""
    fc_params, backbone_params = [], []
    for name, p in model.named_parameters():
        (fc_params if name.startswith("fc.") else backbone_params).append(p)
    optimizer = torch.optim.SGD(
        [{"params": backbone_params, "lr": BACKBONE_LR},   # group 0 = backbone
         {"params": fc_params, "lr": FC_LR}],              # group 1 = fc
        momentum=0.9, nesterov=True, weight_decay=WEIGHT_DECAY,
    )
    return optimizer, sum(p.numel() for p in backbone_params), sum(p.numel() for p in fc_params)


def make_scheduler(optimizer, total_steps):
    """inverse-decay lr 调度（DANN/CDAN 标准）：每步 lr = base_lr · (1 + α·p)^(−β)，
    p = global_step / total_steps **归一化到 [0,1]**（不是迭代号原值！本任务步数少 ~600，
    用原值几乎不衰减）。α=LR_DECAY_ALPHA、β=LR_DECAY_BETA。

    实现细节：LambdaLR 的 lr_lambda 返回的是**相对 base_lr 的缩放因子**（不是绝对 lr），
    LambdaLR 自动用各 param group 各自的 base_lr（=构造时的初始 lr：backbone=BACKBONE_LR、
    fc=FC_LR）乘以同一因子 —— 故两组同步衰减、各按自身基准。每次 optimizer.step() 后调一次
    scheduler.step()，LambdaLR 内部 last_epoch 即 global_step。末端 p→1 时因子 = (1+α)^(−β)
    = 11^(−0.75) ≈ 0.166。total_steps 下限取 1 防除零；min() clamp 防 last_epoch 越界使 p>1。"""
    total_steps = max(int(total_steps), 1)
    lr_lambda = lambda step: (1.0 + LR_DECAY_ALPHA * (min(step, total_steps) / total_steps)) ** (-LR_DECAY_BETA)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ============================================================ 评估
def evaluate_preds(model, loader, device):
    """返回 (y_true, y_pred)（list[int]），用于指标。"""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            y_pred.extend(logits.argmax(1).cpu().tolist())
            y_true.extend(y.tolist())
    return y_true, y_pred


def eval_loss_acc(model, loader, criterion, device):
    """在 loader 上算 (loss, acc)（不增强、不更新参数）；val 过拟合信号用。"""
    model.eval()
    tot_loss, correct, seen = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            tot_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += x.size(0)
    return tot_loss / seen, correct / seen


def churn_predict(model, loader, device):
    """churn（spec §3.4 a · run1 churn-only，无 raw-QP）：模型对固定 churn 100 子集的 argmax 预测向量（长100, int）。
    churn 标量 = 相邻 epoch 此向量的不一致比例（在训练循环里算）。

    ★ 决定论铁律（spec 護栏②）：model.eval() + torch.no_grad() + eval_tfm(无增强) + shuffle=False
      + num_workers=0 -> 零 RNG 消费、零 BN buffer 写（eval 模式 BN 不更新 running stats；resnet18 无 dropout）。
    ★ 红线（spec 護栏①）：只取 batch[0] 图像；batch[1] 是毒丸 FORBIDDEN_LABEL, 绝不索引/绝不读真实标签。
    ★ run1 适配：模型是裸 resnet18(fc 换 2 类)，前向单输出 `logits = model(x)`（非 run2/3 的 `_, logits = model(x)`）。
    调用后 model.train() 复位。返回 np.ndarray(int)。"""
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)        # 只取图像；batch[1] 是毒丸, 绝不碰
            logits = model(x)                                 # run1 单输出 forward
            preds.extend(logits.argmax(1).cpu().tolist())
    model.train()
    return np.array(preds, dtype=np.int64)


def report_metrics(y_true, y_pred):
    """plan §3.2.5：macro-F1 + 各类 P/R/F1（对称）+ 混淆矩阵。"""
    macro_f1 = f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)
    p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc = accuracy_score(y_true, y_pred)
    per_class = {c: {"precision": float(p[i]), "recall": float(r[i]),
                     "f1": float(f[i]), "support": int(sup[i])} for i, c in enumerate(CLASSES)}
    return {"macro_f1": float(macro_f1), "accuracy": float(acc), "per_class": per_class,
            "confusion_matrix": cm.tolist(),
            "cm_axis": "rows=true, cols=pred, order=[airplane, ship]"}


def print_metrics_block(title, m):
    print(f"\n  [{title}]  macro-F1 = {m['macro_f1']:.4f} | accuracy = {m['accuracy']:.4f}")
    print(f"  {'class':>9} | {'precision':>9} | {'recall':>9} | {'f1':>9} | {'support':>7}")
    for c in CLASSES:
        d = m["per_class"][c]
        print(f"  {c:>9} | {d['precision']:>9.4f} | {d['recall']:>9.4f} | {d['f1']:>9.4f} | {d['support']:>7}")
    cm = m["confusion_matrix"]
    print(f"  混淆(行=真,列=预测,[airplane,ship]): "
          f"[[{cm[0][0]},{cm[0][1]}],[{cm[1][0]},{cm[1][1]}]]")


# ============================================================ main
def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    ap.add_argument("--quick", action="store_true", help="快速 smoke（3 epoch、独立产物名 sgd_quick_*）")
    ap.add_argument("--seed", type=int, default=SEED, help="随机种子（默认 0，= D4 基线；改它走 set_seed RNG 四件套）")
    ap.add_argument("--fold", type=int, default=0, help="源 arm2 折号（默认 0）：train/val 同折轮转。"
                    "source-only 无目标训练流，不接 KLSG 训练（KLSG 仅 churn label-free 评估用，块③）")
    ap.add_argument("--save-every-epoch", action="store_true",
                    help="逐 epoch 存全 dict ckpt（标定长跑 post-hoc 用；默认 False，不动现有末 K 存盘逻辑）")
    args = ap.parse_args()
    epochs = 3 if args.quick else args.epochs                 # smoke = 3 epoch（full.py 是 2，本版取 3 以更清楚展示 lr 衰减）
    tag = "sgd_quick" if args.quick else "sgd_full"           # 'sgd_' 前缀：不覆盖 Adam 诊断版 full.py 的产物

    # spec §3.4(d) / 决议表#2：model_dir 含 seed/fold，防标定矩阵 3seed×2fold 互相覆盖
    model_dir = RESULTS_DIR / f"{tag}_s{args.seed}_f{args.fold}"
    metrics_path = model_dir / f"{tag}_metrics.json"
    curve_path = model_dir / f"{tag}_curve.csv"
    churn_argmax_path = model_dir / f"{tag}_churn_argmax.csv"   # spec §3.4(e)：逐 epoch argmax 矩阵
    _run_safety.prepare_empty_run_dirs(model_dir)

    set_seed(args.seed)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"), "CUBLAS_WORKSPACE_CONFIG 未设(须在 import torch 前内联)"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print(f"run1 source-only **SGD 底座版** ({tag}) : arm2 fold0 整网微调 + inverse-decay 调度 (run2 同 regime)")
    print("=" * 78)
    print(f"项目根 : {PROJECT_ROOT}")
    print(f"设备   : {device} | torch {torch.__version__} | "
          f"GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # --- 数据（source-only：只用 arm2 源训练 + 源留出折验证；真实目标 KLSG 不参与监督评估）---
    train_items = read_split_rows(SPLITS_CSV, domain="arm2", fold=args.fold, split="train")  # 施增强；源折由 --fold 选
    val_items = read_split_rows(SPLITS_CSV, domain="arm2", fold=args.fold, split="test")     # 源留出折(同折)，不增强
    tr_cnt = Counter(CLASSES[l] for _, l in train_items)
    va_cnt = Counter(CLASSES[l] for _, l in val_items)
    print(f"\n训练 arm2/fold{args.fold}/train (增强) : {len(train_items)} (airplane={tr_cnt['airplane']}, ship={tr_cnt['ship']})")
    print(f"验证 arm2/fold{args.fold}/test (源留出折,不增强): {len(val_items)} (airplane={va_cnt['airplane']}, ship={va_cnt['ship']})")
    assert len(train_items) and len(val_items), "数据为空，检查 csv 路径/过滤"

    train_tfm, eval_tfm = build_train_transform(), build_eval_transform()
    train_loader = DataLoader(ImageListDataset(train_items, train_tfm), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(ImageListDataset(val_items, eval_tfm), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)
    # churn 固定 100 子集 loader（spec §3.4 a；label-free, 只 churn 用——source-only 训练流不接 KLSG）。
    #   eval_tfm 无增强 / shuffle=False / num_workers=0 -> 零 RNG 消费、零 buffer 写（護栏②）。
    #   盲化入场: read_klsg_churn_subset(sha 锚 + 毒丸 label, 護栏①)。构造在所有 RNG-消费 loader 之后,
    #   shuffle=False 不消费 RNG, 不扰动决定论（run2x 终极证明）。
    churn_items = _klsg_blind.read_klsg_churn_subset(PROJECT_ROOT)
    assert len(churn_items) == _klsg_blind.CHURN_N_EXPECTED, \
        f"churn 子集张数={len(churn_items)} != {_klsg_blind.CHURN_N_EXPECTED}"
    churn_loader = DataLoader(_klsg_blind.UnlabeledImageDataset(churn_items, eval_tfm), batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, collate_fn=_klsg_blind.collate_unlabeled)
    print(f"churn 固定子集 (label-free, sha {_klsg_blind.KLSG_CHURN100_SHA256[:8]}) : {len(churn_items)} 张 "
          f"[eval_tfm 无增强/shuffle=False/num_workers=0; 只取图像、label 位毒丸绝不读]")

    # --- CB-WCE 权重（从源训练集类频）---
    counts = [tr_cnt[c] for c in CLASSES]                       # [n_airplane, n_ship]
    assert counts == [72, 320], f"CB-WCE 护栏（块1钉②）：arm2 各折 train 类频应恒为 [72,320]，实得 {counts}（--fold 放开后防权重静默漂）"
    cb_w = cb_wce_weights(counts, beta=CB_BETA, device=device)
    ratio = (cb_w[0] / cb_w[1]).item()
    print(f"\nCB-WCE (β={CB_BETA}) 权重 (从 arm2 train 类频 {counts}): "
          f"airplane={cb_w[0].item():.4f}, ship={cb_w[1].item():.4f} (airplane:ship = {ratio:.2f}:1)")
    criterion = nn.CrossEntropyLoss(weight=cb_w)

    # --- 模型 / 优化器 / 调度器（整网微调 + 分层 lr + inverse-decay）---
    model = build_model(pretrained=True).to(device)
    optimizer, n_bb, n_fc = make_optimizer(model)
    n_total = sum(p.numel() for p in model.parameters())
    steps_per_epoch = len(train_loader)            # = ceil(n_train / batch)（drop_last=False，与实际 batch 数一致）
    total_steps = epochs * steps_per_epoch         # inverse-decay 归一化分母 = 总 optimizer.step() 次数
    scheduler = make_scheduler(optimizer, total_steps)
    print(f"\n模型 : ResNet18(ImageNet) **全解冻整网微调** | default BN")
    print(f"分层 lr : backbone lr={BACKBONE_LR} ({n_bb:,} 参数) | fc lr={FC_LR} ({n_fc:,} 参数) "
          f"| 比 = 1:{int(FC_LR/BACKBONE_LR)} | 可训/总 = {n_bb+n_fc:,}/{n_total:,}")
    print(f"优化 : SGD(momentum=0.9, nesterov=True) | weight_decay={WEIGHT_DECAY} | batch {BATCH_SIZE} | {epochs} epoch")
    print(f"调度 : inverse-decay lr=base·(1+{int(LR_DECAY_ALPHA)}·p)^(-{LR_DECAY_BETA})，p=global_step/total_steps 归一化[0,1] "
          f"| steps/epoch={steps_per_epoch} total_steps={total_steps} | 末端衰减因子≈{(1.0+LR_DECAY_ALPHA)**(-LR_DECAY_BETA):.4f}")

    # --- 训练循环（每 step 调 scheduler；每 epoch 记 train/val loss & acc）---
    print("-" * 78)
    history = []
    lr_trace = []                                   # (global_step, p, lr_backbone, lr_fc)：抽样前几步 + 末步，验 lr 衰减
    churn_argmax_rows = []     # spec §3.4(e)：逐 epoch (epoch, argmax 长100 向量), epoch1 起
    a_prev = None              # 上一 epoch churn argmax 向量；epoch1 时 None -> churn 标量记 null
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()                       # default BN：train 模式更新 BN running stats
        run_loss, n_correct, n_seen = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # 抽样记录本 step 实际生效的 lr（optimizer.step 前 = base_lr·(1+α·p)^(−β)，p=global_step/total_steps）
            if global_step < 3 or global_step == total_steps - 1:
                lr_trace.append((global_step, global_step / total_steps,
                                 optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"]))
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()                # inverse-decay：每个 optimizer.step() 后步进 1（LambdaLR 内 last_epoch=global_step）
            global_step += 1
            run_loss += loss.item() * x.size(0)
            n_correct += (logits.argmax(1) == y).sum().item()
            n_seen += x.size(0)
        tr_loss, tr_acc = run_loss / n_seen, n_correct / n_seen
        val_loss, val_acc = eval_loss_acc(model, val_loader, criterion, device)   # 源留出折
        # 真实目标 = KLSG 无标签，绝不可读其标签算 F1；过拟合判读以 arm2 源 train/val 为准。
        # churn（spec §3.4 a；紧邻 val/epoch 收尾）：固定100子集 argmax 相邻 epoch 翻转率 —— label-free, 只用模型预测。
        a_e = churn_predict(model, churn_loader, device)                       # 决定论 eval 旁路, 零 RNG/零 buffer 写
        churn_e = float((a_e != a_prev).mean()) if a_prev is not None else None  # epoch1: a_prev=None -> null
        a_prev = a_e
        churn_argmax_rows.append((epoch, a_e.tolist()))                        # argmax 向量 epoch1 起逐 epoch 存
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": val_loss, "val_acc": val_acc, "churn": churn_e})
        churn_str = "n/a" if churn_e is None else f"{churn_e:.3f}"
        print(f"  epoch {epoch:>3}/{epochs} | tr_loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"|| val_loss {val_loss:.4f} acc {val_acc:.4f} | churn {churn_str}")

        # spec §3.2/块④：命名统一 —— 末K 与 --save-every-epoch 都用单一 {tag}_epoch*（去 source_only_ 前缀、单点存盘同名幂等不双写，末K ⊂ every）
        if args.save_every_epoch or epoch > epochs - SAVE_LAST_K:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "train_loss": tr_loss, "val_loss": val_loss,
                        "train_acc": tr_acc, "val_acc": val_acc,
                        "global_step": global_step},
                       model_dir / f"{tag}_epoch{epoch:03d}.pth")

    # --- scheduler sanity 检查：训练结束 p≈1（global_step=total_steps），各 group 实际 lr 应
    #     ≈ base_lr × (1+α)^(−β) = base_lr × 11^(−0.75) ≈ base_lr × 0.166（核验归一化 p + LambdaLR 缩放正确）---
    print("-" * 78)
    print("★ inverse-decay lr 调度 sanity 检查 ★")
    expected_factor = (1.0 + LR_DECAY_ALPHA) ** (-LR_DECAY_BETA)     # 11^(-0.75) ≈ 0.16548
    base_lrs = list(scheduler.base_lrs)                              # [backbone_base, fc_base]，构造时锁定
    group_names = ["backbone", "fc"]
    print(f"  抽样 lr 轨迹 (step, p=step/total_steps, lr_backbone, lr_fc):")
    for gs, p, lb, lf in lr_trace:
        print(f"    step {gs:>6} | p={p:6.4f} | backbone {lb:.6e} | fc {lf:.6e}")
    print(f"  期望末端衰减因子 (1+{int(LR_DECAY_ALPHA)})^(-{LR_DECAY_BETA}) = {expected_factor:.6f}")
    lr_ok = True
    for i, gname in enumerate(group_names):
        cur_lr = optimizer.param_groups[i]["lr"]
        exp_lr = base_lrs[i] * expected_factor
        rel_err = abs(cur_lr - exp_lr) / max(exp_lr, 1e-12)
        flag = "OK" if rel_err < 1e-3 else "MISMATCH"
        print(f"  group[{i}] {gname:>8}: base={base_lrs[i]:.3e} | 末端 lr={cur_lr:.6e} "
              f"| 期望={exp_lr:.6e} | 相对误差={rel_err:.2e} [{flag}]")
        lr_ok = lr_ok and (rel_err < 1e-3)
    assert lr_ok, "scheduler 末端 lr 与 base_lr x (1+a)^(-b) 不符: 归一化 p / LambdaLR 缩放实现有误"
    print(f"  >>> 归一化 inverse-decay 调度核验 : {'通过 PASS' if lr_ok else '失败 FAIL'}")

    # --- 过拟合分析（judge on arm2 源 train/val）---
    tl = [h["train_loss"] for h in history]
    vl = [h["val_loss"] for h in history]
    ta = [h["train_acc"] for h in history]
    vaa = [h["val_acc"] for h in history]
    has_nan = any(np.isnan(v) or np.isinf(v) for v in tl + vl)
    best_val_epoch = int(np.argmin(vl)) + 1
    best_val_loss = float(np.min(vl))
    final_val_loss = float(vl[-1])
    final_train_loss = float(tl[-1])
    val_rebound = final_val_loss - best_val_loss                # >0 = val_loss 已从底回升
    train_still_falling = final_train_loss < tl[best_val_epoch - 1]   # 触底后 train 仍降?
    # 过拟合 = val_loss 明显早于末段触底 + 之后回升 + train 仍降（背离）
    overfit = (best_val_epoch < epochs - 2) and (val_rebound > 1e-3) and train_still_falling
    final_gen_gap = final_val_loss - final_train_loss

    print("-" * 78)
    print("★ 过拟合分析 (arm2 源 train vs 源留出折 val) ★")
    print(f"  val_loss 最低点 : epoch {best_val_epoch}/{epochs}  (val_loss={best_val_loss:.4f})")
    print(f"  末 epoch        : train_loss={final_train_loss:.4f}  val_loss={final_val_loss:.4f}  "
          f"(末段泛化间隙 val-train = {final_gen_gap:+.4f})")
    print(f"  val_loss 从底回升量 = {val_rebound:+.4f} | 触底后 train_loss 仍降 = {train_still_falling}")
    print(f"  >>> 过拟合判据(val_loss 早触底+回升且 train 仍降) : {'是 OVERFIT' if overfit else '否 / 不显著'}")
    print(f"  train_acc 末 = {ta[-1]:.4f} | val_acc 末 = {vaa[-1]:.4f} | val_acc 峰 = {max(vaa):.4f}@epoch{int(np.argmax(vaa))+1}")
    val_acc_peak_ep = int(np.argmax(vaa)) + 1
    if best_val_loss < 0.05 and val_acc_peak_ep <= 3:
        print(f"  [!] 注意: 源留出折过易(val_acc 第 {val_acc_peak_ep} epoch 即达峰、val_loss 始终≈0) -> "
              f"它对'整网过拟合'基本不敏感; 真实 sim->real 退化需在目标域上看(本 source-only 不读目标标签, 留 run2+ 对抗/eval_adabn)。")

    # --- 最终评估：arm2 源留出折（source-only 不评估目标域）---
    print("-" * 78)
    print("最终评估 (末 epoch 模型):")
    va_true, va_pred = evaluate_preds(model, val_loader, device)
    va_metrics = report_metrics(va_true, va_pred)
    print_metrics_block("arm2 源留出折 val", va_metrics)

    # --- 落盘 ---
    out = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run_tag": tag,
        "config": {
            "seed": args.seed, "fold": args.fold, "epochs": epochs, "batch_size": BATCH_SIZE,
            "model": "resnet18_imagenet_full_finetune", "bn": "default",
            "optimizer": {"type": "SGD", "momentum": 0.9, "nesterov": True},
            "layered_lr": {"fc_lr": FC_LR, "backbone_lr": BACKBONE_LR, "ratio": "fc=10x backbone"},
            "weight_decay": WEIGHT_DECAY,
            "weight_decay_note": "SGD default starting point; parameter-layer, station-2 calibration",
            "loss": "CB-WCE", "cb_beta": CB_BETA,
            "cb_weights": {"airplane": float(cb_w[0].item()), "ship": float(cb_w[1].item()),
                           "ratio_airplane_to_ship": float(ratio)},
            "augmentation": {
                "RandomResizedCrop": "224, scale=(0.8,1.0)", "RandomHorizontalFlip": "p=0.5",
                "RandomRotation": "+-15deg", "vertical_flip": "OFF",
                "ColorJitter": "brightness=0.15, contrast=0.15",
                "SpeckleNoise": f"multiplicative, sigma={SPECKLE_SIGMA}",
                "applied_to": "arm2 train only; val/eval no aug"},
            "scheduler": {
                "type": "inverse_decay_LambdaLR",
                "formula": "lr = base_lr * (1 + alpha*p)^(-beta), p = global_step/total_steps in [0,1]",
                "alpha": LR_DECAY_ALPHA, "beta": LR_DECAY_BETA,
                "total_steps": total_steps, "steps_per_epoch": steps_per_epoch,
                "final_decay_factor": float((1.0 + LR_DECAY_ALPHA) ** (-LR_DECAY_BETA)),
                "note": "DANN/CDAN-standard; normalized p so few (~600) steps still decay; "
                        "same optimization regime as run2 CDAN.",
            },
            "save_last_k": SAVE_LAST_K,
        },
        "data": {"train": f"arm2/fold{args.fold}/train (aug)", "source_fold": args.fold, "n_train": len(train_items), "train_counts": dict(tr_cnt),
                 "val": f"arm2/fold{args.fold}/test (source hold-out, no aug)", "n_val": len(val_items), "val_counts": dict(va_cnt),
                 "target_note": "source-only: NO target eval here. KLSG target is unlabeled and its labels are never read. Target transfer evaluation lives in the separate evaluation flow."},
        "train_history": history,
        "overfitting_analysis": {
            "best_val_loss_epoch": best_val_epoch, "best_val_loss": best_val_loss,
            "final_train_loss": final_train_loss, "final_val_loss": final_val_loss,
            "final_generalization_gap_val_minus_train": final_gen_gap,
            "val_loss_rebound_from_min": val_rebound,
            "train_still_falling_after_val_min": bool(train_still_falling),
            "overfit_verdict": bool(overfit),
            "val_acc_peak": float(max(vaa)), "val_acc_peak_epoch": int(np.argmax(vaa)) + 1,
            "note": "source hold-out fold is same-distribution as train and trivially easy "
                    "(val_acc=1.0 by epoch 2, val_loss approx 0) -> insensitive probe for full-net "
                    "overfitting; target-domain sim->real transfer is evaluated in run2+ / eval_adabn, not here.",
        },
        "eval_metrics_arm2_val": va_metrics,
        "has_nan_inf": bool(has_nan),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(curve_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "churn"])
        for h in history:
            churn_cell = "" if h["churn"] is None else f"{h['churn']:.6f}"    # epoch1 null -> 空
            w.writerow([h["epoch"], f"{h['train_loss']:.6f}", f"{h['train_acc']:.6f}",
                        f"{h['val_loss']:.6f}", f"{h['val_acc']:.6f}", churn_cell])
    # spec §3.4(e)：churn argmax 矩阵（首列 epoch + 100 列 argmax，epoch1 起逐 epoch 一行）。
    #   决定论 int 矩阵 -> 字节稳定（run2x byte-compare 命根 §4.4）；churn 标量可由相邻行复算核对（§4.6）。
    with open(churn_argmax_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        n_cols = len(churn_argmax_rows[0][1]) if churn_argmax_rows else _klsg_blind.CHURN_N_EXPECTED
        w.writerow(["epoch"] + [f"a{i}" for i in range(n_cols)])
        for ep, vec in churn_argmax_rows:
            w.writerow([ep] + [int(v) for v in vec])

    print("\n" + "=" * 78)
    print(f"指标/历史 -> {metrics_path}")
    print(f"曲线 CSV  -> {curve_path}")
    print(f"churn argmax -> {churn_argmax_path}")
    print(f"checkpoint(末 {SAVE_LAST_K}) -> {model_dir}")
    print(f"NaN/Inf : {has_nan}")
    print("=" * 78)
    return 1 if has_nan else 0


if __name__ == "__main__":
    sys.exit(main())
