#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_mcc_sgd.py -- 【候补·[E] 事后补充实验】 **MCC × default BN**（SGD 底座）
[SSS sim->real 主实验 B · 事后补充方法 · 非对抗路线代表]

================================  这是什么 / 身份 ================================
**MCC**（Minimum Class Confusion, Jin et al. ECCV 2020, "Minimum Class Confusion for
Versatile Domain Adaptation", arXiv:1912.03699）——**非对抗** DA：不用判别器/GRL，只在
**目标 batch 的预测**上加一个「最小化类混淆」损失（压低类混淆矩阵的非对角项），鼓励目标
预测类别可分。源侧仍 CB-WCE 监督。

★ **身份 = 候补 / [E] 事后补充**：受统一实验协议约束
  （①同冻结协议 ②论文标"事后补充·[E]" ③不替换 M1/M2/M3 主对比）。
  MCC 当初落选主表理由（plan §3.4.7）：**K=2 下类混淆机制退化为"置信催促"；88/12 失衡下有把
  模糊样本推向多数类(ship)的少数类风险（前期方法评估结论）**。→ 故 MCC 在本数据
  **很可能过预测 ship / 坍缩**；若如此 = 该落选风险的实证，**照实记、不 babysit、不剔**（同 frozen
  公平原则）。若发生坍缩，则按统一的逐epoch预测分布规则标记为UNHEALTHY。

================================  MCC 损失（忠实复刻 ECCV2020·逐项注明）================================
给定目标 batch logits Z ∈ R^{B×C}（C=2）：
  ① 温度软化 softmax：Ŷ_ij = softmax(Z_i / T)_j        （T=温度，文献默认 2.5）
  ② 熵不确定性重加权（聚焦自信样本）：
       H(ŷ_i) = -Σ_j Ŷ_ij log Ŷ_ij
       W_ii = B·(1+exp(-H_i)) / Σ_{i'}(1+exp(-H_{i'}))   （W 对角·归一化到 ΣW=B；自信样本权重↑）
  ③ 类混淆矩阵：C = Ŷ^T diag(W) Ŷ ∈ R^{C×C}
  ④ 类别归一化（行归一）：C̃_jj' = C_jj' / Σ_{j''} C_jj''
  ⑤ MCC 损失：L_mcc = (1/C) Σ_j Σ_{j'≠j} |C̃_jj'|        （每类非对角和的均值）
总损失：L = L_CE(源·CB-WCE) + μ·L_mcc(目标)              （μ=trade-off，文献默认常数 1.0）

★ **method-intrinsic 旋钮取文献默认·登记**（严格 UDA·无目标标签可调）：温度 T=2.5、权重 μ=1.0（常数不 ramp）。
★ 数值守卫：softmax + log 用 eps 防 log(0)→NaN；行归一化分母加 eps 防除零（某类预测为空时）。

================================  实现 provenance（派生·对拍门）================================
底座/数据/CB-WCE/inverse-decay/churn-health-ckpt 落盘/决定论护栏/红线盲化 **复用自已封验的
`src/run2_s5_cdan/train_cdan_sgd.py`**（剥掉判别器/GRL/条件化对抗，换上 mcc_loss）。模型骨架
= CDANResNet18 逐字一致（feature_layers + fc·无判别器）→ ckpt 的 model_state_dict 键与
`eval_pipeline.py` ARCH_REGISTRY["cdan"] 匹配 → **MCC ckpt 用 `--arch cdan` 直接评估**。
MCC损失按论文公式实现；通过解析已知case的单元验并与论文公式逐项核对。

================================  数据 / 标签边界 / 合法性（与主矩阵一致）================================
源 = arm2/fold/train（带标签，增强）→ 目标 = KLSG 全 553（无标签，同款增强，transductive）。
目标标签绝不进任何损失（MCC 只用目标【预测】、不用目标真值）；KLSG 经 _klsg_blind 盲化
（label 位毒丸 FORBIDDEN_LABEL，绝不读真实标签/绝不从 filepath 反推）。绝不用真实域表现选模/早停/定参。
==========================================================================
"""

import os
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
from PIL import Image
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support, confusion_matrix, accuracy_score,
)

# [红线护栏] KLSG 盲化入口；本文件在 src/extra/mcc/，src/ 在 parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _klsg_blind
import _run_safety

# ============================================================ 配置
PROJECT_ROOT = Path(__file__).resolve().parents[3]

SPLITS_CSV = PROJECT_ROOT / "data" / "split" / "splits.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "extra" / "mcc"

CLASSES = ["airplane", "ship"]
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

SEED = 0
EPOCHS_DEFAULT = 40                        # 站5 锁定 N=40（候补同冻结协议）
QUICK_EPOCHS = 12
BATCH_SIZE = 64
FC_LR = 1e-2
BACKBONE_LR = 1e-3
WEIGHT_DECAY = 5e-4
LR_DECAY_ALPHA = 10.0
LR_DECAY_BETA = 0.75
CB_BETA = 0.999
SPECKLE_SIGMA = 0.1
SAVE_LAST_K = 5
MCC_TEMP = 2.5                             # ★MCC 温度 T（文献默认·登记）
MCC_MU = 1.0                               # ★MCC trade-off μ（文献默认·常数不 ramp·登记）
_EPS = 1e-5                                # 数值守卫（softmax log / 行归一分母）；对齐规范 MCC + repo Entropy() 的 1e-5


# ============================================================ 数据（与 CDAN/DANN 母版逐字一致）
def read_split_rows(csv_path, domain=None, fold=None, split=None):
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
    def __init__(self, items, transform):
        self.transform = transform
        self.labels = [lab for _, lab in items]
        self.images = []
        for path, _ in items:
            with Image.open(path) as im:
                self.images.append(im.convert("RGB").copy())
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        return self.transform(self.images[i]), self.labels[i]


class SpeckleNoise:
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
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        SpeckleNoise(sigma=SPECKLE_SIGMA),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_eval_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ============================================================ CB-WCE（底座沿用；非 IW）
def cb_wce_weights(class_counts, beta=CB_BETA, num_classes=len(CLASSES), device=None):
    eff_num = [1.0 - (beta ** n) for n in class_counts]
    w = [(1.0 - beta) / en for en in eff_num]
    s = sum(w)
    w = [wi / s * num_classes for wi in w]
    return torch.tensor(w, dtype=torch.float32, device=device)


# ============================================================ ★MCC 损失（忠实复刻 ECCV2020）
def mcc_loss(target_logits, temperature=MCC_TEMP):
    """Minimum Class Confusion 损失（Jin et al. ECCV2020）。输入目标 batch logits [B,C]。
      ① 温度软化 softmax：Ŷ = softmax(Z/T)            （T 越大越平滑）
      ② 熵重加权 W_ii = B(1+e^{-H_i})/Σ(1+e^{-H_{i'}})  （对角·ΣW=B·自信样本权重高）
      ③ 类混淆 C = Ŷ^T diag(W) Ŷ  ∈ [C,C]
      ④ 行归一 C̃_jj' = C_jj' / Σ_j'' C_jj''
      ⑤ L = (1/C) Σ_j Σ_{j'≠j} |C̃_jj'|               （每类非对角和的均值）
    数值守卫：log(Ŷ+eps) 防 NaN；行归一分母 + eps 防除零。返回标量 loss。"""
    B, C = target_logits.shape
    Y = F.softmax(target_logits / temperature, dim=1)            # ① [B,C]
    # ② 熵 + 重加权
    H = -torch.sum(Y * torch.log(Y + _EPS), dim=1)               # [B] 逐样本熵
    W = (1.0 + torch.exp(-H))                                     # [B]
    W = W / (torch.sum(W) + _EPS) * B                            # 归一化到 ΣW=B
    # ★ detach 熵权重：规范 MCC（tllib MinimumClassConfusionLoss）把不确定性重加权当【常数系数】
    #   （entropy(predictions).detach()）——梯度只经类混淆矩阵的 Ŷ 回流，不经熵权重。前向值不变、梯度路径忠实。
    W = W.detach()
    # ③ 类混淆矩阵 C = Ŷ^T diag(W) Ŷ（Ŷ 带梯度·W 常数）
    Yw = Y * W.unsqueeze(1)                                       # [B,C] 行乘权重 = diag(W) Ŷ
    Cmat = torch.mm(Y.t(), Yw)                                   # [C,C]
    # ④ 行归一化
    Cmat = Cmat / (torch.sum(Cmat, dim=1, keepdim=True) + _EPS)  # 每行和=1
    # ⑤ 非对角和 / C
    off_diag = torch.sum(torch.abs(Cmat)) - torch.sum(torch.abs(torch.diag(Cmat)))
    return off_diag / C


# ============================================================ 模型（结构 = CDANResNet18 逐字一致 → eval --arch cdan 兼容）
class MCCResNet18(nn.Module):
    """ResNet18(ImageNet) 整网微调；暴露 512 维 penultimate + fc logits。
    ★ 结构（feature_layers + fc·子模块顺序）与 train_cdan_sgd.CDANResNet18 逐字一致
      → state_dict 键匹配 → ckpt 用 eval_pipeline --arch cdan 直接评估。**MCC 无判别器**。"""
    def __init__(self, num_classes=len(CLASSES)):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.in_features = base.fc.in_features
        self.feature_layers = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4, base.avgpool)
        self.fc = nn.Linear(self.in_features, num_classes)
        for p in self.parameters():
            p.requires_grad = True
    def forward(self, x):
        f = self.feature_layers(x)
        f = f.view(f.size(0), -1)
        y = self.fc(f)
        return f, y
    def output_num(self):
        return self.in_features


def make_mcc_optimizer(model):
    """分层 lr 的 SGD（2 组·MCC 无判别器）：group0=backbone(1e-3) / group1=fc(1e-2=10×)。"""
    fc_params, backbone_params = [], []
    for name, p in model.named_parameters():
        (fc_params if name.startswith("fc.") else backbone_params).append(p)
    optimizer = torch.optim.SGD(
        [{"params": backbone_params, "lr": BACKBONE_LR},
         {"params": fc_params, "lr": FC_LR}],
        momentum=0.9, nesterov=True, weight_decay=WEIGHT_DECAY,
    )
    return (optimizer, sum(p.numel() for p in backbone_params),
            sum(p.numel() for p in fc_params), backbone_params)


def make_scheduler(optimizer, total_steps):
    total_steps = max(int(total_steps), 1)
    lr_lambda = lambda step: (1.0 + LR_DECAY_ALPHA * (min(step, total_steps) / total_steps)) ** (-LR_DECAY_BETA)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ============================================================ 健康/churn 探测（底座沿用·label-free）
def target_pseudolabel_health(model, loader, device, num_classes=len(CLASSES)):
    model.eval()
    hist = [0] * num_classes
    ent_sum, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            _, logits = model(x)
            prob = F.softmax(logits, dim=1)
            for c in prob.argmax(1).cpu().tolist():
                hist[c] += 1
            ent_sum += float((-(prob * prob.clamp_min(1e-12).log()).sum(1)).sum().item())
            n += int(x.size(0))
    return hist, ent_sum / max(n, 1)


def churn_predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            _, logits = model(x)
            preds.extend(logits.argmax(1).cpu().tolist())
    model.train()
    return np.array(preds, dtype=np.int64)


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ============================================================ 单元验（smoke 前先单元验·验 MCC 公式）
def run_unit_tests(device):
    """验 MCC 损失公式（解析已知 case）：
      ① 完美自信 + 类可分（半 batch 全 class0、半 batch 全 class1·极端 logit）→ 混淆矩阵近对角 → L_mcc≈0。
      ② 全混淆（uniform 预测·logit 全 0）→ 行归一后 [[.5,.5],[.5,.5]] → 非对角=0.5+0.5、/C=2 → L_mcc≈0.5。
      ③ 形状/可微：C 矩阵 [C,C]、loss 标量、对 logits 有梯度、无 NaN。"""
    print("=" * 78)
    print("★ 单元验（smoke 前 · MCC 公式） ★")
    ok = True

    # ① 完美自信可分：B=8, 前4全class0、后4全class1，极端 logit（±20）
    z1 = torch.zeros(8, 2, device=device)
    z1[:4, 0] = 20.0; z1[:4, 1] = -20.0
    z1[4:, 1] = 20.0; z1[4:, 0] = -20.0
    L1 = mcc_loss(z1, temperature=MCC_TEMP).item()
    t1 = (L1 < 1e-3)
    print(f"  [①可分 ] 完美自信+类可分 -> L_mcc={L1:.6f} (期望≈0, <1e-3) -> {'PASS' if t1 else 'FAIL'}")
    ok = ok and t1

    # ② 全混淆 uniform：logit 全 0 → Ŷ=[.5,.5] → C 行归一=[[.5,.5],[.5,.5]] → 非对角(0.5+0.5)/2=0.5
    z2 = torch.zeros(8, 2, device=device)
    L2 = mcc_loss(z2, temperature=MCC_TEMP).item()
    t2 = (abs(L2 - 0.5) < 1e-3)
    print(f"  [②混淆 ] uniform 预测 -> L_mcc={L2:.6f} (期望≈0.5, |·-0.5|<1e-3) -> {'PASS' if t2 else 'FAIL'}")
    ok = ok and t2

    # ③ 形状/可微/无 NaN：随机 logit 带梯度
    z3 = torch.randn(16, 2, device=device, requires_grad=True)
    L3 = mcc_loss(z3, temperature=MCC_TEMP)
    L3.backward()
    grad_ok = (z3.grad is not None) and bool(torch.isfinite(z3.grad).all().item())
    scalar_ok = (L3.dim() == 0) and bool(torch.isfinite(L3).item())
    # 手算交叉核 ②：解析期望
    t3 = grad_ok and scalar_ok
    print(f"  [③可微 ] 随机 logit -> loss 标量={scalar_ok} 梯度有限={grad_ok} (L={L3.item():.4f}) -> {'PASS' if t3 else 'FAIL'}")
    ok = ok and t3

    # ④ 单调性 sanity：混淆程度↑ → L_mcc↑（可分 < 中间 < uniform）
    z_mid = torch.zeros(8, 2, device=device)
    z_mid[:4, 0] = 2.0; z_mid[4:, 1] = 2.0       # 温和自信
    Lmid = mcc_loss(z_mid, temperature=MCC_TEMP).item()
    t4 = (L1 < Lmid < L2 + 1e-6)
    print(f"  [④单调 ] L(可分)={L1:.4f} < L(温和)={Lmid:.4f} < L(uniform)={L2:.4f} -> {'PASS' if t4 else 'FAIL'}")
    ok = ok and t4

    print(f"  >>> 单元验总判: {'全部 PASS' if ok else '存在 FAIL'}")
    print("=" * 78)
    assert ok, "单元验失败：MCC 损失公式有误（可分/混淆/可微/单调），不进 smoke"
    return ok


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None, help="epoch 数（默认 full=40；--quick 时 12）")
    ap.add_argument("--quick", action="store_true", help="smoke（默认 12 epoch、产物名 mcc_smoke_*）")
    ap.add_argument("--unittest", action="store_true", help="只跑单元验后退出（不训练）")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--fold", type=int, default=0, help="源 arm2 折号；KLSG 目标全 553 transductive、fold 无关")
    ap.add_argument("--temperature", type=float, default=MCC_TEMP, help="MCC 温度 T（文献默认 2.5）")
    ap.add_argument("--mu", type=float, default=MCC_MU, help="MCC trade-off μ（文献默认 1.0·常数）")
    ap.add_argument("--save-every-epoch", action="store_true", help="逐 epoch 存全 dict ckpt；默认只存末 SAVE_LAST_K")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"), "CUBLAS_WORKSPACE_CONFIG 未设(须在 import torch 前内联)"

    run_unit_tests(device)
    if args.unittest:
        print("仅单元验模式（--unittest），退出。")
        return 0
    # ★ seed 修复：run_unit_tests 消费 torch RNG（randn）→ 重设回 args.seed
    set_seed(args.seed)

    epochs = (QUICK_EPOCHS if args.quick else EPOCHS_DEFAULT) if args.epochs is None else args.epochs
    tag = "mcc_smoke" if args.quick else "mcc_full"

    model_dir = RESULTS_DIR / f"{tag}_s{args.seed}_f{args.fold}"
    metrics_path = model_dir / f"{tag}_metrics.json"
    curve_path = model_dir / f"{tag}_curve.csv"
    churn_argmax_path = model_dir / f"{tag}_churn_argmax.csv"
    _run_safety.prepare_empty_run_dirs(model_dir)

    print("=" * 78)
    print(f"【候补·[E]】 **MCC × default BN** (SGD 底座) ({tag}) : arm2 -> KLSG UDA  | T={args.temperature} μ={args.mu}")
    print("=" * 78)
    print(f"项目根 : {PROJECT_ROOT}")
    print(f"设备   : {device} | torch {torch.__version__} | "
          f"GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    src_items = read_split_rows(SPLITS_CSV, domain="arm2", fold=args.fold, split="train")
    tgt_items = _klsg_blind.read_klsg_unlabeled(PROJECT_ROOT)
    src_cnt = Counter(CLASSES[l] for _, l in src_items)
    print(f"\n源 arm2/fold{args.fold}/train (带标签,增强)     : {len(src_items)} (airplane={src_cnt['airplane']}, ship={src_cnt['ship']})")
    print(f"目标 KLSG 全 {len(tgt_items)} (无标签,同款增强,transductive) : sha {_klsg_blind.KLSG_MANIFEST_SHA256[:8]} "
          f"[label 盲化为毒丸, 绝不入损失/绝不读真实类别; fold 无关]")
    assert len(src_items) and len(tgt_items), "数据为空"
    assert len(tgt_items) == _klsg_blind.KLSG_N_EXPECTED, f"KLSG 目标张数={len(tgt_items)} != 553"

    train_tfm, eval_tfm = build_train_transform(), build_eval_transform()
    src_loader = DataLoader(ImageListDataset(src_items, train_tfm), batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=0, drop_last=True)
    tgt_loader = DataLoader(_klsg_blind.UnlabeledImageDataset(tgt_items, train_tfm), batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=0, drop_last=True, collate_fn=_klsg_blind.collate_unlabeled)
    tgt_probe_loader = DataLoader(_klsg_blind.UnlabeledImageDataset(tgt_items, eval_tfm), batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=0, collate_fn=_klsg_blind.collate_unlabeled)
    churn_items = _klsg_blind.read_klsg_churn_subset(PROJECT_ROOT)
    assert len(churn_items) == _klsg_blind.CHURN_N_EXPECTED, f"churn 子集张数={len(churn_items)} != {_klsg_blind.CHURN_N_EXPECTED}"
    churn_loader = DataLoader(_klsg_blind.UnlabeledImageDataset(churn_items, eval_tfm), batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, collate_fn=_klsg_blind.collate_unlabeled)
    print(f"churn 固定子集 (label-free, sha {_klsg_blind.KLSG_CHURN100_SHA256[:8]}) : {len(churn_items)} 张")
    assert len(src_loader) >= 1 and len(tgt_loader) >= 1, "drop_last=True 后 loader 为空"

    counts = [src_cnt[c] for c in CLASSES]
    assert counts == [72, 320], f"CB-WCE 护栏：arm2 各折 train 类频应恒为 [72,320]，实得 {counts}"
    cb_w = cb_wce_weights(counts, beta=CB_BETA, device=device)
    ratio = (cb_w[0] / cb_w[1]).item()
    print(f"\nCB-WCE (β={CB_BETA}) 权重 (从 arm2 train 类频 {counts}): "
          f"airplane={cb_w[0].item():.4f}, ship={cb_w[1].item():.4f} (airplane:ship = {ratio:.2f}:1)  [非 IW]")
    criterion = nn.CrossEntropyLoss(weight=cb_w)

    steps_per_epoch = len(src_loader)
    total_steps = epochs * steps_per_epoch

    model = MCCResNet18().to(device)
    optimizer, n_bb, n_fc, backbone_params = make_mcc_optimizer(model)
    scheduler = make_scheduler(optimizer, total_steps)
    n_total = sum(p.numel() for p in model.parameters())

    print(f"\n模型 : ResNet18(ImageNet) 全解冻整网微调 | default BN | penultimate=512 维 | **无判别器(MCC 非对抗)**")
    print(f"分层 lr : backbone={BACKBONE_LR}({n_bb:,}) | fc={FC_LR}({n_fc:,}) | 可训/总={n_bb+n_fc:,}/{n_total:,}")
    print(f"优化 : SGD(m=0.9,nesterov) wd={WEIGHT_DECAY} | batch {BATCH_SIZE}(源)+{BATCH_SIZE}(目标) | {epochs} epoch")
    print(f"调度 : inverse-decay lr=base·(1+{int(LR_DECAY_ALPHA)}·p)^(-{LR_DECAY_BETA}) | steps/epoch={steps_per_epoch} total_steps={total_steps}")
    print(f"MCC  : T={args.temperature}(温度) μ={args.mu}(trade-off,常数) | total_loss = cls_loss(CB-WCE源) + μ·L_mcc(目标·非对抗)")

    print("-" * 78)
    print("smoke 每 epoch 诊断: cls_loss/acc | L_mcc(目标·健康>0·趋0=可分/或坍缩) | tgtPred直方 | churn | NaN")
    print("-" * 78)
    history = []
    churn_argmax_rows = []
    a_prev = None
    global_step = 0
    any_nan = False
    target_iter = iter(tgt_loader)
    for epoch in range(1, epochs + 1):
        model.train()
        ep_cls, ep_mcc, ep_gn = 0.0, 0.0, 0.0
        ep_correct, ep_seen, ep_steps = 0, 0, 0
        for x_s, y_s in src_loader:
            try:
                x_t, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(tgt_loader)
                x_t, _ = next(target_iter)
            x_s, y_s = x_s.to(device, non_blocking=True), y_s.to(device, non_blocking=True)
            x_t = x_t.to(device, non_blocking=True)

            _, o_s = model(x_s)                       # 源 logits（监督）
            _, o_t = model(x_t)                       # 目标 logits（MCC·目标标签绝不用）

            cls_loss = criterion(o_s, y_s)            # 源 CB-WCE
            mcc_l = mcc_loss(o_t, temperature=args.temperature)   # 目标 MCC（只用预测）
            total_loss = cls_loss + args.mu * mcc_l

            optimizer.zero_grad()
            total_loss.backward()
            bb_sq = 0.0
            for p in backbone_params:
                if p.grad is not None:
                    bb_sq += float(p.grad.detach().pow(2).sum().item())
            bb_grad_norm = bb_sq ** 0.5
            optimizer.step()
            scheduler.step()
            global_step += 1

            ep_cls += float(cls_loss.item())
            ep_mcc += float(mcc_l.item())
            ep_gn += bb_grad_norm
            ep_correct += int((o_s.argmax(1) == y_s).sum().item())
            ep_seen += int(y_s.size(0))
            ep_steps += 1
            if not np.isfinite(float(total_loss.item())):
                any_nan = True

        cls_m = ep_cls / ep_steps
        mcc_m = ep_mcc / ep_steps
        gn_m = ep_gn / ep_steps
        src_acc = ep_correct / max(ep_seen, 1)
        ep_nan = (not np.isfinite(cls_m)) or (not np.isfinite(mcc_m)) or any_nan
        any_nan = any_nan or ep_nan

        tgt_hist, tgt_ent = target_pseudolabel_health(model, tgt_probe_loader, device)
        tgt_collapsed = (max(tgt_hist) == sum(tgt_hist))
        a_e = churn_predict(model, churn_loader, device)
        churn_e = float((a_e != a_prev).mean()) if a_prev is not None else None
        a_prev = a_e
        churn_argmax_rows.append((epoch, a_e.tolist()))
        model.train()

        history.append({"epoch": epoch, "global_step": global_step,
                        "src_cls_loss_mean": cls_m, "mcc_loss_mean": mcc_m, "src_acc": src_acc,
                        "bb_grad_norm_mean": gn_m, "churn": churn_e,
                        "tgt_pred_hist": list(tgt_hist), "tgt_pred_entropy_mean": tgt_ent,
                        "tgt_pred_collapsed": bool(tgt_collapsed), "nan": bool(ep_nan)})
        churn_str = "n/a" if churn_e is None else f"{churn_e:.3f}"
        print(f"  ep {epoch:>3}/{epochs} | cls {cls_m:.4f} acc {src_acc:.4f} | L_mcc {mcc_m:.4f} "
              f"| bbGrad {gn_m:.3e} | churn {churn_str} | tgtPred {tgt_hist} H{tgt_ent:.3f}"
              f"{' COLLAPSE!' if tgt_collapsed else ''}{' | NaN!' if ep_nan else ''}")
        if args.save_every_epoch or epoch > epochs - SAVE_LAST_K:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "global_step": global_step},
                       model_dir / f"{tag}_epoch{epoch:03d}.pth")

    # --- lr inverse-decay sanity ---
    print("-" * 78)
    print("★ lr inverse-decay sanity（末端 factor = (1+α)^(-β)）★")
    expected_factor = (1.0 + LR_DECAY_ALPHA) ** (-LR_DECAY_BETA)
    base_lrs = list(scheduler.base_lrs)
    lr_ok = True
    for i, gname in enumerate(["backbone", "fc"]):
        cur_lr = optimizer.param_groups[i]["lr"]
        exp_lr = base_lrs[i] * expected_factor
        rel_err = abs(cur_lr - exp_lr) / max(exp_lr, 1e-12)
        print(f"  group[{i}] {gname:>9}: base={base_lrs[i]:.3e} | 末端 lr={cur_lr:.6e} | 期望={exp_lr:.6e} [{'OK' if rel_err<1e-3 else 'MISMATCH'}]")
        lr_ok = lr_ok and (rel_err < 1e-3)
    assert lr_ok, "scheduler 末端 lr 不符"
    print(f"  >>> lr 调度核验: {'PASS' if lr_ok else 'FAIL'}")

    # --- 健康判读（smoke；不读性能、不定参；MCC 坍缩=合法发现，如实记不当 bug）---
    print("-" * 78)
    print("★ 训练健康判读（smoke；**不读性能、不定参**）★")
    cls_series = [h["src_cls_loss_mean"] for h in history]
    mcc_series = [h["mcc_loss_mean"] for h in history]
    acc_series = [h["src_acc"] for h in history]
    src_healthy = acc_series[-1] > 0.5
    tgt_collapse_any = any(h["tgt_pred_collapsed"] for h in history)
    tgt_hist_final = history[-1]["tgt_pred_hist"]
    tgt_ent_final = history[-1]["tgt_pred_entropy_mean"]
    print(f"  ① 源 cls loss   : 首{cls_series[0]:.3f} -> 末{cls_series[-1]:.3f}")
    print(f"  ② MCC loss      : 首{mcc_series[0]:.3f} -> 末{mcc_series[-1]:.3f}  (趋小=目标预测更可分/或坍缩·须配直方)")
    print(f"  ③ 源分类 acc    : 首{acc_series[0]:.3f} -> 末{acc_series[-1]:.3f}  [{'未崩' if src_healthy else '偏低-复核'}]")
    print(f"  ④ NaN/Inf 守卫  : {'发现 NaN/Inf!' if any_nan else '全程无 NaN/Inf'}")
    print(f"  ⑤ backbone 梯度 : 末 {history[-1]['bb_grad_norm_mean']:.3e}")
    print(f"  ⑥ 目标伪标签坍缩: 末 epoch 预测直方 {tgt_hist_final} 熵{tgt_ent_final:.3f}  "
          f"[{'★坍缩过(全预测同类!·MCC 落选风险实证)' if tgt_collapse_any else '未坍缩(两类都有预测)'}]  (label-free)")

    out = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run_tag": tag,
        "method": "MCC (Minimum Class Confusion, Jin et al. ECCV2020, non-adversarial) x default BN, SGD base",
        "identity": "candidate / [E] post-hoc supplementary (station6 S6-8); NOT in confirmatory M1/M2/M3 set",
        "provenance": "scaffold from verified train_cdan_sgd.py (no discriminator/GRL); mcc_loss implemented from "
                      "ECCV2020 formula (temperature softmax -> entropy reweight -> class-confusion C=Y^T W Y -> "
                      "row-normalize -> off-diagonal sum / C); verified by unit tests (analytic cases) + sub review.",
        "config": {
            "seed": args.seed, "fold": args.fold, "epochs": epochs, "batch_size_source": BATCH_SIZE, "batch_size_target": BATCH_SIZE,
            "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
            "model": "resnet18_imagenet_full_finetune", "bn": "default", "penultimate_dim": model.output_num(),
            "adversarial": False, "discriminator": None,
            "mcc": {"temperature": args.temperature, "mu": args.mu, "mu_note": "constant, NOT ramped",
                    "literature_defaults": "T=2.5, mu=1.0 (ECCV2020); strict-UDA: no target labels, no tuning",
                    "loss_formula": "L=(1/C) sum_{j!=j'} |Ctilde_jj'|, Ctilde=row-norm(Y^T diag(W) Y), Y=softmax(Z/T), W=entropy-reweight"},
            "optimizer": {"type": "SGD", "momentum": 0.9, "nesterov": True, "weight_decay": WEIGHT_DECAY},
            "layered_lr": {"backbone_lr": BACKBONE_LR, "fc_lr": FC_LR},
            "scheduler": {"type": "inverse_decay_LambdaLR", "alpha": LR_DECAY_ALPHA, "beta": LR_DECAY_BETA,
                          "formula": "lr=base*(1+alpha*p)^(-beta), p=global_step/total_steps"},
            "source_cls_loss": "CB-WCE", "cb_beta": CB_BETA,
            "cb_weights": {"airplane": float(cb_w[0].item()), "ship": float(cb_w[1].item())},
            "iw_stripped": True, "random_projection": False, "bottleneck": False,
        },
        "data": {"source_train": f"arm2/fold{args.fold}/train (labeled, aug)", "source_fold": args.fold,
                 "n_source": len(src_items), "source_counts": dict(src_cnt),
                 "target": f"klsg full {len(tgt_items)} transductive (sha {_klsg_blind.KLSG_MANIFEST_SHA256[:8]})",
                 "n_target": len(tgt_items),
                 "target_note": "UNLABELED, aug; labels blinded (never read, never inferred); fold-independent transductive; MCC uses target PREDICTIONS only"},
        "train_history": history,
        "smoke_verdict": {
            "src_cls_loss_first": cls_series[0], "src_cls_loss_last": cls_series[-1],
            "mcc_loss_first": mcc_series[0], "mcc_loss_last": mcc_series[-1],
            "src_acc_first": acc_series[0], "src_acc_last": acc_series[-1], "src_cls_healthy": bool(src_healthy),
            "tgt_pred_hist_final": tgt_hist_final, "tgt_pred_entropy_final": tgt_ent_final,
            "tgt_pred_collapsed_any": bool(tgt_collapse_any),
            "has_nan_inf": bool(any_nan), "bb_grad_norm_final": history[-1]["bb_grad_norm_mean"],
            "lr_schedule_ok": bool(lr_ok),
            "note": "MCC training health smoke only; NOT performance/model selection. Collapse (if any) is a legitimate "
                    "finding (K=2/imbalance candidate-rejection risk), recorded as-is, NOT babysat. Target health = "
                    "label-free pseudo-label collapse probe (NO KLSG label read).",
        },
        "has_nan_inf": bool(any_nan),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(curve_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "global_step", "src_cls_loss_mean", "mcc_loss_mean", "src_acc",
                    "bb_grad_norm_mean", "churn",
                    "tgt_pred_airplane", "tgt_pred_ship", "tgt_pred_entropy_mean", "tgt_pred_collapsed", "nan"])
        for h in history:
            churn_cell = "" if h["churn"] is None else f"{h['churn']:.6f}"
            w.writerow([h["epoch"], h["global_step"], f"{h['src_cls_loss_mean']:.6f}", f"{h['mcc_loss_mean']:.6f}",
                        f"{h['src_acc']:.6f}", f"{h['bb_grad_norm_mean']:.6e}", churn_cell,
                        h['tgt_pred_hist'][0], h['tgt_pred_hist'][1], f"{h['tgt_pred_entropy_mean']:.6f}",
                        int(h['tgt_pred_collapsed']), int(h['nan'])])
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
    print(f"checkpoint -> {model_dir}")
    print(f"NaN/Inf : {any_nan}")
    print("=" * 78)
    return 1 if any_nan else 0


if __name__ == "__main__":
    sys.exit(main())
