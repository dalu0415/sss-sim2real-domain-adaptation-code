#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_ceiling_sgd.py -- **#9 supervised ceiling训练器**（SGD底座版）
[SSS sim->real主实验B · 唯一使用目标标签训练的上界]

================================  这是什么 / 与底座 train_source_only_sgd.py 的区别 ================================
ceiling = **全研究唯一合法用 KLSG 真标签【监督训练】的跑法**，用来与 source-only(#1) 比出 headroom
（性能上界锚点）。本文件由同目录 `train_source_only_sgd.py`（SGD 底座）**copy-adapt** 而来——本项目哲学
= 每个 run 自包含完整骨架，故 helper 全部内联拷贝、**绝不 import 底座脚本**。

沿用底座（同 regime、公平对照，逐项不复述细节）：
  - ResNet18 + ImageNet 整网微调（全解冻、default BN）；
  - 分层 lr（fc=1e-2 / backbone=1e-3，比 1:10）；
  - SGD(momentum=0.9, nesterov=True) + weight_decay=5e-4；
  - inverse-decay lr 调度（DANN/CDAN 标准）：lr = base_lr·(1+α·p)^(−β)，α=10、β=0.75，
    p = global_step/total_steps **归一化到 [0,1]**；
  - 增强全套（RandomResizedCrop 0.8~1.0 + HFlip + 旋转±15° + ColorJitter 0.15 + speckle σ=0.1；垂直翻转 OFF）；
  - CB-WCE（Cui 2019 effective-number，β=0.999）；
  - 决定论四件套（CUBLAS_WORKSPACE_CONFIG 内联在 import torch 前 + cudnn deterministic +
    use_deterministic_algorithms(True) + set_seed 四件套 RNG）。

★★ 相对source-only底座的关键差异 ★★
  1. 数据源换 KLSG 真标签：训练 = 该 fold 的 train（经授权核读）、评估 = 该 fold 的 held-out test
     （ceiling 没训过的留出折）。**删掉底座裸的 read_split_rows 函数**——ceiling 一切带标签读取一律
     经 _klsg_blind 授权核，脚本内不留任何裸 csv/read_split_rows 读取口（红线收敛到单门）。
  2. 授权门：CLI `--authorize-klsg-labels`(store_true) + 必填 `--reason`；构造
     auth = _klsg_blind.klsg_label_auth_from_cli(...)。**无授权令牌 → 训练器直接拒跑**
     （ceiling 本质就是要读真标签，无授权无意义）。
  3. CB-WCE 按**折内目标频**动态算：删底座硬编 `assert counts==[72,320]`（那是 arm2 源频），
     从该折训练集真实类频统计（KLSG 飞机少、各折不同，绝不硬编）。
  4. **EPOCHS_DEFAULT=40**（不是底座的 100；从源头灭"忘传 --epochs 跑成 100"整类 bug）。末 K=5。
  5. **全 epoch ckpt 留存**：每个 epoch 都存 → `ceiling/model/s{seed}_f{fold}/epoch{e:03d}.pth`
     （oracle + §5 收敛曲线兜底要吃；底座只存末 K）。
  6. 评估对象换 held-out 折：算 macro-F1 + 对称 airplane/ship P/R/F1 + 混淆矩阵（sklearn, labels=[0,1]）。

★★ 训练器实现选择 ★★
  A. churn **删**：ceiling 是监督训练、有合法 held-out 评估（直接收敛信号），label-free churn 代理冗余。
     〔注：churn_predict 走 eval/shuffle=False/nw=0 = 零 RNG 消费，删它对决定论 RNG 流【无】影响——
       删它的理由是监督held-out信号已足够，不是"少个 RNG 消费点"。〕
  B. 末 K=5 softmax ensemble = 训练器内置"基本级"评估：训练完从盘上**重载末 5 epoch ckpt** →
     held-out上softmax概率平均 → argmax → 基本指标（report_metrics）。完整指标由独立评估包计算。
  C. 底座"arm2 源 train/val 过拟合分析段"**删**：ceiling 评估在 held-out、过拟合天然不虚高；N 已锁、
     ckpt 全留、收敛 sanity 是 §5 事后的活（不据 held-out 早停/选模）。保留 per-epoch held-out
     loss/acc/macro-F1 入 curve（**仅监控/事后 sanity，绝不参与任何选择**），但删"过拟合判据"启发式。

★ 标签授权边界：`--authorize-klsg-labels`与非空`--reason`均必需；每次读取写入domain=klsg审计记录。
==========================================================================
"""

import os
# [determinism①] CUBLAS_WORKSPACE_CONFIG 必须在 import torch 之前设置(cuBLAS GEMM 决定论); 用 setdefault 让 driver 级 export 优先、幂等。
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
# [determinism②③] 进程全局 cuDNN/算法决定论(放模块顶: 覆盖所有路径)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)   # 硬模式 warn_only=False
from PIL import Image
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support, confusion_matrix, accuracy_score,
)

# [红线护栏] KLSG 授权读取的【唯一合法门】（单点维护）；src/ 在 parents[1]。
#   ★ ceiling 一切带标签读取（train + held-out）都经此模块的授权核；脚本内【不留】任何裸 read_split_rows / 裸 csv 读取。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _klsg_blind  # klsg_label_auth_from_cli / read_klsg_labeled
import _run_safety

# ============================================================ 配置
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results" / "run1_s4_baseline_ceiling"
CEILING_DIR = RESULTS_DIR / "ceiling"

CLASSES = ["airplane", "ship"]            # 索引: airplane=0, ship=1（与 _klsg_blind.CLASSES 一致）
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

# --- SGD 底座超参（与 #1/主矩阵同 regime；RTX 5090，算力充沛）---
SEED = 0
EPOCHS_DEFAULT = 40                        # ★必改⑤：ceiling 锁 40（不沿用底座 100；从源头灭"忘传 --epochs"）
BATCH_SIZE = 64
FC_LR = 1e-2                               # 新层（fc）SGD 量级
BACKBONE_LR = 1e-3                         # backbone = fc 的 1/10（分层 lr）
WEIGHT_DECAY = 5e-4                        # 论文正式配置
LR_DECAY_ALPHA = 10.0                      # inverse-decay α：lr = base_lr·(1+α·p)^(−β)
LR_DECAY_BETA = 0.75                       # inverse-decay β（DANN/CDAN 标准 0.75）
CB_BETA = 0.999                            # CB-WCE effective-number β（论文标准默认、不调）
SPECKLE_SIGMA = 0.1                        # 乘性 speckle 噪声标准差
EVAL_LAST_K = 5                            # ★必改⑤：末 K=5（全 epoch 留存，末 K 用于 softmax ensemble 基本评估）
N_KLSG_FOLDS = 5                           # KLSG 5 折（5 seed × 5 fold = 25 跑）


# ============================================================ 数据
# ★必改①：底座的裸 read_split_rows 函数【已删除】——ceiling 不读 arm2、一切带标签读取经授权核（见 load_labeled_split）。
#   保留裸 read_split_rows 反而是红线隐患（它能读 splits.csv 的 klsg 真标签明文），故整函数移除。

def load_labeled_split(auth, fold, split):
    """经 _klsg_blind 授权核读取一个带标签 split → [(abspath, label_idx), ...]。
    脚本内【唯一】带标签读取入口；固定读取splits.csv的KLSG正式ceiling折，
    内部含按折路径锚一致性校验 + 追加写审计 _LABEL_ACCESS_LOG.txt。
    无授权令牌(auth 非 KlsgLabelAuth) → 授权核自身 raise（红线）。"""
    return _klsg_blind.read_klsg_labeled(auth, PROJECT_ROOT, fold, split)


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
    """训练增强（在线）：几何 + 光度 + speckle。垂直翻转 OFF。"""
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
    """held-out 评估：**不增强**。resize + ImageNet 归一化。"""
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
def build_model():
    """ResNet18 + ImageNet 预训练；fc 换 2 类；**全解冻（整网微调）**；default BN。"""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # 未缓存时由torchvision下载ImageNet权重。
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
    p = global_step / total_steps **归一化到 [0,1]**。α=LR_DECAY_ALPHA、β=LR_DECAY_BETA。
    LambdaLR 的 lr_lambda 返回相对 base_lr 的缩放因子；每次 optimizer.step() 后调一次 scheduler.step()。
    total_steps 下限取 1 防除零；min() clamp 防 last_epoch 越界使 p>1。"""
    total_steps = max(int(total_steps), 1)
    lr_lambda = lambda step: (1.0 + LR_DECAY_ALPHA * (min(step, total_steps) / total_steps)) ** (-LR_DECAY_BETA)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ============================================================ 评估
def forward_collect(model, loader, device, criterion=None):
    """单次前向，收集 (y_true:list[int], probs:np[N,2], mean_loss or None)。
    ★ 决定论旁路：model.eval() + no_grad + 无增强 loader(shuffle=False/num_workers=0)
      -> 零 RNG 消费、零 BN buffer 写（eval 模式 BN 不更新 running stats；resnet18 无 dropout）。
      => per-epoch held-out 评估 与 末K ensemble 重载评估 都不扰动训练 RNG 流。
    probs = softmax(logits)（末K 概率平均要用概率，不是 argmax）。criterion 给则同时算加权 loss。"""
    model.eval()
    ys, prob_chunks = [], []
    tot_loss, seen = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            if criterion is not None:
                yl = y.to(device, non_blocking=True)
                tot_loss += criterion(logits, yl).item() * x.size(0)
            seen += x.size(0)
            prob_chunks.append(torch.softmax(logits, dim=1).cpu())
            ys.extend(int(v) for v in y.tolist())
    probs = torch.cat(prob_chunks, dim=0).numpy()
    mean_loss = (tot_loss / seen) if (criterion is not None and seen) else None
    return ys, probs, mean_loss


def report_metrics(y_true, y_pred):
    """plan §3.2.5：macro-F1 + 各类 P/R/F1（对称）+ 混淆矩阵（sklearn, labels=[0,1]）。"""
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
    """[determinism④] set_seed 四件套 RNG。"""
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser(description="run1 #9 ceiling 训练器（KLSG 真标签监督上界）")
    ap.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT,
                    help=f"训练 epoch 数（默认 {EPOCHS_DEFAULT}=ceiling 锁定；不靠人记得传 40）")
    ap.add_argument("--seed", type=int, default=SEED, help="随机种子（默认 0；走 set_seed 四件套 RNG）")
    ap.add_argument("--fold", type=int, default=0,
                    help="KLSG 折号：0..4（5 折）")
    # ★必改②：授权门——显式难误触开关 + 必填 reason；无令牌训练器拒跑。
    ap.add_argument("--authorize-klsg-labels", action="store_true",
                    help="授权读取KLSG真标签并写入domain=klsg审计行")
    ap.add_argument("--reason", type=str, default=None,
                    help="授权理由（必填、写入审计日志 _LABEL_ACCESS_LOG.txt）")
    args = ap.parse_args()

    # --- 参数校验 ---
    if not (0 <= args.fold < N_KLSG_FOLDS):
        ap.error(f"KLSG 折号须 ∈ [0,{N_KLSG_FOLDS-1}]，实得 {args.fold}")
    if args.authorize_klsg_labels and not (args.reason and args.reason.strip()):
        ap.error("--authorize-klsg-labels 须配非空 --reason（写入审计日志）")

    epochs = args.epochs

    # ★必改②：构造授权令牌；无授权 -> 训练器直接拒跑（ceiling 本质=读真标签监督训练，无授权无意义）。
    auth = _klsg_blind.klsg_label_auth_from_cli(
        args.authorize_klsg_labels, reason=args.reason, caller="train_ceiling_sgd.main")
    if auth is None:
        raise SystemExit(
            "RED LINE / ceiling 无意义: ceiling = 用 KLSG 真标签监督训练的上界跑法，"
            "必须传 --authorize-klsg-labels 且 --reason '<理由>' 才可执行。")

    out_root = CEILING_DIR
    model_dir = out_root / "model" / f"s{args.seed}_f{args.fold}"   # ★必改⑥：全 epoch ckpt 落此
    metrics_dir = out_root / f"s{args.seed}_f{args.fold}"
    metrics_path = metrics_dir / "metrics.json"
    curve_path = metrics_dir / "curve.csv"
    _run_safety.prepare_empty_run_dirs(model_dir, metrics_dir)

    set_seed(args.seed)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"), \
        "CUBLAS_WORKSPACE_CONFIG 未设(须在 import torch 前内联)"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print("#9 **supervised ceiling** 训练器 | KLSG 真标签监督上界")
    print("=" * 78)
    print(f"项目根 : {PROJECT_ROOT}")
    print(f"产物根 : {out_root}")
    print(f"设备   : {device} | torch {torch.__version__} | "
          f"GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"授权   : --authorize-klsg-labels=ON | reason={args.reason!r} | caller=train_ceiling_sgd.main "
          "| 审计 -> _LABEL_ACCESS_LOG.txt (domain=klsg)")

    # --- 数据（★必改①③：训练=该折 train、评估=该折 held-out；全经授权核读，2 条审计行）---
    train_items = load_labeled_split(auth, args.fold, "train")
    heldout_items = load_labeled_split(auth, args.fold, "test")   # ceiling 没训过的留出折
    tr_cnt = Counter(CLASSES[l] for _, l in train_items)
    ho_cnt = Counter(CLASSES[l] for _, l in heldout_items)
    print(f"\n训练 klsg/fold{args.fold}/train (增强) : {len(train_items)} "
          f"(airplane={tr_cnt['airplane']}, ship={tr_cnt['ship']})")
    print(f"评估 klsg/fold{args.fold}/test (held-out, 不增强): {len(heldout_items)} "
          f"(airplane={ho_cnt['airplane']}, ship={ho_cnt['ship']})")
    assert len(train_items) and len(heldout_items), "数据为空，检查 KLSG fold/splits.csv"

    train_tfm, eval_tfm = build_train_transform(), build_eval_transform()
    train_loader = DataLoader(ImageListDataset(train_items, train_tfm), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, drop_last=False)
    ho_loader = DataLoader(ImageListDataset(heldout_items, eval_tfm), batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=0)
    # ceiling 有合法 held-out 收敛信号，label-free churn 代理冗余；不构造 churn loader。

    # --- CB-WCE（★必改④：从该折训练集真实类频【动态】算；删底座硬编 assert counts==[72,320]）---
    counts = [tr_cnt[c] for c in CLASSES]                       # [n_airplane, n_ship]，按折实测
    assert all(c > 0 for c in counts), \
        f"CB-WCE：某类训练样本为 0（counts={counts}），effective-number 权重会发散——检查 fold"
    cb_w = cb_wce_weights(counts, beta=CB_BETA, device=device)
    ratio = (cb_w[0] / cb_w[1]).item()
    print(f"\nCB-WCE (β={CB_BETA}) 权重 (从 KLSG fold{args.fold} train 真实类频 {counts}): "
          f"airplane={cb_w[0].item():.4f}, ship={cb_w[1].item():.4f} (airplane:ship = {ratio:.2f}:1)")
    criterion = nn.CrossEntropyLoss(weight=cb_w)

    # --- 模型 / 优化器 / 调度器（沿用底座：整网微调 + 分层 lr + inverse-decay）---
    model = build_model().to(device)
    optimizer, n_bb, n_fc = make_optimizer(model)
    n_total = sum(p.numel() for p in model.parameters())
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    scheduler = make_scheduler(optimizer, total_steps)
    print(f"\n模型 : ResNet18(ImageNet) **全解冻整网微调** | default BN")
    print(f"分层 lr : backbone lr={BACKBONE_LR} ({n_bb:,} 参数) | fc lr={FC_LR} ({n_fc:,} 参数) "
          f"| 比 = 1:{int(FC_LR/BACKBONE_LR)} | 可训/总 = {n_bb+n_fc:,}/{n_total:,}")
    print(f"优化 : SGD(momentum=0.9, nesterov=True) | weight_decay={WEIGHT_DECAY} | batch {BATCH_SIZE} | {epochs} epoch")
    print(f"调度 : inverse-decay lr=base·(1+{int(LR_DECAY_ALPHA)}·p)^(-{LR_DECAY_BETA})，p=global_step/total_steps 归一化[0,1] "
          f"| steps/epoch={steps_per_epoch} total_steps={total_steps} | 末端衰减因子≈{(1.0+LR_DECAY_ALPHA)**(-LR_DECAY_BETA):.4f}")

    # --- 训练循环（每 step 调 scheduler；每 epoch 记 train + held-out 信号；★必改⑥ 全 epoch 存 ckpt）---
    print("-" * 78)
    history = []
    lr_trace = []                                   # (global_step, p, lr_backbone, lr_fc)：验 lr 衰减
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()                       # default BN：train 模式更新 BN running stats
        run_loss, n_correct, n_seen = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if global_step < 3 or global_step == total_steps - 1:
                lr_trace.append((global_step, global_step / total_steps,
                                 optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"]))
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()                # inverse-decay：每个 optimizer.step() 后步进 1
            global_step += 1
            run_loss += loss.item() * x.size(0)
            n_correct += (logits.argmax(1) == y).sum().item()
            n_seen += x.size(0)
        tr_loss, tr_acc = run_loss / n_seen, n_correct / n_seen

        # ★设计点 D：held-out 评估 = ceiling 的真评估对象。每 epoch 记 loss/acc/macro-F1 入 curve
        #   —— 仅监控 + §5 事后 sanity 兜底；**绝不据此早停/选模**（N=40 锁死、末K 固定、ckpt 全留）。
        #   决定论旁路（forward_collect: eval/no_grad/shuffle=False/nw=0 -> 零 RNG/零 buffer 写）。
        ho_true, ho_probs, ho_loss = forward_collect(model, ho_loader, device, criterion)
        ho_pred = ho_probs.argmax(1)
        ho_acc = float((ho_pred == np.asarray(ho_true)).mean())
        ho_f1 = float(f1_score(ho_true, ho_pred, labels=[0, 1], average="macro", zero_division=0))
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "heldout_loss": ho_loss, "heldout_acc": ho_acc, "heldout_macro_f1": ho_f1})
        print(f"  epoch {epoch:>3}/{epochs} | tr_loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"|| ho_loss {ho_loss:.4f} acc {ho_acc:.4f} macroF1 {ho_f1:.4f}")

        # ★必改⑥：每个 epoch 都存 ckpt（不是底座的只存末K）；oracle + §5 收敛曲线兜底要吃。
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "train_loss": tr_loss, "train_acc": tr_acc,
                    "heldout_loss": ho_loss, "heldout_acc": ho_acc, "heldout_macro_f1": ho_f1,
                    "global_step": global_step, "seed": args.seed, "fold": args.fold},
                   model_dir / f"epoch{epoch:03d}.pth")

    # --- scheduler sanity（训练结束 p≈1，各 group 实际 lr 应 ≈ base_lr×(1+α)^(−β)）---
    print("-" * 78)
    print("★ inverse-decay lr 调度 sanity 检查 ★")
    expected_factor = (1.0 + LR_DECAY_ALPHA) ** (-LR_DECAY_BETA)     # 11^(-0.75) ≈ 0.16548
    base_lrs = list(scheduler.base_lrs)
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

    # ★设计点 D：底座的"arm2 源 train/val 过拟合分析段"【删】。
    #   理由：ceiling 评估在 held-out，过拟合只会在没训过的折上考差、不虚高；N 已锁、ckpt 全留、
    #   收敛 sanity 是 §5 事后从 ckpt 复算的活；训练器内不再烤"过拟合判据"启发式（避诱导早停思维）。

    # --- 最终评估① 末 epoch 单模型（held-out）---
    print("-" * 78)
    print("最终评估① 末 epoch 单模型 (held-out):")
    fin_true, fin_probs, _ = forward_collect(model, ho_loader, device)
    final_metrics = report_metrics(fin_true, fin_probs.argmax(1).tolist())
    print_metrics_block(f"KLSG fold{args.fold} held-out · 末 epoch 单模型", final_metrics)

    # --- 最终评估② ★设计点 B：末 K=5 softmax ensemble（基本级、训练器内置）---
    #   从盘上【重载】末 K epoch ckpt -> held-out 上 softmax 概率平均 -> argmax -> 基本指标。
    #   重载评估 = 决定论旁路、零RNG；与独立评估包从checkpoint复算的口径一致。
    print("-" * 78)
    k_eff = min(EVAL_LAST_K, epochs)
    ens_epochs = list(range(epochs - k_eff + 1, epochs + 1))
    print(f"最终评估② 末 K={k_eff} softmax ensemble (held-out) | epochs={ens_epochs}:")
    ens_model = build_model().to(device)               # 复用一个壳，逐 ckpt load_state_dict（省去重复 ImageNet 初始化）
    ens_true, ens_prob_sum = None, None
    for e in ens_epochs:
        ckpt = torch.load(model_dir / f"epoch{e:03d}.pth", map_location=device)
        ens_model.load_state_dict(ckpt["model_state_dict"])
        yt, pr, _ = forward_collect(ens_model, ho_loader, device)
        if ens_true is None:
            ens_true = yt                              # 同 loader/顺序(shuffle=False) -> 各 ckpt y_true 恒同
        else:
            assert yt == ens_true, \
                "ensemble 重载各 ckpt 的 held-out y_true 顺序不一致(ho_loader shuffle 应为 False)"  # 嫁接 draft_B 防御断言
        ens_prob_sum = pr if ens_prob_sum is None else (ens_prob_sum + pr)
    ens_prob = ens_prob_sum / float(k_eff)
    ens_pred = ens_prob.argmax(1).tolist()
    ensemble_metrics = report_metrics(ens_true, ens_pred)
    print_metrics_block(f"KLSG fold{args.fold} held-out · 末K={k_eff} softmax ensemble", ensemble_metrics)

    # --- 落盘 ---
    tl = [h["train_loss"] for h in history]
    vl = [h["heldout_loss"] for h in history]
    has_nan = any(np.isnan(v) or np.isinf(v) for v in tl + vl)

    out = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run_tag": "ceiling",
        "authorization": {"reason": args.reason, "caller": "train_ceiling_sgd.main",
                          "audit_log": str((PROJECT_ROOT / "results" / "run1_s4_baseline_ceiling" / "_LABEL_ACCESS_LOG.txt"))},
        "config": {
            "seed": args.seed, "fold": args.fold, "epochs": epochs, "batch_size": BATCH_SIZE,
            "eval_last_k": EVAL_LAST_K, "k_eff": k_eff,
            "model": "resnet18_imagenet_full_finetune", "bn": "default",
            "optimizer": {"type": "SGD", "momentum": 0.9, "nesterov": True},
            "layered_lr": {"fc_lr": FC_LR, "backbone_lr": BACKBONE_LR, "ratio": "fc=10x backbone"},
            "weight_decay": WEIGHT_DECAY,
            "loss": "CB-WCE", "cb_beta": CB_BETA,
            "cb_weights": {"airplane": float(cb_w[0].item()), "ship": float(cb_w[1].item()),
                           "ratio_airplane_to_ship": float(ratio),
                           "counts_source": f"klsg fold{args.fold} train (dynamic, per-fold target freq)"},
            "augmentation": {
                "RandomResizedCrop": "224, scale=(0.8,1.0)", "RandomHorizontalFlip": "p=0.5",
                "RandomRotation": "+-15deg", "vertical_flip": "OFF",
                "ColorJitter": "brightness=0.15, contrast=0.15",
                "SpeckleNoise": f"multiplicative, sigma={SPECKLE_SIGMA}",
                "applied_to": "train only; held-out eval no aug"},
            "scheduler": {
                "type": "inverse_decay_LambdaLR",
                "formula": "lr = base_lr * (1 + alpha*p)^(-beta), p = global_step/total_steps in [0,1]",
                "alpha": LR_DECAY_ALPHA, "beta": LR_DECAY_BETA,
                "total_steps": total_steps, "steps_per_epoch": steps_per_epoch,
                "final_decay_factor": float((1.0 + LR_DECAY_ALPHA) ** (-LR_DECAY_BETA)),
            },
            "ckpt_retention": "every_epoch",
            "churn": "removed (ceiling supervised; label-free churn proxy redundant)",
        },
        "data": {
            "domain": "klsg", "fold": args.fold,
            "train": f"klsg/fold{args.fold}/train (aug)",
            "n_train": len(train_items), "train_counts": dict(tr_cnt),
            "heldout": f"klsg/fold{args.fold}/test (held-out, no aug; ceiling never trained on)",
            "n_heldout": len(heldout_items), "heldout_counts": dict(ho_cnt),
            "read_via": "_klsg_blind authorized core (no bare read_split_rows / no bare csv read)",
        },
        "train_history": history,
        "eval_heldout_final_epoch": final_metrics,
        "eval_heldout_lastk_softmax_ensemble": {
            "k": k_eff, "epochs_averaged": ens_epochs, "metrics": ensemble_metrics,
            "note": "basic ensemble is computed in the trainer; the full metric suite is handled by the separate evaluation package.",
        },
        "has_nan_inf": bool(has_nan),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(curve_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "heldout_loss", "heldout_acc", "heldout_macro_f1"])
        for h in history:
            w.writerow([h["epoch"], f"{h['train_loss']:.6f}", f"{h['train_acc']:.6f}",
                        f"{h['heldout_loss']:.6f}", f"{h['heldout_acc']:.6f}", f"{h['heldout_macro_f1']:.6f}"])

    print("\n" + "=" * 78)
    print(f"指标/历史 -> {metrics_path}")
    print(f"曲线 CSV  -> {curve_path}")
    print(f"checkpoint(全 {epochs} epoch) -> {model_dir}")
    print(f"末 epoch 单模型 macro-F1 = {final_metrics['macro_f1']:.4f} | "
          f"末K={k_eff} ensemble macro-F1 = {ensemble_metrics['macro_f1']:.4f}")
    print(f"NaN/Inf : {has_nan}")
    print("=" * 78)
    return 1 if has_nan else 0


if __name__ == "__main__":
    sys.exit(main())
