#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_dann_sgd.py -- 【候补·[E] 事后补充实验】 **DANN × default BN**（SGD 底座）
[SSS sim->real 主实验 B · 事后补充方法 · 边际对齐代表]

================================  这是什么 / 身份 ================================
**DANN**（Domain-Adversarial Neural Network, Ganin et al., JMLR 2016）——边际域对抗对齐的
代表方法：在 source-only 底座上加一个**域判别器**，经梯度反转层（GRL）让 backbone 学到
"域不可分"的特征。**与 CDAN 的唯一区别 = 判别器吃【裸 512 维 penultimate 特征】、不做
条件化**（CDAN 把判别器输入条件化为 f⊗g = 特征 ⊗ softmax 的外积；DANN 不条件化）。

★ **身份 = 候补 / [E] 事后补充实验**：
  - 受 §3.4.7 三护栏约束：① 同冻结协议（同折/同 seed/同底座/同判据）② 论文如实标注
    "事后补充实验·[E]" ③ **不替换** §3.3.3 预指定的主报告对比（M1/M2/M3）。
  - DANN 当初落选主表理由（plan §3.4.7）：领域已大量使用（novelty 疲劳），其对照功能由
    M2 配对与文献引用覆盖。补跑作 [E] 旁证、不进 confirmatory 头条。

================================  实现 provenance（忠实派生·对拍门）================================
本文件**派生自已封验的 `src/run2_s5_cdan/train_cdan_sgd.py`**（该文件的对抗部件逐位移植自微软
GLS-IWCDAN参考实现、并通过单元验与训练健康检查）。DANN = **CDAN 去掉条件化**：
  - 复用（逐字不动）：calc_coeff GRL ramp / grl_hook / AdversarialNetwork（3 层 MLP + 内置 GRL
    register_hook）/ CB-WCE 源分类损失 / inverse-decay LambdaLR / 训练循环骨架（源·目标各自过
    backbone → cat → 判别器吃 concat 单 batch）/ churn·health·逐 epoch ckpt 落盘 / 决定论护栏 /
    KLSG 目标域盲化红线。
  - **改（DANN 专属）**：① 判别器输入 = **裸 512 维特征**（in_feature=512，非 CDAN 的 512×C=1024
    条件化外积）；② **删除 entropy 条件化分支**（那是 CDAN-E·与 DANN 无关）；③ tag=dann_*。
  这等价于 Ganin 原版 DANN（GRL 作用于特征、判别器二分类源/目标）；不引外部 repo 逐位对拍
  （无匹配 pipeline，与 CDAN 当初同—靠忠实派生 + 单元验 + smoke 健康验收）。

数据：源 = arm2/fold/train（带标签，过增强）→ 目标 = KLSG 全 553（**无标签**，过同款增强，transductive）。

================================  相对 GLS/CDAN 的【有意偏离】（继承自 CDAN 母版，显式申报）================================
  D1. 【特征维度】丢 bottleneck、直接用 avgpool 后 512 维 penultimate（plan §3.3.2 锁 512）。
      ⇒ 判别器输入 = **512 维**（DANN 不条件化、无 ×C；< 4096 不随机投影）。
  D2. 【分类损失】沿用底座 CB-WCE（β=0.999），与 source-only 基线同损失 → 公平对照（非 IW）。
  D3. 【lr 调度】沿用底座 inverse-decay LambdaLR（lr=base·(1+10·p)^-0.75）；weight_decay 恒 5e-4。
  D4. 【fc 初始化】沿用底座 nn.Linear 默认初始化；判别器 ad_net 按 GLS 套 xavier。

================================  对抗调度 = 两个独立旋钮（与 CDAN 母版一致）================================
  旋钮 A · trade_off = **常数 1.0**：total_loss = 1.0 * transfer_loss + cls_loss（**不 ramp**）。
  旋钮 B · GRL 系数 coeff = **0→1 ramp**：calc_coeff = 2/(1+e^(-10·p))-1，p=global_step/total_steps。
      仅经 register_hook 作用于反传到特征的梯度（grad <- -coeff*grad）。
  ★ ad_net.max_iter = total_steps（非硬编码 10000）；否则几百步时 coeff 只爬到 ~0.3、对抗打不开。
  ★ 绝不把 ramp 的 λ 乘在 transfer_loss 上（会连判别器学习信号一起压住、偏离 DANN 正解）。

================================  batch 契约 / 红线（与 CDAN 母版一致）================================
  判别器/dc_target 用 size(0)//2 硬切两半，死假设 源 batch == 目标 batch。故源/目标 train loader
  等长 batch + drop_last=True；cat 顺序固定 (source, target) 对齐 dc_target=[1..,0..]。
  目标标签绝不进任何损失（inputs_target, _ = ...）—— UDA 命根。KLSG 目标经 _klsg_blind 盲化
  （label 位是毒丸 FORBIDDEN_LABEL，绝不读真实标签、绝不从 filepath 反推类别）。

合法性护栏：源带标签合法；目标(KLSG)无标签仅供对抗对齐 + label-free 坍缩探测；
**绝不**用真实域表现选模/早停/定参。

================================  eval 兼容（命根）================================
  模型 = DANNResNet18（结构 = CDANResNet18 逐字一致：feature_layers + fc）→ ckpt 的 model_state_dict
  键与 `eval_pipeline.py` 的 ARCH_REGISTRY["cdan"]（_make_cdan→CDANResNet18）逐字匹配 →
  **DANN ckpt 用 `eval_pipeline.py --arch cdan` 直接评估**（前向 _forward_da 返 (f,y) 逐位同）。
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
import torch.nn.functional as F
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

# [红线护栏] KLSG 目标域盲化入口（单点维护，绝不读真实标签）；本文件在 src/extra/dann/，src/ 在 parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _klsg_blind  # read_klsg_unlabeled / UnlabeledImageDataset / collate_unlabeled / FORBIDDEN_LABEL
import _run_safety

# ============================================================ 配置
# 本文件在 src/extra/dann/ -> parents[3] = 项目根（CDAN 母版在 src/run2_s5_cdan/ 用 parents[2]，本文件深一层）
PROJECT_ROOT = Path(__file__).resolve().parents[3]

SPLITS_CSV = PROJECT_ROOT / "data" / "split" / "splits.csv"            # 源 arm2
# DANN 候补产物落 results/extra/dann（与主矩阵 run1-5 物理隔离；脚本位置无关，PROJECT_ROOT 自适应）
RESULTS_DIR = PROJECT_ROOT / "results" / "extra" / "dann"

CLASSES = ["airplane", "ship"]            # 索引: airplane=0, ship=1
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

# --- SGD 底座超参（与 CDAN 母版同 regime，公平对照）---
SEED = 0
EPOCHS_DEFAULT = 40                        # 站5 锁定 N=40（与主矩阵一致；候补同冻结协议·§3.4.7 护栏①）
QUICK_EPOCHS = 12                          # smoke：12 epoch，足以展现 coeff ramp + 判别器动力学
BATCH_SIZE = 64
FC_LR = 1e-2                               # 新层 fc
BACKBONE_LR = 1e-3                         # backbone = fc 的 1/10
DISC_LR = 1e-2                             # 判别器 ad_net（= fc lr = 10× backbone；GLS ad_net lr_mult=10）
WEIGHT_DECAY = 5e-4
LR_DECAY_ALPHA = 10.0                      # inverse-decay α
LR_DECAY_BETA = 0.75                       # inverse-decay β
GRL_ALPHA = 10.0                           # GRL ramp α（calc_coeff，与 lr 的 α 同值但独立旋钮）
CB_BETA = 0.999                            # CB-WCE effective-number β
SPECKLE_SIGMA = 0.1
SAVE_LAST_K = 5                            # 存末 K 个 epoch ckpt（与 --save-every-epoch 单点存盘、同名幂等）
AD_HIDDEN = 1024                           # 判别器隐藏层宽度（GLS 恒用 1024）
TRADE_OFF = 1.0                            # 旋钮 A：常数，乘在 transfer_loss 上，不 ramp


# ============================================================ 数据（与 CDAN 母版逐字一致）
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
    """从 (abspath, label_idx) 列表建数据集；小数据集——预载 PIL(RGB) 进内存。"""
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
    """乘性 speckle 噪声：x <- x * (1 + ε), ε~N(0,σ²)，作用于 [0,1] tensor，clamp 回 [0,1]。"""
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
    """训练增强（在线）：几何 + 光度 + speckle。**源与目标同款增强**。垂直翻转 OFF。"""
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
    """评估：**不增强**。resize + ImageNet 归一化。"""
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ============================================================ CB-WCE（底座沿用；D2：非 IW）
def cb_wce_weights(class_counts, beta=CB_BETA, num_classes=len(CLASSES), device=None):
    """Class-Balanced 权重（Cui et al. 2019, effective number）：w_c=(1-β)/(1-β^{n_c})，归一化 mean=1。"""
    eff_num = [1.0 - (beta ** n) for n in class_counts]
    w = [(1.0 - beta) / en for en in eff_num]
    s = sum(w)
    w = [wi / s * num_classes for wi in w]
    return torch.tensor(w, dtype=torch.float32, device=device)


# ============================================================ DANN 对抗组件（派生自 CDAN 母版·去条件化）
def init_weights(m):
    """GLS network.py L25-35：判别器 Linear 用 xavier_normal（仅施于 ad_net；模型 fc 走底座默认初始化）。"""
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=GRL_ALPHA, max_iter=10000.0):
    """GLS network.py L21-22 的 GRL ramp 系数（**np.float -> float**，numpy>=1.24 修）：
    coeff = 2(high-low)/(1+e^(-alpha*iter_num/max_iter)) - (high-low) + low。
    high=1,low=0 时 = 2/(1+e^(-alpha*p))-1，p=iter_num/max_iter；iter_num=0→0、iter_num=max_iter→~0.9999。
    ★ max_iter 必须传 total_steps（不是 10000），否则本任务几百步时 coeff 只到 ~0.3。"""
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def grl_hook(coeff):
    """GLS network.py L198-201：反传时把流入特征的梯度取负并按 coeff 缩放（grad <- -coeff*grad）。"""
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1


class AdversarialNetwork(nn.Module):
    """域判别器（移植自 GLS network.py L367-406，剥 IW，无改动语义）。内置 GRL register_hook：
    forward 时（training）self.iter_num 自增 1 → 算 coeff → 对输入挂 grl_hook。
    ★ DANN：in_feature = 512（裸 penultimate），非 CDAN 的 512×C 条件化外积。
    ★ 全程每个训练 step **只前向一次**（吃 concat 后的单 batch），故 iter_num == 训练步数。
    ★ max_iter 由外部置为 total_steps。"""
    def __init__(self, in_feature, hidden_size, max_iter=10000.0):
        super().__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = GRL_ALPHA
        self.low = 0.0
        self.high = 1.0
        self.max_iter = float(max_iter)

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))      # GRL：只作用于反传到特征的梯度
        x = self.dropout1(self.relu1(self.ad_layer1(x)))
        x = self.dropout2(self.relu2(self.ad_layer2(x)))
        y = self.sigmoid(self.ad_layer3(x))
        return y

    def current_coeff(self):
        """读取当前 iter_num 对应的 coeff（= 上一次 forward 实际用的值；forward 先自增再算 coeff）。"""
        return calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)


def dann_loss(features, ad_net, device):
    """DANN 域对抗损失（Ganin et al. JMLR 2016；= CDAN **去条件化**）。
    判别器直接吃 **512 维 penultimate features**（无 f⊗g 外积、无 softmax 条件化、无 entropy 分支）。
    GRL 经 ad_net 内置 register_hook 作用于回流特征的梯度。
    batch_size = size(0)//2 硬切（死假设源==目标 batch）；dc_target=[1]*bs+[0]*bs 对齐 cat(源,目标)。
    返回 (loss, ad_out)：loss = 判别器 BCE = transfer_loss；ad_out 供诊断（源/目标半 ad_out 均值）。"""
    ad_out = ad_net(features)                                     # GRL register_hook 在 ad_net 内部完成
    batch_size = features.size(0) // 2
    dc_target = torch.from_numpy(
        np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(device)
    loss = nn.BCELoss()(ad_out, dc_target)
    return loss, ad_out


# ============================================================ 模型（结构 = CDANResNet18 逐字一致 → eval --arch cdan 兼容）
class DANNResNet18(nn.Module):
    """ResNet18(ImageNet) 整网微调；暴露 **avgpool 后 512 维 penultimate** + fc logits。
    ★ 结构（attribute 名 feature_layers + fc、子模块顺序）与 `train_cdan_sgd.CDANResNet18` 逐字一致
      → state_dict 键匹配 → ckpt 用 `eval_pipeline.py --arch cdan` 直接评估。
    GRL/判别器都挂在这 512 维 penultimate（avgpool 后、fc 前）。default BN。无 bottleneck（D1）。"""
    def __init__(self, num_classes=len(CLASSES)):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # 未缓存时由torchvision下载ImageNet权重。
        self.in_features = base.fc.in_features                    # 512
        self.feature_layers = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4, base.avgpool)
        self.fc = nn.Linear(self.in_features, num_classes)        # 底座默认初始化（D4，不套 GLS xavier）
        for p in self.parameters():
            p.requires_grad = True                                # 全解冻整网微调（含 BN gamma/beta）

    def forward(self, x):
        f = self.feature_layers(x)
        f = f.view(f.size(0), -1)                                 # 512 维 penultimate
        y = self.fc(f)
        return f, y

    def output_num(self):
        return self.in_features


def make_dann_optimizer(model, ad_net):
    """分层 lr 的 SGD（+momentum +nesterov）。3 组（**在构造 scheduler 之前**加判别器组，
    否则 LambdaLR.base_lrs 不含它、不被 inverse-decay 缩放）：
      group 0 = backbone (lr=BACKBONE_LR=1e-3)
      group 1 = fc       (lr=FC_LR=1e-2 = 10× backbone)
      group 2 = 判别器   (lr=DISC_LR=1e-2 = 10× backbone；GLS ad_net lr_mult=10)
    返回 (optimizer, n_bb, n_fc, n_ad, backbone_params)。backbone_params 供 grad-norm 诊断。"""
    fc_params, backbone_params = [], []
    for name, p in model.named_parameters():
        (fc_params if name.startswith("fc.") else backbone_params).append(p)
    ad_params = list(ad_net.parameters())
    optimizer = torch.optim.SGD(
        [{"params": backbone_params, "lr": BACKBONE_LR},   # group 0
         {"params": fc_params, "lr": FC_LR},               # group 1
         {"params": ad_params, "lr": DISC_LR}],            # group 2
        momentum=0.9, nesterov=True, weight_decay=WEIGHT_DECAY,
    )
    return (optimizer, sum(p.numel() for p in backbone_params),
            sum(p.numel() for p in fc_params), sum(p.numel() for p in ad_params), backbone_params)


def make_scheduler(optimizer, total_steps):
    """底座 inverse-decay（LambdaLR）：每步 factor = (1+α·p)^(-β)，p=global_step/total_steps 归一化[0,1]。
    LambdaLR 用各组各自 base_lr × 同一 factor → 三组同步衰减。total_steps 同时是 GRL max_iter。"""
    total_steps = max(int(total_steps), 1)
    lr_lambda = lambda step: (1.0 + LR_DECAY_ALPHA * (min(step, total_steps) / total_steps)) ** (-LR_DECAY_BETA)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ============================================================ 评估 / 指标（底座沿用）
def evaluate_preds(model, loader, device):
    """返回 (y_true, y_pred)；model 返回 (feature, logits)，取 logits。"""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            _, logits = model(x)
            y_pred.extend(logits.argmax(1).cpu().tolist())
            y_true.extend(y.tolist())
    return y_true, y_pred


def target_pseudolabel_health(model, loader, device, num_classes=len(CLASSES)):
    """label-free 目标域伪标签坍缩探测。**只用模型自身在目标【图像】上的预测**，绝不读目标真实 label
    （loader 的 label 位是毒丸 FORBIDDEN_LABEL，这里只取 batch[0] 图像、永不触碰 batch[1]）。
    返回 (pred_hist[list], mean_entropy)。判读：pred_hist 全压一类 = 坍缩；mean_entropy 趋 0 = 过度自信。"""
    model.eval()
    hist = [0] * num_classes
    ent_sum, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)        # 只取图像；batch[1] 是毒丸, 绝不碰
            _, logits = model(x)
            prob = F.softmax(logits, dim=1)
            for c in prob.argmax(1).cpu().tolist():
                hist[c] += 1
            ent_sum += float((-(prob * prob.clamp_min(1e-12).log()).sum(1)).sum().item())
            n += int(x.size(0))
    return hist, ent_sum / max(n, 1)


def churn_predict(model, loader, device):
    """churn：模型对固定 churn 100 子集的 argmax 预测向量（长100, int）。
    churn 标量 = 相邻 epoch 此向量的不一致比例（在训练循环里算）。
    ★ 决定论铁律：model.eval() + torch.no_grad() + eval_tfm(无增强) + shuffle=False + num_workers=0
      -> 零 RNG 消费、零 BN buffer 写。
    ★ 红线：只取 batch[0] 图像；batch[1] 是毒丸 FORBIDDEN_LABEL, 绝不索引/绝不读真实标签。
    调用后 model.train() 复位。返回 np.ndarray(int)。"""
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)        # 只取图像；batch[1] 是毒丸, 绝不碰
            _, logits = model(x)
            preds.extend(logits.argmax(1).cpu().tolist())
    model.train()
    return np.array(preds, dtype=np.int64)


def report_metrics(y_true, y_pred):
    macro_f1 = f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)
    p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc = accuracy_score(y_true, y_pred)
    per_class = {c: {"precision": float(p[i]), "recall": float(r[i]),
                     "f1": float(f[i]), "support": int(sup[i])} for i, c in enumerate(CLASSES)}
    return {"macro_f1": float(macro_f1), "accuracy": float(acc), "per_class": per_class,
            "confusion_matrix": cm.tolist(),
            "cm_axis": "rows=true, cols=pred, order=[airplane, ship]"}


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ============================================================ 单元验（smoke 前先单元验）
def run_unit_tests(device):
    """smoke 前先单元验：① calc_coeff(max_iter=total_steps) 末端≈1.0 ② 判别器吃【512 维裸特征】+ batch 切分
    ③ grl_hook 真取负+缩放。★ 与 CDAN 母版的差异点 = ② 判别器输入维度 = 512（非 1024 条件化）。"""
    print("=" * 78)
    print("★ 单元验（smoke 前 · DANN） ★")
    ok = True

    # ① calc_coeff ramp 端点 + 'max_iter=10000 vs total_steps' 对照
    ts = 240                                  # 代表性 total_steps（DANN N=40 × ~6 step/epoch ≈ 240）
    c0 = calc_coeff(0, max_iter=ts)
    c_end = calc_coeff(ts, max_iter=ts)
    c_bug = calc_coeff(ts, max_iter=10000.0)  # 若误用硬编码 10000：训练结束 coeff 只爬到这
    print(f"  [①coeff] calc_coeff(0,max_iter={ts})           = {c0:.6f}  (期望≈0)")
    print(f"  [①coeff] calc_coeff({ts},max_iter={ts})        = {c_end:.6f}  (期望≈1, 阈值>0.999)")
    print(f"  [①coeff] calc_coeff({ts},max_iter=10000) [坑]  = {c_bug:.6f}  (若不改 max_iter, 末端只到这)")
    t1 = (abs(c0) < 1e-6) and (c_end > 0.999)
    print(f"  [①coeff] -> {'PASS' if t1 else 'FAIL'}")
    ok = ok and t1

    # ② DANN 判别器吃 512 维裸特征（非 CDAN 的 512×C=1024 条件化）+ batch 等长切分
    bs, fdim = 8, 512
    feat = torch.randn(2 * bs, fdim, requires_grad=True, device=device)
    ad = AdversarialNetwork(fdim, AD_HIDDEN, max_iter=ts).to(device)   # ★ in_feature=512（DANN 不条件化）
    ad.train()
    loss_v, ad_out = dann_loss(feat, ad, device)
    expect_dc = np.array([[1]] * bs + [[0]] * bs)
    split_ok = (ad_out.shape == (2 * bs, 1)) and \
               ((2 * bs) // 2 == bs) and \
               (feat.size(0) // 2 == bs) and \
               (expect_dc[:bs].sum() == bs and expect_dc[bs:].sum() == 0)
    # 判别器输入维度 = 512（裸特征，非条件化）
    ad_in_dim = ad.ad_layer1.in_features
    dim_ok = (ad_in_dim == fdim == 512)
    print(f"  [②disc ] concat 2x{bs}={2*bs} -> //2={ (2*bs)//2 } (期望{bs}) | ad_out shape={tuple(ad_out.shape)} (期望({2*bs},1))")
    print(f"  [②disc ] dc_target=[1]*{bs}+[0]*{bs} | 判别器输入维度={ad_in_dim} (期望512=裸特征,非1024条件化) | BCE={loss_v.item():.4f}")
    print(f"  [②disc ] -> {'PASS' if (split_ok and dim_ok) else 'FAIL'}")
    ok = ok and split_ok and dim_ok

    # ③ grl_hook 取负+缩放：loss=(t2*2).sum(), 流入 t2 的 grad=2，hook 应输出 -coeff*2
    t = torch.ones(3, requires_grad=True, device=device)
    t2 = t * 1.0
    t2.register_hook(grl_hook(0.5))
    (t2 * 2.0).sum().backward()
    grl_ok = torch.allclose(t.grad, torch.full_like(t.grad, -1.0), atol=1e-6)  # -0.5*2 = -1.0
    print(f"  [③grl ] grad(t)={t.grad.tolist()} (期望[-1,-1,-1] = -coeff*2, coeff=0.5) -> {'PASS' if grl_ok else 'FAIL'}")
    ok = ok and grl_ok

    print(f"  >>> 单元验总判: {'全部 PASS' if ok else '存在 FAIL'}")
    print("=" * 78)
    assert ok, "单元验失败：calc_coeff 端点 / 判别器512维切分 / grl_hook 语义有误，不进 smoke"
    return ok


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None, help="epoch 数（默认 full=40；--quick 时默认 12）")
    ap.add_argument("--quick", action="store_true", help="smoke（默认 12 epoch、产物名 dann_smoke_*）")
    ap.add_argument("--unittest", action="store_true", help="只跑单元验后退出（不训练）")
    ap.add_argument("--seed", type=int, default=SEED, help="随机种子（默认 0）；改它走 set_seed RNG 四件套")
    ap.add_argument("--fold", type=int, default=0, help="源 arm2 折号（默认 0）。**仅作用于源 arm2 留出折**；"
                    "KLSG 目标流走全 553 transductive，fold 无关、绝不接此参数")
    ap.add_argument("--save-every-epoch", action="store_true",
                    help="逐 epoch 存全 dict ckpt（候补 oracle/末K 用；默认 False，只存末 SAVE_LAST_K）")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"), "CUBLAS_WORKSPACE_CONFIG 未设(须在 import torch 前内联)"

    # ---- 单元验先行 ----
    run_unit_tests(device)
    if args.unittest:
        print("仅单元验模式（--unittest），退出。")
        return 0

    # ★ seed 修复：run_unit_tests **消费** torch RNG（randn + 判别器 xavier 初始化）→ RNG 被推进；
    #   若不重设，训练 RNG 起点会随单元验的消费而偏。故在单元验之后、建数据/模型/训练之前重设回 args.seed。
    set_seed(args.seed)

    epochs = (QUICK_EPOCHS if args.quick else EPOCHS_DEFAULT) if args.epochs is None else args.epochs
    tag = "dann_smoke" if args.quick else "dann_full"      # dann_ 前缀，与主矩阵产物物理隔离

    model_dir = RESULTS_DIR / f"{tag}_s{args.seed}_f{args.fold}"
    metrics_path = model_dir / f"{tag}_metrics.json"
    curve_path = model_dir / f"{tag}_curve.csv"
    churn_argmax_path = model_dir / f"{tag}_churn_argmax.csv"
    _run_safety.prepare_empty_run_dirs(model_dir)

    print("=" * 78)
    print(f"【候补·[E]】 **DANN × default BN** (SGD 底座) ({tag}) : arm2 -> KLSG UDA")
    print("=" * 78)
    print(f"项目根 : {PROJECT_ROOT}")
    print(f"设备   : {device} | torch {torch.__version__} | "
          f"GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # --- 数据：源 arm2/fold/train(带标签,增强) | 目标 KLSG 全 553(无标签,同款增强,transductive,fold无关) ---
    src_items = read_split_rows(SPLITS_CSV, domain="arm2", fold=args.fold, split="train")  # 源折由 --fold 选
    tgt_items = _klsg_blind.read_klsg_unlabeled(PROJECT_ROOT)                               # KLSG 全 553（无标签）
    src_cnt = Counter(CLASSES[l] for _, l in src_items)
    print(f"\n源 arm2/fold{args.fold}/train (带标签,增强)     : {len(src_items)} (airplane={src_cnt['airplane']}, ship={src_cnt['ship']})")
    print(f"目标 KLSG 全 {len(tgt_items)} (无标签,同款增强,transductive) : sha {_klsg_blind.KLSG_MANIFEST_SHA256[:8]} "
          f"[label 盲化为毒丸 FORBIDDEN_LABEL, 绝不入损失/绝不读真实类别; fold 无关]")
    assert len(src_items) and len(tgt_items), "数据为空，检查 csv 路径/manifest"
    assert len(tgt_items) == _klsg_blind.KLSG_N_EXPECTED, f"KLSG 目标张数={len(tgt_items)} != 553"

    train_tfm, eval_tfm = build_train_transform(), build_eval_transform()
    # 源/目标 train loader 等长 batch + drop_last=True（dann_loss size(0)//2 硬切的契约）
    src_loader = DataLoader(ImageListDataset(src_items, train_tfm), batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=0, drop_last=True)
    # 目标：盲化数据集 + 盲化 collate（绝不 stack 标签）；增强+shuffle+drop_last
    tgt_loader = DataLoader(_klsg_blind.UnlabeledImageDataset(tgt_items, train_tfm), batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=0, drop_last=True, collate_fn=_klsg_blind.collate_unlabeled)
    # 目标域伪标签坍缩探测 loader（不增强、确定性、shuffle=False、无标签）
    tgt_probe_loader = DataLoader(_klsg_blind.UnlabeledImageDataset(tgt_items, eval_tfm), batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=0, collate_fn=_klsg_blind.collate_unlabeled)
    # churn 固定 100 子集 loader（eval_tfm 无增强/shuffle=False/num_workers=0 -> 零 RNG 消费、零 buffer 写）
    #   ★ 构造在所有 RNG-消费 loader 之后, shuffle=False 不消费 RNG, 不扰动决定论。
    churn_items = _klsg_blind.read_klsg_churn_subset(PROJECT_ROOT)
    assert len(churn_items) == _klsg_blind.CHURN_N_EXPECTED, \
        f"churn 子集张数={len(churn_items)} != {_klsg_blind.CHURN_N_EXPECTED}"
    churn_loader = DataLoader(_klsg_blind.UnlabeledImageDataset(churn_items, eval_tfm), batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, collate_fn=_klsg_blind.collate_unlabeled)
    print(f"churn 固定子集 (label-free, sha {_klsg_blind.KLSG_CHURN100_SHA256[:8]}) : {len(churn_items)} 张 "
          f"[eval_tfm 无增强/shuffle=False/num_workers=0; 只取图像、label 位毒丸绝不读]")
    assert len(src_loader) >= 1 and len(tgt_loader) >= 1, "drop_last=True 后 loader 为空：batch 太大/数据太少"

    # --- CB-WCE 权重（从源训练集类频；底座配方，非 IW）---
    counts = [src_cnt[c] for c in CLASSES]
    assert counts == [72, 320], f"CB-WCE 护栏：arm2 各折 train 类频应恒为 [72,320]，实得 {counts}（防权重静默漂）"
    cb_w = cb_wce_weights(counts, beta=CB_BETA, device=device)
    ratio = (cb_w[0] / cb_w[1]).item()
    print(f"\nCB-WCE (β={CB_BETA}) 权重 (从 arm2 train 类频 {counts}): "
          f"airplane={cb_w[0].item():.4f}, ship={cb_w[1].item():.4f} (airplane:ship = {ratio:.2f}:1)  [非 IW]")
    criterion = nn.CrossEntropyLoss(weight=cb_w)                # 源分类损失（CB-WCE）

    # --- 步数预算：扁平步数，源 loader 驱动 epoch，目标 loader 循环复用 ---
    steps_per_epoch = len(src_loader)                          # = floor(n_src / batch)
    total_steps = epochs * steps_per_epoch                     # lr inverse-decay 的 p 分母 + GRL max_iter

    # --- 模型 / 判别器 / 优化器 / 调度器 ---
    model = DANNResNet18().to(device)
    # 判别器：DANN 输入 = 512 维裸 penultimate（D1，不条件化、不随机投影）；max_iter=total_steps
    ad_in_dim = model.output_num()                             # 512（DANN：非 ×C 条件化）
    ad_net = AdversarialNetwork(ad_in_dim, AD_HIDDEN, max_iter=total_steps).to(device)
    optimizer, n_bb, n_fc, n_ad, backbone_params = make_dann_optimizer(model, ad_net)  # 3 组在 scheduler 前
    scheduler = make_scheduler(optimizer, total_steps)         # base_lrs 含判别器组 -> 同步 inverse-decay
    n_total = sum(p.numel() for p in model.parameters())

    print(f"\n模型 : ResNet18(ImageNet) 全解冻整网微调 | default BN | penultimate=512 维 (无 bottleneck, D1)")
    print(f"判别器 : AdversarialNetwork(in={ad_in_dim}=512 裸特征[DANN 不条件化], hidden={AD_HIDDEN}) | 内置 GRL | max_iter={total_steps}")
    print(f"分层 lr : backbone={BACKBONE_LR}({n_bb:,}) | fc={FC_LR}({n_fc:,}) | 判别器={DISC_LR}({n_ad:,}) | 可训/总={n_bb+n_fc:,}/{n_total:,}")
    print(f"优化 : SGD(m=0.9,nesterov) wd={WEIGHT_DECAY} | batch {BATCH_SIZE}(源)+{BATCH_SIZE}(目标) | {epochs} epoch")
    print(f"调度 : inverse-decay lr=base·(1+{int(LR_DECAY_ALPHA)}·p)^(-{LR_DECAY_BETA}) | steps/epoch={steps_per_epoch} total_steps={total_steps}")
    print(f"对抗 : 旋钮A trade_off={TRADE_OFF}(常数,不ramp) | 旋钮B coeff=2/(1+e^(-{int(GRL_ALPHA)}·p))-1 (0→1 ramp)")
    print(f"  ★ total_loss = {TRADE_OFF}·transfer_loss(判别器BCE,DANN边际对齐) + cls_loss(CB-WCE源); GRL 仅经 register_hook 作用于特征梯度")

    # --- 训练循环（源/目标各自单独过 backbone → cat → 判别器吃 concat 单 batch；DANN 判别器吃裸特征）---
    print("-" * 78)
    print("smoke 每 epoch 诊断: coeff(末步) | 判别器BCE(均, 健康≈0.693=ln2) | ad_out 源/目标均值(健康→0.5) | "
          "源cls_loss/acc | backbone梯度范数 | NaN")
    print("-" * 78)
    history = []
    coeff_trace = []                                # 全程逐步 coeff（首/中/末汇总用）
    churn_argmax_rows = []     # 逐 epoch (epoch, argmax 长100 向量), epoch1 起
    a_prev = None              # 上一 epoch churn argmax 向量；epoch1 时 None -> churn 标量记 null
    global_step = 0
    any_nan = False
    target_iter = iter(tgt_loader)
    for epoch in range(1, epochs + 1):
        model.train(); ad_net.train()               # default BN：train 模式更新 running stats
        ep_bce, ep_cls, ep_adS, ep_adT, ep_gn = 0.0, 0.0, 0.0, 0.0, 0.0
        ep_correct, ep_seen, ep_steps = 0, 0, 0
        ep_coeff_last = 0.0
        for x_s, y_s in src_loader:                 # 源 loader 驱动 epoch
            # 目标 batch：循环复用；**丢标签**（UDA 命根）
            try:
                x_t, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(tgt_loader)
                x_t, _ = next(target_iter)
            x_s, y_s = x_s.to(device, non_blocking=True), y_s.to(device, non_blocking=True)
            x_t = x_t.to(device, non_blocking=True)

            # 源、目标各自单独过 backbone（BN 各吃各域统计 = default 混合语义）
            f_s, o_s = model(x_s)
            f_t, _ = model(x_t)        # ★目标 logits 故意丢弃：DANN 判别器只吃裸特征 f_t、目标 logits 不进任何损失（UDA 命根）
            assert f_s.size(0) == f_t.size(0), "契约破：源/目标 batch 不等长（检查 drop_last/batch_size）"
            # cat：判别器吃 concat 后的单 batch（DANN：吃裸 512 维特征、无条件化）
            features = torch.cat((f_s, f_t), dim=0)

            cls_loss = criterion(o_s, y_s)          # 源分类（CB-WCE），仅源、仅 logits
            transfer_loss, ad_out = dann_loss(features, ad_net, device)
            total_loss = TRADE_OFF * transfer_loss + cls_loss   # 旋钮 A：常数 1.0

            optimizer.zero_grad()
            total_loss.backward()
            # backbone 梯度范数（在 step 前）——对抗信号经 GRL 反传到 backbone 的强度
            bb_sq = 0.0
            for p in backbone_params:
                if p.grad is not None:
                    bb_sq += float(p.grad.detach().pow(2).sum().item())
            bb_grad_norm = bb_sq ** 0.5
            optimizer.step()
            scheduler.step()                        # 每 optimizer.step() 后步进
            global_step += 1

            # 诊断（不额外前向；ad_out 复用训练前向）
            cur_coeff = ad_net.current_coeff()      # = 本 step forward 实际用的 coeff
            coeff_trace.append((global_step, cur_coeff))
            ep_coeff_last = cur_coeff
            bs = ad_out.size(0) // 2
            ep_bce += float(transfer_loss.item())
            ep_cls += float(cls_loss.item())
            ep_adS += float(ad_out[:bs].mean().item())   # 源半（域标签 1）
            ep_adT += float(ad_out[bs:].mean().item())   # 目标半（域标签 0）
            ep_gn += bb_grad_norm
            ep_correct += int((o_s.argmax(1) == y_s).sum().item())
            ep_seen += int(y_s.size(0))
            ep_steps += 1
            if not np.isfinite(float(total_loss.item())):
                any_nan = True

        # epoch 汇总
        bce_m = ep_bce / ep_steps
        cls_m = ep_cls / ep_steps
        adS_m = ep_adS / ep_steps
        adT_m = ep_adT / ep_steps
        gn_m = ep_gn / ep_steps
        src_acc = ep_correct / max(ep_seen, 1)
        ep_nan = (not np.isfinite(bce_m)) or (not np.isfinite(cls_m)) or any_nan
        any_nan = any_nan or ep_nan

        # label-free 目标域伪标签坍缩探测（只用模型预测、绝不读目标真实标签）
        tgt_hist, tgt_ent = target_pseudolabel_health(model, tgt_probe_loader, device)
        tgt_collapsed = (max(tgt_hist) == sum(tgt_hist))   # 全预测同一类 = 坍缩
        # churn（固定100子集 argmax 相邻 epoch 翻转率 —— label-free, 只用模型预测）
        a_e = churn_predict(model, churn_loader, device)                       # 决定论 eval 旁路
        churn_e = float((a_e != a_prev).mean()) if a_prev is not None else None  # epoch1: a_prev=None -> null
        a_prev = a_e
        churn_argmax_rows.append((epoch, a_e.tolist()))
        model.train(); ad_net.train()               # 探测切了 eval，切回

        history.append({"epoch": epoch, "global_step": global_step, "coeff_last": ep_coeff_last,
                        "disc_bce_mean": bce_m, "ad_out_src_mean": adS_m, "ad_out_tgt_mean": adT_m,
                        "src_cls_loss_mean": cls_m, "src_acc": src_acc, "bb_grad_norm_mean": gn_m,
                        "churn": churn_e,
                        "tgt_pred_hist": list(tgt_hist), "tgt_pred_entropy_mean": tgt_ent,
                        "tgt_pred_collapsed": bool(tgt_collapsed), "nan": bool(ep_nan)})
        churn_str = "n/a" if churn_e is None else f"{churn_e:.3f}"
        print(f"  ep {epoch:>3}/{epochs} | coeff {ep_coeff_last:.4f} | D_BCE {bce_m:.4f} "
              f"| adOut S{adS_m:.3f}/T{adT_m:.3f} | cls {cls_m:.4f} acc {src_acc:.4f} "
              f"| bbGrad {gn_m:.3e} | churn {churn_str} | tgtPred {tgt_hist} H{tgt_ent:.3f}"
              f"{' COLLAPSE!' if tgt_collapsed else ''}{' | NaN!' if ep_nan else ''}")
        # 单点存盘：默认存末 SAVE_LAST_K，--save-every-epoch 则全 epoch（末K ⊂ every、同名幂等不双写）
        if args.save_every_epoch or epoch > epochs - SAVE_LAST_K:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "ad_net_state_dict": ad_net.state_dict(),
                        "global_step": global_step},
                       model_dir / f"{tag}_epoch{epoch:03d}.pth")

    # --- coeff ramp 轨迹（首/中/末）+ lr inverse-decay sanity ---
    print("-" * 78)
    print("★ GRL coeff ramp 轨迹（验 max_iter=total_steps 修没修）★")
    if coeff_trace:
        mid = len(coeff_trace) // 2
        for label, (gs, cv) in [("首", coeff_trace[0]), ("中", coeff_trace[mid]), ("末", coeff_trace[-1])]:
            print(f"  {label}步 global_step={gs:>5} (p={gs/total_steps:6.4f}) -> coeff = {cv:.6f}")
        coeff_end_ok = coeff_trace[-1][1] > 0.9
        print(f"  末端 coeff {'已爬到 ~1.0 (对抗已打开)' if coeff_end_ok else '偏低 (检查 max_iter)'} | "
              f"对照: 若误用 max_iter=10000 末端只会到 {calc_coeff(total_steps, max_iter=10000.0):.4f}")

    print("★ lr inverse-decay sanity（末端 factor = (1+α)^(-β)）★")
    expected_factor = (1.0 + LR_DECAY_ALPHA) ** (-LR_DECAY_BETA)
    base_lrs = list(scheduler.base_lrs)                # [backbone, fc, 判别器]
    group_names = ["backbone", "fc", "discriminator"]
    lr_ok = True
    for i, gname in enumerate(group_names):
        cur_lr = optimizer.param_groups[i]["lr"]
        exp_lr = base_lrs[i] * expected_factor
        rel_err = abs(cur_lr - exp_lr) / max(exp_lr, 1e-12)
        flag = "OK" if rel_err < 1e-3 else "MISMATCH"
        print(f"  group[{i}] {gname:>13}: base={base_lrs[i]:.3e} | 末端 lr={cur_lr:.6e} | 期望={exp_lr:.6e} [{flag}]")
        lr_ok = lr_ok and (rel_err < 1e-3)
    assert lr_ok, "scheduler 末端 lr 不符：判别器组未入 base_lrs 或归一化 p 实现有误"
    print(f"  >>> lr 调度（含判别器组）核验: {'PASS' if lr_ok else 'FAIL'}")

    # --- 对抗动力学健康判读（smoke 判据；**不读性能、不定参**）---
    print("-" * 78)
    print("★ 对抗动力学健康判读（smoke；**不读性能、不定参**）★")
    bce_series = [h["disc_bce_mean"] for h in history]
    coeff_series = [h["coeff_last"] for h in history]
    acc_series = [h["src_acc"] for h in history]
    ln2 = float(np.log(2.0))
    coeff_ramped = coeff_series[-1] > 0.9 and coeff_series[-1] > coeff_series[0]
    bce_not_collapsed = bce_series[-1] > 0.2        # 判别器 BCE 未坠向 0（坠 0 = 判别器碾压/对抗失效）
    adS_final, adT_final = history[-1]["ad_out_src_mean"], history[-1]["ad_out_tgt_mean"]
    src_healthy = acc_series[-1] > 0.5              # 源分类没崩（极小数据,宽松阈）
    tgt_collapse_any = any(h["tgt_pred_collapsed"] for h in history)
    tgt_hist_final = history[-1]["tgt_pred_hist"]
    tgt_ent_final = history[-1]["tgt_pred_entropy_mean"]
    print(f"  ① coeff ramp     : 首{coeff_series[0]:.3f} -> 末{coeff_series[-1]:.3f}  [{'真 ramp 到~1' if coeff_ramped else '未爬升'}]")
    print(f"  ② 判别器 BCE     : 首{bce_series[0]:.3f} -> 末{bce_series[-1]:.3f} (参照 ln2={ln2:.3f})  [{'未塌(健康)' if bce_not_collapsed else '坠向0(对抗失效?)'}]")
    print(f"  ③ ad_out 均值末  : 源{adS_final:.3f} / 目标{adT_final:.3f}  (健康→趋 0.5)")
    print(f"  ④ 源分类 acc     : 首{acc_series[0]:.3f} -> 末{acc_series[-1]:.3f}  [{'未崩' if src_healthy else '偏低-复核'}]")
    print(f"  ⑤ NaN/Inf 守卫   : {'发现 NaN/Inf!' if any_nan else '全程无 NaN/Inf'}")
    print(f"  ⑥ backbone 梯度  : 末 epoch 范数 {history[-1]['bb_grad_norm_mean']:.3e} (有限即对抗信号经 GRL 正常回传)")
    print(f"  ⑦ 目标伪标签坍缩 : 末 epoch 预测直方 {tgt_hist_final} 熵{tgt_ent_final:.3f}  "
          f"[{'坍缩过(全预测同类!)' if tgt_collapse_any else '未坍缩(两类都有预测)'}]  (label-free, 只用模型预测、不读目标真实标签)")

    # --- 落盘 ---
    out = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run_tag": tag,
        "method": "DANN (Ganin et al. JMLR 2016, feature-level adversarial, no conditioning) x default BN, SGD base",
        "identity": "candidate / [E] post-hoc supplementary (station6 S6-8); NOT in confirmatory M1/M2/M3 set",
        "provenance": "derived from verified src/run2_s5_cdan/train_cdan_sgd.py by removing CDAN conditioning "
                      "(discriminator on raw 512-d features) and the entropy branch; = canonical DANN",
        "config": {
            "seed": args.seed, "fold": args.fold, "epochs": epochs, "batch_size_source": BATCH_SIZE, "batch_size_target": BATCH_SIZE,
            "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
            "model": "resnet18_imagenet_full_finetune", "bn": "default", "penultimate_dim": model.output_num(),
            "discriminator": {"in_feature": ad_in_dim, "hidden": AD_HIDDEN, "max_iter": total_steps,
                              "lr": DISC_LR, "note": "DANN: raw penultimate(512), NO conditioning, no random projection"},
            "optimizer": {"type": "SGD", "momentum": 0.9, "nesterov": True, "weight_decay": WEIGHT_DECAY},
            "layered_lr": {"backbone_lr": BACKBONE_LR, "fc_lr": FC_LR, "disc_lr": DISC_LR},
            "scheduler": {"type": "inverse_decay_LambdaLR", "alpha": LR_DECAY_ALPHA, "beta": LR_DECAY_BETA,
                          "formula": "lr=base*(1+alpha*p)^(-beta), p=global_step/total_steps"},
            "adversarial": {"trade_off": TRADE_OFF, "trade_off_note": "constant, NOT ramped",
                            "grl_coeff": "calc_coeff=2/(1+exp(-alpha*p))-1", "grl_alpha": GRL_ALPHA,
                            "grl_max_iter": total_steps, "grl_max_iter_note": "= total_steps (NOT hardcoded 10000)",
                            "conditioning": "NONE (DANN: discriminator on raw features)"},
            "source_cls_loss": "CB-WCE", "cb_beta": CB_BETA,
            "cb_weights": {"airplane": float(cb_w[0].item()), "ship": float(cb_w[1].item())},
            "iw_stripped": True, "entropy_conditioning": False, "conditioned": False, "random_projection": False, "bottleneck": False,
            "literature_default_knobs": "DANN has no method-intrinsic hyperparameter to tune beyond GRL ramp "
                                        "(shared with CDAN母版); strict-UDA: no target labels, no tuning.",
        },
        "data": {"source_train": f"arm2/fold{args.fold}/train (labeled, aug)", "source_fold": args.fold,
                 "n_source": len(src_items), "source_counts": dict(src_cnt),
                 "target": f"klsg full {len(tgt_items)} transductive (sha {_klsg_blind.KLSG_MANIFEST_SHA256[:8]})",
                 "n_target": len(tgt_items),
                 "target_note": "UNLABELED, aug; labels blinded by _klsg_blind.FORBIDDEN_LABEL (never read, never inferred from filepath); fold-independent transductive"},
        "train_history": history,
        "smoke_verdict": {
            "coeff_first": coeff_series[0], "coeff_last": coeff_series[-1], "coeff_ramped": bool(coeff_ramped),
            "disc_bce_first": bce_series[0], "disc_bce_last": bce_series[-1], "ln2_ref": ln2,
            "disc_bce_not_collapsed": bool(bce_not_collapsed),
            "ad_out_src_final": adS_final, "ad_out_tgt_final": adT_final,
            "src_acc_first": acc_series[0], "src_acc_last": acc_series[-1], "src_cls_healthy": bool(src_healthy),
            "tgt_pred_hist_final": tgt_hist_final, "tgt_pred_entropy_final": tgt_ent_final,
            "tgt_pred_collapsed_any": bool(tgt_collapse_any),
            "has_nan_inf": bool(any_nan), "bb_grad_norm_final": history[-1]["bb_grad_norm_mean"],
            "lr_schedule_ok": bool(lr_ok),
            "note": "adversarial dynamics smoke only; NOT performance / model selection. Target health = label-free pseudo-label collapse probe (NO KLSG label read).",
        },
        "has_nan_inf": bool(any_nan),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(curve_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "global_step", "coeff_last", "disc_bce_mean", "ad_out_src_mean",
                    "ad_out_tgt_mean", "src_cls_loss_mean", "src_acc", "bb_grad_norm_mean",
                    "churn",
                    "tgt_pred_airplane", "tgt_pred_ship", "tgt_pred_entropy_mean", "tgt_pred_collapsed", "nan"])
        for h in history:
            churn_cell = "" if h["churn"] is None else f"{h['churn']:.6f}"    # epoch1 null -> 空
            w.writerow([h["epoch"], h["global_step"], f"{h['coeff_last']:.6f}", f"{h['disc_bce_mean']:.6f}",
                        f"{h['ad_out_src_mean']:.6f}", f"{h['ad_out_tgt_mean']:.6f}", f"{h['src_cls_loss_mean']:.6f}",
                        f"{h['src_acc']:.6f}", f"{h['bb_grad_norm_mean']:.6e}",
                        churn_cell,
                        h['tgt_pred_hist'][0], h['tgt_pred_hist'][1], f"{h['tgt_pred_entropy_mean']:.6f}",
                        int(h['tgt_pred_collapsed']), int(h['nan'])])
    # churn argmax 矩阵（首列 epoch + 100 列 argmax，epoch1 起逐 epoch 一行）。决定论 int 矩阵 -> 字节稳定。
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
