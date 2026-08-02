#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_iwcdan_frozen_sgd.py -- **IW-CDAN（CDAN + IW 实例加权）× frozen BN**（SGD 底座）
[SSS sim->real 主实验 B · UDA 对抗训练 · 主角方法 · BN 诊断四档之 frozen]

================================  这是什么 ================================
本文件由 **run3 IW-CDAN × default BN**（`run3_s5_iwcdan/train_iwcdan_sgd.py`, 943 行）复制而来——
**除 BN 处理外与 run3 完全相同**：IW QP+EMA / CDAN GRL / 分域双前向 / SGD+分层lr / CB-WCE / 增强 /
momentum 全部逐字保留。唯一方法改动 = 把训练期 BN 从 default（两域交替更新的混合滑动统计）改成
**frozen（只冻统计、仿射照学）**；另加 provenance 改 run4/frozen + 3 条 frozen 专属断言。
（run3本身由run2 vanilla CDAN复制并加入IW；IW部分移植自Microsoft GLS-IWCDAN参考实现。）

frozen 语义（plan §3.3.2，对抗审已定）：BN running mean/var 自第 0 步冻结（全程保持 ImageNet 预训练统计、
适配期不吸任何域统计），可学习仿射 γ/β 照常随整网微调学。**只冻统计、不冻仿射**——连仿射一起冻会引入
"可学习容量被剥夺"的第二变量、污染 BN 诊断轴（全冻是被弃轴、留作 §3.5 可选消融）。

frozen实现遵循MMDetection ResNet的`norm_eval`惯例——见下方 **FrozenBNCDANResNet18**：
  ① __init__ 末尾遍历 self.modules() 把所有 _BatchNorm m.eval()（belt-and-suspenders，防"构造后未 train() 先 forward"窗口）；
  ② 重载 train(mode)：先 super().train(mode)，再把所有 _BatchNorm m.eval()、return self
     （mmdet 自愈写法：训练期 BN 保持 eval 用冻结 running stats，防任何 .train() 把 BN 翻回更新态）；
  ③ 命门 `requires_grad=True` 原样从基类继承（γ/β 照学）；
  ★ 绝不用 momentum=0 冻统计：train 模式 BN 无论 momentum 多少都按当前 batch 统计归一、momentum=0 只是不写回
     running stats，前向仍不吃 ImageNet 统计；**只有 .eval() 才真按冻结 running stats 归一**。冻结只靠 eval+train()自愈。
  pretrained：真训练/smoke 用 pretrained=True（冻的必须是 ImageNet 真值）；pretrained=False 仅 --unittest 离线用（冻 0/1 废值）。

IW 思路（承自 run3，未改）：用 QP（cvxopt）估计 源/目标 类分布比 ŵ（在广义标签漂移假设下），
用 ŵ **按样本展开后加权对抗对齐损失**，使对抗对齐对标签漂移更鲁棒。

正式数据（承自 IW-CDAN）：源 = 所选半合成训练折（带标签、施加增强）→ 目标 = 全部 553 张
KLSG-II 图像（**无标签**、施加同款增强）。目标真标签不进入 UDA 训练或模型选择。

================================  frozen 专属验证（run3 没有，本文件新增）================================
  --unittest 加 selftest_frozen_bn（pretrained=False 离线、不依赖 ImageNet）：
    (a) model.train() 后所有 BN（resnet18=20 个）全 training==False；(b) BN γ/β requires_grad==True；
    (c) 两次 forward 间 running stats 不变。run3 原有单元验（calc_coeff/batch/grl/IW-1/2/3/4）全保留。
  --quick smoke 加 3 条 frozen 断言：① running stats 全程不动(max|Δ|<1e-6)；② γ/β 在学(max|Δ|>1e-6,宽容差)；
    ③ 每 epoch model.train() 后所有 BN 仍 eval。run3 对抗动力学 + IW 机制 smoke 判读全保留。

================================  去重方案 (b)：IW 只加在对齐损失 ================================
★ 分类损失侧 **保持 run2 的 CB-WCE 一字不改**（criterion(o_s, y_s)），**不**移植 GLS 官方的分类侧 IW 加权
  （GLS train_image.py L284-286: CE(weight=class_weights)×im_weights/class_num）—— 整块不移植（spec ⑦）。
  IW 只作用于 **对抗对齐损失**（cdan_loss 的加权分支）。

================================  从 GLS 移植了什么（IW 5 块）================================
  ① 加权对抗分支         loss.py L58-61            -> cdan_loss(weights!=None) 加权分支（+两处 log clamp, spec ②）
  ② im_weights buffer    network.py L500-501       -> CDANResNet18.register_buffer（**buffer 非 Parameter**, spec ⑤）
  ③ QP+EMA               network.py L496-531       -> solve_im_weights_qp + im_weights_ema_update（spec ⑧⑨④⑤）
  ④ cov/pseudo 累积器    train_image.py L249-255   -> 训练循环 step 累积（verbatim, spec ⑥）
  ⑤ QP 触发+归一+清零    train_image.py L321-341   -> epoch 末触发（mult=1, spec ③⑥）

继承自 run2 的 vanilla CDAN 全部组件（calc_coeff/AdversarialNetwork/GRL register_hook/条件化 multilinear/
GLS 式训练循环/CB-WCE/inverse-decay/分层 lr）—— 与 run2 逐字一致，详见 run2 头注。

================================  IW 落地的关键护栏（最易踩的坑）================================
  ★ spec ① ŵ 按样本展开：传给 cdan_loss 的 weights = ys_onehot(y_s) @ im_weights = (bs,1) 每样本向量，
    **不是 K=2 向量**（直接传 K 会 shape 崩）；ys_onehot 用源 batch 的 y_s，(bs,1) 对齐 ad_out 源半 [:bs]。
  ★ spec ② 两处 log 都 clamp：源半 log(ad_out[:bs].clamp_min(1e-7)) + 目标半 log((1-ad_out[bs:]).clamp_min(1e-7))。
    GLS L59-60 是裸 log 无 clamp；run2 vanilla 走 nn.BCELoss（两侧内置 clamp），手写加权分支没有——
    对抗使目标 ad_out 饱和→1 时 log(1-1)→-inf 会爆 NaN。**两处都 clamp**（别只 clamp 源半）。
  ★ spec ⑤ im_weights = register_buffer（不是 nn.Parameter）——否则被 named_parameters() 扫到混进 backbone 组。
  ★ spec ⑩ 防泄漏：目标标签绝不进损失/不参与 ŵ；不移植 GLS 的 true_weights/ORACLE 作弊分支；
    pseudo_target_label 用模型对目标的**软预测**(softmax)，**不是目标真标签**。run2 已 `x_t,_=next(...)` 丢目标标签。

================================  IW 落地的【有意偏离 / 实测发现】（显式申报）================================
  IW-D1. spec ④ QP 健壮性：实测 cvxopt 对【有限的】病态 cov（全 0 / 秩 1 / 近奇异）**返回 status='optimal'
         且 x=[1,1]（均匀,良性 fallback）**，并不会"静默返坏解"，故这些不触发跳过（良性）。真正触发跳过的是
         **非有限 cov（NaN/Inf，如 logits 爆炸）**——cvxopt 此时 raise。本实现 ④ 三层防护：
         (i) 解前查输入 finite（NaN/Inf→跳过,确定性,不靠 cvxopt raise）；(ii) try/except 包住；
         (iii) 解后查 status=='optimal' 且 x 全 finite。任一不过 -> 跳过本次更新、保留上次 ŵ、打印告警。
  IW-D2. spec IW-2 单测阈值：实测 cvxopt 内点法把"塌 0"那维解到 ~6.8e-4（非严格 0），spec 原阈 ŵ[1]<1e-6
         偏紧会误判。本实现把该阈放宽到 ŵ[1]<1e-2（仍清楚体现"从 1.0 塌向 0"），ŵ[0]<=2+1e-6 保留。
  IW-D3. CDANResNet18 增 `pretrained=True` 形参（默认 True，真训练行为与 run2 完全一致）；仅单测用 pretrained=False
         构造随机权重 backbone，使 --unittest 离线可跑（不依赖 ImageNet 权重缓存/联网）。
  IW-D4. EMA ma=IW_EMA_MA=0.5 —— 与 GLS --ma 默认值一致（已核 train_image.py L368），非偏离。
  IW-D5. 新增 IW-4 集成自测（合成张量过真实 CDANResNet18, 离线）：forward→加权 cdan_loss→累积→QP→EMA→backward
         全链路不崩 + shape/grad 正确。超出 spec 的"IW 三测"，但属"验机制健康"职责内，**新增项已申报**。

继承 run2 的 D1-D4 偏离（512 penultimate 无 bottleneck / CB-WCE / inverse-decay / fc 默认初始化）不再赘述。
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
from torch.nn.modules.batchnorm import _BatchNorm   # frozen: 用 _BatchNorm 基类(覆盖 BN1d/2d/3d)、别用窄 nn.BatchNorm2d
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
from cvxopt import matrix, solvers          # spec (e)：IW QP 求解器
solvers.options['show_progress'] = False    # spec ④：静默 QP 进度（GLS network.py L15 同款）

# [红线护栏] KLSG 目标域盲化入口（单点维护，绝不读真实标签）；src/ 在 parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _klsg_blind  # read_klsg_unlabeled / UnlabeledImageDataset / collate_unlabeled / FORBIDDEN_LABEL
import _run_safety

# ============================================================ 配置
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLITS_CSV = PROJECT_ROOT / "data" / "split" / "splits.csv"            # 源 arm2
# IW-CDAN × frozen 产物落 run4_s5_iwcdan_frozen（与 run1 source-only / run2 vanilla CDAN / run3 IW-CDAN×default 物理隔离）
RESULTS_DIR = PROJECT_ROOT / "results" / "run4_s5_iwcdan_frozen"

CLASSES = ["airplane", "ship"]            # 索引: airplane=0, ship=1
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

# --- SGD 底座超参（与 run2 / source-only 同 regime）---
SEED = 0
EPOCHS_DEFAULT = 40                        # 论文正式配置 N=40
QUICK_EPOCHS = 12                          # smoke：12 epoch × 6 step/epoch = 72 步
BATCH_SIZE = 64
FC_LR = 1e-2                               # 新层 fc
BACKBONE_LR = 1e-3                         # backbone = fc 的 1/10
DISC_LR = 1e-2                             # 判别器 ad_net（= fc lr = 10× backbone；GLS ad_net lr_mult=10）
WEIGHT_DECAY = 5e-4
LR_DECAY_ALPHA = 10.0                      # inverse-decay α
LR_DECAY_BETA = 0.75                       # inverse-decay β
GRL_ALPHA = 10.0                           # GRL ramp α（calc_coeff）
CB_BETA = 0.999                            # CB-WCE effective-number β
SPECKLE_SIGMA = 0.1
SAVE_LAST_K = 5                            # spec §3.1/块④：存末 K 个 epoch ckpt（与 --save-every-epoch 单点存盘、同名幂等）
AD_HIDDEN = 1024                           # 判别器隐藏层宽度（GLS 恒用 1024）
TRADE_OFF = 1.0                            # 旋钮 A：常数，乘在 transfer_loss 上，不 ramp

# --- IW 超参（spec ③⑤b）---
IW_EMA_MA = 0.5                            # spec (b)/⑤：im_weights EMA 动量（= GLS --ma 默认 0.5，IW-D4）
IW_UPDATE_MULT = 1                         # 正式配置：每个 epoch 末触发一次 QP 更新。
                                           #   ★ 绝不设 0：GLS 触发判据含 i%(mult*len) 会除零崩。


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


# ============================================================ CB-WCE（底座沿用；spec ⑦ 分类损失不改）
def cb_wce_weights(class_counts, beta=CB_BETA, num_classes=len(CLASSES), device=None):
    """Class-Balanced 权重（Cui et al. 2019, effective number）：w_c=(1-β)/(1-β^{n_c})，归一化 mean=1。"""
    eff_num = [1.0 - (beta ** n) for n in class_counts]
    w = [(1.0 - beta) / en for en in eff_num]
    s = sum(w)
    w = [wi / s * num_classes for wi in w]
    return torch.tensor(w, dtype=torch.float32, device=device)


# ============================================================ CDAN 对抗组件（移植自 GLS，承自 run2）
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
    ★ max_iter 必须传 total_steps（不是 10000），否则本任务几百步时 coeff 只到 ~0.3。"""
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def grl_hook(coeff):
    """GLS network.py L198-201：反传时把流入特征的梯度取负并按 coeff 缩放（grad <- -coeff*grad）。"""
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1


class AdversarialNetwork(nn.Module):
    """移植自 GLS network.py L367-406（剥 IW Parameter，无改动语义）。内置 GRL register_hook。
    ★ 每个训练 step 只前向一次（吃 concat 后的单 batch），故 iter_num == 训练步数。
    ★ max_iter 由外部置为 total_steps（spec 1）。"""
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


def cdan_loss(features, softmax_output, ad_net, device, weights=None):
    """CDAN 对抗损失（移植自 GLS loss.py CDAN, entropy=None 分支）。spec (a)：
      - weights is None  -> 走 run2 原样 nn.BCELoss 分支（GLS L63；vanilla 兼容，供 IW-1 连续性回归）。
      - weights not None -> 走 **IW 加权分支**（GLS L58-61）+ **两处 log clamp**（spec ②，防对抗饱和爆 NaN）。
    条件化 multilinear f⊗g：bmm 外积拉平（512×C，不随机投影）；g=softmax.detach()（GLS L28）。
    batch_size = size(0)//2 硬切（spec：死假设源==目标 batch）；dc_target=[1]*bs+[0]*bs 对齐 cat(源,目标)。
    spec ①：weights 须为 (bs,1) 每样本向量（= ys_onehot @ im_weights），对齐 ad_out 源半 ad_out[:bs]。
    返回 (loss, ad_out)：ad_out 供诊断（源/目标半均值），避免额外前向。"""
    softmax_output = softmax_output.detach()                      # 对抗梯度只回特征、不回分类头
    batch_size = softmax_output.size(0) // 2
    feature = features
    # multilinear 外积 f⊗g：(2B, C, 1) bmm (2B, 1, 512) -> (2B, C, 512) -> view (2B, C*512)
    op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
    ad_in = op_out.view(-1, softmax_output.size(1) * feature.size(1))
    ad_out = ad_net(ad_in)                                        # GRL register_hook 在 ad_net 内部完成
    if weights is not None:
        # === IW 加权分支（spec ①②；对应 GLS loss.py L58-61，**两处 log 均加 clamp_min**）===
        # 源半：每样本 weights(bs,1) 加权 -log(D(源));  目标半：-log(1 - D(目标))
        weighted_nll_source = -weights * torch.log(ad_out[:batch_size].clamp_min(1e-7))          # spec ② clamp 源半
        nll_target = -torch.log((1.0 - ad_out[batch_size:]).clamp_min(1e-7))                     # spec ② clamp 目标半
        loss = (torch.mean(weighted_nll_source) + torch.mean(nll_target)) / 2.0
        return loss, ad_out
    # === vanilla 分支（GLS L63，run2 原样；nn.BCELoss 两侧内置 clamp）===
    dc_target = torch.from_numpy(
        np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(device)
    loss = nn.BCELoss()(ad_out, dc_target)
    return loss, ad_out


# ============================================================ IW: QP + EMA（移植自 GLS network.py L496-531）
def solve_im_weights_qp(source_y, target_y, cov):
    """在广义标签漂移假设下解 QP 求最优实例权重 ŵ（移植自 GLS create_im_weights_update 内的 QP, network.py L512-525）。
    spec ⑧ 三输入 + 约束：
        source_y : 源类分布(和=1, np.double)  -> 等式约束 A=source_y(1,K), b=[1] (source_y·w=1, 令 ŵ 上界=1/source_y)
        target_y : 目标伪标签分布(K×1, 归一)
        cov      : 源 预测⊗真标签 混淆协方差(K×K, 归一)
        不等式约束 G=-I, h=0  -> w >= 0
    spec ⑨ 两行 **verbatim 别改**（K=2 时 cov 不对称、与最小二乘正规方程不完全配对，但保复现；靠单测 IW-2 兜底）：
        P = matrix(np.dot(cov.T, cov), tc="d");  q = -matrix(np.dot(cov, target_y), tc="d")
    spec ④ 健壮性（GLS 无、极小数据必加）：三层防护（详见头注 IW-D1）。
    返回 (status:str, w_raw or None)：w_raw 为 (K,1) np.array（**QP 原始解, 未经 EMA**）；不可用时返回 (status, None)。"""
    dim = cov.shape[0]
    source_y = np.asarray(source_y).reshape(-1, 1).astype(np.double)
    target_y = np.asarray(target_y).reshape(-1, 1).astype(np.double)
    cov = np.asarray(cov).astype(np.double)
    # ④-(i) 解前查输入 finite（NaN/Inf,如 logits 爆炸 -> 跳过；确定性,不靠 cvxopt 是否 raise）
    if not (np.all(np.isfinite(source_y)) and np.all(np.isfinite(target_y)) and np.all(np.isfinite(cov))):
        return "nonfinite_input", None
    try:
        P = matrix(np.dot(cov.T, cov), tc="d")          # spec ⑨ verbatim
        q = -matrix(np.dot(cov, target_y), tc="d")      # spec ⑨ verbatim
        G = matrix(-np.eye(dim), tc="d")                # w >= 0
        h = matrix(np.zeros(dim), tc="d")
        A = matrix(source_y.reshape(1, -1), tc="d")     # source_y·w = 1
        b = matrix([1.0], tc="d")
        sol = solvers.qp(P, q, G, h, A, b)
    except Exception as e:                              # ④-(ii) QP 病态/数值错 -> 跳过
        return f"exception:{type(e).__name__}", None
    status = sol.get("status", "unknown")
    x = np.array(sol["x"]).flatten()
    # ④-(iii) 解后查 status & finite
    if status != "optimal" or not np.all(np.isfinite(x)):
        return status, None
    return status, x.reshape(-1, 1)


def im_weights_ema_update(im_weights, source_y, target_y, cov, device, ma=IW_EMA_MA):
    """解 QP 得原始 ŵ；若可用则 **EMA 写回 im_weights buffer**（spec ⑤ .copy_ in-place；GLS network.py L528-529）：
        im_weights <- (1-ma)*new + ma*im_weights   (ma=IW_EMA_MA=0.5, 同 GLS --ma 默认)
    若 QP 不可用（spec ④ 跳过）：**保留上次 ŵ 不动**，仅返回 status。
    返回 (status:str, updated:bool, w_raw)：w_raw = QP 原始解 (K,1) np.array（**EMA 前**, 块③落盘用）；
    非 optimal / 跳过时 w_raw=None（spec §3.4 b①）。im_weights 为模型 buffer（K,1）；用 buffer 而非 Parameter 见 spec ⑤。"""
    status, w_raw = solve_im_weights_qp(source_y, target_y, cov)
    if w_raw is None:
        return status, False, None
    new = torch.as_tensor(w_raw, dtype=torch.float32, device=device)
    im_weights.copy_((1.0 - ma) * new + ma * im_weights)        # spec ⑤ EMA in-place 写 buffer
    return status, True, w_raw


# ============================================================ 模型（返回 (feature, logits) + IW buffer）
class CDANResNet18(nn.Module):
    """ResNet18(ImageNet) 整网微调；暴露 **avgpool 后 512 维 penultimate** + fc logits（仿 GLS ResNetFc.forward）。
    条件化/GRL/判别器都挂在这 512 维 penultimate（avgpool 后、fc 前）。无 bottleneck（run2 D1）。
    ★ run4 说明：本类 = run3 原样基类（**default BN**，BN 随 batch 更新 running stats）；本文件训练**不**直接用它，
      而用下方子类 **FrozenBNCDANResNet18**（frozen BN）。保留基类是为：① 让 frozen 子类逐字继承 IW-CDAN 全部内核（只覆盖 BN）；
      ② run3 原有 IW-4 集成单测仍在此基类上跑（IW 机制与 BN 档无关，verbatim 保留）。
    ★ spec ⑤：im_weights 注册为 **register_buffer**（torch.ones(K,1)，**非 nn.Parameter**）——
      否则被优化器 named_parameters() 扫到混进 backbone 组。EMA 用 .copy_ 写（见 im_weights_ema_update）。
    pretrained: True(默认,真训练,与 run2 一致) 载 ImageNet 权重；False 仅供 --unittest 离线构造随机 backbone(IW-D3)。"""
    def __init__(self, num_classes=len(CLASSES), pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet18(weights=weights)  # pretrained=True且未缓存时由torchvision下载；单元验传False。
        self.in_features = base.fc.in_features                    # 512
        self.feature_layers = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4, base.avgpool)
        self.fc = nn.Linear(self.in_features, num_classes)        # 底座默认初始化（run2 D4，不套 GLS xavier）
        # spec ⑤：IW 实例权重 = buffer（非 Parameter）；初始 ones(K,1) = 无重加权
        self.register_buffer("im_weights", torch.ones(num_classes, 1))
        for p in self.parameters():
            p.requires_grad = True                                # 命门：全解冻整网微调（含 BN γ/β；buffer 不在 parameters()）。
            #   ★ frozen 子类原样继承此行：统计由 eval+train()重载冻结、仿射 γ/β 仍可学；绝不借"冻结"把 γ/β 设 requires_grad=False

    def forward(self, x):
        f = self.feature_layers(x)
        f = f.view(f.size(0), -1)                                 # 512 维 penultimate
        y = self.fc(f)
        return f, y

    def output_num(self):
        return self.in_features

    def im_weights_update(self, source_y, target_y, cov, device):
        """spec (b)：QP+EMA 的 model 方法（薄封装 im_weights_ema_update，作用于 self.im_weights buffer）。
        返回 (status:str, updated:bool, w_raw)：透传 im_weights_ema_update 的三元组（w_raw = EMA 前 QP 原始解, 块③落盘）。"""
        return im_weights_ema_update(self.im_weights, source_y, target_y, cov, device)


# ============================================================ frozen BN 模型（#6；只冻统计、仿射照学）
def iter_bn_modules(module):
    """遍历 module 下所有 BN 层；用 _BatchNorm 基类（覆盖 BatchNorm1d/2d/3d），不用窄 nn.BatchNorm2d。frozen 各处复用。"""
    return [m for m in module.modules() if isinstance(m, _BatchNorm)]


class FrozenBNCDANResNet18(CDANResNet18):
    """#6 frozen 档模型 = CDANResNet18（IW-CDAN 内核逐字继承）+ **BN frozen（只冻统计、仿射照学）**。
    frozen 语义：BN running mean/var 冻在 ImageNet 预训练值（适配期不吸任何域统计），γ/β 仿射照常随整网微调学。
    实现遵循MMDetection ResNet的`norm_eval`惯例：
      ① __init__ 末尾把所有 _BatchNorm 置 eval（belt-and-suspenders，防"构造后未调 train() 先 forward"的窗口吸了 batch 统计）；
      ② 重载 train()：super().train(mode) 后再把所有 _BatchNorm 置 eval、return self（自愈：任何 .train() 都不会把 BN 翻回更新态）。
    命门：基类 __init__ 的 `requires_grad=True` 原样继承（γ/β 仍可学）；★ 绝不用 momentum=0 冻统计（见文件头注 ★：
      train 模式下 momentum 多少都按当前 batch 统计归一，只有 eval 才真按冻结 running stats 归一）。"""
    def __init__(self, num_classes=len(CLASSES), pretrained=True):
        super().__init__(num_classes=num_classes, pretrained=pretrained)   # 含命门 requires_grad=True（γ/β 照学）
        # ① belt-and-suspenders：构造期即把 BN 置 eval（防"构造后未 train() 先 forward"窗口）
        for m in iter_bn_modules(self):
            m.eval()

    def train(self, mode=True):
        """mmdet norm_eval 标准写法：训练期 BN 保持 eval、用冻结 running stats（防任何 .train() 把 BN 翻回更新态）。
        本 override 只碰 _BatchNorm，不影响 fc/conv/判别器等其余模块的 train/eval 切换。"""
        super().train(mode)                          # 先按常规把整网切到 mode（BN 也会被翻成 mode）
        for m in iter_bn_modules(self):              # ② 再把所有 BN 强制拨回 eval（冻结 running stats）
            m.eval()
        return self                                  # mmdet 自愈写法：return self


def make_cdan_optimizer(model, ad_net, lr_mult=1.0):  # lr_mult由main/diag论文配置显式约束。
    """分层 lr 的 SGD（+momentum +nesterov）。3 组（在构造 scheduler 之前加判别器组）：
      group 0 = backbone (lr=1e-3) | group 1 = fc (lr=1e-2) | group 2 = 判别器 (lr=1e-2)
    ★ spec ⑤ 护栏：im_weights 是 buffer 不在 named_parameters()，故不会混进 backbone 组（IW-4 单测验证）。
    返回 (optimizer, n_bb, n_fc, n_ad, backbone_params)。"""
    fc_params, backbone_params = [], []
    for name, p in model.named_parameters():
        (fc_params if name.startswith("fc.") else backbone_params).append(p)
    ad_params = list(ad_net.parameters())
    optimizer = torch.optim.SGD(
        [{"params": backbone_params, "lr": BACKBONE_LR * lr_mult},   # group 0（× lr_mult，默认 1.0→逐位一致）
         {"params": fc_params, "lr": FC_LR * lr_mult},               # group 1
         {"params": ad_params, "lr": DISC_LR * lr_mult}],            # group 2
        momentum=0.9, nesterov=True, weight_decay=WEIGHT_DECAY,
    )
    return (optimizer, sum(p.numel() for p in backbone_params),
            sum(p.numel() for p in fc_params), sum(p.numel() for p in ad_params), backbone_params)


def make_scheduler(optimizer, total_steps):
    """底座 inverse-decay（LambdaLR）：每步 factor = (1+α·p)^(-β)，p=global_step/total_steps 归一化[0,1]。
    total_steps 同时是 GRL max_iter。"""
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
    """label-free 目标域伪标签坍缩探测。
    **只用模型自身在目标【图像】上的预测**，绝不读目标真实 label（loader 的 label 位是毒丸 FORBIDDEN_LABEL，
    这里只取 batch[0] 图像、永不触碰 batch[1]）。返回 (pred_hist[list], mean_entropy)。
    判读：pred_hist 全压一类 = 坍缩（不健康）；mean_entropy 趋 0 = 过度自信/可能坍缩。"""
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
    """churn（spec §3.4 a）：模型对固定 churn 100 子集的 argmax 预测向量（长100, int）。
    churn 标量 = 相邻 epoch 此向量的不一致比例（在训练循环里算）。

    ★ 决定论铁律（spec 護栏②）：model.eval() + torch.no_grad() + eval_tfm(无增强) + shuffle=False
      + num_workers=0 -> 零 RNG 消费、零 BN buffer 写（eval 模式 BN 不更新 running stats）。
    ★ frozen 差异（run4）：model.eval() 下 frozen BN 自动用冻结 ImageNet running stats（与训练期 frozen 一致）；
      末尾 model.train() 被 FrozenBNCDANResNet18.train() override 再设回 BN-eval —— 无需任何特判, 照搬 run3。
    ★ 红线（spec 護栏①）：只取 batch[0] 图像；batch[1] 是毒丸 FORBIDDEN_LABEL, 绝不索引/绝不读真实标签。
    调用后 model.train() 复位（沿用 target_pseudolabel_health 的 eval 旁路先例）。返回 np.ndarray(int)。"""
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


def print_metrics_block(title, m):
    print(f"\n  [{title}]  macro-F1 = {m['macro_f1']:.4f} | accuracy = {m['accuracy']:.4f}")
    print(f"  {'class':>9} | {'precision':>9} | {'recall':>9} | {'f1':>9} | {'support':>7}")
    for c in CLASSES:
        d = m["per_class"][c]
        print(f"  {c:>9} | {d['precision']:>9.4f} | {d['recall']:>9.4f} | {d['f1']:>9.4f} | {d['support']:>7}")
    cm = m["confusion_matrix"]
    print(f"  混淆(行=真,列=预测,[airplane,ship]): [[{cm[0][0]},{cm[0][1]}],[{cm[1][0]},{cm[1][1]}]]")


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ============================================================ frozen BN 离线单测（#6 新增）
def selftest_frozen_bn(device):
    """#6 frozen 离线单测（pretrained=False、不依赖 ImageNet 权重/联网）：构造 frozen 模型 -> model.train() ->
       (a) 所有 BN training==False（resnet18=20 个全覆盖）；(b) BN γ/β requires_grad==True（命门：仿射仍学）；
       (c) 两次 forward 间 running stats 不变（eval 模式用冻结 running stats、不写回）。
    返回 ok:bool（不内部 assert，结果交 run_unit_tests 总判）。"""
    print("  " + "-" * 74)
    print("  ★ frozen BN 单测（selftest_frozen_bn, pretrained=False 离线）★")
    fok = True
    torch.manual_seed(0)
    m = FrozenBNCDANResNet18(pretrained=False).to(device)
    m.train()                                                   # 关键：frozen 的 train() override 须让 BN 留在 eval
    bns = iter_bn_modules(m)
    n_bn = len(bns)
    # (a) model.train() 后所有 BN 仍 eval（training==False）—— train() override 自愈
    n_eval = sum(1 for b in bns if not b.training)
    a_ok = (n_bn == 20) and (n_eval == n_bn)
    print(f"  [F-a] BN 模块数={n_bn}(期望20) | model.train() 后 training==False 的={n_eval}/{n_bn} -> {'PASS' if a_ok else 'FAIL'}")
    fok = fok and a_ok
    # (b) BN γ/β requires_grad==True（命门：仿射仍学、不被冻）
    n_aff = sum(1 for b in bns if (b.weight is not None and b.weight.requires_grad
                                   and b.bias is not None and b.bias.requires_grad))
    b_ok = (n_aff == n_bn)
    print(f"  [F-b] BN γ/β requires_grad==True 的={n_aff}/{n_bn}(命门：仿射仍学) -> {'PASS' if b_ok else 'FAIL'}")
    fok = fok and b_ok
    # (c) 两次 forward 间 running stats 不变（eval 模式：按冻结 running stats 归一、不写回）
    before = [(b.running_mean.detach().cpu().clone(), b.running_var.detach().cpu().clone()) for b in bns]
    with torch.no_grad():
        for _ in range(2):
            _ = m(torch.randn(2, 3, IMG_SIZE, IMG_SIZE, device=device))
    after = [(b.running_mean.detach().cpu().clone(), b.running_var.detach().cpu().clone()) for b in bns]
    max_d = (max(max(float((a0 - b0).abs().max()), float((a1 - b1).abs().max()))
                 for (b0, b1), (a0, a1) in zip(before, after)) if bns else 0.0)
    c_ok = max_d < 1e-6
    print(f"  [F-c] 2 次 forward 后 running stats max|Δ(mean,var)|={max_d:.2e}(期望<1e-6, eval 不更新统计) -> {'PASS' if c_ok else 'FAIL'}")
    fok = fok and c_ok
    print(f"  [frozen] selftest_frozen_bn 总判 -> {'PASS' if fok else 'FAIL'}")
    return fok


# ============================================================ 单元验（run2 三测 + IW 三测 + IW-4 集成 + frozen 单测）
def run_unit_tests(device):
    """run2 三测（calc_coeff 端点 / batch 切分 / grl_hook 语义）+ IW 三测（spec 构造给死）+ IW-4 集成(IW-D5)
    + selftest_frozen_bn（#6 frozen 离线单测，新增）。"""
    print("=" * 78)
    print("★ 单元验（smoke 前） ★")
    ok = True

    # ① calc_coeff ramp 端点 + 'max_iter=10000 vs total_steps' 对照
    ts = 600
    c0 = calc_coeff(0, max_iter=ts)
    c_end = calc_coeff(ts, max_iter=ts)
    c_bug = calc_coeff(ts, max_iter=10000.0)
    print(f"  [①coeff] calc_coeff(0,max_iter={ts})           = {c0:.6f}  (期望≈0)")
    print(f"  [①coeff] calc_coeff({ts},max_iter={ts})        = {c_end:.6f}  (期望≈1, 阈值>0.999)")
    print(f"  [①coeff] calc_coeff({ts},max_iter=10000) [坑]  = {c_bug:.6f}  (若不改 max_iter, 末端只到这≈0.29)")
    t1 = (abs(c0) < 1e-6) and (c_end > 0.999)
    print(f"  [①coeff] -> {'PASS' if t1 else 'FAIL'}")
    ok = ok and t1

    # ② batch 等长切分：源/目标各 bs=8, C=2, 512 维 -> concat 16 -> //2 = 8；dc_target=[1]*8+[0]*8
    bs, C, fdim = 8, len(CLASSES), 512
    feat = torch.randn(2 * bs, fdim, requires_grad=True, device=device)
    logit = torch.randn(2 * bs, C, device=device)
    smax = F.softmax(logit, dim=1)
    ad = AdversarialNetwork(fdim * C, AD_HIDDEN, max_iter=ts).to(device)
    ad.train()
    loss_v, ad_out = cdan_loss(feat, smax, ad, device)          # weights=None -> vanilla 分支
    expect_dc = np.array([[1]] * bs + [[0]] * bs)
    split_ok = (ad_out.shape == (2 * bs, 1)) and \
               ((2 * bs) // 2 == bs) and \
               (feat.size(0) // 2 == bs) and \
               (expect_dc[:bs].sum() == bs and expect_dc[bs:].sum() == 0)
    ad_in_dim = smax.size(1) * feat.size(1)
    dim_ok = (ad_in_dim == fdim * C == 1024)
    print(f"  [②split] concat 2x{bs}={2*bs} -> //2={ (2*bs)//2 } (期望{bs}) | ad_out shape={tuple(ad_out.shape)} (期望({2*bs},1))")
    print(f"  [②split] dc_target=[1]*{bs}+[0]*{bs} | 条件化输入维度 C*512={ad_in_dim} (期望1024) | BCE={loss_v.item():.4f}")
    print(f"  [②split] -> {'PASS' if (split_ok and dim_ok) else 'FAIL'}")
    ok = ok and split_ok and dim_ok

    # ③ grl_hook 取负+缩放：loss=(t2*2).sum(), 流入 t2 的 grad=2，hook 应输出 -coeff*2
    t = torch.ones(3, requires_grad=True, device=device)
    t2 = t * 1.0
    t2.register_hook(grl_hook(0.5))
    (t2 * 2.0).sum().backward()
    grl_ok = torch.allclose(t.grad, torch.full_like(t.grad, -1.0), atol=1e-6)  # -0.5*2 = -1.0
    print(f"  [③grl ] grad(t)={t.grad.tolist()} (期望[-1,-1,-1] = -coeff*2, coeff=0.5) -> {'PASS' if grl_ok else 'FAIL'}")
    ok = ok and grl_ok

    # ============ IW 三测（spec 构造给死）============
    # IW-1 连续性恒等：ŵ=ones(bs,1) 时 cdan_loss(weights=ones) == cdan_loss(weights=None)（数学恒等）。
    #   两者都 = (Σ-log(src) + Σ-log(1-tgt))/(2·bs)。喂随机初始化、ad_out≈0.5 不触发 clamp。
    #   ★ ad_net 置 eval()：关 dropout 使两次 forward 确定性同 ad_out（否则 dropout 随机 -> 两次 ad_out 不同, 误判）。
    bs1 = 8
    feat1 = torch.randn(2 * bs1, fdim, device=device, requires_grad=True)   # 须 require grad: ad_net.forward 无条件 register_hook
    smax1 = F.softmax(torch.randn(2 * bs1, C, device=device), dim=1)
    ad1 = AdversarialNetwork(fdim * C, AD_HIDDEN, max_iter=ts).to(device)
    ad1.eval()                                                  # 确定性 forward
    loss_none, ad_o_n = cdan_loss(feat1, smax1, ad1, device, weights=None)
    ones_w = torch.ones(bs1, 1, device=device)
    loss_ones, ad_o_w = cdan_loss(feat1, smax1, ad1, device, weights=ones_w)
    adout_mid = float(ad_o_n.mean().item())                    # 应≈0.5（不触发 clamp）
    iw1_ok = torch.allclose(loss_none, loss_ones, atol=1e-5)
    print(f"  [IW-1] ad_out 均值={adout_mid:.4f}(≈0.5) | loss(weights=None)={loss_none.item():.6f} "
          f"loss(weights=ones)={loss_ones.item():.6f} | |Δ|={abs(loss_none.item()-loss_ones.item()):.2e} (atol=1e-5)")
    print(f"  [IW-1] 连续性恒等(ones==None) -> {'PASS' if iw1_ok else 'FAIL'}")
    ok = ok and iw1_ok

    # IW-2 人造正控：source_y=[.5,.5], target_y=[1,0](目标偏 airplane), cov=diag([.5,.5])(完美分类) -> ŵ≈[2,0]。
    #   验"有漂移时 ŵ 真会动"（当前单元验分布的label shift≈0，只能测试机制）。上界 1/source_y=2。
    #   ★ IW-D2：cvxopt 内点法把塌 0 维解到 ~6.8e-4(非严格0)，spec 原阈 ŵ[1]<1e-6 偏紧 -> 放宽到 <1e-2。
    st2, w2 = solve_im_weights_qp(np.array([.5, .5]), np.array([1., 0.]), np.diag([.5, .5]))
    w2f = None if w2 is None else w2.flatten()
    iw2_ok = (st2 == "optimal") and (w2f is not None) and \
             (w2f[0] > 1.0) and (w2f[0] <= 2.0 + 1e-6) and (w2f[1] < 1e-2)
    print(f"  [IW-2] QP status={st2!r} | ŵ={None if w2f is None else [round(float(v),6) for v in w2f]} "
          f"(期望≈[2,0]: ŵ[0]>1 且 ≤2+1e-6 且 ŵ[1]<1e-2[IW-D2放宽])")
    print(f"  [IW-2] 漂移时 ŵ 朝目标多的方向偏 -> {'PASS' if iw2_ok else 'FAIL'}")
    ok = ok and iw2_ok

    # IW-3 QP 健壮性：喂【非有限】病态 cov(NaN) -> spec ④ 跳过更新 + 不崩 + im_weights 保持上次值。
    #   ★ IW-D1：实测有限病态 cov(全0/秩1)cvxopt 返 optimal+[1,1]（良性,不跳过）；真正触发跳过的是 NaN/Inf。
    buf = torch.tensor([[1.5], [0.5]], dtype=torch.float32, device=device)   # "上次" ŵ
    buf_before = buf.detach().cpu().clone()
    bad_cov = np.array([[np.nan, 0.0], [0.0, 1.0]])                          # 非有限病态 cov（模拟 logits 爆炸）
    st3, up3, _w3raw = im_weights_ema_update(buf, np.array([.5, .5]), np.array([.5, .5]), bad_cov, device)  # 块③: 返回元数 +1（w_raw）
    buf_after = buf.detach().cpu()
    kept = torch.allclose(buf_after, buf_before, atol=0)                     # buffer 一字未动
    # 附核：有限病态 cov(全0) 实测 cvxopt 返 optimal（良性, 文档 IW-D1）——仅打印不 assert
    st_zero, w_zero = solve_im_weights_qp(np.array([.5, .5]), np.array([1., 0.]), np.zeros((2, 2)))
    iw3_ok = (up3 is False) and kept
    print(f"  [IW-3] 非有限 cov: status={st3!r} updated={up3} | ŵ buffer {buf_before.flatten().tolist()}"
          f"->{buf_after.flatten().tolist()} (期望未动)")
    print(f"  [IW-3] 附(IW-D1): 全0 cov -> status={st_zero!r}, ŵ={None if w_zero is None else w_zero.flatten().tolist()} (有限病态=良性均匀,不跳过)")
    print(f"  [IW-3] 病态 QP 跳过更新 + 保留上次 ŵ + 不崩 -> {'PASS' if iw3_ok else 'FAIL'}")
    ok = ok and iw3_ok

    # ============ IW-4 集成自测（IW-D5，合成张量过真实 CDANResNet18, 离线 pretrained=False）============
    #   验全链路: buffer 注册(⑤红线) + forward + 加权 cdan_loss(①②) + cov/pseudo 累积(⑥) + QP+EMA(③) + backward。
    torch.manual_seed(0)
    m = CDANResNet18(pretrained=False).to(device)
    # (a) spec ⑤ 红线：im_weights 必须在 named_buffers 而**非** named_parameters，且不被优化器扫到
    param_names = dict(m.named_parameters())
    buffer_names = dict(m.named_buffers())
    ad_tmp = AdversarialNetwork(m.output_num() * C, AD_HIDDEN, max_iter=100).to(device)
    opt_tmp, _, _, _, _ = make_cdan_optimizer(m, ad_tmp)
    opt_param_ids = {id(p) for g in opt_tmp.param_groups for p in g["params"]}
    buffer_ok = ("im_weights" in buffer_names) and ("im_weights" not in param_names) \
                and (id(m.im_weights) not in opt_param_ids)
    print(f"  [IW-4a] im_weights ∈ named_buffers={('im_weights' in buffer_names)} | "
          f"∈ named_parameters={('im_weights' in param_names)}(期望False) | 入优化器={id(m.im_weights) in opt_param_ids}(期望False)")
    print(f"  [IW-4a] buffer 非 Parameter(spec⑤红线) -> {'PASS' if buffer_ok else 'FAIL'}")
    ok = ok and buffer_ok

    # (b) 全链路集成：2 源 + 2 目标 合成图过真实 backbone
    m.train()
    bs4 = 2
    xs = torch.randn(bs4, 3, IMG_SIZE, IMG_SIZE, device=device)
    xt = torch.randn(bs4, 3, IMG_SIZE, IMG_SIZE, device=device)
    ys = torch.tensor([0, 1], dtype=torch.long, device=device)
    ad4 = AdversarialNetwork(m.output_num() * C, AD_HIDDEN, max_iter=100).to(device); ad4.train()
    f_s, o_s = m(xs); f_t, o_t = m(xt)
    feats = torch.cat((f_s, f_t), 0); outs = torch.cat((o_s, o_t), 0)
    smax4 = F.softmax(outs, 1)
    yoh = torch.zeros(bs4, C, device=device); yoh.scatter_(1, ys.view(-1, 1), 1.0)
    w4 = torch.mm(yoh, m.im_weights)                            # spec ①：(bs,1) 每样本权重；buffer×onehot 无 grad
    w4_ok = (w4.shape == (bs4, 1)) and (not w4.requires_grad)
    tloss, ad_o4 = cdan_loss(feats, smax4, ad4, device, weights=w4)
    closs = F.cross_entropy(o_s, ys)                            # 占位分类损失(集成测;真训练走 CB-WCE criterion)
    (TRADE_OFF * tloss + closs).backward()                     # 验 backward 不崩 + 梯度经 GRL 回 backbone
    grad_ok = any((p.grad is not None and torch.isfinite(p.grad).all().item())
                  for p in m.feature_layers.parameters())
    # cov/pseudo 累积(⑥) + QP + EMA(③)：合成分布
    with torch.no_grad():
        cov_acc = torch.mm(F.softmax(outs[:bs4], 1).t(), yoh).detach()           # (K,K)
        pseudo_acc = F.softmax(outs[bs4:], 1).sum(0).view(-1, 1).detach()        # (K,1)
    cov_shape_ok = (tuple(cov_acc.shape) == (C, C)) and (tuple(pseudo_acc.shape) == (C, 1))
    sld = np.array([0.5, 0.5], dtype=np.double)
    st4, up4, _w4raw = m.im_weights_update(sld, (pseudo_acc / bs4).cpu().numpy(), (cov_acc / bs4).cpu().numpy(), device)  # 块③: 解包 +1（w_raw）
    loss_finite = bool(torch.isfinite(tloss).item() and torch.isfinite(closs).item())
    iw4_ok = w4_ok and (tuple(ad_o4.shape) == (2 * bs4, 1)) and grad_ok and cov_shape_ok and loss_finite
    print(f"  [IW-4b] w4 shape={tuple(w4.shape)}(期望({bs4},1)) grad={w4.requires_grad}(期望False) | "
          f"ad_out={tuple(ad_o4.shape)} | cov{tuple(cov_acc.shape)} pseudo{tuple(pseudo_acc.shape)}")
    print(f"  [IW-4b] transfer_loss={tloss.item():.4f} finite={loss_finite} | backbone梯度经GRL有限={grad_ok} | "
          f"QP集成 status={st4!r} updated={up4}")
    print(f"  [IW-4b] IW 全链路集成(forward→加权loss→累积→QP→EMA→backward) -> {'PASS' if iw4_ok else 'FAIL'}")
    ok = ok and iw4_ok

    # ============ frozen BN 单测（#6 新增；离线 pretrained=False）============
    frozen_ok = selftest_frozen_bn(device)
    ok = ok and frozen_ok

    print(f"  >>> 单元验总判: {'全部 PASS' if ok else '存在 FAIL'}")
    print("=" * 78)
    assert ok, "单元验失败：run2 三测 / IW-1 连续性 / IW-2 正控 / IW-3 健壮性 / IW-4 集成 / frozen 单测 有误，不进 smoke"
    return ok


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None, help="epoch 数（默认 full=40；--quick 时默认 12）")
    ap.add_argument("--quick", action="store_true", help="smoke（默认 12 epoch、产物名 iwcdan_frozen_smoke_*）")
    ap.add_argument("--unittest", action="store_true", help="只跑单元验后退出（不训练）")
    ap.add_argument("--seed", type=int, default=SEED, help="随机种子（默认 0，= D4 基线；改它走 set_seed RNG 四件套）")
    ap.add_argument("--fold", type=int, default=0, help="源 arm2 折号（默认 0）。**仅作用于源 arm2 留出折**；"
                    "KLSG 目标流走全 553 transductive，fold 无关、绝不接此参数")
    ap.add_argument("--save-every-epoch", action="store_true",
                    help="逐epoch存全dict checkpoint（post-hoc分析用；默认只存末K）")
    # main/diag两套论文配置共用同一训练器；下方一致性检查禁止任意组合。
    ap.add_argument("--lr-mult", type=float, default=1.0,
                    help="分层base lr统一乘子：main=1.0，diag=0.1")
    ap.add_argument("--trade-off", type=float, default=1.0,
                    help="对抗 trade_off 乘子（乘在 TRADE_OFF=1.0 上；默认 1.0 = 不改）")
    ap.add_argument("--config-name", choices=["main", "diag"], default=None,
                    help="论文训练配置名（正式训练必填）：main=(lr_mult=1, trade_off=1)；diag=(0.1, 0.25)")
    args = ap.parse_args()

    if args.config_name is None:
        if args.unittest:
            args.config_name = "main"  # 单元验不启动训练；仅用于通过后续配置一致性检查。
        else:
            ap.error("正式训练必须显式提供 --config-name main 或 --config-name diag")

    expected_pair = {"main": (1.0, 1.0), "diag": (0.1, 0.25)}[args.config_name]
    if not (np.isclose(args.lr_mult, expected_pair[0]) and np.isclose(args.trade_off, expected_pair[1])):
        ap.error(
            f"--config-name {args.config_name} 要求 --lr-mult {expected_pair[0]} "
            f"--trade-off {expected_pair[1]}，实得 {args.lr_mult}, {args.trade_off}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"), "CUBLAS_WORKSPACE_CONFIG 未设(须在 import torch 前内联)"

    # ---- 单元验先行（spec：先单元验再 smoke）----
    run_unit_tests(device)
    if args.unittest:
        print("仅单元验模式（--unittest），退出。")
        return 0

    # ★ seed 修复（2026-06-18）：run_unit_tests 内部 torch.manual_seed(0) 会覆盖上面的
    #   set_seed(args.seed) → 训练 RNG 恒从 seed0 起、--seed 失效（跨 seed curve/ckpt 字节相同）。
    #   故在单元验之后、建数据/模型/训练之前重设回 args.seed。
    set_seed(args.seed)

    # 有效trade_off = 公共常数 × 当前论文配置乘子。
    trade_off_eff = TRADE_OFF * args.trade_off
    epochs = (QUICK_EPOCHS if args.quick else EPOCHS_DEFAULT) if args.epochs is None else args.epochs
    tag = "iwcdan_frozen_smoke" if args.quick else "iwcdan_frozen_full"   # iwcdan_frozen_ 前缀，不覆盖 run1/run2/run3 产物

    # spec §3.4(d) / 决议表#2：model_dir 含 seed/fold，防标定矩阵 3seed×2fold 互相覆盖
    model_dir = RESULTS_DIR / f"{tag}_{args.config_name}_s{args.seed}_f{args.fold}"
    metrics_path = model_dir / f"{tag}_metrics.json"
    curve_path = model_dir / f"{tag}_curve.csv"
    churn_argmax_path = model_dir / f"{tag}_churn_argmax.csv"   # spec §3.4(e)：逐 epoch argmax 矩阵
    _run_safety.prepare_empty_run_dirs(model_dir)

    print("=" * 78)
    print(f"run4 #6 **IW-CDAN (CDAN + IW 实例加权) × frozen BN** "
          f"(SGD 底座; config={args.config_name}) ({tag}) : arm2 -> KLSG UDA")
    print("=" * 78)
    print(f"项目根 : {PROJECT_ROOT}")
    print(f"设备   : {device} | torch {torch.__version__} | "
          f"GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # --- 数据：源 arm2/fold/train(带标签,增强) | 目标 KLSG 全 553(无标签,同款增强,transductive,fold无关) ---
    #   ★ 红线：目标 = 全 KLSG 553 张，经 _klsg_blind 盲化（label 位是毒丸 FORBIDDEN_LABEL，绝不读真实标签、
    #     绝不从 filepath 反推类别）；目标流绝不接 args.fold（transductive 全集，fold 无关）。
    src_items = read_split_rows(SPLITS_CSV, domain="arm2", fold=args.fold, split="train")  # 源折由 --fold 选
    tgt_items = _klsg_blind.read_klsg_unlabeled(PROJECT_ROOT)                               # KLSG 全 553（无标签）
    src_cnt = Counter(CLASSES[l] for _, l in src_items)
    print(f"\n源 arm2/fold{args.fold}/train (带标签,增强)     : {len(src_items)} (airplane={src_cnt['airplane']}, ship={src_cnt['ship']})")
    print(f"目标 KLSG 全 {len(tgt_items)} (无标签,同款增强,transductive) : sha {_klsg_blind.KLSG_MANIFEST_SHA256[:8]} "
          f"[label 盲化为毒丸 FORBIDDEN_LABEL, 绝不入损失/不参与 ŵ/绝不读真实类别; fold 无关]")
    assert len(src_items) and len(tgt_items), "数据为空，检查 csv 路径/manifest"
    assert len(tgt_items) == _klsg_blind.KLSG_N_EXPECTED, f"KLSG 目标张数={len(tgt_items)} != 553"

    train_tfm, eval_tfm = build_train_transform(), build_eval_transform()
    # 源/目标 train loader 等长 batch + drop_last=True（cdan_loss size(0)//2 硬切的契约 + IW 累积分母靠 batch 恒定）
    src_loader = DataLoader(ImageListDataset(src_items, train_tfm), batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=0, drop_last=True)
    # 目标：盲化数据集 + 盲化 collate（绝不 stack 标签、绝不 default_collate）；增强+shuffle+drop_last
    tgt_loader = DataLoader(_klsg_blind.UnlabeledImageDataset(tgt_items, train_tfm), batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=0, drop_last=True, collate_fn=_klsg_blind.collate_unlabeled)
    # 目标域伪标签坍缩探测 loader（不增强、确定性、shuffle=False、无标签）——label-free 健康检查用
    tgt_probe_loader = DataLoader(_klsg_blind.UnlabeledImageDataset(tgt_items, eval_tfm), batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=0, collate_fn=_klsg_blind.collate_unlabeled)
    # churn 固定 100 子集 loader（spec §3.4 a；与 tgt_probe_loader 同口径：eval_tfm 无增强/shuffle=False/num_workers=0
    #   -> 零 RNG 消费、零 buffer 写, 護栏②）。盲化入场: read_klsg_churn_subset(sha 锚 + 毒丸 label, 護栏①)。
    #   ★ 构造在所有 RNG-消费 loader 之后, shuffle=False 不消费 RNG, 不扰动决定论（run2x 终极证明）。
    churn_items = _klsg_blind.read_klsg_churn_subset(PROJECT_ROOT)
    assert len(churn_items) == _klsg_blind.CHURN_N_EXPECTED, \
        f"churn 子集张数={len(churn_items)} != {_klsg_blind.CHURN_N_EXPECTED}"
    churn_loader = DataLoader(_klsg_blind.UnlabeledImageDataset(churn_items, eval_tfm), batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, collate_fn=_klsg_blind.collate_unlabeled)
    print(f"churn 固定子集 (label-free, sha {_klsg_blind.KLSG_CHURN100_SHA256[:8]}) : {len(churn_items)} 张 "
          f"[eval_tfm 无增强/shuffle=False/num_workers=0; 只取图像、label 位毒丸绝不读]")
    assert len(src_loader) >= 1 and len(tgt_loader) >= 1, "drop_last=True 后 loader 为空：batch 太大/数据太少"

    # --- CB-WCE 权重（从源训练集类频；spec ⑦ 分类损失侧, 与 run2 一字不改）---
    counts = [src_cnt[c] for c in CLASSES]
    assert counts == [72, 320], f"CB-WCE 护栏（块1钉②）：arm2 各折 train 类频应恒为 [72,320]，实得 {counts}（--fold 放开后防权重静默漂）"
    cb_w = cb_wce_weights(counts, beta=CB_BETA, device=device)
    ratio = (cb_w[0] / cb_w[1]).item()
    print(f"\nCB-WCE (β={CB_BETA}) 权重 (从 arm2 train 类频 {counts}): "
          f"airplane={cb_w[0].item():.4f}, ship={cb_w[1].item():.4f} (airplane:ship = {ratio:.2f}:1)  [分类损失侧, 非 IW]")
    criterion = nn.CrossEntropyLoss(weight=cb_w)                # 源分类损失（CB-WCE；spec ⑦ 不动）

    # --- IW: 源类分布(spec ⑧/d)：arm2 train 类频归一(和=1, np.double)。QP 等式约束 source_y·w=1 用它 ---
    source_label_distribution = np.array([src_cnt[c] for c in CLASSES], dtype=np.double)
    source_label_distribution /= source_label_distribution.sum()
    print(f"源类分布 source_label_distribution (QP 约束 source_y·w=1, ŵ 上界=1/source_y): "
          f"{[round(float(v),4) for v in source_label_distribution]}")

    # --- 步数预算：扁平步数，源 loader 驱动 epoch，目标 loader 循环复用 ---
    steps_per_epoch = len(src_loader)                          # = floor(n_src / batch)
    total_steps = epochs * steps_per_epoch                     # lr inverse-decay 的 p 分母 + GRL max_iter

    # --- 模型 / 判别器 / 优化器 / 调度器 ---
    model = FrozenBNCDANResNet18().to(device)                  # frozen BN：pretrained=True（冻的必须是 ImageNet 真值）
    ad_in_dim = model.output_num() * len(CLASSES)              # 512 × C(=2) = 1024（不随机投影）
    ad_net = AdversarialNetwork(ad_in_dim, AD_HIDDEN, max_iter=total_steps).to(device)
    optimizer, n_bb, n_fc, n_ad, backbone_params = make_cdan_optimizer(model, ad_net, lr_mult=args.lr_mult)
    scheduler = make_scheduler(optimizer, total_steps)
    n_total = sum(p.numel() for p in model.parameters())

    print(f"\n模型 : ResNet18(ImageNet) 全解冻整网微调 | frozen BN (running stats 冻 ImageNet, γ/β 仿射照学) | penultimate=512 维 (无 bottleneck) | im_weights=buffer(K,1)")
    print(f"判别器 : AdversarialNetwork(in={ad_in_dim}=512×{len(CLASSES)}, hidden={AD_HIDDEN}) | 内置 GRL | max_iter={total_steps}")
    print(f"分层 lr : backbone={BACKBONE_LR}({n_bb:,}) | fc={FC_LR}({n_fc:,}) | 判别器={DISC_LR}({n_ad:,}) | 可训/总={n_bb+n_fc:,}/{n_total:,} | lr_mult={args.lr_mult}（有效 bb={BACKBONE_LR*args.lr_mult:.3e}/fc={FC_LR*args.lr_mult:.3e}/disc={DISC_LR*args.lr_mult:.3e}）")
    print(f"优化 : SGD(m=0.9,nesterov) wd={WEIGHT_DECAY} | batch {BATCH_SIZE}(源)+{BATCH_SIZE}(目标) | {epochs} epoch")
    print(f"调度 : inverse-decay lr=base·(1+{int(LR_DECAY_ALPHA)}·p)^(-{LR_DECAY_BETA}) | steps/epoch={steps_per_epoch} total_steps={total_steps}")
    print(f"对抗 : 旋钮A trade_off={trade_off_eff}(=TRADE_OFF×{args.trade_off},常数不ramp) | 旋钮B coeff=2/(1+e^(-{int(GRL_ALPHA)}·p))-1 (0→1 ramp)")
    print(f"IW  : IW 只加权对齐损失(分类侧保持 CB-WCE) | QP+EMA(ma={IW_EMA_MA}) | 更新频率 mult={IW_UPDATE_MULT}(每个 epoch 末)")
    print(f"  ★ total_loss = {trade_off_eff}·transfer_loss(IW 加权判别器) + cls_loss(CB-WCE源); ŵ=ys_onehot@im_weights 按样本加权对齐")

    # --- 训练循环（run2 GLS 式 + IW 累积/QP）---
    print("-" * 78)
    print("smoke 每 epoch 诊断: coeff | 判别器对齐loss(均) | ad_out 源/目标均值(健康→0.5) | "
          "源cls_loss/acc | backbone梯度范数 | ŵ(EMA后) | QP status | NaN")
    print("-" * 78)
    history = []
    coeff_trace = []
    churn_argmax_rows = []     # spec §3.4(e)：逐 epoch (epoch, argmax 长100 向量), epoch1 起
    a_prev = None              # 上一 epoch churn argmax 向量；epoch1 时 None -> churn 标量记 null
    global_step = 0
    any_nan = False
    K = len(CLASSES)
    target_iter = iter(tgt_loader)
    # IW 累积器（spec ⑥；跨 mult-epoch 窗口累积, 触发后清零）
    cov_mat = torch.zeros(K, K, device=device)
    pseudo_target_label = torch.zeros(K, 1, device=device)
    iw_acc_samples = 0                                          # 窗口内累积样本数(= Σ bs)，做归一分母
    last_qp_status = "init(ŵ=ones)"
    # === frozen 断言快照（训练前；#6 专属，run3 无）：①running stats ②γ/β 训练末对比 ===
    frozen_bns = iter_bn_modules(model)
    assert len(frozen_bns) == 20, f"frozen: 预期 resnet18 有 20 个 BN，实得 {len(frozen_bns)}"
    bn_run_before = [(b.running_mean.detach().cpu().clone(), b.running_var.detach().cpu().clone()) for b in frozen_bns]
    bn_aff_before = [(b.weight.detach().cpu().clone(), b.bias.detach().cpu().clone()) for b in frozen_bns]
    for epoch in range(1, epochs + 1):
        model.train(); ad_net.train()
        # frozen 断言③：model.train() 后所有 BN 仍 eval（train() override 自愈；run3 无此断言）
        _bn_on = [i for i, b in enumerate(frozen_bns) if b.training]
        assert not _bn_on, f"frozen③破：epoch {epoch} model.train() 后仍有 BN 处于 training 态 idx={_bn_on}（train() override 未生效？）"
        ep_bce, ep_cls, ep_adS, ep_adT, ep_gn = 0.0, 0.0, 0.0, 0.0, 0.0
        ep_correct, ep_seen, ep_steps = 0, 0, 0
        ep_coeff_last = 0.0
        for x_s, y_s in src_loader:
            try:
                x_t, _ = next(target_iter)                     # **丢目标标签**（spec ⑩ 防泄漏）
            except StopIteration:
                target_iter = iter(tgt_loader)
                x_t, _ = next(target_iter)
            x_s, y_s = x_s.to(device, non_blocking=True), y_s.to(device, non_blocking=True)
            x_t = x_t.to(device, non_blocking=True)

            f_s, o_s = model(x_s)
            f_t, o_t = model(x_t)
            assert f_s.size(0) == f_t.size(0), "契约破：源/目标 batch 不等长（检查 drop_last/batch_size）"
            features = torch.cat((f_s, f_t), dim=0)
            outputs = torch.cat((o_s, o_t), dim=0)
            softmax_out = F.softmax(outputs, dim=1)
            bs = x_s.size(0)

            # === IW: 每样本权重 ŵ (spec ①⑩)：ys_onehot(源 y_s) @ im_weights -> (bs,1)，对齐 ad_out 源半 ===
            ys_onehot = torch.zeros(bs, K, device=device)
            ys_onehot.scatter_(1, y_s.view(-1, 1), 1.0)                 # 仅用源标签做 onehot（目标标签从不参与）
            weights = torch.mm(ys_onehot, model.im_weights)            # (bs,1)；buffer×onehot 天然 requires_grad=False (spec c)

            cls_loss = criterion(o_s, y_s)                            # spec ⑦：CB-WCE，仅源、仅 logits（一字不改）
            transfer_loss, ad_out = cdan_loss(features, softmax_out, ad_net, device, weights=weights)  # IW 加权对齐
            total_loss = trade_off_eff * transfer_loss + cls_loss         # 旋钮 A：trade_off_eff=TRADE_OFF×--trade-off（默认 1.0→逐位一致）

            # === IW: 累积 cov/pseudo (spec ⑥, verbatim GLS train_image L251-255；全 detach)===
            with torch.no_grad():
                src_logits = outputs[:bs]                             # = o_s
                tgt_logits = outputs[bs:]                             # = o_t
                cov_mat += torch.mm(F.softmax(src_logits, dim=1).t(), ys_onehot).detach()        # (K,K)
                pseudo_target_label += F.softmax(tgt_logits, dim=1).sum(0).view(-1, 1).detach()  # (K,1)
                iw_acc_samples += bs

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

            cur_coeff = ad_net.current_coeff()
            coeff_trace.append((global_step, cur_coeff))
            ep_coeff_last = cur_coeff
            adb = ad_out.size(0) // 2
            ep_bce += float(transfer_loss.item())
            ep_cls += float(cls_loss.item())
            ep_adS += float(ad_out[:adb].mean().item())
            ep_adT += float(ad_out[adb:].mean().item())
            ep_gn += bb_grad_norm
            ep_correct += int((o_s.argmax(1) == y_s).sum().item())
            ep_seen += int(y_s.size(0))
            ep_steps += 1
            if not np.isfinite(float(total_loss.item())):
                any_nan = True

        # === IW: epoch 末按 mult 触发 QP 更新 (spec ③⑥⑤④)===
        w_raw_ep = None                                               # spec §3.4(b)：本 epoch EMA 前 QP 原始解；非 fire/非 optimal 留 None
        if epoch % IW_UPDATE_MULT == 0:
            denom = max(iw_acc_samples, 1)                            # = Σ bs(窗口内) = 实际累积 step 数 × batch
            cov_np = (cov_mat / denom).cpu().numpy()                  # 归一(分母=实际累积样本数)
            pseudo_np = (pseudo_target_label / denom).cpu().numpy()
            last_qp_status, _updated, w_raw_ep = model.im_weights_update(   # 解 QP + EMA 写回 buffer; w_raw_ep=EMA 前原始解(块③)
                source_label_distribution, pseudo_np, cov_np, device)
            cov_mat.zero_(); pseudo_target_label.zero_(); iw_acc_samples = 0   # 清零(spec ⑥)
            if not _updated:
                print(f"  [IW] ⚠ epoch {epoch}: QP 未更新(status={last_qp_status!r}) -> 保留上次 ŵ (spec ④)")

        cur_w = model.im_weights.detach().cpu().numpy().flatten().tolist()

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
        # churn（spec §3.4 a；紧邻 health 探测）：固定100子集 argmax 相邻 epoch 翻转率 —— label-free, 只用模型预测
        a_e = churn_predict(model, churn_loader, device)                       # 决定论 eval 旁路, 零 RNG/零 buffer 写
        churn_e = float((a_e != a_prev).mean()) if a_prev is not None else None  # epoch1: a_prev=None -> null
        a_prev = a_e
        churn_argmax_rows.append((epoch, a_e.tolist()))                        # argmax 向量 epoch1 起逐 epoch 存
        model.train(); ad_net.train()

        im_weights_raw = (w_raw_ep.flatten().tolist() if w_raw_ep is not None else None)  # spec §3.4(e)：EMA 前 QP 原始解或 null
        history.append({"epoch": epoch, "global_step": global_step, "coeff_last": ep_coeff_last,
                        "disc_bce_mean": bce_m, "ad_out_src_mean": adS_m, "ad_out_tgt_mean": adT_m,
                        "src_cls_loss_mean": cls_m, "src_acc": src_acc, "bb_grad_norm_mean": gn_m,
                        "im_weights": cur_w, "im_weights_raw": im_weights_raw, "qp_status": last_qp_status,
                        "churn": churn_e,
                        "tgt_pred_hist": list(tgt_hist), "tgt_pred_entropy_mean": tgt_ent,
                        "tgt_pred_collapsed": bool(tgt_collapsed), "nan": bool(ep_nan)})
        churn_str = "n/a" if churn_e is None else f"{churn_e:.3f}"
        print(f"  ep {epoch:>3}/{epochs} | coeff {ep_coeff_last:.4f} | D_loss {bce_m:.4f} "
              f"| adOut S{adS_m:.3f}/T{adT_m:.3f} | cls {cls_m:.4f} acc {src_acc:.4f} "
              f"| bbGrad {gn_m:.2e} | ŵ[{cur_w[0]:.3f},{cur_w[1]:.3f}] | QP {last_qp_status} | churn {churn_str} "
              f"| tgtPred {tgt_hist} H{tgt_ent:.3f}{' COLLAPSE!' if tgt_collapsed else ''}{' | NaN!' if ep_nan else ''}")
        # spec §3.1/块④：单点存盘 —— 默认存末 SAVE_LAST_K，--save-every-epoch 则全 epoch（末K ⊂ every、同名幂等不双写）
        if args.save_every_epoch or epoch > epochs - SAVE_LAST_K:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "ad_net_state_dict": ad_net.state_dict(),
                        "im_weights": model.im_weights.detach().cpu(),
                        "global_step": global_step},
                       model_dir / f"{tag}_epoch{epoch:03d}.pth")

    # --- coeff ramp 轨迹（首/中/末）---
    print("-" * 78)
    print("★ GRL coeff ramp 轨迹（max_iter=total_steps 修没修）★")
    if coeff_trace:
        mid = len(coeff_trace) // 2
        for label, (gs, cv) in [("首", coeff_trace[0]), ("中", coeff_trace[mid]), ("末", coeff_trace[-1])]:
            print(f"  {label}步 global_step={gs:>5} (p={gs/total_steps:6.4f}) -> coeff = {cv:.6f}")
        coeff_end_ok = coeff_trace[-1][1] > 0.9
        print(f"  末端 coeff {'已爬到 ~1.0 (对抗已打开)' if coeff_end_ok else '偏低 (检查 max_iter)'} | "
              f"对照: 若误用 max_iter=10000 末端只会到 {calc_coeff(total_steps, max_iter=10000.0):.4f}")

    print("★ lr inverse-decay sanity（末端 factor = (1+α)^(-β)）★")
    expected_factor = (1.0 + LR_DECAY_ALPHA) ** (-LR_DECAY_BETA)
    base_lrs = list(scheduler.base_lrs)
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

    # --- 对抗 + IW 动力学健康判读（smoke 判据；**不读性能、不定参**）---
    print("-" * 78)
    print("★ 对抗 + IW 机制健康判读（smoke；**不读性能、不下科学结论、不定参**）★")
    bce_series = [h["disc_bce_mean"] for h in history]
    coeff_series = [h["coeff_last"] for h in history]
    acc_series = [h["src_acc"] for h in history]
    ln2 = float(np.log(2.0))
    coeff_ramped = coeff_series[-1] > 0.9 and coeff_series[-1] > coeff_series[0]
    bce_not_collapsed = bce_series[-1] > 0.2
    adS_final, adT_final = history[-1]["ad_out_src_mean"], history[-1]["ad_out_tgt_mean"]
    src_healthy = acc_series[-1] > 0.5
    w_first, w_last = history[0]["im_weights"], history[-1]["im_weights"]
    w_moved = abs(w_last[0] - 1.0) > 1e-4 or abs(w_last[1] - 1.0) > 1e-4          # ŵ 是否从初始 ones 动过
    qp_statuses = [h["qp_status"] for h in history]
    qp_ever_optimal = any(s == "optimal" for s in qp_statuses)
    print(f"  ① coeff ramp     : 首{coeff_series[0]:.3f} -> 末{coeff_series[-1]:.3f}  [{'真 ramp 到~1' if coeff_ramped else '未爬升'}]")
    print(f"  ② 判别器对齐loss : 首{bce_series[0]:.3f} -> 末{bce_series[-1]:.3f} (参照 ln2={ln2:.3f})  [{'未塌(健康)' if bce_not_collapsed else '坠向0(对抗失效?)'}]")
    print(f"  ③ ad_out 均值末  : 源{adS_final:.3f} / 目标{adT_final:.3f}  (健康→趋 0.5)")
    print(f"  ④ 源分类 acc     : 首{acc_series[0]:.3f} -> 末{acc_series[-1]:.3f}  [{'未崩' if src_healthy else '偏低-复核'}]")
    print(f"  ⑤ NaN/Inf 守卫   : {'发现 NaN/Inf!' if any_nan else '全程无 NaN/Inf'}")
    print(f"  ⑥ backbone 梯度  : 末 epoch 范数 {history[-1]['bb_grad_norm_mean']:.3e} (有限即对抗信号经 GRL 正常回传)")
    print(f"  ⑦ IW: ŵ 轨迹     : 首{[round(v,3) for v in w_first]} -> 末{[round(v,3) for v in w_last]}  "
          f"[{'ŵ 已随 QP 移动(IW 在动)' if w_moved else 'ŵ 仍≈ones(QP 未生效?)'}] | QP 曾 optimal={qp_ever_optimal}")
    tgt_collapse_any = any(h["tgt_pred_collapsed"] for h in history)
    tgt_hist_final = history[-1]["tgt_pred_hist"]
    tgt_ent_final = history[-1]["tgt_pred_entropy_mean"]
    print(f"  ⑧ 目标伪标签坍缩 : 末 epoch 预测直方 {tgt_hist_final} 熵{tgt_ent_final:.3f}  "
          f"[{'坍缩过(全预测同类!)' if tgt_collapse_any else '未坍缩(两类都有预测)'}]  (label-free, 只用模型预测、不读目标真实标签)")
    print(f"  [!] ŵ 轨迹/目标伪标签坍缩探测 仅机制健康(label-free)，**非成功判据、不用于选模/调参**（绝不读 KLSG 真实标签）")

    # --- frozen BN 三断言（#6 专属；run3 无；冻结只靠 eval+train()自愈，仿射 γ/β 照学）---
    print("-" * 78)
    print("★ frozen BN 三断言（#6 专属）: running stats 冻 ImageNet / γ/β 在学 / train() 后 BN 恒 eval ★")
    bn_run_after = [(b.running_mean.detach().cpu().clone(), b.running_var.detach().cpu().clone()) for b in frozen_bns]
    bn_aff_after = [(b.weight.detach().cpu().clone(), b.bias.detach().cpu().clone()) for b in frozen_bns]
    frozen_run_max_d = max(max(float((a0 - b0).abs().max()), float((a1 - b1).abs().max()))
                           for (b0, b1), (a0, a1) in zip(bn_run_before, bn_run_after))
    frozen_aff_max_d = max(max(float((a0 - b0).abs().max()), float((a1 - b1).abs().max()))
                           for (b0, b1), (a0, a1) in zip(bn_aff_before, bn_aff_after))
    frozen_stats_unchanged = frozen_run_max_d < 1e-6
    frozen_affine_moved = frozen_aff_max_d > 1e-6
    print(f"  ① running stats 全程不动 : max|Δ(mean,var)| = {frozen_run_max_d:.3e}  (期望<1e-6) -> {'PASS' if frozen_stats_unchanged else 'FAIL'}")
    print(f"  ② γ/β 仿射在学           : max|Δ(γ,β)|       = {frozen_aff_max_d:.3e}  (期望>1e-6, 宽容差: lr小步少, wd 保非零) -> {'PASS' if frozen_affine_moved else 'FAIL'}")
    print(f"  ③ train() 后 BN 恒 eval   : 全 {epochs} epoch × {len(frozen_bns)} BN 逐 epoch 硬 assert 已通过 -> PASS")
    assert frozen_stats_unchanged, f"frozen①破：running stats 动了 max|Δ|={frozen_run_max_d:.3e}（eval/train override 未生效？）"
    assert frozen_affine_moved, f"frozen②破：γ/β 未移动 max|Δ|={frozen_aff_max_d:.3e}（仿射被冻？检查命门 requires_grad）"

    # --- 存末 epoch checkpoint 已上移进训练循环（spec §3.1/块④：单点存盘 if save_every_epoch or epoch>epochs-SAVE_LAST_K，循环后不再单存）---

    # --- 落盘 ---
    out = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run_tag": tag,
        "method": "IW-CDAN (CDAN + IW instance weighting on alignment loss only) x frozen BN, SGD base",
        "config": {
            "config_name": args.config_name,
            "seed": args.seed, "fold": args.fold, "epochs": epochs, "batch_size_source": BATCH_SIZE, "batch_size_target": BATCH_SIZE,
            "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
            "model": "resnet18_imagenet_full_finetune", "bn": "frozen", "penultimate_dim": model.output_num(),
            "bn_note": "frozen running stats (ImageNet pretrained), affine gamma/beta still learnable; eval-mode + train() override (mmdet norm_eval), NOT momentum=0",
            "discriminator": {"in_feature": ad_in_dim, "hidden": AD_HIDDEN, "max_iter": total_steps,
                              "lr": DISC_LR, "note": "conditioning = penultimate(512) x class(2) = 1024, no random projection"},
            "optimizer": {"type": "SGD", "momentum": 0.9, "nesterov": True, "weight_decay": WEIGHT_DECAY},
            "layered_lr": {"backbone_lr": BACKBONE_LR, "fc_lr": FC_LR, "disc_lr": DISC_LR,
                           "lr_mult": args.lr_mult, "effective_backbone_lr": BACKBONE_LR*args.lr_mult,
                           "effective_fc_lr": FC_LR*args.lr_mult, "effective_disc_lr": DISC_LR*args.lr_mult},
            "scheduler": {"type": "inverse_decay_LambdaLR", "alpha": LR_DECAY_ALPHA, "beta": LR_DECAY_BETA,
                          "formula": "lr=base*(1+alpha*p)^(-beta), p=global_step/total_steps"},
            "adversarial": {"trade_off": trade_off_eff, "trade_off_base": TRADE_OFF, "trade_off_mult": args.trade_off,
                            "trade_off_note": "effective = base x --trade-off mult; constant, NOT ramped",
                            "grl_coeff": "calc_coeff=2/(1+exp(-alpha*p))-1", "grl_alpha": GRL_ALPHA,
                            "grl_max_iter": total_steps, "conditioning": "multilinear bmm f(x)g, softmax detached"},
            "iw": {"enabled": True, "applied_to": "alignment loss only (dedup plan b)",
                   "classification_loss_iw": False, "classification_loss": "CB-WCE (run2 unchanged)",
                   "im_weights": "register_buffer (NOT nn.Parameter)", "ema_ma": IW_EMA_MA,
                   "update_mult": IW_UPDATE_MULT, "update_mult_note": "final setting: QP update at the end of every epoch",
                   "qp": "cvxopt, A=source_y(source_y.w=1), G=-I(w>=0), P=cov.T@cov, q=-cov@target_y (verbatim GLS)",
                   "weighted_branch_clamp": "both logs clamp_min(1e-7) (anti-NaN, beyond GLS)",
                   "source_label_distribution": [float(v) for v in source_label_distribution],
                   "anti_leak": "target labels never in loss / never in weight estimation; pseudo-target = softmax (not true labels); no ORACLE"},
            "source_cls_loss": "CB-WCE", "cb_beta": CB_BETA,
            "cb_weights": {"airplane": float(cb_w[0].item()), "ship": float(cb_w[1].item())},
            "entropy_conditioning": False, "random_projection": False, "bottleneck": False,
        },
        "data": {"source_train": f"arm2/fold{args.fold}/train (labeled, aug)", "source_fold": args.fold,
                 "n_source": len(src_items), "source_counts": dict(src_cnt),
                 "target": f"klsg full {len(tgt_items)} transductive (sha {_klsg_blind.KLSG_MANIFEST_SHA256[:8]})",
                 "n_target": len(tgt_items),
                 "target_note": "UNLABELED, aug; labels blinded by _klsg_blind.FORBIDDEN_LABEL (never read, never inferred from filepath, never in ŵ); fold-independent transductive"},
        "train_history": history,
        "smoke_verdict": {
            "coeff_first": coeff_series[0], "coeff_last": coeff_series[-1], "coeff_ramped": bool(coeff_ramped),
            "disc_bce_first": bce_series[0], "disc_bce_last": bce_series[-1], "ln2_ref": ln2,
            "disc_bce_not_collapsed": bool(bce_not_collapsed),
            "ad_out_src_final": adS_final, "ad_out_tgt_final": adT_final,
            "src_acc_first": acc_series[0], "src_acc_last": acc_series[-1], "src_cls_healthy": bool(src_healthy),
            "has_nan_inf": bool(any_nan), "bb_grad_norm_final": history[-1]["bb_grad_norm_mean"],
            "lr_schedule_ok": bool(lr_ok),
            "im_weights_first": w_first, "im_weights_last": w_last, "im_weights_moved": bool(w_moved),
            "qp_ever_optimal": bool(qp_ever_optimal), "qp_status_last": qp_statuses[-1],
            "frozen_bn_count": len(frozen_bns),
            "frozen_running_stats_max_delta": frozen_run_max_d,
            "frozen_running_stats_unchanged": bool(frozen_stats_unchanged),
            "frozen_affine_max_delta": frozen_aff_max_d,
            "frozen_affine_moved": bool(frozen_affine_moved),
            "frozen_bn_stayed_eval_all_epochs": True,
            "tgt_pred_hist_final": tgt_hist_final, "tgt_pred_entropy_final": tgt_ent_final,
            "tgt_pred_collapsed_any": bool(tgt_collapse_any),
            "note": "adversarial + IW + frozen-BN mechanism smoke only; NOT performance / model selection. im_weights trajectory + label-free target pseudo-label collapse probe are diagnostics only (NO KLSG label read).",
        },
        "has_nan_inf": bool(any_nan),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(curve_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "global_step", "coeff_last", "disc_bce_mean", "ad_out_src_mean",
                    "ad_out_tgt_mean", "src_cls_loss_mean", "src_acc", "bb_grad_norm_mean",
                    "im_weight_airplane", "im_weight_ship", "raw_w_airplane", "raw_w_ship", "qp_status",
                    "churn",
                    "tgt_pred_airplane", "tgt_pred_ship", "tgt_pred_entropy_mean", "tgt_pred_collapsed", "nan"])
        for h in history:
            rw = h["im_weights_raw"]                                          # [w0,w1] 或 None
            raw_a = "" if rw is None else f"{rw[0]:.6f}"
            raw_s = "" if rw is None else f"{rw[1]:.6f}"
            churn_cell = "" if h["churn"] is None else f"{h['churn']:.6f}"    # epoch1 null -> 空
            w.writerow([h["epoch"], h["global_step"], f"{h['coeff_last']:.6f}", f"{h['disc_bce_mean']:.6f}",
                        f"{h['ad_out_src_mean']:.6f}", f"{h['ad_out_tgt_mean']:.6f}", f"{h['src_cls_loss_mean']:.6f}",
                        f"{h['src_acc']:.6f}", f"{h['bb_grad_norm_mean']:.6e}",
                        f"{h['im_weights'][0]:.6f}", f"{h['im_weights'][1]:.6f}", raw_a, raw_s, h["qp_status"],
                        churn_cell,
                        h['tgt_pred_hist'][0], h['tgt_pred_hist'][1], f"{h['tgt_pred_entropy_mean']:.6f}",
                        int(h['tgt_pred_collapsed']), int(h['nan'])])
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
    print(f"checkpoint -> {model_dir}")
    print(f"NaN/Inf : {any_nan}")
    print("=" * 78)
    return 1 if any_nan else 0


if __name__ == "__main__":
    sys.exit(main())
