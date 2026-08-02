#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_mdd_sgd.py -- 【候补·[E] 事后补充实验】 **MDD × default BN**（SGD 底座）
[SSS sim->real 主实验 B · 事后补充方法 · margin理论方法]

================================  这是什么 / 身份 ================================
**MDD**（Margin Disparity Discrepancy, Zhang et al. ICML 2019, "Bridging Theory and Algorithm
for Domain Adaptation", arXiv:1904.05801）——基于 margin 理论的对抗 DA：在主分类头 f 之外引入一个
**辅助对抗分类头 f'**（经 GRL），用「f' 在源上认同 f、在目标上背离 f 的程度」度量域差（margin
disparity discrepancy），GRL 使特征提取器 F 最小化此差（对齐域）。

★ **身份 = 候补 / [E] 事后补充**：受统一实验协议约束
  （①同冻结协议 ②论文标"事后补充·[E]" ③不替换 M1/M2/M3 主对比）。
  MDD 当初落选主表理由（plan §3.4.7）：**关键超参 γ(margin) 跨数据集变、严格 UDA 下无合法调参信号；
  无 label-shift 机制；应用叙事弱**。→ 补跑用文献默认 γ=4（登记·不调），观测其在本数据落点。

================================  MDD 损失（忠实复刻 ICML2019 / tllib MarginDisparityDiscrepancy）================================
设 y_main = 主头 f 的 logits、y_adv = 辅助头 f' 的 logits（f' 输入经 GRL）；pred = argmax(主头)：
  源项（f' 认同主头·margin 放大）：  γ · CE(y_adv_s, pred_s)
  目标项（f' 背离主头）：           nll( shift_log(1 − softmax(y_adv_t)), pred_t )
  L_mdd = γ·CE(y_adv_s, pred_s) + nll(shift_log(1−softmax(y_adv_t)), pred_t)
  其中 shift_log(x) = log(clamp(x + 1e-6, max=1.0))（数值守卫·防 log(0)/log(>1)）。
总损失：L = L_CE(主头·源·CB-WCE) + η · L_mdd     （η=trade-off，文献默认常数 1.0）
机制：f' 经 GRL → backward 时 F 收到**反向**梯度 → F 最小化判别差（对齐域）；f' 收正常梯度 →
  最大化判别能力（源认同/目标背离）= 对抗。GRL coeff 0→1 ramp（max_iter=total_steps·同 CDAN 母版修）。

★ **method-intrinsic 旋钮取文献默认·登记**（严格 UDA·无目标标签可调）：margin γ=4、trade-off η=1.0(常数)。
★ **辅助头只 forward 一次**：concat(源,目标) 特征喂 f' 后 split → GRL 内置 iter_num 每 step +1（非 +2），
  coeff ramp 不翻倍（CDAN 母版 spec2 同款坑·已避）。

================================  实现 provenance / eval 兼容（命根）================================
底座/数据/CB-WCE/GRL(calc_coeff·grl_hook)/inverse-decay/churn-health-ckpt/决定论/红线盲化 **复用自
已封验的 `src/run2_s5_cdan/train_cdan_sgd.py`**。MDD 损失从 ICML2019 公式现写（参照 tllib）→ 正确性靠
**单元验（解析已知case + shift_log + GRL）和论文公式逐项核对**共同验证。
★ **主头 f = 单 fc(512→2)**：结构 = CDANResNet18 逐字一致（项目 D1 约定:无 bottleneck·单 fc·跨臂同骨架公平）
  → 主模型 MDDResNet18 的 state_dict 键(feature_layers + fc)与 ARCH_REGISTRY["cdan"] 匹配 →
  **ckpt 的 model_state_dict 用 eval_pipeline --arch cdan 直接评估**。辅助头 f'(adv_head)是独立模块、
  **单独存 adv_head_state_dict、不进 model_state_dict**（eval 忽略·同 CDAN 的 ad_net）。
  〔偏离申报 D-MDD：主头单 fc（项目约定·非 MDD 原配的 bottleneck+2层）；f' = 2 层 MLP(512→1024→2)作辅助
   对抗头（类比 CDAN ad_net 的辅助判别角色）。〕

================================  数据 / 标签边界 / 合法性（与主矩阵一致）================================
源 = arm2/fold/train（带标签,增强）→ 目标 = KLSG 全 553（无标签,同款增强,transductive）。
目标标签绝不进任何损失（MDD 目标项只用 f'/f 的【预测】、不用目标真值）；KLSG 经 _klsg_blind 盲化
（label 位毒丸 FORBIDDEN_LABEL,绝不读/绝不反推）。绝不用真实域表现选模/早停/定参。
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
# 注：本训练器不算最终指标（eval 由下游 eval_pipeline + metrics_lib 出），故不 import sklearn.metrics。

# [红线护栏] KLSG 盲化入口；本文件在 src/extra/mdd/，src/ 在 parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _klsg_blind
import _run_safety

# ============================================================ 配置
PROJECT_ROOT = Path(__file__).resolve().parents[3]

SPLITS_CSV = PROJECT_ROOT / "data" / "split" / "splits.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "extra" / "mdd"

CLASSES = ["airplane", "ship"]
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

SEED = 0
EPOCHS_DEFAULT = 40
QUICK_EPOCHS = 12
BATCH_SIZE = 64
FC_LR = 1e-2
BACKBONE_LR = 1e-3
ADV_LR = 1e-2                              # 辅助头 f'（= fc lr = 10× backbone；同 CDAN ad_net lr_mult=10）
WEIGHT_DECAY = 5e-4
LR_DECAY_ALPHA = 10.0
LR_DECAY_BETA = 0.75
GRL_ALPHA = 10.0
CB_BETA = 0.999
SPECKLE_SIGMA = 0.1
SAVE_LAST_K = 5
ADV_HIDDEN = 1024                          # 辅助头隐藏宽度（类比 CDAN AD_HIDDEN）
MDD_MARGIN = 4.0                           # ★margin γ（文献默认·登记）
MDD_TRADEOFF = 1.0                         # ★trade-off η（文献默认·常数不 ramp·登记）
_EPS = 1e-6                                # shift_log offset（tllib MDD 同值）


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


# ============================================================ GRL（移植自 CDAN 母版·逐字）
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight); nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02); nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=GRL_ALPHA, max_iter=10000.0):
    """GLS GRL ramp 系数：2/(1+e^(-α·p))-1，p=iter/max_iter；max_iter 必须传 total_steps。"""
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1


# ============================================================ MDD 损失 + shift_log（忠实复刻 ICML2019）
def shift_log(x, offset=_EPS):
    """tllib MDD：log(clamp(x+offset, max=1.0))——防 log(0)→-inf / log(>1)。"""
    return torch.log(torch.clamp(x + offset, max=1.0))


def mdd_loss(y_main_s, y_adv_s, y_main_t, y_adv_t, margin=MDD_MARGIN):
    """Margin Disparity Discrepancy（Zhang et al. ICML2019·参照 tllib MarginDisparityDiscrepancy）。
      pred_s = argmax(y_main_s)、pred_t = argmax(y_main_t)（主头伪标签·argmax 天然不可微）。
      L = γ·CE(y_adv_s, pred_s) + nll( shift_log(1 − softmax(y_adv_t)), pred_t )
        源项：f' 认同主头（margin 放大）；目标项：f' 背离主头（压低 f' 在 pred_t 上的概率）。
    返回 (loss, src_term, tgt_term)：src/tgt 供诊断。GRL 在 adv_head 内部对输入特征生效（不在此处）。"""
    pred_s = y_main_s.argmax(dim=1)
    pred_t = y_main_t.argmax(dim=1)
    src_term = margin * F.cross_entropy(y_adv_s, pred_s)
    tgt_prob = F.softmax(y_adv_t, dim=1)                          # f' 在目标上的类概率
    tgt_term = F.nll_loss(shift_log(1.0 - tgt_prob), pred_t)     # -log(1 - p_{pred_t})：背离 pred_t
    return src_term + tgt_term, float(src_term.item()), float(tgt_term.item())


# ============================================================ 模型：主头(eval 兼容) + 辅助对抗头
class MDDResNet18(nn.Module):
    """主模型：ResNet18(ImageNet) + 主分类头 fc（512→2）。结构 = CDANResNet18 逐字一致
    （feature_layers + fc）→ ckpt model_state_dict 键匹配 → eval --arch cdan 直接评估。
    forward 返 (feature[512], main_logits)。**辅助对抗头 f' 是独立模块、不在本类**。"""
    def __init__(self, num_classes=len(CLASSES)):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.in_features = base.fc.in_features
        self.feature_layers = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4, base.avgpool)
        self.fc = nn.Linear(self.in_features, num_classes)       # 主头（eval 用这个 fc）
        for p in self.parameters():
            p.requires_grad = True
    def forward(self, x):
        f = self.feature_layers(x)
        f = f.view(f.size(0), -1)
        y = self.fc(f)
        return f, y
    def output_num(self):
        return self.in_features


class MDDAdvHead(nn.Module):
    """辅助对抗头 f'（2 层 MLP 512→hidden→C）+ 内置 GRL register_hook（同 CDAN AdversarialNetwork）。
    forward(features)：training 时 iter_num+1 → coeff → 对输入挂 grl_hook（反传到特征的梯度反向）→ MLP。
    ★ 每 step **只 forward 一次**（concat 源+目标）→ iter_num +1（非 +2），coeff ramp 不翻倍。
    ★ f' 不进主模型 state_dict（单独存 adv_head_state_dict·eval 忽略）。"""
    def __init__(self, in_feature, hidden_size, num_classes=len(CLASSES), max_iter=10000.0):
        super().__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, num_classes)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
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
        x.register_hook(grl_hook(coeff))                         # GRL：反传到特征的梯度反向（F 对齐域）
        x = self.dropout1(self.relu1(self.ad_layer1(x)))
        y = self.ad_layer2(x)                                    # 返 logits（无 softmax；mdd_loss 内处理）
        return y
    def current_coeff(self):
        return calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)


def make_mdd_optimizer(model, adv_head):
    """3 组 SGD（在 scheduler 前）：group0=backbone(1e-3) / group1=主头fc(1e-2) / group2=辅助头(1e-2)。"""
    fc_params, backbone_params = [], []
    for name, p in model.named_parameters():
        (fc_params if name.startswith("fc.") else backbone_params).append(p)
    adv_params = list(adv_head.parameters())
    optimizer = torch.optim.SGD(
        [{"params": backbone_params, "lr": BACKBONE_LR},
         {"params": fc_params, "lr": FC_LR},
         {"params": adv_params, "lr": ADV_LR}],
        momentum=0.9, nesterov=True, weight_decay=WEIGHT_DECAY,
    )
    return (optimizer, sum(p.numel() for p in backbone_params),
            sum(p.numel() for p in fc_params), sum(p.numel() for p in adv_params), backbone_params)


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


# ============================================================ 单元验（smoke 前·验 MDD 损失 + GRL）
def run_unit_tests(device):
    """验 ① shift_log 端点 ② mdd_loss 解析 case（margin 放大·源认同/目标背离）③ GRL 取负缩放
       ④ calc_coeff ramp 端点 ⑤ 可微/无 NaN。"""
    print("=" * 78)
    print("★ 单元验（smoke 前 · MDD 公式 + GRL） ★")
    ok = True

    # ① shift_log: shift_log(1)=log(1)=0; shift_log(0)=log(1e-6)≈-13.8155
    s1 = shift_log(torch.tensor([1.0], device=device)).item()
    s0 = shift_log(torch.tensor([0.0], device=device)).item()
    t_sl = (abs(s1) < 1e-4) and (abs(s0 - np.log(1e-6)) < 1e-3)
    print(f"  [①shiftlog] shift_log(1)={s1:.5f}(期望0) shift_log(0)={s0:.4f}(期望≈{np.log(1e-6):.4f}) -> {'PASS' if t_sl else 'FAIL'}")
    ok = ok and t_sl

    # ② mdd_loss 解析：C=2,B=4。主头 pred_s=[0,0,1,1],pred_t=[0,0,1,1]（极端 logit 定 argmax）。
    #   case-A（f' 认同源+背离目标=低 disparity loss）：y_adv_s 强匹配 pred_s；y_adv_t 强【反】pred_t。
    ymain_s = torch.tensor([[20.,-20.],[20.,-20.],[-20.,20.],[-20.,20.]], device=device)
    ymain_t = torch.tensor([[20.,-20.],[20.,-20.],[-20.,20.],[-20.,20.]], device=device)
    yadv_s_agree = ymain_s.clone()                          # f' 认同源 pred -> CE≈0
    yadv_t_disagree = -ymain_t.clone()                      # f' 在目标上反 pred -> softmax[pred_t]≈0 -> 1-·≈1 -> shift_log≈0
    LA, sA, tA = mdd_loss(ymain_s, yadv_s_agree, ymain_t, yadv_t_disagree, margin=MDD_MARGIN)
    t_a = (LA.item() < 1e-2)
    print(f"  [②mddA  ] f'认同源+背离目标 -> L_mdd={LA.item():.5f}(src={sA:.4f}+tgt={tA:.4f}, 期望≈0) -> {'PASS' if t_a else 'FAIL'}")
    ok = ok and t_a

    #   case-B（uniform f'）：y_adv 全 0 -> 源 CE=ln2≈0.693·×γ=4 -> 2.7726；目标 softmax=0.5 -> 1-0.5=0.5 ->
    #     shift_log(0.5)=ln0.5≈-0.6931 -> nll=-(-0.6931)=0.6931。L=4·0.6931+0.6931=3.4657。
    yadv_unif = torch.zeros(4, 2, device=device)
    LB, sB, tB = mdd_loss(ymain_s, yadv_unif, ymain_t, yadv_unif, margin=MDD_MARGIN)
    expect_B = MDD_MARGIN * np.log(2) + (-np.log(0.5))
    t_b = (abs(LB.item() - expect_B) < 1e-2)
    print(f"  [②mddB  ] uniform f' -> L_mdd={LB.item():.5f}(src={sB:.4f}+tgt={tB:.4f}, 期望≈{expect_B:.4f}) -> {'PASS' if t_b else 'FAIL'}")
    ok = ok and t_b

    #   margin 放大核：源项 = γ·CE，γ=4 时 src_B 应 = 4·ln2
    t_m = (abs(sB - MDD_MARGIN * np.log(2)) < 1e-3)
    print(f"  [②margin] src 项 = γ·CE: {sB:.4f} 期望 {MDD_MARGIN*np.log(2):.4f}(=4·ln2) -> {'PASS' if t_m else 'FAIL'}")
    ok = ok and t_m

    # ③ GRL register_hook（adv_head 内置）取负+缩放：构 max_iter 使 coeff 已知；验梯度反向
    adv = MDDAdvHead(8, 16, num_classes=2, max_iter=100).to(device)
    adv.train(); adv.iter_num = 99                          # forward 后 iter_num=100 -> coeff=calc_coeff(100,max_iter=100)
    feat = torch.ones(2, 8, requires_grad=True, device=device)
    out = adv(feat)
    coeff_used = adv.current_coeff()
    out.sum().backward()
    # 梯度方向：经 GRL 应与无 GRL 反号；这里只验 coeff>0 且梯度有限（精确符号验见 grl_hook 单测）
    t_grl = bool(torch.isfinite(feat.grad).all().item()) and (coeff_used > 0.9)
    print(f"  [③GRL   ] adv_head forward coeff={coeff_used:.4f}(期望≈1) 梯度有限={bool(torch.isfinite(feat.grad).all().item())} -> {'PASS' if t_grl else 'FAIL'}")
    ok = ok and t_grl

    # ③b grl_hook 直验取负缩放（同 DANN 单测）
    t = torch.ones(3, requires_grad=True, device=device)
    t2 = t * 1.0; t2.register_hook(grl_hook(0.5)); (t2 * 2.0).sum().backward()
    t_grlb = torch.allclose(t.grad, torch.full_like(t.grad, -1.0), atol=1e-6)
    print(f"  [③bhook ] grl_hook(0.5) grad={t.grad.tolist()}(期望[-1,-1,-1]=-0.5·2) -> {'PASS' if t_grlb else 'FAIL'}")
    ok = ok and t_grlb

    # ④ calc_coeff ramp 端点
    ts = 240
    c0, ce = calc_coeff(0, max_iter=ts), calc_coeff(ts, max_iter=ts)
    t_c = (abs(c0) < 1e-6) and (ce > 0.999)
    print(f"  [④coeff ] calc_coeff(0)={c0:.4f}(≈0) ({ts})={ce:.6f}(≈1) -> {'PASS' if t_c else 'FAIL'}")
    ok = ok and t_c

    # ⑤ 可微/无 NaN：随机 logit 全链路
    zms = torch.randn(8, 2, device=device, requires_grad=True)
    zas = torch.randn(8, 2, device=device, requires_grad=True)
    zmt = torch.randn(8, 2, device=device, requires_grad=True)
    zat = torch.randn(8, 2, device=device, requires_grad=True)
    L5, _, _ = mdd_loss(zms, zas, zmt, zat, margin=MDD_MARGIN)
    L5.backward()
    t_d = bool(torch.isfinite(L5).item()) and bool(torch.isfinite(zas.grad).all().item()) and bool(torch.isfinite(zat.grad).all().item())
    print(f"  [⑤可微 ] 随机 logit L_mdd={L5.item():.4f} 标量+梯度有限={t_d} -> {'PASS' if t_d else 'FAIL'}")
    ok = ok and t_d

    print(f"  >>> 单元验总判: {'全部 PASS' if ok else '存在 FAIL'}")
    print("=" * 78)
    assert ok, "单元验失败：MDD 损失/shift_log/GRL/coeff 有误，不进 smoke"
    return ok


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None, help="epoch 数（默认 full=40；--quick 时 12）")
    ap.add_argument("--quick", action="store_true", help="smoke（默认 12 epoch、产物名 mdd_smoke_*）")
    ap.add_argument("--unittest", action="store_true", help="只跑单元验后退出（不训练）")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--fold", type=int, default=0, help="源 arm2 折号；KLSG 目标全 553 transductive、fold 无关")
    ap.add_argument("--margin", type=float, default=MDD_MARGIN, help="MDD margin γ（文献默认 4）")
    ap.add_argument("--trade-off", type=float, default=MDD_TRADEOFF, help="MDD trade-off η（文献默认 1.0·常数）")
    ap.add_argument("--save-every-epoch", action="store_true", help="逐 epoch 存全 dict ckpt；默认只存末 SAVE_LAST_K")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"), "CUBLAS_WORKSPACE_CONFIG 未设(须在 import torch 前内联)"

    run_unit_tests(device)
    if args.unittest:
        print("仅单元验模式（--unittest），退出。")
        return 0
    set_seed(args.seed)   # run_unit_tests 消费 RNG（randn + adv_head xavier 初始化）→ 重设回 args.seed

    epochs = (QUICK_EPOCHS if args.quick else EPOCHS_DEFAULT) if args.epochs is None else args.epochs
    tag = "mdd_smoke" if args.quick else "mdd_full"

    model_dir = RESULTS_DIR / f"{tag}_s{args.seed}_f{args.fold}"
    metrics_path = model_dir / f"{tag}_metrics.json"
    curve_path = model_dir / f"{tag}_curve.csv"
    churn_argmax_path = model_dir / f"{tag}_churn_argmax.csv"
    _run_safety.prepare_empty_run_dirs(model_dir)

    print("=" * 78)
    print(f"【候补·[E]】 **MDD × default BN** (SGD 底座) ({tag}) : arm2 -> KLSG UDA  | γ={args.margin} η={args.trade_off}")
    print("=" * 78)
    print(f"项目根 : {PROJECT_ROOT}")
    print(f"设备   : {device} | torch {torch.__version__} | "
          f"GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    src_items = read_split_rows(SPLITS_CSV, domain="arm2", fold=args.fold, split="train")
    tgt_items = _klsg_blind.read_klsg_unlabeled(PROJECT_ROOT)
    src_cnt = Counter(CLASSES[l] for _, l in src_items)
    print(f"\n源 arm2/fold{args.fold}/train (带标签,增强)     : {len(src_items)} (airplane={src_cnt['airplane']}, ship={src_cnt['ship']})")
    print(f"目标 KLSG 全 {len(tgt_items)} (无标签,同款增强,transductive) : sha {_klsg_blind.KLSG_MANIFEST_SHA256[:8]}")
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
    print(f"\nCB-WCE (β={CB_BETA}) 权重 {counts}: airplane={cb_w[0].item():.4f}, ship={cb_w[1].item():.4f}  [非 IW]")
    criterion = nn.CrossEntropyLoss(weight=cb_w)

    steps_per_epoch = len(src_loader)
    total_steps = epochs * steps_per_epoch

    model = MDDResNet18().to(device)
    adv_head = MDDAdvHead(model.output_num(), ADV_HIDDEN, num_classes=len(CLASSES), max_iter=total_steps).to(device)
    optimizer, n_bb, n_fc, n_adv, backbone_params = make_mdd_optimizer(model, adv_head)
    scheduler = make_scheduler(optimizer, total_steps)
    n_total = sum(p.numel() for p in model.parameters())

    print(f"\n模型 : ResNet18(ImageNet) 全解冻 | default BN | 主头 fc(512->2, eval用) | 辅助头 f' MLP(512->{ADV_HIDDEN}->2)+GRL")
    print(f"分层 lr : backbone={BACKBONE_LR}({n_bb:,}) | 主头fc={FC_LR}({n_fc:,}) | 辅助头={ADV_LR}({n_adv:,})")
    print(f"优化 : SGD(m=0.9,nesterov) wd={WEIGHT_DECAY} | batch {BATCH_SIZE}+{BATCH_SIZE} | {epochs} epoch | total_steps={total_steps}")
    print(f"MDD  : γ(margin)={args.margin} η(trade-off)={args.trade_off}(常数) | L = cls(CB-WCE源) + η·[γ·CE(f'_s,pred_s) + nll(shift_log(1-σ(f'_t)),pred_t)]")
    print(f"  ★ GRL coeff 0->1 ramp(max_iter={total_steps}) 内置 adv_head·只 forward 一次(concat 源+目标)")

    print("-" * 78)
    print("smoke 每 epoch 诊断: cls/acc | L_mdd(src/tgt) | coeff | tgtPred直方 | churn | NaN")
    print("-" * 78)
    history = []
    churn_argmax_rows = []
    a_prev = None
    global_step = 0
    any_nan = False
    target_iter = iter(tgt_loader)
    for epoch in range(1, epochs + 1):
        model.train(); adv_head.train()
        ep_cls, ep_mdd, ep_mdd_s, ep_mdd_t, ep_gn = 0.0, 0.0, 0.0, 0.0, 0.0
        ep_correct, ep_seen, ep_steps = 0, 0, 0
        ep_coeff_last = 0.0
        for x_s, y_s in src_loader:
            try:
                x_t, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(tgt_loader)
                x_t, _ = next(target_iter)
            x_s, y_s = x_s.to(device, non_blocking=True), y_s.to(device, non_blocking=True)
            x_t = x_t.to(device, non_blocking=True)

            f_s, y_main_s = model(x_s)
            f_t, y_main_t = model(x_t)
            assert f_s.size(0) == f_t.size(0), "契约破：源/目标 batch 不等长"
            # ★ 辅助头只 forward 一次（concat 源+目标）→ GRL iter_num +1（非 +2）
            f_cat = torch.cat((f_s, f_t), dim=0)
            y_adv_cat = adv_head(f_cat)
            bs = f_s.size(0)
            y_adv_s, y_adv_t = y_adv_cat[:bs], y_adv_cat[bs:]

            cls_loss = criterion(y_main_s, y_s)                  # 主头·源监督（CB-WCE）
            mdd_l, mdd_s_v, mdd_t_v = mdd_loss(y_main_s, y_adv_s, y_main_t, y_adv_t, margin=args.margin)
            total_loss = cls_loss + args.trade_off * mdd_l

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

            cur_coeff = adv_head.current_coeff()
            ep_coeff_last = cur_coeff
            ep_cls += float(cls_loss.item())
            ep_mdd += float(mdd_l.item())
            ep_mdd_s += mdd_s_v
            ep_mdd_t += mdd_t_v
            ep_gn += bb_grad_norm
            ep_correct += int((y_main_s.argmax(1) == y_s).sum().item())
            ep_seen += int(y_s.size(0))
            ep_steps += 1
            if not np.isfinite(float(total_loss.item())):
                any_nan = True

        cls_m = ep_cls / ep_steps
        mdd_m = ep_mdd / ep_steps
        mdd_s_m = ep_mdd_s / ep_steps
        mdd_t_m = ep_mdd_t / ep_steps
        gn_m = ep_gn / ep_steps
        src_acc = ep_correct / max(ep_seen, 1)
        ep_nan = (not np.isfinite(cls_m)) or (not np.isfinite(mdd_m)) or any_nan
        any_nan = any_nan or ep_nan

        tgt_hist, tgt_ent = target_pseudolabel_health(model, tgt_probe_loader, device)
        tgt_collapsed = (max(tgt_hist) == sum(tgt_hist))
        a_e = churn_predict(model, churn_loader, device)
        churn_e = float((a_e != a_prev).mean()) if a_prev is not None else None
        a_prev = a_e
        churn_argmax_rows.append((epoch, a_e.tolist()))
        model.train(); adv_head.train()

        history.append({"epoch": epoch, "global_step": global_step, "coeff_last": ep_coeff_last,
                        "src_cls_loss_mean": cls_m, "mdd_loss_mean": mdd_m,
                        "mdd_src_mean": mdd_s_m, "mdd_tgt_mean": mdd_t_m,
                        "src_acc": src_acc, "bb_grad_norm_mean": gn_m, "churn": churn_e,
                        "tgt_pred_hist": list(tgt_hist), "tgt_pred_entropy_mean": tgt_ent,
                        "tgt_pred_collapsed": bool(tgt_collapsed), "nan": bool(ep_nan)})
        churn_str = "n/a" if churn_e is None else f"{churn_e:.3f}"
        print(f"  ep {epoch:>3}/{epochs} | cls {cls_m:.4f} acc {src_acc:.4f} | L_mdd {mdd_m:.4f}(s{mdd_s_m:.3f}/t{mdd_t_m:.3f}) "
              f"| coeff {ep_coeff_last:.3f} | bbGrad {gn_m:.2e} | churn {churn_str} | tgtPred {tgt_hist}"
              f"{' COLLAPSE!' if tgt_collapsed else ''}{' | NaN!' if ep_nan else ''}")
        if args.save_every_epoch or epoch > epochs - SAVE_LAST_K:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "adv_head_state_dict": adv_head.state_dict(),
                        "global_step": global_step},
                       model_dir / f"{tag}_epoch{epoch:03d}.pth")

    # --- lr inverse-decay sanity ---
    print("-" * 78)
    print("★ lr inverse-decay sanity ★")
    expected_factor = (1.0 + LR_DECAY_ALPHA) ** (-LR_DECAY_BETA)
    base_lrs = list(scheduler.base_lrs)
    lr_ok = True
    for i, gname in enumerate(["backbone", "main_fc", "adv_head"]):
        cur_lr = optimizer.param_groups[i]["lr"]
        exp_lr = base_lrs[i] * expected_factor
        rel_err = abs(cur_lr - exp_lr) / max(exp_lr, 1e-12)
        print(f"  group[{i}] {gname:>9}: base={base_lrs[i]:.3e} | 末端 lr={cur_lr:.6e} | 期望={exp_lr:.6e} [{'OK' if rel_err<1e-3 else 'MISMATCH'}]")
        lr_ok = lr_ok and (rel_err < 1e-3)
    assert lr_ok, "scheduler 末端 lr 不符（辅助头组未入 base_lrs？）"
    print(f"  >>> lr 调度核验: {'PASS' if lr_ok else 'FAIL'}")

    # --- 健康判读（smoke；MDD 不稳/坍缩=合法发现，如实记不当 bug）---
    print("-" * 78)
    print("★ 训练健康判读（smoke；**不读性能、不定参**）★")
    cls_series = [h["src_cls_loss_mean"] for h in history]
    mdd_series = [h["mdd_loss_mean"] for h in history]
    coeff_series = [h["coeff_last"] for h in history]
    acc_series = [h["src_acc"] for h in history]
    coeff_ramped = coeff_series[-1] > 0.9 and coeff_series[-1] > coeff_series[0]
    src_healthy = acc_series[-1] > 0.5
    tgt_collapse_any = any(h["tgt_pred_collapsed"] for h in history)
    tgt_hist_final = history[-1]["tgt_pred_hist"]
    print(f"  ① coeff ramp    : 首{coeff_series[0]:.3f} -> 末{coeff_series[-1]:.3f}  [{'真 ramp' if coeff_ramped else '未爬升'}]")
    print(f"  ② 源 cls loss   : 首{cls_series[0]:.3f} -> 末{cls_series[-1]:.3f}")
    print(f"  ③ L_mdd         : 首{mdd_series[0]:.3f} -> 末{mdd_series[-1]:.3f}")
    print(f"  ④ 源分类 acc    : 首{acc_series[0]:.3f} -> 末{acc_series[-1]:.3f}  [{'未崩' if src_healthy else '偏低-复核'}]")
    print(f"  ⑤ NaN/Inf 守卫  : {'发现 NaN/Inf!' if any_nan else '全程无 NaN/Inf'}")
    print(f"  ⑥ backbone 梯度 : 末 {history[-1]['bb_grad_norm_mean']:.3e}")
    print(f"  ⑦ 目标伪标签坍缩: 末 {tgt_hist_final}  [{'★坍缩过(全预测同类!)' if tgt_collapse_any else '未坍缩'}]  (label-free)")

    out = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run_tag": tag,
        "method": "MDD (Margin Disparity Discrepancy, Zhang et al. ICML2019, adversarial aux-classifier) x default BN, SGD base",
        "identity": "candidate / [E] post-hoc supplementary (station6 S6-8); NOT in confirmatory M1/M2/M3 set",
        "provenance": "scaffold from verified train_cdan_sgd.py (GRL/calc_coeff reused); mdd_loss + aux adv head from "
                      "ICML2019 / tllib MarginDisparityDiscrepancy; main head = single fc (project D1, eval --arch cdan); "
                      "aux head f' = separate 2-layer MLP (saved adv_head_state_dict, ignored by eval); verified by unit tests + sub review.",
        "config": {
            "seed": args.seed, "fold": args.fold, "epochs": epochs, "batch_size_source": BATCH_SIZE, "batch_size_target": BATCH_SIZE,
            "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
            "model": "resnet18_imagenet_full_finetune", "bn": "default", "penultimate_dim": model.output_num(),
            "main_head": "single Linear(512,2) = CDANResNet18.fc (eval --arch cdan compatible)",
            "adv_head": {"type": "MLP(512->%d->2)" % ADV_HIDDEN, "hidden": ADV_HIDDEN, "lr": ADV_LR,
                         "note": "auxiliary adversarial classifier f'; GRL on input; saved as adv_head_state_dict, NOT in model_state_dict"},
            "mdd": {"margin": args.margin, "trade_off": args.trade_off, "trade_off_note": "constant, NOT ramped",
                    "grl_coeff": "calc_coeff=2/(1+exp(-alpha*p))-1 (0->1 ramp)", "grl_alpha": GRL_ALPHA,
                    "grl_alpha_note": "alpha=10 follows project CDAN/DANN母版 GRL convention (shared across ALL adversarial "
                                      "arms #3/#5/#7/DANN/MDD for within-study fairness); DEVIATES from tllib MDD default alpha=1.0. "
                                      "0->1 ramp over total_steps satisfied either way; holding alpha constant isolates the "
                                      "MDD-specific machinery (margin disparity + aux head) from GRL-schedule confounds.",
                    "grl_max_iter": total_steps, "grl_max_iter_note": "= total_steps (NOT hardcoded 10000)",
                    "literature_defaults": "margin=4, trade_off=1.0 (ICML2019); strict-UDA: no target labels, no tuning",
                    "loss_formula": "L=margin*CE(y_adv_s, argmax y_main_s) + nll(shift_log(1-softmax(y_adv_t)), argmax y_main_t); shift_log=log(clamp(x+1e-6,max=1))"},
            "optimizer": {"type": "SGD", "momentum": 0.9, "nesterov": True, "weight_decay": WEIGHT_DECAY},
            "layered_lr": {"backbone_lr": BACKBONE_LR, "main_fc_lr": FC_LR, "adv_head_lr": ADV_LR},
            "scheduler": {"type": "inverse_decay_LambdaLR", "alpha": LR_DECAY_ALPHA, "beta": LR_DECAY_BETA},
            "source_cls_loss": "CB-WCE", "cb_beta": CB_BETA,
            "cb_weights": {"airplane": float(cb_w[0].item()), "ship": float(cb_w[1].item())},
            "iw_stripped": True, "random_projection": False, "bottleneck": False,
            "deviation_D_MDD": "main head single fc (project D1 convention, not MDD original bottleneck+2-layer); aux head f' is the 2-layer auxiliary classifier",
        },
        "data": {"source_train": f"arm2/fold{args.fold}/train (labeled, aug)", "source_fold": args.fold,
                 "n_source": len(src_items), "source_counts": dict(src_cnt),
                 "target": f"klsg full {len(tgt_items)} transductive (sha {_klsg_blind.KLSG_MANIFEST_SHA256[:8]})",
                 "n_target": len(tgt_items),
                 "target_note": "UNLABELED, aug; labels blinded (never read/inferred); transductive; MDD target term uses f'/f PREDICTIONS only"},
        "train_history": history,
        "smoke_verdict": {
            "coeff_first": coeff_series[0], "coeff_last": coeff_series[-1], "coeff_ramped": bool(coeff_ramped),
            "src_cls_loss_first": cls_series[0], "src_cls_loss_last": cls_series[-1],
            "mdd_loss_first": mdd_series[0], "mdd_loss_last": mdd_series[-1],
            "src_acc_first": acc_series[0], "src_acc_last": acc_series[-1], "src_cls_healthy": bool(src_healthy),
            "tgt_pred_hist_final": tgt_hist_final, "tgt_pred_collapsed_any": bool(tgt_collapse_any),
            "has_nan_inf": bool(any_nan), "bb_grad_norm_final": history[-1]["bb_grad_norm_mean"],
            "lr_schedule_ok": bool(lr_ok),
            "note": "MDD training health smoke only; NOT performance/model selection. Instability/collapse (if any) "
                    "recorded as-is, NOT babysat (margin gamma fixed at literature default 4, no tuning per strict-UDA). "
                    "Target health = label-free pseudo-label collapse probe (NO KLSG label read).",
        },
        "has_nan_inf": bool(any_nan),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(curve_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "global_step", "coeff_last", "src_cls_loss_mean", "mdd_loss_mean",
                    "mdd_src_mean", "mdd_tgt_mean", "src_acc", "bb_grad_norm_mean", "churn",
                    "tgt_pred_airplane", "tgt_pred_ship", "tgt_pred_entropy_mean", "tgt_pred_collapsed", "nan"])
        for h in history:
            churn_cell = "" if h["churn"] is None else f"{h['churn']:.6f}"
            w.writerow([h["epoch"], h["global_step"], f"{h['coeff_last']:.6f}", f"{h['src_cls_loss_mean']:.6f}",
                        f"{h['mdd_loss_mean']:.6f}", f"{h['mdd_src_mean']:.6f}", f"{h['mdd_tgt_mean']:.6f}",
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
    print(f"checkpoint -> {model_dir}")
    print(f"NaN/Inf : {any_nan}")
    print("=" * 78)
    return 1 if any_nan else 0


if __name__ == "__main__":
    sys.exit(main())
