#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_adabn.py -- run1 #2 **AdaBN（搭车 source-only SGD 基线）零训练评估脚本**
[SSS sim->real 主实验 · 域适配 BN 统计重估 · AdaBN 搭车评估]

================================  这是什么 ================================
AdaBN = Adaptive Batch Normalization（Li et al. 2016, https://arxiv.org/abs/1603.04779）。
**零训练（zero-train）**：取宿主 source-only SGD 基线（run1）**已训** checkpoint，**重置 + 用无标签目标图重估**
所有 BatchNorm 的 running 统计（running_mean / running_var），推理时改用**目标域统计**——
与宿主唯一差别 = BN 统计来源（源域训练统计 -> 目标域统计）。**不训练 / 不动任何权重 / 不碰目标标签。**

宿主脚本 = `src/run1_s4_baseline_ceiling/train_source_only_sgd.py`（本脚本复用其 build_model、
build_eval_transform、set_seed及正式配置常量）。
AdaBN机制参照Dassl.pytorch的`dassl/engine/da/adabn.py`（reset_running_stats + no_grad forward）。

================================  transductive 口径（plan §3.3.2#2 + §3.2.4/§3.2.1）================================
AdaBN 走 **transductive** 口径：用"全部可得无标签目标图（含将被评估那些图的【像素】，不含标签）"
重估 BN 统计、并在其上评估。
  ★ 正式生产路径：目标 = **全部 KLSG，无 train/test 之分**——
    用【全部 KLSG 无标签图】重估 + 在【全部 KLSG】评估（目标折只服务 ceiling）。
================================  AdaBN 语义 / 重估口径（论文最终配置）================================
  - reset 所有 BN running stats（reset_running_stats: running_mean->0 / running_var->1 / num_batches_tracked->0；
    **不碰 γ/β 仿射**——已核 PyTorch reset_running_stats 源码只动这三个 buffer）。
  - ★★ momentum = None（**锁定项**）：reset 后把所有 BN 的 m.momentum 置 None == 累积平均（CMA, cumulative
    moving average）= 等权全集统计。**绝不用默认 0.1**（那是 EMA、偏向最后几个 batch、非目标域真实分布；
    对抗审实证：默认 0.1 均值偏差量级远大、momentum=None 精确命中全集均值 ~1e-6）。这是语义正确性、非占位旋钮。
    （与Dassl adabn.py的**有意偏离**：Dassl用默认momentum，本研究最终实现锁定为None。）
  - 重估前向：model.train()（BN 更新统计）+ torch.no_grad()（不算梯度、不更新权重），前向目标图。
    当显存允许时优先全量单batch一次forward（一个batch=全集，running stats为精确全集统计）。
    通用实现支持分块；**分块时重估 loader shuffle=True**（防数据按类排序时每 batch 只见一类、BN 丢类间方差使
    var 低估；shuffle 让 batch 近 i.i.d. -> CMA 的 mean/var 都近无偏）。
  - 末尾 model.eval()：做成函数自包含契约（防后续概率平均循环忘切 eval、BN 继续吃 eval batch 统计污染结果）。
  - **只动 BN 统计、绝不动权重**（conv / fc / γ / β 全不变）。

================================  公开实现边界 ================================
  - `reset_and_prime_bn()` + `adabn_recompute()`：重置并用无标签KLSG图像重估BN统计；
  - `load_host_model()`：构造同结构模型并严格加载checkpoint中的`model_state_dict`；
  - `build_recompute_loader_unlabeled()` / `build_eval_loader_unlabeled()`：只建立盲化KLSG loader；
  - `adabn_eval_ensemble_unlabeled()`：每个末K checkpoint独立重估BN，再平均softmax概率；
  - `RECOMPUTE_PASSES=1`且`momentum=None`为论文正式锁定配置。

=========================================================================="""

import os
# [determinism] CUBLAS_WORKSPACE_CONFIG 必须在 import torch 之前设置(cuBLAS GEMM 决定论); 用 setdefault 让 driver 级 export 优先、幂等。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys
import json
import argparse
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.modules.batchnorm import _BatchNorm    # 统一 BN 基类（BatchNorm1d/2d/3d/SyncBN 都继承它）

# ---- 复用宿主脚本（同目录） ----
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import train_source_only_sgd as host                  # 宿主：build_model / build_eval_transform / set_seed / ...
# [红线护栏] KLSG 目标域盲化入口（生产路径用；src/ 在 parents[1]）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _klsg_blind  # read_klsg_unlabeled / UnlabeledImageDataset / collate_unlabeled / FORBIDDEN_LABEL

# ============================================================ 配置
RESULTS_DIR = host.RESULTS_DIR                         # results/run1_s4_baseline_ceiling
CLASSES = host.CLASSES
SEED = host.SEED
# pass 数 = 重估时遍历目标集的次数；论文正式配置锁定为 1。
#   注：全量单 batch 模式下，passes>=1 给出完全相同的精确全集统计（CMA 单 batch 幂等）；passes 仅在分块模式下
#   影响 BN var 稳定度。**momentum 已锁 None、非占位**（见模块头注 / reset_and_prime_bn）。
RECOMPUTE_PASSES = 1

# momentum 锁定值（对抗审实证：None=CMA 等权全集统计、精确命中均值 ~1e-6；0.1=EMA 偏后批、错）。
#   **锁定项，绝不改回 0.1。** 仅用作常量记录 / 不暴露为可调旋钮。
ADABN_MOMENTUM = None


# ============================================================ ① 重估核心（自包含 A，可复制给 #2/#4）
def reset_and_prime_bn(model):
    """spec ①：reset 所有 BN 的 running 统计 + 设 momentum=None（CMA）。返回处理的 BN 模块数。
      - reset_running_stats(): running_mean->0, running_var->1, num_batches_tracked->0。
        **不碰 γ(weight)/β(bias) 仿射**（已核 PyTorch 源码该函数只动上述三个 buffer）。
      - m.momentum = None: 累积平均（CMA）= 等权全集统计；**锁定，非默认 0.1**（见头注对抗审）。
    与 Dassl adabn.py 对照：Dassl 用 classname.find('BatchNorm') 选模块且保留默认 momentum；
    本实现用 isinstance(_BatchNorm) 选模块（更稳）+ **有意锁 momentum=None**（对抗审修正）。
    注：用 isinstance 遍历，裸 resnet18 / CDAN wrapper 都命中（宿主无关）。"""
    n_bn = 0
    for m in model.modules():
        if isinstance(m, _BatchNorm) and m.track_running_stats:
            m.reset_running_stats()        # 只清 running_mean/var/num_batches_tracked，不动 γ/β
            m.momentum = ADABN_MOMENTUM    # None -> CMA 等权全集统计（锁定项）
            n_bn += 1
    return n_bn


def adabn_recompute(model, recompute_loader, device, passes=RECOMPUTE_PASSES):
    """spec ①：AdaBN 重估核心（自包含契约）。reset+prime BN -> train()+no_grad 前向目标图重估统计 -> eval()。
    **只动 BN 统计、绝不动权重**（no_grad 不算梯度、无 optimizer.step、γ/β 不在 reset 范围内）。
    spec ⑥：只吃目标【图像】、丢弃 loader 的 label（绝不用目标标签）。
    ★ 重估前向 model(x) **丢弃输出**（只触发 BN forward 更新统计），单/双输出不影响重估——故宿主无关、不用改。
    返回 dict（处理的 BN 数 / 前向 batch 数 / passes），供诊断。"""
    n_bn = reset_and_prime_bn(model)
    model.train()                                       # BN 进入"更新 running 统计"模式
    n_fwd = 0
    with torch.no_grad():                               # 不算梯度、不更新权重
        for _ in range(int(passes)):
            for batch in recompute_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch   # 只取图像，丢标签(spec ⑥)
                x = x.to(device, non_blocking=True)
                model(x)                                # 前向 -> BN 累积目标域统计（CMA）；输出丢弃(单/双输出均可)
                n_fwd += 1
    model.eval()                                        # 自包含契约：离开时 BN 处 eval（用刚重估的统计）
    return {"n_bn": n_bn, "n_forward_batches": n_fwd, "passes": int(passes)}


# ============================================================ ④ 加载宿主 checkpoint（只取 model_state_dict）
def load_host_model(ckpt_path, device):
    """spec ④：host.build_model() 构造（run1 无 CDANResNet18；build_model 返回裸 resnet18 + 改 fc(2类)、
    全解冻；不带 device 故补 .to(device)）；**只加载 ckpt['model_state_dict']**，忽略顶层 train_loss/val_loss 等键。
    strict load 自洽：run1 ckpt 的裸键 conv1./bn1./layerX./fc. ↔ build_model 的 resnet18 键一一对应。
    注：model_state_dict 内部含训练期 BN 统计，载入无害——随即在 adabn_recompute 里 reset BN。"""
    ckpt_path = Path(ckpt_path)
    assert ckpt_path.is_file(), f"checkpoint 不存在: {ckpt_path}"
    model = host.build_model(pretrained=False).to(device)   # checkpoint strict load提供全部训练权重
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    assert "model_state_dict" in ckpt, f"ckpt 缺 model_state_dict 键: {list(ckpt.keys())}"
    model.load_state_dict(ckpt["model_state_dict"])          # strict=True；忽略其余顶层键
    return model


# ============================================================ 生产路径（KLSG 无标签）label-free loaders / 评估
def build_recompute_loader_unlabeled(items, recompute_batch=0):
    """生产重估 loader（KLSG 无标签，盲化 collate；不增强、确定性优先）。与 build_recompute_loader 同口径，
    但用 _klsg_blind.UnlabeledImageDataset + collate_unlabeled（绝不 stack 标签 / 绝不 default_collate）。"""
    ds = _klsg_blind.UnlabeledImageDataset(items, host.build_eval_transform())
    n = len(ds)
    if recompute_batch and 0 < recompute_batch < n:
        loader = DataLoader(ds, batch_size=int(recompute_batch), shuffle=True, num_workers=0,
                            drop_last=False, collate_fn=_klsg_blind.collate_unlabeled)
        return loader, n, True
    loader = DataLoader(ds, batch_size=n, shuffle=False, num_workers=0,
                        drop_last=False, collate_fn=_klsg_blind.collate_unlabeled)
    return loader, n, False


def build_eval_loader_unlabeled(items, batch_size=None):
    """生产评估 loader（KLSG 无标签）：不增强、shuffle=False、盲化 collate。"""
    bs = int(batch_size) if batch_size else host.BATCH_SIZE
    ds = _klsg_blind.UnlabeledImageDataset(items, host.build_eval_transform())
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0, collate_fn=_klsg_blind.collate_unlabeled)


def evaluate_probs_unlabeled(model, loader, device):
    """label-free 概率评估：只取 batch[0] 图像（batch[1] 是毒丸, 绝不碰）。返回 probs[N,K]，**无 y_true**。"""
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)        # 只取图像；绝不读目标 label
            logits = model(x)                                 # run1 裸 resnet18 单输出 logits（**非**双解包）
            probs.append(F.softmax(logits, dim=1).cpu())
    return torch.cat(probs, 0).numpy()


def adabn_eval_ensemble_unlabeled(ckpt_paths, recompute_loader, eval_loader, device, passes=RECOMPUTE_PASSES):
    """末 K AdaBN 集成（KLSG 无标签生产路径）：每 ckpt 独立 reload -> reset+recompute BN -> 概率 -> 平均概率 -> argmax。
    ★ 绝不收集/返回 y_true（目标 label 盲化）。返回 (y_pred, avg_probs[N,K], per_ckpt[list], recompute_infos)。"""
    assert len(ckpt_paths) >= 1, "ckpt_paths 至少 1 个"
    prob_sum, per_ckpt, recompute_infos = None, [], []
    for path in ckpt_paths:
        model = load_host_model(path, device)                # 独立 reload（无 BN 串味）
        recompute_infos.append(adabn_recompute(model, recompute_loader, device, passes))
        probs = evaluate_probs_unlabeled(model, eval_loader, device)
        per_ckpt.append(probs)
        prob_sum = probs.copy() if prob_sum is None else prob_sum + probs
    avg_probs = prob_sum / float(len(ckpt_paths))
    return avg_probs.argmax(1).tolist(), avg_probs, per_ckpt, recompute_infos


# ============================================================ 输出辅助
def _hr(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)

# ============================================================ main
def main():
    ap = argparse.ArgumentParser(description="AdaBN（搭车 source-only SGD 基线）零训练评估")
    ap.add_argument("--ckpt", nargs="+", required=True,
                    help="checkpoint 路径（正式配置传入末5个 checkpoint）")
    ap.add_argument("--recompute-batch", type=int, default=0,
                    help="重估 batch：0=全量单 batch(默认/优先,精确全集统计)；>0 且 <N=分块(shuffle=True)")
    ap.add_argument("--passes", type=int, default=RECOMPUTE_PASSES,
                    help="重估遍历目标集次数（正式配置=1；全量单batch模式幂等）")
    ap.add_argument("--seed", type=int, default=SEED, help="随机种子（默认 0）→ host.set_seed")
    ap.add_argument("--fold", type=int, default=0, help="折号（默认 0）：记录所评估 ckpt 的源折。"
                    "AdaBN 重估目标=KLSG 全 553 transductive，fold 无关、绝不接此参数")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    host.set_seed(args.seed)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"), "CUBLAS_WORKSPACE_CONFIG 未设(须在 import torch 前内联)"

    _hr("run1 #2 AdaBN（搭车 source-only SGD 基线）零训练评估")
    print(f"项目根 : {host.PROJECT_ROOT}")
    print(f"设备   : {device} | torch {torch.__version__}")
    print(f"AdaBN  : zero-train | reset BN running stats + momentum={ADABN_MOMENTUM}(锁定CMA) + 目标图重估 | 不训练/不动权重/不碰目标标签")
    print(f"ckpt   : {len(args.ckpt)} 个 (N={len(args.ckpt)}) -> {args.ckpt}")

    # ---- 生产：KLSG 全 553 无标签（transductive，fold 无关）AdaBN 重估 + label-free 评估 ----
    prod_items = _klsg_blind.read_klsg_unlabeled(host.PROJECT_ROOT)
    assert len(prod_items) == _klsg_blind.KLSG_N_EXPECTED, f"KLSG 张数={len(prod_items)} != 553"
    prod_recompute_loader, n_re, is_chunked = build_recompute_loader_unlabeled(prod_items, args.recompute_batch)
    prod_eval_loader = build_eval_loader_unlabeled(prod_items)
    print(f"\n[生产] AdaBN 目标 = KLSG 全 {len(prod_items)} transductive (sha {_klsg_blind.KLSG_MANIFEST_SHA256[:8]}) | "
          f"重估 n={n_re} {'分块(shuffle=True)' if is_chunked else '全量单 batch(精确全集统计)'} | passes={args.passes} | label-free")

    _hr("★ AdaBN 评估（KLSG 全 553 transductive；label-free，绝不读真实标签） ★")
    y_pred, avg_probs, per_ckpt, recompute_infos = adabn_eval_ensemble_unlabeled(
        args.ckpt, prod_recompute_loader, prod_eval_loader, device, passes=args.passes)
    has_nan = bool(not np.isfinite(avg_probs).all())
    pred_hist = [int((np.asarray(y_pred) == c).sum()) for c in range(len(CLASSES))]
    pred_collapsed = (max(pred_hist) == sum(pred_hist))
    _p = np.clip(avg_probs, 1e-12, 1.0)
    mean_pred_entropy = float((-(_p * np.log(_p)).sum(1)).mean())
    print(f"  目标预测直方 [airplane,ship] = {pred_hist} (和={sum(pred_hist)}=553) | 预测熵均值={mean_pred_entropy:.4f} | "
          f"{'坍缩(全预测同类!)' if pred_collapsed else '未坍缩'} | 概率含 NaN/Inf={has_nan}")
    print("  [!] 本入口只输出label-free预测健康信息；真实标签指标由独立评估包计算。")

    # ---- 落盘 ----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"adabn_eval_s{args.seed}_f{args.fold}.json"
    out = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "method": "AdaBN (zero-train BN re-estimation) on top of source-only SGD baseline (run1) checkpoint",
        "adabn_semantics": {
            "zero_train": True, "trains_weights": False, "uses_target_labels_for_recompute": False,
            "bn_reset": "reset_running_stats (mean->0,var->1,num_batches->0; gamma/beta untouched)",
            "momentum": "None (CMA, equal-weight full-set stats; LOCKED, NOT 0.1)",
            "recompute_mode": ("chunked(shuffle=True)" if is_chunked else "single_full_batch(shuffle=False)"),
            "recompute_passes": int(args.passes),
            "passes_note": "final setting: one target-set pass; momentum=None is locked",
        },
        "seed": args.seed, "fold": args.fold,
        "transductive_protocol": {
            "production": "KLSG full 553, NO train/test split, transductive (fold-independent): recompute on ALL KLSG unlabeled images + eval on ALL KLSG; LABEL-FREE",
        },
        "production_target": {"data": f"klsg full {len(prod_items)} transductive (sha {_klsg_blind.KLSG_MANIFEST_SHA256[:8]})",
                              "n_images": len(prod_items), "labels": "blinded (FORBIDDEN_LABEL); never read"},
        "checkpoints": [str(p) for p in args.ckpt], "n_checkpoints": len(args.ckpt),
        "host_reuse": "build_model / build_eval_transform / set_seed (train_source_only_sgd.py)",
        "recompute_infos": recompute_infos,
        "production_label_free_health": {
            "pred_hist_airplane_ship": pred_hist, "pred_collapsed": bool(pred_collapsed),
            "mean_pred_entropy": mean_pred_entropy, "has_nan_inf": has_nan,
            "uses_target_labels": False,
            "DISCLAIMER": "production path is LABEL-FREE; no KLSG true-label metric is computed here. Predictions/health only; scoring is handled by the separate evaluation package.",
        },
        "has_nan_inf": has_nan,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n落盘 -> {out_path}")
    _hr("AdaBN 评估完成（KLSG transductive；label-free）")
    return 1 if has_nan else 0


if __name__ == "__main__":
    sys.exit(main())
