#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
_lastk_eval.py -- 块2批4b **末K softmax 概率平均**评估共享核（host 无关 / default-BN·frozen·DSBN 通用）
================================================================================
定位（spec §3.3 / §7）：5 个 run（run1 default-BN、run2 CDAN、run3 IW-CDAN、run4 frozen-BN、
run5 DSBN）共用的"末 K 个 epoch checkpoint -> 各自 load -> softmax 概率 -> 概率平均 -> argmax"机制
的**单点实现**。AdaBN 宿主（#2/#4/#7）的末 K 已在 eval_adabn.py 建好（各自 reset+recompute BN），
本核**不做 BN 重估**——default/frozen/DSBN 直接 load -> forward -> softmax -> 概率平均 -> argmax。

★ 与 eval_adabn.py 的关系：两者使用相同的概率平均和盲化 loader；
  本核不执行 `adabn_recompute`。

================================  三护栏（spec §1 / §7）================================
① 真实标签红线：KLSG 生产路只经 `_klsg_blind.read_klsg_unlabeled` + UnlabeledImageDataset +
   collate_unlabeled；评估函数**只取 batch[0] 图像、永不收集/返回 y_true**。
   本入口只输出label-free预测健康信息，真实标签指标由独立评估包计算。
② 决定论：薄入口在 import torch 前内联 CUBLAS env（本核顶部亦 setdefault 兜底）；select_lastk 用
   严格正则取 epoch 整数、按整数升序确定排序取 top-K、固定顺序传平均器；逐 checkpoint softmax
   概率转为 **CPU numpy float64** 后按固定顺序累加，与论文最终评分链一致。
★ host 无关边界：核心平均函数（lastk_prob_average_unlabeled、eval_probs_unlabeled、
  select_lastk_ckpts、load_model_for_eval）**只吃 loader + model_factory + forward_logits**，不引用任何
  host 符号。`lastk_eval_main` 是 glue（薄入口传 host 模块进来），允许读 host.RESULTS_DIR 等常量。
"""

import os
# [determinism] 兜底：主钉在每个薄 eval_lastk.py 的第一段可执行代码（import torch 前）。此处 setdefault 幂等。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import re
import sys
import json
import argparse
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 让 host glue 能 import 同目录 _klsg_blind（核心选池/平均函数本身不依赖它）。
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


# ============================================================ select_lastk（按 epoch 整数 top-K）
_CKPT_RE_TMPL = r"^{tag}_epoch(\d+)\.pth$"


def select_lastk_ckpts(model_dir, tag, k=5):
    """spec §3.3 / 决议表#5：glob `{tag}_epoch*.pth` -> 严格正则 `^{tag}_epoch\\d+\\.pth$` 取 epoch 整数 ->
    **按整数升序确定排序** -> 取最高 K 个（top-K，K=min(k, 命中数)）-> 以**升序固定顺序**返回路径。
    ★ assert 命中 >=1；空集 fail-loud + 打印实际 glob 路径与目录内容（§7.3）。
    返回 (ckpt_paths[list,升序], epochs[list,升序])。**不靠文件名前缀排序（只靠 epoch 整数）。**"""
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or int(k) <= 0:
        raise ValueError("k must be a positive integer")
    k = int(k)
    model_dir = Path(model_dir)
    pat = re.compile(_CKPT_RE_TMPL.format(tag=re.escape(tag)))
    found = []
    for p in model_dir.glob(f"{tag}_epoch*.pth"):
        m = pat.match(p.name)
        if m:
            found.append((int(m.group(1)), p))
    glob_str = str(model_dir / f"{tag}_epoch*.pth")
    if not found:
        dir_listing = ([q.name for q in sorted(model_dir.glob("*"))]
                       if model_dir.is_dir() else "<目录不存在>")
        raise AssertionError(
            "select_lastk_ckpts: 未命中任何末K ckpt（空集 fail-loud）\n"
            f"  glob   : {glob_str}\n"
            f"  正则   : ^{tag}_epoch\\d+\\.pth$\n"
            f"  目录内容: {dir_listing}")
    epoch_values = [epoch for epoch, _ in found]
    duplicate_epochs = sorted({epoch for epoch in epoch_values if epoch_values.count(epoch) > 1})
    if duplicate_epochs:
        duplicate_files = [p.name for epoch, p in found if epoch in duplicate_epochs]
        raise ValueError(
            "select_lastk_ckpts: duplicate parsed epoch values are not allowed; "
            f"epochs={duplicate_epochs}, files={duplicate_files}"
        )
    found.sort(key=lambda t: t[0])              # 按 epoch 整数升序（确定排序，不靠文件名前缀）
    k_eff = min(k, len(found))
    selected = found[-k_eff:]                    # top-K = 最高 K 个 epoch（已升序切片，仍升序）
    return [str(p) for _, p in selected], [e for e, _ in selected]


# ============================================================ 加载 ckpt（只取 model_state_dict）
def load_model_for_eval(model_factory, ckpt_path, device):
    """spec §7.3/§7.7：model_factory() 构造本 host 原生模型 -> **只取 ckpt['model_state_dict']** strict load
    （批4a 的 run3/4/5 ckpt 另含 ad_net_state_dict/im_weights/global_step 顶层键，一律忽略）-> eval()。
    host 无关：model_factory 由薄入口提供（各 host 原生构造）。"""
    ckpt_path = Path(ckpt_path)
    assert ckpt_path.is_file(), f"checkpoint 不存在: {ckpt_path}"
    model = model_factory().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    assert "model_state_dict" in ckpt, f"ckpt 缺 model_state_dict 键: {list(ckpt.keys())}"
    model.load_state_dict(ckpt["model_state_dict"])                          # strict=True；忽略其余顶层键
    model.eval()
    return model


# ============================================================ 概率评估
def eval_probs_unlabeled(model, loader, device, forward_logits):
    """【生产函数族·KLSG 无标签】只取 batch[0] 图像（batch[1] 是毒丸 FORBIDDEN_LABEL, 绝不碰）；
    softmax -> 概率。**绝不收集/返回 y_true**。返回 probs[N,K]（numpy float64；softmax 后转换）。
    host 无关：forward_logits(model, x) -> logits 由薄入口提供（run1 单输出 / run2-4 双解包 / run5 domain=1）。"""
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)        # 只取图像；绝不读目标 label（batch[1] 永不索引）
            logits = forward_logits(model, x)
            probs.append(F.softmax(logits, dim=1).cpu())
    return torch.cat(probs, 0).numpy().astype(np.float64)


# ============================================================ 末K 概率平均（核心机制）
def lastk_prob_average_unlabeled(ckpt_paths, eval_loader, device, model_factory, forward_logits):
    """【生产·KLSG 无标签】spec §3.3/§7.4：每 ckpt 独立 load（只取 model_state_dict）-> softmax 概率（在
    平均**前**逐 ckpt 施加）-> **CPU numpy float64 按固定顺序累加** -> 平均概率 -> argmax。
    ★ 概率平均（非 logit 平均、非权重平均）；K=1 退化为单 ckpt softmax-argmax。
    ★ 绝不收集/返回 y_true。返回 (y_pred[list], avg_probs[N,K], per_ckpt_probs[list])。"""
    assert len(ckpt_paths) >= 1, "ckpt_paths 至少 1 个"
    prob_sum, per_ckpt = None, []
    for path in ckpt_paths:                              # 固定顺序（epoch 升序，select_lastk 已定序）
        model = load_model_for_eval(model_factory, path, device)
        probs = np.asarray(
            eval_probs_unlabeled(model, eval_loader, device, forward_logits),
            dtype=np.float64,
        )
        per_ckpt.append(probs)
        prob_sum = probs.copy() if prob_sum is None else prob_sum + probs
    avg_probs = prob_sum / np.float64(len(ckpt_paths))
    return avg_probs.argmax(1).tolist(), avg_probs, per_ckpt


# ============================================================ 输出辅助
def _hr(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# ============================================================ 薄入口统一 main（glue；薄入口传 host 进来）
def lastk_eval_main(*, host, model_factory, forward_logits,
                    tag_full, tag_quick, tag_variants=None, default_variant=None,
                    prog=None, argv=None):
    """5 个薄 eval_lastk.py 共用的 main（default-BN/frozen/DSBN 通用；**不做 BN 重估**）。

    薄入口职责（host 相关、传进本 glue）：
      - host            : 宿主训练模块（取 RESULTS_DIR/CLASSES/BATCH_SIZE/PROJECT_ROOT/
                          set_seed/build_eval_transform）
      - model_factory   : ()->本 host 原生模型（run1 build_model / run2 CDANResNet18() / run3-5 各类 pretrained=False）
      - forward_logits  : (model,x)->logits（run1 单输出 / run2-4 双解包 / run5 domain=1 目标支）
      - tag_full/tag_quick : 各 host tag 规约（run1 sgd_full/sgd_quick；run2-5 *_full/*_smoke）
      - tag_variants    : 可选的 {CLI名称: (full_tag, quick_tag)}；用于同一训练器产生多个正式变体
    """
    import _klsg_blind  # label-blind data access is needed only by this host glue

    ap = argparse.ArgumentParser(prog=prog, description="末K softmax 概率平均评估（default-BN/frozen/DSBN；不重估 BN）")
    ap.add_argument("--seed", type=int, default=getattr(host, "SEED", 0), help="随机种子 -> host.set_seed")
    ap.add_argument("--fold", type=int, default=0, help="折号（重建 model_dir）；KLSG 生产 transductive、fold 无关")
    ap.add_argument("--quick", action="store_true", help="用 quick/smoke tag 的 ckpt 池")
    if tag_variants:
        if default_variant not in tag_variants:
            raise ValueError("default_variant 必须是 tag_variants 中的键")
        ap.add_argument("--variant", choices=list(tag_variants), default=default_variant,
                        help=f"训练变体（默认 {default_variant}）")
    ap.add_argument("--k", type=int, default=5, help="末K（K=min(k, 命中 epoch 数)）")
    ap.add_argument("--label", choices=["main", "diag"], default=None,
                    help="可选配置目录标签；frozen BN 的 main/diag 需要显式提供")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    host.set_seed(args.seed)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") in (":4096:8", ":16:8"), \
        "CUBLAS_WORKSPACE_CONFIG 未设(须在薄入口 import torch 前内联)"

    RESULTS_DIR = host.RESULTS_DIR
    CLASSES = host.CLASSES
    BATCH_SIZE = host.BATCH_SIZE
    if tag_variants:
        selected_full, selected_quick = tag_variants[args.variant]
    else:
        selected_full, selected_quick = tag_full, tag_quick
    tag = selected_quick if args.quick else selected_full

    _hr(f"末K 概率平均评估（{tag} | seed={args.seed} fold={args.fold} | K={args.k}；不重估 BN）")
    print(f"项目根 : {host.PROJECT_ROOT}")
    print(f"设备   : {device} | torch {torch.__version__}")
    print(f"机制   : 末K 个 epoch ckpt -> 各自 load(只取 model_state_dict) -> softmax -> 概率平均 -> argmax | label-free 生产")

    # 生产路径仅依赖 KLSG 无标签目标数据。
    prod_items = _klsg_blind.read_klsg_unlabeled(host.PROJECT_ROOT)
    assert len(prod_items) == _klsg_blind.KLSG_N_EXPECTED, f"KLSG 张数={len(prod_items)} != 553"
    prod_loader = DataLoader(_klsg_blind.UnlabeledImageDataset(prod_items, host.build_eval_transform()),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                             collate_fn=_klsg_blind.collate_unlabeled)
    print(f"[生产] 末K 目标 = KLSG 全 {len(prod_items)} transductive "
          f"(sha {_klsg_blind.KLSG_MANIFEST_SHA256[:8]}) | label-free")

    # ==================== 生产：选真末K ckpt -> label-free 概率平均 ====================
    directory_tag = f"{tag}_{args.label}" if args.label else tag
    model_dir = RESULTS_DIR / f"{directory_tag}_s{args.seed}_f{args.fold}"
    ckpt_paths, epochs = select_lastk_ckpts(model_dir, tag, k=args.k)
    print(f"\n[末K 选池] model_dir={model_dir}")
    print(f"[末K 选池] K={len(ckpt_paths)} (epochs={epochs}, 升序固定序) -> {ckpt_paths}")

    _hr("★ 末K 概率平均评估（KLSG 全 553 transductive；label-free，绝不读真实标签）★")
    y_pred, avg_probs, per_ckpt = lastk_prob_average_unlabeled(
        ckpt_paths, prod_loader, device, model_factory, forward_logits)
    has_nan = bool(not np.isfinite(avg_probs).all())
    pred_hist = [int((np.asarray(y_pred) == c).sum()) for c in range(len(CLASSES))]
    pred_collapsed = (max(pred_hist) == sum(pred_hist))
    _p = np.clip(avg_probs, 1e-12, 1.0)
    mean_pred_entropy = float((-(_p * np.log(_p)).sum(1)).mean())
    print(f"  目标预测直方 {CLASSES} = {pred_hist} (和={sum(pred_hist)}={len(prod_items)}) | "
          f"预测熵均值={mean_pred_entropy:.4f} | {'坍缩(全预测同类!)' if pred_collapsed else '未坍缩'} | "
          f"概率含 NaN/Inf={has_nan}")
    print("  [!] 本入口只输出label-free预测健康信息；真实标签指标由独立评估包计算。")

    # ---- 落盘（与 eval_adabn 同款字段命名 + ignore generated_utc，复用 _determinism_acceptance json）----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_tag = directory_tag
    out_path = RESULTS_DIR / f"lastk_eval_{output_tag}_s{args.seed}_f{args.fold}.json"
    out = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "method": "last-K softmax probability averaging (NO BN re-estimation) on default-BN/frozen/DSBN checkpoints",
        "lastk_semantics": {
            "bn_recompute": False,
            "averaging": "per-ckpt softmax applied BEFORE averaging (probability average, NOT logit/weight average)",
            "accumulation": "CPU numpy float64 after per-checkpoint softmax, fixed order (epoch ascending)",
            "k_requested": args.k, "k_used": len(ckpt_paths), "epochs": epochs,
        },
        "tag": tag, "label": args.label, "seed": args.seed, "fold": args.fold,
        "model_dir": str(model_dir),
        "checkpoints": ckpt_paths, "n_checkpoints": len(ckpt_paths),
        "production_target": {"data": f"klsg full {len(prod_items)} transductive (sha {_klsg_blind.KLSG_MANIFEST_SHA256[:8]})",
                              "n_images": len(prod_items), "labels": "blinded (FORBIDDEN_LABEL); never read"},
        "production_label_free_health": {
            "pred_hist": pred_hist, "pred_hist_classes": CLASSES, "pred_collapsed": bool(pred_collapsed),
            "mean_pred_entropy": mean_pred_entropy, "has_nan_inf": has_nan,
            "uses_target_labels": False,
            "DISCLAIMER": "production path is LABEL-FREE; NO KLSG true-label F1 computed "
                          "here. Health only, NOT a judgment metric; scoring is handled by the separate evaluation package.",
        },
        "has_nan_inf": has_nan,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n落盘 -> {out_path}")
    _hr(f"末K 概率平均评估完成（{tag}；label-free）")
    return 1 if has_nan else 0
