#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_lastk.py -- run1 default-BN 末K softmax 概率平均评估（薄入口）
[SSS sim->real 主实验 · 末K 概率平均 · default-BN 直接前向（不重估 BN）]

这是什么：取 run1 source-only SGD 基线**末 K 个 epoch checkpoint**，各自 load -> forward -> softmax ->
概率平均 -> argmax。**不做 AdaBN 那样的 BN 重估**（default-BN 用 ckpt 自带的训练期 BN running stats）。
机制单点实现在 `src/_lastk_eval.py`（host 无关共享核）；本文件只传 run1 的 per-host 回调。

★ AdaBN 末K（#2/#4/#7）在 `eval_adabn.py`，不在本文件 scope（本核去掉 adabn_recompute 那步）。
★ KLSG评估全程label-free（经 `_klsg_blind`）；真实标签指标由独立评估包计算。
"""

import os
# [determinism] 第一段可执行代码：必须在任何拉进 torch 的 import（host / _lastk_eval）之前内联设置。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys
from pathlib import Path

# ---- 复用宿主脚本（同目录）----
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import train_source_only_sgd as host          # build_model / build_eval_transform / set_seed / ...
# ---- 末K 概率平均共享核（src/ 在 parents[1]）----
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _lastk_eval as lastk


# ============================================================ per-host 回调（spec §7.3 wiring）
def model_factory():
    """构造同结构模型；checkpoint strict load会提供全部训练权重。"""
    return host.build_model(pretrained=False)


def forward_logits(model, x):
    """run1 裸 resnet18 forward **单输出 logits**（非 run2-5 的双解包）。"""
    return model(x)


if __name__ == "__main__":
    sys.exit(lastk.lastk_eval_main(
        host=host,
        model_factory=model_factory,
        forward_logits=forward_logits,
        tag_full="sgd_full", tag_quick="sgd_quick",     # run1 tag 规约
        prog="run1 eval_lastk (default-BN last-K prob average)"))
