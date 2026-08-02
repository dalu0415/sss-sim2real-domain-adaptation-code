#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_lastk.py -- run2 CDAN / CDAN+E 末K softmax 概率平均评估（薄入口）
[SSS sim->real 主实验 · 末K 概率平均 · default-BN 直接前向（不重估 BN）]

取 run2 CDAN 或 CDAN+E 训练**末 K 个 epoch checkpoint**，各自 load -> forward -> softmax -> 概率平均 -> argmax。
机制单点在 `src/_lastk_eval.py`；本文件只传 run2 per-host 回调。AdaBN 末K（搭 run2）在 eval_adabn.py，不在此 scope。

★ wiring：model_factory = host.CDANResNet18(pretrained=False)；forward双解包`_, logits = m(x)`。
  checkpoint随后strict load，因此评估无需先下载会被覆盖的ImageNet初始化。
★ KLSG评估全程label-free（经 `_klsg_blind`）；真实标签指标由独立评估包计算。
"""

import os
# [determinism] 第一段可执行代码：必须在任何拉进 torch 的 import（host / _lastk_eval）之前内联设置。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import train_cdan_sgd as host                 # CDANResNet18 / build_eval_transform / set_seed / ...
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _lastk_eval as lastk


# ============================================================ per-host 回调（spec §7.3 wiring）
def model_factory():
    """构造同结构模型；checkpoint strict load会提供全部训练权重。"""
    return host.CDANResNet18(pretrained=False)


def forward_logits(model, x):
    """run2 CDANResNet18 forward 双输出 (feature, logits)，取 logits。"""
    _, logits = model(x)
    return logits


if __name__ == "__main__":
    sys.exit(lastk.lastk_eval_main(
        host=host,
        model_factory=model_factory,
        forward_logits=forward_logits,
        tag_full="cdan_full", tag_quick="cdan_smoke",   # run2 tag 规约
        tag_variants={
            "cdan": ("cdan_full", "cdan_smoke"),
            "cdan-e": ("cdan_e_full", "cdan_e_smoke"),
        },
        default_variant="cdan",
        prog="run2 eval_lastk (CDAN/CDAN+E last-K prob average)"))
