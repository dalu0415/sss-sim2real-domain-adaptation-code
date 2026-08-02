#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_lastk.py -- run4 IW-CDAN frozen-BN 末K softmax 概率平均评估（薄入口）
[SSS sim->real 主实验 · 末K 概率平均 · frozen-BN 直接前向（不重估 BN）]

取 run4 IW-CDAN frozen-BN 训练**末 K 个 epoch checkpoint**，各自 load -> forward -> softmax -> 概率平均 -> argmax。
frozen 档：BN running stats 冻在 ImageNet 预训练值；推理 model.eval() 下两者都用冻结 running stats，前向**不改统计**。
机制单点在 `src/_lastk_eval.py`；本文件只传 run4 的模型构造与前向回调。

★ wiring（spec §7.3）：model_factory = host.FrozenBNCDANResNet18(pretrained=False)；forward 双解包 `_, logits = m(x)`；
  load 只取 ckpt['model_state_dict']（批4a ckpt 另含 ad_net/im_weights/global_step 顶层键，忽略）。
★ run4 此前完全无 eval 脚本 —— 本文件 net-new。
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
import train_iwcdan_frozen_sgd as host        # FrozenBNCDANResNet18 / build_eval_transform / set_seed / ...
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _lastk_eval as lastk


# ============================================================ per-host 回调（spec §7.3 wiring）
def model_factory():
    """run4 原生构造：host.FrozenBNCDANResNet18(pretrained=False)（frozen BN；离线随机 backbone，载 ckpt 覆盖）。"""
    return host.FrozenBNCDANResNet18(pretrained=False)


def forward_logits(model, x):
    """run4 FrozenBNCDANResNet18 forward 双输出 (feature, logits)，取 logits。"""
    _, logits = model(x)
    return logits


if __name__ == "__main__":
    sys.exit(lastk.lastk_eval_main(
        host=host,
        model_factory=model_factory,
        forward_logits=forward_logits,
        tag_full="iwcdan_frozen_full", tag_quick="iwcdan_frozen_smoke",   # run4 tag 规约
        prog="run4 eval_lastk (frozen-BN last-K prob average)"))
