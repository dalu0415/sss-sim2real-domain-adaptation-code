#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_lastk.py -- run3 IW-CDAN 末K softmax 概率平均评估（薄入口）
[SSS sim->real 主实验 · 末K 概率平均 · default-BN 直接前向（不重估 BN）]

取 run3 IW-CDAN 训练**末 K 个 epoch checkpoint**，各自 load -> forward -> softmax -> 概率平均 -> argmax。
机制单点在 `src/_lastk_eval.py`；本文件只传 run3 per-host 回调。AdaBN 末K（搭 run3）在 eval_adabn.py，不在此 scope。

★ wiring（spec §7.3）：model_factory = host.CDANResNet18(pretrained=False)（省 ImageNet 载入/离线/决定论不变，
  载 ckpt 覆盖）；forward 双解包 `_, logits = m(x)`。load 只取 ckpt['model_state_dict']（批4a ckpt 另含
  ad_net_state_dict/im_weights/global_step 顶层键，忽略）。
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
import train_iwcdan_sgd as host               # CDANResNet18 / build_eval_transform / set_seed / ...
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _lastk_eval as lastk


# ============================================================ per-host 回调（spec §7.3 wiring）
def model_factory():
    """run3 原生构造：host.CDANResNet18(pretrained=False)（离线随机 backbone，载 ckpt 覆盖；决定论不变）。"""
    return host.CDANResNet18(pretrained=False)


def forward_logits(model, x):
    """run3 CDANResNet18 forward 双输出 (feature, logits)，取 logits。"""
    _, logits = model(x)
    return logits


if __name__ == "__main__":
    sys.exit(lastk.lastk_eval_main(
        host=host,
        model_factory=model_factory,
        forward_logits=forward_logits,
        tag_full="iwcdan_full", tag_quick="iwcdan_smoke",   # run3 tag 规约
        prog="run3 eval_lastk (IW-CDAN last-K prob average)"))
