#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_lastk.py -- run5 IW-CDAN × DSBN 末K softmax 概率平均评估（薄入口）
[SSS sim->real 主实验 · 末K 概率平均 · DSBN 目标支 domain=1 直接前向（不重估 BN）]

取 run5 IW-CDAN×DSBN 训练**末 K 个 epoch checkpoint**，各自 load -> forward(domain=1 目标支) -> softmax ->
概率平均 -> argmax。DSBN：两套完整 BN（源支 bns[0] / 目标支 bns[1]）；评估**显式走 domain=1 目标支**（已训）。
机制单点在 `src/_lastk_eval.py`；本文件只传 run5 的模型构造与目标分支前向回调。

★ wiring（spec §7.3）：model_factory = host.DSBNCDANResNet18(pretrained=False)（含 bns.0/1）；
  forward **显式传 domain=1** `_, logits = m(x, domain=1)`（domain 无默认、漏传 TypeError）；
  load 只取 ckpt['model_state_dict']（批4a ckpt 另含 ad_net/im_weights/global_step + bns.0/1，只取 model_state_dict）。
★ run5 此前完全无 eval 脚本 —— 本文件 net-new。
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
import train_iwcdan_dsbn_sgd as host          # DSBNCDANResNet18 / build_eval_transform / set_seed / ...
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _lastk_eval as lastk

_TARGET_DOMAIN = 1   # 目标支（DSBN 仓 trainval_multi.py L533；别抄 evaluate_multi.py 的 domain=0）


# ============================================================ per-host 回调（spec §7.3 wiring）
def model_factory():
    """run5 原生构造：host.DSBNCDANResNet18(pretrained=False)（含 bns.0/1 两支完整 BN；载 ckpt 覆盖）。"""
    return host.DSBNCDANResNet18(pretrained=False)


def forward_logits(model, x):
    """run5 DSBN forward **显式传 domain=1 目标支**（domain 无默认、漏传 TypeError）；取 logits。"""
    _, logits = model(x, domain=_TARGET_DOMAIN)
    return logits


if __name__ == "__main__":
    sys.exit(lastk.lastk_eval_main(
        host=host,
        model_factory=model_factory,
        forward_logits=forward_logits,
        tag_full="iwcdan_dsbn_full", tag_quick="iwcdan_dsbn_smoke",   # run5 tag 规约
        prog="run5 eval_lastk (DSBN domain=1 last-K prob average)"))
