#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
_make_churn_subset.py -- 生成 churn(目标预测翻转率)固定 100 张 KLSG 子集 manifest（standalone，跑一次）
================================================================================
站2 块2 第③批。churn 标量 = 相邻 epoch 对【同一固定 100 张 KLSG 子集】argmax 预测的不一致比例。
本脚本**一次性**从冻结的 553 张 KLSG transductive 全集 manifest 里**盲采**100 张、按同配方落盘为
`data/split/klsg_churn100_manifest.txt`，供 5 个训练脚本经 `_klsg_blind.read_klsg_churn_subset()` 恒同入场。

★ 红线（碰一条即作废）：
  - **类盲**：只对【行索引】抽样。**绝不**解析 filepath 里的 airplane/ship 目录名或任何类别信息来
    采样/分层/排序。manifest 行只当 **filepath 字符串**搬运。
  - 抽样 RNG 与训练 RNG **物理隔离**：独立脚本 + 独立 `np.random.default_rng(CHURN_SUBSET_SEED=42)`，
    与训练 `--seed`、make_splits 的 SPLIT_SEED 互不干扰（恰好同为 42 是 provenance 对齐，非耦合）。
  - 跨档/跨 seed/跨 fold 恒为同一份 100 张（seed 固定 + 落盘冻结 + 下游 sha 不可变锚）。

★ sha 自校（不可变锚）：
  先按 553 配方（去重 -> 字典序升序 -> UTF-8 + LF 无尾换行）对源 manifest 复算 sha256，断言 ==
  f7241c60…b66bc23（与 _klsg_blind.KLSG_MANIFEST_SHA256 同源），不符则 raise —— 防有人偷换源 manifest。

产物（写入 data/split/）：
  - klsg_churn100_manifest.txt  : 100 行 filepath（与 553 同配方：去重->字典序升序->UTF-8+LF无尾换行）。
  - klsg_churn100_manifest.json : churn_subset_seed / churn manifest sha256 / n=100 / 源553 sha / 采样行索引。

环境：直连 sonarimage env 的 python.exe + PYTHONIOENCODING=utf-8 PYTHONUTF8=1（别 conda run = GBK 崩）。
================================================================================
"""

import json
import hashlib
import datetime
from collections import OrderedDict
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------- 配置
# 项目根 = 本文件所在 src/ 的上一级。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = PROJECT_ROOT / "data" / "split"
SRC_MANIFEST_PATH = SPLIT_DIR / "klsg_unlabeled_manifest.txt"
OUT_TXT_PATH = SPLIT_DIR / "klsg_churn100_manifest.txt"
OUT_JSON_PATH = SPLIT_DIR / "klsg_churn100_manifest.json"

# 源 553 manifest 的内容哈希（不可变锚；与 _klsg_blind.KLSG_MANIFEST_SHA256 同源）
SRC_MANIFEST_SHA256 = "f7241c601543999f1cc14a7947df5962e25a94c44f687980b8184ea64b66bc23"
SRC_N_EXPECTED = 553            # KLSG transductive 全集张数
CHURN_SUBSET_SEED = 42          # churn 子集抽样 seed（独立于训练 --seed；跨档恒同一份）
CHURN_N = 100                   # churn 子集张数


# ----------------------------------------------------------------------------- 553 配方
def _canon_bytes(lines):
    """553 落盘配方：去重 -> 字典序升序 -> '\\n' 连接（LF）-> 无尾换行 -> UTF-8 编码。
    **只对 filepath 字符串排序/去重，绝不解析类别。**"""
    return "\n".join(sorted(set(lines))).encode("utf-8")


def _parse_lines(raw_bytes):
    """raw bytes -> filepath 字符串列表（UTF-8 解码、split LF、过滤空行、strip）。只搬字符串，不解析类别。"""
    return [ln.strip() for ln in raw_bytes.decode("utf-8").split("\n") if ln.strip()]


# ----------------------------------------------------------------------------- 主流程
def main():
    # --- 1. 读源 manifest + 按 553 配方复算 sha256 自校（不可变锚，不符 raise） ---
    assert SRC_MANIFEST_PATH.is_file(), f"源 KLSG manifest 不存在: {SRC_MANIFEST_PATH}"
    raw = SRC_MANIFEST_PATH.read_bytes()
    src_lines = _parse_lines(raw)
    assert len(src_lines) == SRC_N_EXPECTED, (
        f"RED LINE: 源 manifest 行数={len(src_lines)} != {SRC_N_EXPECTED}")
    # 按 553 配方对源行复算（不直接用 raw bytes，确保配方一致性）
    src_canon_digest = hashlib.sha256(_canon_bytes(src_lines)).hexdigest()
    assert src_canon_digest == SRC_MANIFEST_SHA256, (
        f"RED LINE: 源 manifest 553 配方复算 sha256 不符（源被改动？）\n"
        f"  期望 {SRC_MANIFEST_SHA256}\n  实得 {src_canon_digest}")
    print(f"[1] 源 manifest 自校通过: {len(src_lines)} 行, 553 配方复算 sha256 == 锚 {src_canon_digest}")

    # 采样源 = 553 配方的规范顺序（与冻结 manifest 字节一致），保证行索引语义恒定
    src_canon_lines = sorted(set(src_lines))
    assert len(src_canon_lines) == SRC_N_EXPECTED, "RED LINE: 去重后行数变化（源有重复？）"

    # --- 2. 类盲抽样：仅对【行索引】抽 100，绝不解析 filepath 类别 ---
    rng = np.random.default_rng(CHURN_SUBSET_SEED)
    idx = sorted(int(i) for i in rng.choice(SRC_N_EXPECTED, size=CHURN_N, replace=False))
    assert len(idx) == CHURN_N and len(set(idx)) == CHURN_N, "抽样行索引数/去重异常"
    selected = [src_canon_lines[i] for i in idx]   # 仅按索引取 filepath 字符串
    print(f"[2] 类盲抽样: default_rng({CHURN_SUBSET_SEED}) -> {CHURN_N} 个行索引 "
          f"(min={idx[0]}, max={idx[-1]}); 仅索引抽样, 未解析任何类别")

    # --- 3. 按 553 同配方落盘 churn manifest（去重->字典序升序->UTF-8+LF无尾换行） ---
    churn_body = _canon_bytes(selected)
    churn_lines = sorted(set(selected))
    assert len(churn_lines) == CHURN_N, f"RED LINE: churn 子集去重后 != {CHURN_N}（抽样有重复？）"
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TXT_PATH.write_bytes(churn_body)           # write_bytes 保证 LF + 无尾换行 + UTF-8, 无平台换行翻译
    churn_sha256 = hashlib.sha256(churn_body).hexdigest()
    print(f"[3] 已写 {OUT_TXT_PATH}  ({len(churn_lines)} 行, UTF-8+LF无尾换行)")
    print(f"    churn manifest sha256 = {churn_sha256}")

    # --- 4. 写 json provenance（seed / sha / n / 源553 sha / 采样行索引；无任何类别信息） ---
    manifest = OrderedDict([
        ("description",
         "SSS sim->real 站2块2第③批 churn 固定100张 KLSG 子集 manifest（类盲行索引抽样，跨档恒同一份）"),
        ("generated_utc", datetime.datetime.now(datetime.timezone.utc).isoformat()),
        ("churn_subset_seed", CHURN_SUBSET_SEED),
        ("sampler", f"numpy.random.default_rng({CHURN_SUBSET_SEED}).choice({SRC_N_EXPECTED}, size={CHURN_N}, replace=False)"),
        ("sampling_note", "仅对行索引抽样（类盲）；绝不解析 filepath 里的 airplane/ship 目录名或任何类别"),
        ("n", CHURN_N),
        ("recipe", "去重 -> 字典序升序 -> UTF-8 + LF 无尾换行（与 553 manifest 同配方）"),
        ("source_manifest_file", SRC_MANIFEST_PATH.name),
        ("source_manifest_sha256", SRC_MANIFEST_SHA256),
        ("source_n", SRC_N_EXPECTED),
        ("churn_manifest_file", OUT_TXT_PATH.name),
        ("churn_manifest_sha256", churn_sha256),
        ("sampled_row_indices", idx),   # 进 553 规范顺序的行索引（整数，类盲、可复现）
    ])
    with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[4] 已写 {OUT_JSON_PATH}")

    print("\n" + "=" * 78)
    print("CHURN 子集生成完成。回填用 sha256（填入 _klsg_blind.KLSG_CHURN100_SHA256）：")
    print(f"  {churn_sha256}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
