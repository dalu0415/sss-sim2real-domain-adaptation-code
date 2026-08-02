#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_splits.py -- 站1 数据划分（两套分层 5 折）  [SSS sim->real 主实验 B]

================================  定盘星（划分规格，严格按 plan §3.2.1/§3.2.2/§3.2.4）================================
本脚本**只做数据划分、记录、自检**，不碰任何数据增强（增强留站2），不消费/产出任何真实标签指标。

切两套相互独立的分层 5 折：
  - 源域 arm2  = data/raw/ai4simulate   (airplane 90 / ship 400)
  - 目标域 KLSG = data/raw/SeabedObjects (airplane 66 / ship 487)

规格：
  1. 两侧各自用 sklearn.StratifiedKFold **按类别(airplane/ship)分层**，每折两类比例 ≈ 全局。
  2. **划分一次固定**：固定划分 seed = 42，shuffle=True，切一次。
     —— 所有训练 seed(0..4) 共用这一套划分；训练 seed 是 run 时的训练随机性，**不进本 csv**（§3.2.2）。
  3. **4 训 1 测**：每折 = 4 折并为 train + 1 折为 test，5 折轮转 —— 每个样本恰好当一次 test。
  4. 两套折各管各（本脚本不管谁用，只切+记录）：
       源 arm2 折 -> 源训练跑法用；目标 KLSG 折 -> 仅 ceiling 用。
  5. 标签来源 = **子文件夹名**(airplane/ship)。目标侧分层读 KLSG 类别 = §3.2.4 合法程序性例外
     （只用于切分、不产指标、不参与训练/调参/选模）。
  6. 决定论：喂入 StratifiedKFold 前样本按相对路径排序，保证跨机器/跨次重跑得到同一划分 -> 同一内容哈希。

产物（写入 data/split/）：
  - splits.csv    : 方案(b) 单一总表，列 = domain,fold,split,filepath,label（见下方"为什么选(b)"）。
  - splits_manifest.json : 划分 seed、各数据集每折每类计数、splits.csv 的 sha256 内容哈希（后续 run 比对防篡改）。

为什么 csv 选方案(b)单一总表（而非(a)每折每 split 一个文件）：
  splits.csv 是"所有 run 读同一份"的承重路径(README: 配对Δ 成立的命根)，单文件 = 单一真相源 =
  **单个内容哈希**即可冻结/校验（plan §3.2.1 + README 要求内容哈希 manifest）；按 (domain,fold,split)
  过滤一行 pandas/csv 即可，列名自解释、人可读。20 个小文件反而要管 20 个哈希、易漂。
====================================================================================================================
"""

import os
import csv
import json
import hashlib
import datetime
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

# ----------------------------------------------------------------------------- 配置
# 项目根 = 本文件所在 src/datapreprocess/ 的上上级。
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "split"

# 两套折：domain -> 数据集根目录
DATASETS = OrderedDict([
    ("arm2", RAW_DIR / "ai4simulate"),    # 源域（半合成）
    ("klsg", RAW_DIR / "SeabedObjects"),  # 目标域（真实 KLSG）
])
CLASSES = ["airplane", "ship"]   # 分层依据 = 类别；标签取子文件夹名
IMG_EXTS = {".png"}              # 只收 png；自动排除 *_labels.csv 等非图片
N_SPLITS = 5
SPLIT_SEED = 42                  # 划分 seed（固定、切一次、所有训练 seed 共用）

CSV_COLUMNS = ["domain", "fold", "split", "filepath", "label"]


# ----------------------------------------------------------------------------- 采样收集
def collect_samples(dataset_dir):
    """枚举 dataset_dir 下 airplane/ ship/ 内的 png，返回按相对路径排序的 [(relpath, label), ...]。
    relpath 相对项目根、正斜杠（跨平台）。非 png（如 *_labels.csv）一律剔除。"""
    samples = []
    for cls in CLASSES:
        cls_dir = dataset_dir / cls
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"缺类别子目录: {cls_dir}")
        for p in cls_dir.iterdir():
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                rel = os.path.relpath(p, PROJECT_ROOT).replace(os.sep, "/")
                samples.append((rel, cls))
    # 决定论：按相对路径排序，喂入 StratifiedKFold 前固定输入顺序
    samples.sort(key=lambda t: t[0])
    return samples


# ----------------------------------------------------------------------------- 切折
def make_folds(samples):
    """两类分层 5 折；返回 list[fold] -> {"train": [idx...], "test": [idx...]}（idx 指向 samples）。"""
    y = np.array([1 if lab == "ship" else 0 for _, lab in samples])  # 仅内部用于分层
    X = np.arange(len(samples))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SPLIT_SEED)
    folds = []
    for train_idx, test_idx in skf.split(X, y):
        folds.append({"train": train_idx.tolist(), "test": test_idx.tolist()})
    return folds


# ----------------------------------------------------------------------------- 主流程：建表
def build_rows():
    """切两套折，返回 (rows, per_domain_folds, per_domain_samples)。
    rows 按 (domain, fold, split=train|test, filepath排序) 决定论展开。"""
    rows = []
    per_domain_folds = OrderedDict()
    per_domain_samples = OrderedDict()
    for domain, ddir in DATASETS.items():
        samples = collect_samples(ddir)
        folds = make_folds(samples)
        per_domain_samples[domain] = samples
        per_domain_folds[domain] = folds
        for fold_id, fold in enumerate(folds):
            for split in ("train", "test"):
                idxs = sorted(fold[split], key=lambda i: samples[i][0])  # 行内按路径排序
                for i in idxs:
                    relpath, label = samples[i]
                    rows.append([domain, fold_id, split, relpath, label])
    return rows, per_domain_folds, per_domain_samples


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        w.writerows(rows)


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(per_domain_folds, per_domain_samples, csv_path):
    """每折每类计数 + 全局计数 + splits.csv 内容哈希。"""
    datasets_block = OrderedDict()
    for domain, folds in per_domain_folds.items():
        samples = per_domain_samples[domain]
        global_counts = Counter(lab for _, lab in samples)
        fold_block = []
        for fold_id, fold in enumerate(folds):
            entry = {"fold": fold_id}
            for split in ("train", "test"):
                c = Counter(samples[i][1] for i in fold[split])
                entry[split] = {cls: int(c.get(cls, 0)) for cls in CLASSES}
                entry[split]["total"] = int(sum(c.values()))
            fold_block.append(entry)
        datasets_block[domain] = {
            "root": os.path.relpath(DATASETS[domain], PROJECT_ROOT).replace(os.sep, "/"),
            "n_samples": len(samples),
            "global_class_counts": {cls: int(global_counts.get(cls, 0)) for cls in CLASSES},
            "folds": fold_block,
        }
    manifest = OrderedDict([
        ("description", "SSS sim->real 主实验B 站1 数据划分（两套分层5折，划分一次固定，所有训练seed共用）"),
        ("spec_refs", ["plan §3.2.1", "plan §3.2.2", "plan §3.2.4"]),
        ("generated_utc", datetime.datetime.now(datetime.timezone.utc).isoformat()),
        ("split_seed", SPLIT_SEED),
        ("n_splits", N_SPLITS),
        ("shuffle", True),
        ("sklearn_splitter", "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)"),
        ("stratify_by", "class label (airplane/ship), derived from subfolder name"),
        ("scheme", "4 folds -> train, 1 fold -> test; 5-fold rotation; each sample tests exactly once"),
        ("training_seeds_note", "训练 seed 0..4 是 run 时随机性(§3.2.2)，不在本划分内"),
        ("fold_usage", {"arm2": "源训练跑法", "klsg": "仅 ceiling"}),
        ("csv_file", csv_path.name),
        ("csv_columns", CSV_COLUMNS),
        ("csv_sha256", sha256_of_file(csv_path)),
        ("datasets", datasets_block),
    ])
    return manifest


# ----------------------------------------------------------------------------- 自检
def selfcheck(rows, per_domain_folds, per_domain_samples, csv_path):
    print("\n" + "=" * 78)
    print("PHASE 自检")
    print("=" * 78)
    ok_all = True

    # --- 1. 分层对：每折 train/test 每类计数 + airplane 占比（应 ≈ 全局占比） ---
    print("\n[1] 分层（每折 train/test 的 airplane/ship 计数；af% = airplane 占比）")
    for domain, folds in per_domain_folds.items():
        samples = per_domain_samples[domain]
        gc = Counter(lab for _, lab in samples)
        g_af = gc["airplane"] / len(samples)
        print(f"  {domain}: 全局 airplane={gc['airplane']} ship={gc['ship']} "
              f"(共 {len(samples)}, 全局 af%={g_af:.3f})")
        print(f"    {'fold':>4} | {'train(air/ship/af%)':>26} | {'test(air/ship/af%)':>26}")
        for fid, fold in enumerate(folds):
            tr = Counter(samples[i][1] for i in fold["train"])
            te = Counter(samples[i][1] for i in fold["test"])
            tr_af = tr["airplane"] / max(1, sum(tr.values()))
            te_af = te["airplane"] / max(1, sum(te.values()))
            print(f"    {fid:>4} | {tr['airplane']:>5}/{tr['ship']:>4}/{tr_af:>6.3f}{'':>5} "
                  f"| {te['airplane']:>5}/{te['ship']:>4}/{te_af:>6.3f}")

    # --- 2. 无泄漏：每套每折 train ∩ test = ∅ ---
    print("\n[2] 无泄漏（每套每折 train ∩ test 应为空）")
    for domain, folds in per_domain_folds.items():
        bad = 0
        for fid, fold in enumerate(folds):
            inter = set(fold["train"]) & set(fold["test"])
            if inter:
                bad += 1
                print(f"    [FAIL] {domain} fold{fid}: 交集大小 {len(inter)}")
        status = "PASS" if bad == 0 else "FAIL"
        ok_all &= (bad == 0)
        print(f"    {domain}: {status}（5 折全部 train∩test=空）")

    # --- 3. 覆盖：5 折的 test 并起来 = 全集，且每样本恰好当一次 test ---
    print("\n[3] 覆盖（5 折 test 并集=全集，每样本恰好 test 一次）")
    for domain, folds in per_domain_folds.items():
        samples = per_domain_samples[domain]
        test_counter = Counter()
        for fold in folds:
            for i in fold["test"]:
                test_counter[i] += 1
        union = set(test_counter.keys())
        full = set(range(len(samples)))
        cover_ok = (union == full)
        once_ok = all(v == 1 for v in test_counter.values()) and len(test_counter) == len(samples)
        # 每折 train 大小应 = N - test 大小
        train_ok = all(len(f["train"]) + len(f["test"]) == len(samples) for f in folds)
        ok = cover_ok and once_ok and train_ok
        ok_all &= ok
        print(f"    {domain}: 并集=={len(union)}/{len(samples)} cover={cover_ok} "
              f"each_once={once_ok} train+test==N={train_ok} -> {'PASS' if ok else 'FAIL'}")

    # --- 4. 可读回 + UTF-8 + 路径有效 ---
    print("\n[4] csv 读回（UTF-8、行数、表头、路径存在）")
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rd = list(csv.reader(f))
    header, data = rd[0], rd[1:]
    header_ok = header == CSV_COLUMNS
    nrows_ok = len(data) == len(rows)
    # 路径存在性（抽检全部 distinct 路径）
    distinct_paths = {r[3] for r in data}
    missing = [rp for rp in distinct_paths if not (PROJECT_ROOT / rp).is_file()]
    path_ok = (len(missing) == 0)
    ok_all &= (header_ok and nrows_ok and path_ok)
    print(f"    表头匹配={header_ok}  行数={len(data)}(期望 {len(rows)}) 匹配={nrows_ok}")
    print(f"    distinct 路径数={len(distinct_paths)}  缺失文件数={len(missing)} -> "
          f"{'PASS' if path_ok else 'FAIL'}")
    if missing:
        for m in missing[:5]:
            print(f"      [MISSING] {m}")

    print("\n" + "-" * 78)
    print(f"自检总判：{'ALL PASS' if ok_all else 'HAS FAILURE'}")
    print("-" * 78)
    return ok_all


# ----------------------------------------------------------------------------- main
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "splits.csv"
    manifest_path = OUT_DIR / "splits_manifest.json"

    rows, per_domain_folds, per_domain_samples = build_rows()
    write_csv(rows, csv_path)

    manifest = build_manifest(per_domain_folds, per_domain_samples, csv_path)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"项目根       : {PROJECT_ROOT}")
    print(f"splits.csv   : {csv_path}  ({len(rows)} 行 + 表头)")
    print(f"manifest     : {manifest_path}")
    print(f"csv sha256   : {manifest['csv_sha256']}")

    ok = selfcheck(rows, per_domain_folds, per_domain_samples, csv_path)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
