# ============================================================================
# Step 1.5b (de-normalization, JOINT mean+std) — Part 1: AI4 bg JOINT pool builder
# ============================================================================
# 目的 (PURPOSE):
#   Step 1.5b 在 1.5 的基础上, 不只放开 per-image 背景 MEAN(平移), 还放开
#   per-image 背景 STD(对比度)。两者【成对】从同一个 AI4 块抽 (mu_target,
#   sigma_target), 用一次仿射变换同时定背景均值和标准差, 直接攻击
#   "对比度异常一致" 这条 C3 指纹的另一半 (1.5 只解了均值那一半)。
#
#   本脚本 = ai4_bg_brightness_pool.py 的【逐字拷贝】, 唯一行为改动:
#     block_means_from_strip -> 同时收集每个通过块的 (mean, std)。
#   crop 网格 + nadir 检测 + 质量过滤 (MIN_BLOCK_MEAN=20 / MIN_BLOCK_STD=5)
#   逐字不变 -> 产出的 block_means 数组必须与 1.5 池逐元素一致 (同块同序)。
#
# 输出新文件 (不覆盖 1.5 的 bg_brightness_pool.npz):
#   step1p5b_denorm\bg_brightness_pool_joint.npz
#     block_means (数组), block_stds (数组, 与 block_means 同序成对),
#     n_blocks, p5/p50/p95 (对 means), std_p5/std_p50/std_p95 (对 stds),
#     crop 参数, source_dir, n_source_images。
#
# 诚实铁律: 本脚本绝无任何 KLSG / SeabedObjects 路径或其 mean/std 数值。
#   池只来自 AI4 terrain 源 (synth\data\terrain\images, normalize 之前)。
#   AI4 源块 mean≈58.5、within-block std 与块间 mean-std≈35.6 都比 KLSG 的
#   80/30 还暗/异质 —— 朝它走天然不可能 gaming。
# ============================================================================

import os
import glob
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime


# ============ 路径配置 ============
# 读: AI4 terrain 源 (= bg_prepare 的 INPUT_DIR, normalize 之前的原图)
INPUT_DIR = r"PATH_TO_AI4_TERRAIN_IMAGES"
# 写: Step1.5b 子树 (新建, 不碰 1.5 的 step1p5_denorm)
OUTPUT_POOL_PATH = r"PATH_TO_SYNTH_IMPROVED_WORKSPACE\data\outputs\step1p5b_denorm\bg_brightness_pool_joint.npz"
OUTPUT_STATS_DIR = r"PATH_TO_SYNTH_IMPROVED_WORKSPACE\data\outputs\step1p5b_denorm\stats"

# ---- 裁剪参数: 逐字取自 bg_prepare.py (与 ai4_bg_brightness_pool.py 完全相同) ----
BLOCK_SIZE = 700              # bg_prepare.BLOCK_SIZE
OVERLAP_RATIO = 0.2           # bg_prepare.OVERLAP_RATIO
STRIDE = int(BLOCK_SIZE * (1 - OVERLAP_RATIO))  # = 560
NADIR_THRESHOLD_RATIO = 0.4   # bg_prepare.NADIR_THRESHOLD_RATIO
SAFETY_MARGIN_RATIO = 0.5     # bg_prepare.SAFETY_MARGIN_RATIO
MIN_BLOCK_MEAN = 20           # bg_prepare.MIN_BLOCK_MEAN
MIN_BLOCK_STD = 5             # bg_prepare.MIN_BLOCK_STD


# ---- fail-fast: 写路径必在 step1p5b_denorm 下; 读路径必在 AI4 terrain 源白名单内 ----
_STEP1P5B_ROOT = r"PATH_TO_SYNTH_IMPROVED_WORKSPACE\data\outputs\step1p5b_denorm"
_AI4_SOURCE = INPUT_DIR

def _assert_paths():
    sr = os.path.normcase(os.path.abspath(_STEP1P5B_ROOT))
    src = os.path.normcase(os.path.abspath(_AI4_SOURCE))
    for p in [OUTPUT_POOL_PATH, OUTPUT_STATS_DIR]:
        assert os.path.normcase(os.path.abspath(p)).startswith(sr), \
            f"WRITE path must be under step1p5b_denorm: {p}"
    assert os.path.normcase(os.path.abspath(INPUT_DIR)).startswith(src), \
        f"READ path must be the AI4 terrain source: {INPUT_DIR}"
    # 诚实自检: 本脚本任何路径绝不含 KLSG / SeabedObjects
    for p in [INPUT_DIR, OUTPUT_POOL_PATH, OUTPUT_STATS_DIR]:
        low = p.lower()
        assert "klsg" not in low and "seabedobject" not in low, \
            f"FORBIDDEN: pool must NOT touch any KLSG/SeabedObjects path: {p}"


# ---------------------------------------------------------------------------
# crop 逻辑 (逐字镜像 bg_prepare.py 的 detect_nadir_gap + crop_blocks_from_strip,
#           但去掉 normalize_blocks —— 现在每块同时记 (mean, std))
# ---------------------------------------------------------------------------

def detect_nadir_gap(img):
    """逐字镜像 bg_prepare.detect_nadir_gap。返回 (usable_left_end, usable_right_start, nadir_width)。"""
    col_means = np.mean(img, axis=0)
    global_mean = np.mean(col_means)
    threshold = global_mean * NADIR_THRESHOLD_RATIO

    dark_cols = col_means < threshold
    if not np.any(dark_cols):
        return None, None, None

    changes = np.diff(dark_cols.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    if dark_cols[0]:
        starts = np.insert(starts, 0, 0)
    if dark_cols[-1]:
        ends = np.append(ends, len(dark_cols))
    if len(starts) == 0 or len(ends) == 0:
        return None, None, None

    widths = ends - starts
    max_idx = np.argmax(widths)
    nadir_left = starts[max_idx]
    nadir_right = ends[max_idx]
    nadir_width = nadir_right - nadir_left

    margin = int(nadir_width * SAFETY_MARGIN_RATIO)
    usable_left_end = nadir_left - margin
    usable_right_start = nadir_right + margin
    return usable_left_end, usable_right_start, nadir_width


def block_stats_from_strip(img, x_start, x_end):
    """逐字镜像 bg_prepare.crop_blocks_from_strip 的网格 + 质量过滤,
       但【对每个通过块同时返回 RAW (mean, std)】(不 normalize、不存块本身)。
       网格 / 过滤逻辑与 ai4_bg_brightness_pool.block_means_from_strip 逐字相同,
       仅多收集 std -> 保证 means 序列逐元素与 1.5 池一致。"""
    h = img.shape[0]
    strip_width = x_end - x_start
    means = []
    stds = []
    if strip_width < BLOCK_SIZE or h < BLOCK_SIZE:
        return means, stds

    n_cols = (strip_width - BLOCK_SIZE) // STRIDE + 1
    n_rows = (h - BLOCK_SIZE) // STRIDE + 1
    for row in range(n_rows):
        for col in range(n_cols):
            y = row * STRIDE
            x = x_start + col * STRIDE
            block = img[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]
            block_mean = np.mean(block)
            block_std = np.std(block)
            # 同 bg_prepare: 丢弃暗带残余/黑边 与 纯色无纹理块
            if block_mean < MIN_BLOCK_MEAN:
                continue
            if block_std < MIN_BLOCK_STD:
                continue
            means.append(float(block_mean))
            stds.append(float(block_std))
    return means, stds


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    _assert_paths()
    os.makedirs(OUTPUT_STATS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_POOL_PATH), exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.png")))
    assert len(img_paths) > 0, f"No PNG terrain images in {INPUT_DIR}"

    print(f"{'='*60}")
    print(f"STEP 1.5b / Part1: AI4 BG JOINT (mean,std) Pool Builder (read-only)")
    print(f"{'='*60}")
    print(f"Source (AI4 terrain, pre-normalize): {INPUT_DIR}")
    print(f"  {len(img_paths)} terrain images")
    print(f"Block: {BLOCK_SIZE}x{BLOCK_SIZE}, stride {STRIDE}, "
          f"min_mean {MIN_BLOCK_MEAN}, min_std {MIN_BLOCK_STD}")
    print(f"NOTE: NO normalization -> recovers natural per-image mean AND std spread")

    all_means = []
    all_stds = []
    per_image_log = []
    for i, p in enumerate(img_paths):
        name = Path(p).stem
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            per_image_log.append(f"  [{i+1}/{len(img_paths)}] {name} - CANNOT READ, skipped")
            continue
        h, w = img.shape
        le, rs, nw = detect_nadir_gap(img)
        if le is None:
            m, s = block_stats_from_strip(img, 0, w)
            per_image_log.append(f"  [{i+1}/{len(img_paths)}] {name} ({w}x{h}) "
                                 f"No nadir - {len(m)} blocks")
        else:
            ml, sl = block_stats_from_strip(img, 0, max(0, le))
            mr, sr = block_stats_from_strip(img, min(w, rs), w)
            m = ml + mr
            s = sl + sr
            per_image_log.append(f"  [{i+1}/{len(img_paths)}] {name} ({w}x{h}) "
                                 f"Nadir w={nw} - L:{len(ml)}+R:{len(mr)}={len(m)} blocks")
        all_means.extend(m)
        all_stds.extend(s)

    pool_means = np.array(all_means, dtype=np.float64)
    pool_stds = np.array(all_stds, dtype=np.float64)
    assert pool_means.size > 0, "No qualifying blocks -> empty pool"
    assert pool_means.size == pool_stds.size, \
        f"mean/std count mismatch: {pool_means.size} vs {pool_stds.size}"

    n_blocks = int(pool_means.size)
    # mean-of-means stats (对 block_means)
    pool_mean = float(np.mean(pool_means))
    pool_std = float(np.std(pool_means))         # = std-of-means (应≈35.6, 与 1.5 一致)
    pool_min = float(pool_means.min())
    pool_max = float(pool_means.max())
    p5 = float(np.percentile(pool_means, 5))
    p50 = float(np.percentile(pool_means, 50))
    p95 = float(np.percentile(pool_means, 95))
    # within-block std stats (对 block_stds, 即每块的对比度)
    std_mean = float(np.mean(pool_stds))
    std_min = float(pool_stds.min())
    std_max = float(pool_stds.max())
    std_p5 = float(np.percentile(pool_stds, 5))
    std_p50 = float(np.percentile(pool_stds, 50))
    std_p95 = float(np.percentile(pool_stds, 95))

    # ---- 落盘 (成对存 means+stds + 两组分位 + crop 参数, 便于 denorm_b 复用与审计) ----
    np.savez(
        OUTPUT_POOL_PATH,
        block_means=pool_means,     # 全部 per-block-mean 样本 (与 1.5 池逐元素一致)
        block_stds=pool_stds,       # 与 block_means 同序成对的 per-block-std (对比度)
        n_blocks=n_blocks,
        pool_mean=pool_mean,
        pool_std=pool_std,
        pool_min=pool_min,
        pool_max=pool_max,
        p5=p5, p50=p50, p95=p95,
        std_mean=std_mean,
        std_min=std_min,
        std_max=std_max,
        std_p5=std_p5, std_p50=std_p50, std_p95=std_p95,
        block_size=BLOCK_SIZE,
        source_dir=INPUT_DIR,
        n_source_images=len(img_paths),
    )

    print(f"\n[Pool statistics — JOINT (mean,std)]")
    print(f"  n_blocks          = {n_blocks}")
    print(f"  mean-of-means     = {pool_mean:.4f}")
    print(f"  std-of-means      = {pool_std:.4f}  <-- natural spread normalize erased (≈35.6)")
    print(f"  mean min/max      = {pool_min:.2f} / {pool_max:.2f}")
    print(f"  mean p5/p50/p95   = {p5:.2f} / {p50:.2f} / {p95:.2f}")
    print(f"  within-block std  = mean {std_mean:.4f}  min {std_min:.2f}  max {std_max:.2f}")
    print(f"  std p5/p50/p95    = {std_p5:.2f} / {std_p50:.2f} / {std_p95:.2f}")
    print(f"\n[Saved] {OUTPUT_POOL_PATH}")

    report_path = os.path.join(OUTPUT_STATS_DIR, "bg_brightness_pool_joint_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Step 1.5b / Part1: AI4 Background JOINT (mean,std) Pool Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")
        f.write("[Provenance]\n")
        f.write(f"  Source (AI4 terrain, PRE-normalize): {INPUT_DIR}\n")
        f.write(f"  Source images: {len(img_paths)}\n")
        f.write(f"  NO normalization applied (recovers natural per-image mean AND std spread).\n")
        f.write(f"  Pool contains ONLY AI4 terrain block (mean,std). NO KLSG/SeabedObjects data.\n\n")
        f.write("[Crop parameters (mirrored verbatim from bg_prepare.py / 1.5 pool)]\n")
        f.write(f"  BLOCK_SIZE={BLOCK_SIZE}, STRIDE={STRIDE}, OVERLAP_RATIO={OVERLAP_RATIO}\n")
        f.write(f"  NADIR_THRESHOLD_RATIO={NADIR_THRESHOLD_RATIO}, SAFETY_MARGIN_RATIO={SAFETY_MARGIN_RATIO}\n")
        f.write(f"  MIN_BLOCK_MEAN={MIN_BLOCK_MEAN}, MIN_BLOCK_STD={MIN_BLOCK_STD}\n\n")
        f.write("[Pool statistics — block MEAN]\n")
        f.write(f"  n_blocks      = {n_blocks}\n")
        f.write(f"  mean-of-means = {pool_mean:.4f}\n")
        f.write(f"  std-of-means  = {pool_std:.4f}\n")
        f.write(f"  min/max       = {pool_min:.4f} / {pool_max:.4f}\n")
        f.write(f"  p5/p50/p95    = {p5:.4f} / {p50:.4f} / {p95:.4f}\n\n")
        f.write("[Pool statistics — block STD (contrast)]\n")
        f.write(f"  mean   = {std_mean:.4f}\n")
        f.write(f"  min/max= {std_min:.4f} / {std_max:.4f}\n")
        f.write(f"  p5/p50/p95 = {std_p5:.4f} / {std_p50:.4f} / {std_p95:.4f}\n\n")
        f.write("[Per-image cropping log]\n")
        for line in per_image_log:
            f.write(line + "\n")
    print(f"[Saved] {report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
