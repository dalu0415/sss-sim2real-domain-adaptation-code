# ============================================================================
# 阶段1：背景池准备
# ============================================================================
# 功能：从AI4shipwreck terrain地形图中裁剪700×700背景块并做灰度归一化
# 流程：暗带自动检测 → 可用区域裁剪 → 质量检查 → 灰度归一化 → 自检
#
# 输入：terrain原始PNG图像
#   路径：PATH_TO_AI4_TERRAIN_IMAGES\*.png
#
# 输出：
#   背景块图片：outputs\stage1_bg\blocks\bg_0001.png ~ bg_XXXX.png
#   统计报告：  outputs\stage1_bg\stats\stage1_report.txt
#   抽检图：    outputs\stage1_bg\stats\qc_samples.png
# ============================================================================

import os
import glob
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime


# ============ 路径配置 ============
INPUT_DIR = r"PATH_TO_AI4_TERRAIN_IMAGES"
OUTPUT_BLOCKS_DIR = r"PATH_TO_BACKGROUND_BLOCKS_OUTPUT"
OUTPUT_STATS_DIR = r"PATH_TO_BACKGROUND_STATS_OUTPUT"

# ============ 裁剪参数 ============
BLOCK_SIZE = 700              # 裁剪块尺寸（像素）
OVERLAP_RATIO = 0.2           # 相邻块重叠比例
STRIDE = int(BLOCK_SIZE * (1 - OVERLAP_RATIO))  # 裁剪步长 = 700 * 0.8 = 560

# ============ 暗带检测参数 ============
NADIR_THRESHOLD_RATIO = 0.4   # 列均值 < 全局均值 * 此比例 → 判定为暗带
SAFETY_MARGIN_RATIO = 0.5     # 暗带两侧安全边距 = 暗带宽度 * 此比例

# ============ 质量检查参数 ============
MIN_BLOCK_MEAN = 20           # 块均值低于此值 → 丢弃（暗带残余/黑边）
MIN_BLOCK_STD = 5             # 块标准差低于此值 → 丢弃（纯色无纹理）

# ============ 自检抽样参数 ============
QC_SAMPLE_COUNT = 12          # 抽检展示的背景块数量


# ---------------------------------------------------------------------------
# 核心函数
# ---------------------------------------------------------------------------

def detect_nadir_gap(img):
    """
    自动检测SSS水瀑图的nadir暗带位置。
    方法：逐列计算灰度均值，低于全局均值*阈值的连续区间即为暗带。
    返回：(可用左条带右边界, 可用右条带左边界, 暗带宽度)
          如果未检测到暗带，返回 (None, None, None)
    """
    col_means = np.mean(img, axis=0)
    global_mean = np.mean(col_means)
    threshold = global_mean * NADIR_THRESHOLD_RATIO

    dark_cols = col_means < threshold

    if not np.any(dark_cols):
        return None, None, None

    # 找所有连续暗带区间
    changes = np.diff(dark_cols.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1

    if dark_cols[0]:
        starts = np.insert(starts, 0, 0)
    if dark_cols[-1]:
        ends = np.append(ends, len(dark_cols))

    if len(starts) == 0 or len(ends) == 0:
        return None, None, None

    # 取最宽的暗带
    widths = ends - starts
    max_idx = np.argmax(widths)
    nadir_left = starts[max_idx]
    nadir_right = ends[max_idx]
    nadir_width = nadir_right - nadir_left

    # 安全边距
    margin = int(nadir_width * SAFETY_MARGIN_RATIO)
    usable_left_end = nadir_left - margin
    usable_right_start = nadir_right + margin

    return usable_left_end, usable_right_start, nadir_width


def crop_blocks_from_strip(img, x_start, x_end):
    """
    从图像的一个垂直条带中按网格裁剪700×700块。
    返回：通过质量检查的块列表，每个元素为numpy数组。
    """
    h = img.shape[0]
    strip_width = x_end - x_start
    blocks = []

    if strip_width < BLOCK_SIZE or h < BLOCK_SIZE:
        return blocks

    n_cols = (strip_width - BLOCK_SIZE) // STRIDE + 1
    n_rows = (h - BLOCK_SIZE) // STRIDE + 1

    for row in range(n_rows):
        for col in range(n_cols):
            y = row * STRIDE
            x = x_start + col * STRIDE
            block = img[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]

            # 质量检查
            block_mean = np.mean(block)
            block_std = np.std(block)

            if block_mean < MIN_BLOCK_MEAN:
                continue
            if block_std < MIN_BLOCK_STD:
                continue

            blocks.append(block)

    return blocks


def normalize_blocks(blocks):
    """
    对所有块做灰度归一化：统一到全局平均均值和平均标准差。
    保留每块内部的纹理空间结构，只调整整体亮度和对比度。
    返回：归一化后的块列表 + (target_mean, target_std)
    """
    # 统计每块的均值和标准差
    means = np.array([np.mean(b) for b in blocks])
    stds = np.array([np.std(b) for b in blocks])

    target_mean = np.mean(means)
    target_std = np.mean(stds)

    normalized = []
    for b, bm, bs in zip(blocks, means, stds):
        if bs < 1e-6:
            norm = np.full_like(b, int(target_mean), dtype=np.uint8)
        else:
            norm = (b.astype(np.float64) - bm) / bs * target_std + target_mean
            norm = np.clip(norm, 0, 255).astype(np.uint8)
        normalized.append(norm)

    return normalized, target_mean, target_std, means, stds


def generate_qc_image(blocks, count=12):
    """
    生成抽检拼图：随机挑选若干块拼成网格，用于目视检查。
    """
    n = min(count, len(blocks))
    indices = np.random.choice(len(blocks), n, replace=False)
    indices.sort()

    # 排成 3行4列（或自适应）
    cols = 4
    rows = (n + cols - 1) // cols
    gap = 4  # 块之间的间隔

    canvas_h = rows * BLOCK_SIZE + (rows - 1) * gap
    canvas_w = cols * BLOCK_SIZE + (cols - 1) * gap
    canvas = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 200  # 灰色底

    for k, idx in enumerate(indices):
        r = k // cols
        c = k % cols
        y = r * (BLOCK_SIZE + gap)
        x = c * (BLOCK_SIZE + gap)
        canvas[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE] = blocks[idx]

        # 在左上角标注编号
        label = f"bg_{idx + 1:04d}"
        cv2.putText(canvas, label, (x + 8, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,), 2)

    return canvas


def run_self_check(blocks, means_before, stds_before, target_mean, target_std):
    """
    自动自检，逐条验证质量标准，返回检查报告文本列表。
    """
    checks = []
    all_pass = True

    # 检查1：总量是否 >= 419（之前验证过的数量）
    passed = len(blocks) >= 400
    checks.append(f"[{'PASS' if passed else 'WARN'}] Total blocks: {len(blocks)} (expected >= 400)")
    if not passed:
        all_pass = False

    # 检查2：所有块尺寸均为700×700
    size_ok = all(b.shape == (BLOCK_SIZE, BLOCK_SIZE) for b in blocks)
    checks.append(f"[{'PASS' if size_ok else 'FAIL'}] All blocks are {BLOCK_SIZE}x{BLOCK_SIZE}")
    if not size_ok:
        all_pass = False

    # 检查3：归一化后均值一致性
    norm_means = np.array([np.mean(b) for b in blocks])
    norm_stds = np.array([np.std(b) for b in blocks])
    mean_spread = np.std(norm_means)
    checks.append(f"[INFO] Normalized mean range: {norm_means.min():.1f} - {norm_means.max():.1f} (std of means: {mean_spread:.1f})")

    # 检查4：归一化前后的改善
    before_spread = np.std(means_before)
    checks.append(f"[INFO] Before normalization: mean spread = {before_spread:.1f}")
    checks.append(f"[INFO] After normalization:  mean spread = {mean_spread:.1f}")
    improved = mean_spread < before_spread
    checks.append(f"[{'PASS' if improved else 'WARN'}] Normalization reduced mean spread: {improved}")

    # 检查5：无全黑或全白块
    min_mean = norm_means.min()
    max_mean = norm_means.max()
    no_extreme = min_mean > 5 and max_mean < 250
    checks.append(f"[{'PASS' if no_extreme else 'WARN'}] No extreme blocks: min_mean={min_mean:.1f}, max_mean={max_mean:.1f}")

    # 检查6：像素值范围
    pixel_min = min(b.min() for b in blocks)
    pixel_max = max(b.max() for b in blocks)
    checks.append(f"[{'PASS' if pixel_min >= 0 and pixel_max <= 255 else 'FAIL'}] Pixel range: [{pixel_min}, {pixel_max}]")

    # 总结
    checks.append("")
    checks.append(f"{'='*40}")
    checks.append(f"SELF-CHECK RESULT: {'ALL PASSED' if all_pass else 'ISSUES DETECTED - REVIEW ABOVE'}")

    return checks


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_BLOCKS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_STATS_DIR, exist_ok=True)

    # 获取所有terrain图像
    img_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.png")))
    if not img_paths:
        print(f"[ERROR] No PNG files found in {INPUT_DIR}")
        return

    print(f"{'='*60}")
    print(f"STAGE 1: Background Pool Preparation")
    print(f"{'='*60}")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_BLOCKS_DIR}")
    print(f"Found {len(img_paths)} terrain images")
    print(f"Block: {BLOCK_SIZE}x{BLOCK_SIZE}, stride: {STRIDE}, overlap: {OVERLAP_RATIO}")
    print(f"{'='*60}")

    # ---- Phase 1: 暗带检测 + 裁剪 ----
    print("\n[Phase 1] Nadir detection and block cropping...")
    all_blocks = []
    crop_log = []

    for i, img_path in enumerate(img_paths):
        filename = Path(img_path).stem
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [{i+1}/{len(img_paths)}] {filename} - CANNOT READ, skipped")
            continue

        h, w = img.shape

        # 检测暗带
        usable_left_end, usable_right_start, nadir_width = detect_nadir_gap(img)

        if usable_left_end is None:
            # 无暗带，整图可用
            blocks = crop_blocks_from_strip(img, 0, w)
            log_entry = f"  [{i+1}/{len(img_paths)}] {filename} ({w}x{h}) - No nadir - {len(blocks)} blocks"
        else:
            blocks_left = crop_blocks_from_strip(img, 0, max(0, usable_left_end))
            blocks_right = crop_blocks_from_strip(img, min(w, usable_right_start), w)
            blocks = blocks_left + blocks_right
            log_entry = (f"  [{i+1}/{len(img_paths)}] {filename} ({w}x{h}) - "
                        f"Nadir width={nadir_width}, margin={int(nadir_width * SAFETY_MARGIN_RATIO)} - "
                        f"L:{len(blocks_left)} + R:{len(blocks_right)} = {len(blocks)} blocks")

        print(log_entry)
        crop_log.append(log_entry)
        all_blocks.extend(blocks)

    print(f"\n  Total raw blocks (pre-normalization): {len(all_blocks)}")

    if len(all_blocks) == 0:
        print("[ERROR] No blocks extracted. Check terrain images and parameters.")
        return

    # ---- Phase 2: 灰度归一化 ----
    print("\n[Phase 2] Grayscale normalization...")
    normalized_blocks, target_mean, target_std, means_before, stds_before = normalize_blocks(all_blocks)

    print(f"  Target mean: {target_mean:.1f}")
    print(f"  Target std:  {target_std:.1f}")
    print(f"  Before: mean range [{means_before.min():.1f}, {means_before.max():.1f}]")

    norm_means = np.array([np.mean(b) for b in normalized_blocks])
    print(f"  After:  mean range [{norm_means.min():.1f}, {norm_means.max():.1f}]")

    # ---- Phase 3: 保存 ----
    print(f"\n[Phase 3] Saving {len(normalized_blocks)} blocks...")
    for idx, block in enumerate(normalized_blocks):
        output_path = os.path.join(OUTPUT_BLOCKS_DIR, f"bg_{idx+1:04d}.png")
        cv2.imwrite(output_path, block)

    # ---- Phase 4: 自检 ----
    print("\n[Phase 4] Self-check...")
    check_results = run_self_check(
        normalized_blocks, means_before, stds_before, target_mean, target_std
    )
    for line in check_results:
        print(f"  {line}")

    # ---- Phase 5: 生成抽检图 ----
    print("\n[Phase 5] Generating QC sample image...")
    qc_img = generate_qc_image(normalized_blocks, QC_SAMPLE_COUNT)
    qc_path = os.path.join(OUTPUT_STATS_DIR, "qc_samples.png")
    cv2.imwrite(qc_path, qc_img)
    print(f"  Saved: {qc_path}")

    # ---- Phase 6: 保存统计报告 ----
    print("\n[Phase 6] Saving statistics report...")
    report_path = os.path.join(OUTPUT_STATS_DIR, "stage1_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Stage 1: Background Pool Preparation Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")

        f.write(f"[Parameters]\n")
        f.write(f"  Block size: {BLOCK_SIZE}x{BLOCK_SIZE}\n")
        f.write(f"  Stride: {STRIDE} (overlap: {OVERLAP_RATIO})\n")
        f.write(f"  Nadir threshold: global_mean * {NADIR_THRESHOLD_RATIO}\n")
        f.write(f"  Safety margin: nadir_width * {SAFETY_MARGIN_RATIO}\n")
        f.write(f"  Min block mean: {MIN_BLOCK_MEAN}\n")
        f.write(f"  Min block std: {MIN_BLOCK_STD}\n\n")

        f.write(f"[Input]\n")
        f.write(f"  Terrain images: {len(img_paths)}\n\n")

        f.write(f"[Cropping Log]\n")
        for entry in crop_log:
            f.write(f"{entry}\n")
        f.write(f"\n")

        f.write(f"[Normalization]\n")
        f.write(f"  Target mean: {target_mean:.2f}\n")
        f.write(f"  Target std:  {target_std:.2f}\n")
        f.write(f"  Before normalization - mean range: [{means_before.min():.1f}, {means_before.max():.1f}]\n")
        f.write(f"  After normalization  - mean range: [{norm_means.min():.1f}, {norm_means.max():.1f}]\n\n")

        f.write(f"[Output]\n")
        f.write(f"  Total blocks: {len(normalized_blocks)}\n")
        f.write(f"  Block files: bg_0001.png ~ bg_{len(normalized_blocks):04d}.png\n")
        f.write(f"  Blocks dir: {OUTPUT_BLOCKS_DIR}\n\n")

        f.write(f"[Self-Check]\n")
        for line in check_results:
            f.write(f"  {line}\n")

    print(f"  Saved: {report_path}")

    # ---- 完成 ----
    print(f"\n{'='*60}")
    print(f"STAGE 1 COMPLETE")
    print(f"  Blocks: {len(normalized_blocks)}")
    print(f"  Output: {OUTPUT_BLOCKS_DIR}")
    print(f"  Report: {report_path}")
    print(f"  QC img: {qc_path}")
    print(f"{'='*60}")
    print(f"\nPlease visually inspect:")
    print(f"  1. qc_samples.png - Are blocks free of nadir artifacts?")
    print(f"  2. qc_samples.png - Is brightness roughly uniform across blocks?")
    print(f"  3. stage1_report.txt - Are all self-checks PASS?")


if __name__ == "__main__":
    main()
