# ============================================================================
# 阶段2：URM参数提取
# ============================================================================
# 功能：从Marine-Pulse URM参考图像中提取Weibull灰度参数和功率谱
# 流程：去噪→中值滤波→GMM三类分割→形态学清理→从原图提取像素→Weibull拟合→功率谱估计
#
# 输入：URM原始JPG图像
#   路径：PATH_TO_MARINE_PULSE_URM_IMAGES\*.jpg
#
# 输出：
#   参数文件：outputs\stage2_params\data\urm_params.npz（所有URM的参数打包）
#   统计报告：outputs\stage2_params\stats\stage2_report.txt
#   抽检图：  outputs\stage2_params\stats\qc_XX.png（随机10张的分割+拟合可视化）
# ============================================================================

import os
import glob
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from sklearn.mixture import GaussianMixture
from scipy.stats import weibull_min
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt


# ============ 路径配置 ============
INPUT_DIR = r"PATH_TO_MARINE_PULSE_URM_IMAGES"
OUTPUT_DATA_DIR = r"PATH_TO_URM_PARAMETER_OUTPUT"
OUTPUT_STATS_DIR = r"PATH_TO_URM_STATS_OUTPUT"

# ============ 分割参数（经阶段性实测验证） ============
DENOISE_H = 35
DENOISE_TEMPLATE = 7
DENOISE_SEARCH = 21
MEDIAN_KERNEL = 11
GMM_N_COMPONENTS = 3
GMM_COVARIANCE = 'full'
GMM_MAX_ITER = 300
GMM_N_INIT = 5
MORPH_KERNEL_SIZE = 15

# ============ 功率谱参数 ============
PSD_PATCH_SIZE_PRIMARY = 32    # 默认patch尺寸
PSD_PATCH_SIZE_FALLBACK = 16   # 退化patch尺寸
PSD_MIN_PATCHES = 5            # 每个区域最少patch数

# ============ 抽检参数 ============
QC_SAMPLE_COUNT = 10


# ---------------------------------------------------------------------------
# 分割相关函数
# ---------------------------------------------------------------------------

def segment_urm(img):
    """
    对单张URM图像执行三区域分割。
    输入：灰度图像（原图）
    返回：标签图（0=阴影, 1=背景, 2=高亮），各簇均值
    """
    # 去噪
    denoised = cv2.fastNlMeansDenoising(
        img, None, h=DENOISE_H,
        templateWindowSize=DENOISE_TEMPLATE,
        searchWindowSize=DENOISE_SEARCH
    )

    # 中值滤波
    filtered = cv2.medianBlur(denoised, MEDIAN_KERNEL)

    # GMM分割
    pixels = filtered.reshape(-1, 1).astype(np.float64)
    gmm = GaussianMixture(
        n_components=GMM_N_COMPONENTS,
        covariance_type=GMM_COVARIANCE,
        max_iter=GMM_MAX_ITER,
        n_init=GMM_N_INIT,
        random_state=42
    )
    gmm.fit(pixels)
    labels = gmm.predict(pixels)

    # 区域重标记：按簇均值排序（低→阴影, 中→背景, 高→高亮）
    means = gmm.means_.flatten()
    sorted_indices = np.argsort(means)
    label_map = np.zeros(GMM_N_COMPONENTS, dtype=int)
    for semantic_label, original_label in enumerate(sorted_indices):
        label_map[original_label] = semantic_label
    semantic_labels = label_map[labels].reshape(img.shape)

    # 形态学后处理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    cleaned = semantic_labels.copy()
    for label_val in range(GMM_N_COMPONENTS):
        mask = (semantic_labels == label_val).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cleaned[mask == 1] = label_val

    sorted_means = means[sorted_indices]  # [阴影均值, 背景均值, 高亮均值]
    return cleaned, sorted_means


# ---------------------------------------------------------------------------
# Weibull拟合函数
# ---------------------------------------------------------------------------

def fit_weibull(pixel_values):
    """
    对一组像素灰度值做三参数Weibull MLE拟合。
    返回：(C, loc, scale) 即 (形状, 位置/min, 尺度)
    如果拟合失败返回 (None, None, None)
    """
    if len(pixel_values) < 50:
        return None, None, None
    try:
        params = weibull_min.fit(pixel_values.astype(np.float64))
        c, loc, scale = params
        # 基本合理性检查
        if c <= 0 or scale <= 0 or not np.isfinite(c) or not np.isfinite(loc) or not np.isfinite(scale):
            return None, None, None
        return c, loc, scale
    except Exception:
        return None, None, None


# ---------------------------------------------------------------------------
# 功率谱估计函数
# ---------------------------------------------------------------------------

def estimate_psd(img, label_map, target_label, patch_size=PSD_PATCH_SIZE_PRIMARY):
    """
    从原图的指定区域中提取多个不重叠正方形patch，估计平均功率谱。
    输入：原图, 标签图, 目标标签(0或2), patch尺寸
    返回：平均功率谱（二维数组），实际patch数量，使用的patch尺寸
           如果失败返回 (None, 0, patch_size)
    """
    mask = (label_map == target_label).astype(np.uint8)
    h, w = img.shape

    # 在掩膜区域内寻找可用的不重叠patch
    patches = []
    occupied = np.zeros_like(mask)  # 标记已被占用的位置

    # 扫描所有可能的patch位置
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            # 检查该patch是否完全在目标区域内
            patch_mask = mask[y:y+patch_size, x:x+patch_size]
            if np.all(patch_mask == 1):
                # 检查未被占用
                if np.all(occupied[y:y+patch_size, x:x+patch_size] == 0):
                    patch_data = img[y:y+patch_size, x:x+patch_size].astype(np.float64)
                    patches.append(patch_data)
                    occupied[y:y+patch_size, x:x+patch_size] = 1

                    if len(patches) >= 50:  # 上限，避免过多计算
                        break
        if len(patches) >= 50:
            break

    if len(patches) < PSD_MIN_PATCHES:
        return None, len(patches), patch_size

    # 计算每个patch的功率谱并取平均
    psds = []
    for patch in patches:
        # 去均值
        patch_centered = patch - np.mean(patch)
        # 2D FFT
        fft_result = np.fft.fft2(patch_centered)
        # 功率谱 = |FFT|^2
        psd = np.abs(fft_result) ** 2
        psds.append(psd)

    avg_psd = np.mean(psds, axis=0)
    return avg_psd, len(patches), patch_size


def estimate_psd_with_fallback(img, label_map, target_label):
    """
    功率谱估计，带退化策略：先尝试32×32，不足5个patch则退化到16×16。
    返回：(psd_array, num_patches, actual_patch_size)
    """
    # 尝试主patch尺寸
    psd, n, ps = estimate_psd(img, label_map, target_label, PSD_PATCH_SIZE_PRIMARY)
    if psd is not None:
        return psd, n, ps

    # 退化到小patch尺寸
    psd, n, ps = estimate_psd(img, label_map, target_label, PSD_PATCH_SIZE_FALLBACK)
    if psd is not None:
        return psd, n, ps

    # 都失败
    return None, 0, PSD_PATCH_SIZE_FALLBACK


# ---------------------------------------------------------------------------
# 可视化函数
# ---------------------------------------------------------------------------

def generate_qc_figure(img, label_map, sorted_means,
                       pixels_highlight, pixels_shadow,
                       wb_highlight, wb_shadow,
                       urm_name):
    """
    生成单张URM的抽检图：原图 | 分割结果 | 高亮区直方图+拟合 | 阴影区直方图+拟合
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 原图
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f'Original: {urm_name}')
    axes[0].axis('off')

    # 分割结果（伪彩色）
    seg_vis = np.zeros_like(img, dtype=np.uint8)
    seg_vis[label_map == 0] = 0
    seg_vis[label_map == 1] = 128
    seg_vis[label_map == 2] = 255
    axes[1].imshow(seg_vis, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(f'Segmentation\nMeans: {sorted_means[0]:.0f}/{sorted_means[1]:.0f}/{sorted_means[2]:.0f}')
    axes[1].axis('off')

    # 高亮区直方图 + Weibull拟合曲线
    if len(pixels_highlight) > 0:
        axes[2].hist(pixels_highlight, bins=80, density=True, alpha=0.6, color='orange', label='Histogram')
        if wb_highlight[0] is not None:
            c, loc, scale = wb_highlight
            x_range = np.linspace(max(0, loc), min(255, loc + scale * 4), 200)
            pdf_vals = weibull_min.pdf(x_range, c, loc=loc, scale=scale)
            axes[2].plot(x_range, pdf_vals, 'r-', linewidth=2,
                        label=f'Weibull\nC={c:.2f}\nloc={loc:.1f}\nscale={scale:.1f}')
        axes[2].set_title('Highlight Region')
        axes[2].set_xlabel('Gray value')
        axes[2].legend(fontsize=7)
    else:
        axes[2].text(0.5, 0.5, 'No highlight pixels', ha='center', va='center')

    # 阴影区直方图 + Weibull拟合曲线
    if len(pixels_shadow) > 0:
        axes[3].hist(pixels_shadow, bins=80, density=True, alpha=0.6, color='steelblue', label='Histogram')
        if wb_shadow[0] is not None:
            c, loc, scale = wb_shadow
            x_range = np.linspace(max(0, loc), min(255, loc + scale * 4), 200)
            pdf_vals = weibull_min.pdf(x_range, c, loc=loc, scale=scale)
            axes[3].plot(x_range, pdf_vals, 'r-', linewidth=2,
                        label=f'Weibull\nC={c:.2f}\nloc={loc:.1f}\nscale={scale:.1f}')
        axes[3].set_title('Shadow Region')
        axes[3].set_xlabel('Gray value')
        axes[3].legend(fontsize=7)
    else:
        axes[3].text(0.5, 0.5, 'No shadow pixels', ha='center', va='center')

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 自检函数
# ---------------------------------------------------------------------------

def run_self_check(all_results):
    """
    对全部URM的提取结果做自动自检。
    """
    checks = []
    all_pass = True

    total = len(all_results)
    checks.append(f"[INFO] Total URM images processed: {total}")

    # 检查1：成功率
    seg_success = sum(1 for r in all_results if r['seg_ok'])
    checks.append(f"[{'PASS' if seg_success == total else 'WARN'}] Segmentation success: {seg_success}/{total}")

    # 检查2：Weibull拟合成功率
    wb_hl_ok = sum(1 for r in all_results if r['wb_highlight'][0] is not None)
    wb_sh_ok = sum(1 for r in all_results if r['wb_shadow'][0] is not None)
    checks.append(f"[{'PASS' if wb_hl_ok == total else 'WARN'}] Weibull highlight fit: {wb_hl_ok}/{total}")
    checks.append(f"[{'PASS' if wb_sh_ok == total else 'WARN'}] Weibull shadow fit: {wb_sh_ok}/{total}")

    # 检查3：功率谱成功率
    psd_hl_ok = sum(1 for r in all_results if r['psd_highlight'] is not None)
    psd_sh_ok = sum(1 for r in all_results if r['psd_shadow'] is not None)
    checks.append(f"[{'PASS' if psd_hl_ok > total * 0.8 else 'WARN'}] PSD highlight estimated: {psd_hl_ok}/{total}")
    checks.append(f"[{'PASS' if psd_sh_ok > total * 0.8 else 'WARN'}] PSD shadow estimated: {psd_sh_ok}/{total}")

    # 检查4：灰度排序正确（高亮均值 > 背景均值 > 阴影均值）
    order_ok = sum(1 for r in all_results if r['seg_ok'] and
                   r['sorted_means'][2] > r['sorted_means'][1] > r['sorted_means'][0])
    checks.append(f"[{'PASS' if order_ok == seg_success else 'WARN'}] Gray level order correct: {order_ok}/{seg_success}")

    # 检查5：Weibull参数范围合理性
    valid_wb = [r for r in all_results if r['wb_highlight'][0] is not None]
    if valid_wb:
        c_vals = [r['wb_highlight'][0] for r in valid_wb]
        loc_vals = [r['wb_highlight'][1] for r in valid_wb]
        scale_vals = [r['wb_highlight'][2] for r in valid_wb]
        checks.append(f"[INFO] Highlight C range: [{min(c_vals):.2f}, {max(c_vals):.2f}]")
        checks.append(f"[INFO] Highlight loc range: [{min(loc_vals):.1f}, {max(loc_vals):.1f}]")
        checks.append(f"[INFO] Highlight scale range: [{min(scale_vals):.1f}, {max(scale_vals):.1f}]")
        all_positive = all(c > 0 and s > 0 for c, s in zip(c_vals, scale_vals))
        checks.append(f"[{'PASS' if all_positive else 'FAIL'}] All C > 0 and scale > 0 (highlight)")
        if not all_positive:
            all_pass = False

    valid_wb_sh = [r for r in all_results if r['wb_shadow'][0] is not None]
    if valid_wb_sh:
        c_vals = [r['wb_shadow'][0] for r in valid_wb_sh]
        loc_vals = [r['wb_shadow'][1] for r in valid_wb_sh]
        scale_vals = [r['wb_shadow'][2] for r in valid_wb_sh]
        checks.append(f"[INFO] Shadow C range: [{min(c_vals):.2f}, {max(c_vals):.2f}]")
        checks.append(f"[INFO] Shadow loc range: [{min(loc_vals):.1f}, {max(loc_vals):.1f}]")
        checks.append(f"[INFO] Shadow scale range: [{min(scale_vals):.1f}, {max(scale_vals):.1f}]")
        all_positive = all(c > 0 and s > 0 for c, s in zip(c_vals, scale_vals))
        checks.append(f"[{'PASS' if all_positive else 'FAIL'}] All C > 0 and scale > 0 (shadow)")
        if not all_positive:
            all_pass = False

    # 检查6：PSD patch数量统计
    hl_patches = [r['psd_highlight_n'] for r in all_results if r['psd_highlight'] is not None]
    sh_patches = [r['psd_shadow_n'] for r in all_results if r['psd_shadow'] is not None]
    if hl_patches:
        checks.append(f"[INFO] Highlight PSD patches: min={min(hl_patches)}, max={max(hl_patches)}, mean={np.mean(hl_patches):.1f}")
    if sh_patches:
        checks.append(f"[INFO] Shadow PSD patches: min={min(sh_patches)}, max={max(sh_patches)}, mean={np.mean(sh_patches):.1f}")

    checks.append("")
    checks.append(f"{'='*40}")
    checks.append(f"SELF-CHECK RESULT: {'ALL PASSED' if all_pass else 'ISSUES DETECTED - REVIEW ABOVE'}")

    return checks


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_STATS_DIR, exist_ok=True)

    # 获取所有URM图像
    all_files = glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
    # 排除可能存在的非数字文件名
    valid_files = []
    for f in all_files:
        stem = Path(f).stem
        try:
            int(stem)
            valid_files.append(f)
        except ValueError:
            # 非纯数字文件名，也尝试包含
            valid_files.append(f)
    img_paths = sorted(valid_files, key=lambda x: Path(x).stem)

    if not img_paths:
        print(f"[ERROR] No JPG files found in {INPUT_DIR}")
        return

    print(f"{'='*60}")
    print(f"STAGE 2: URM Parameter Extraction")
    print(f"{'='*60}")
    print(f"Input:  {INPUT_DIR}")
    print(f"Found {len(img_paths)} URM images")
    print(f"Segmentation: NLMeans(h={DENOISE_H}) + MedianBlur({MEDIAN_KERNEL}) + GMM(k={GMM_N_COMPONENTS})")
    print(f"PSD patches: {PSD_PATCH_SIZE_PRIMARY}x{PSD_PATCH_SIZE_PRIMARY} (fallback {PSD_PATCH_SIZE_FALLBACK}x{PSD_PATCH_SIZE_FALLBACK})")
    print(f"{'='*60}")

    # 随机选择抽检样本
    np.random.seed(42)
    qc_indices = set(np.random.choice(len(img_paths), min(QC_SAMPLE_COUNT, len(img_paths)), replace=False))

    all_results = []

    for i, img_path in enumerate(img_paths):
        filename = Path(img_path).stem
        print(f"[{i+1}/{len(img_paths)}] {filename}.jpg ... ", end="", flush=True)

        # 读取原图
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("CANNOT READ")
            all_results.append({
                'name': filename, 'seg_ok': False,
                'sorted_means': [0, 0, 0],
                'wb_highlight': (None, None, None), 'wb_shadow': (None, None, None),
                'psd_highlight': None, 'psd_shadow': None,
                'psd_highlight_n': 0, 'psd_shadow_n': 0,
                'psd_highlight_size': 0, 'psd_shadow_size': 0,
                'n_highlight_px': 0, 'n_shadow_px': 0
            })
            continue

        # ---- Phase 1: 分割 ----
        label_map, sorted_means = segment_urm(img)

        # ---- Phase 2: 从原图提取像素（不是去噪图） ----
        pixels_highlight = img[label_map == 2].flatten()
        pixels_shadow = img[label_map == 0].flatten()

        # ---- Phase 3: Weibull拟合 ----
        wb_hl = fit_weibull(pixels_highlight)
        wb_sh = fit_weibull(pixels_shadow)

        # ---- Phase 4: 功率谱估计 ----
        psd_hl, psd_hl_n, psd_hl_size = estimate_psd_with_fallback(img, label_map, 2)
        psd_sh, psd_sh_n, psd_sh_size = estimate_psd_with_fallback(img, label_map, 0)

        # 记录结果
        result = {
            'name': filename,
            'seg_ok': True,
            'sorted_means': sorted_means,
            'wb_highlight': wb_hl,
            'wb_shadow': wb_sh,
            'psd_highlight': psd_hl,
            'psd_shadow': psd_sh,
            'psd_highlight_n': psd_hl_n,
            'psd_shadow_n': psd_sh_n,
            'psd_highlight_size': psd_hl_size,
            'psd_shadow_size': psd_sh_size,
            'n_highlight_px': len(pixels_highlight),
            'n_shadow_px': len(pixels_shadow)
        }
        all_results.append(result)

        # 状态输出
        wb_hl_str = f"C={wb_hl[0]:.2f}" if wb_hl[0] is not None else "FAIL"
        wb_sh_str = f"C={wb_sh[0]:.2f}" if wb_sh[0] is not None else "FAIL"
        psd_hl_str = f"{psd_hl_n}x{psd_hl_size}" if psd_hl is not None else "FAIL"
        psd_sh_str = f"{psd_sh_n}x{psd_sh_size}" if psd_sh is not None else "FAIL"
        print(f"WB_hl:{wb_hl_str} WB_sh:{wb_sh_str} PSD_hl:{psd_hl_str} PSD_sh:{psd_sh_str}")

        # ---- 抽检图生成 ----
        if i in qc_indices:
            fig = generate_qc_figure(
                img, label_map, sorted_means,
                pixels_highlight, pixels_shadow,
                wb_hl, wb_sh, filename
            )
            qc_path = os.path.join(OUTPUT_STATS_DIR, f"qc_{filename}.png")
            fig.savefig(qc_path, dpi=120, bbox_inches='tight')
            plt.close(fig)

    # ---- Phase 5: 保存参数文件 ----
    print(f"\n[Saving] Parameter file...")

    # 构建可序列化的参数字典
    save_dict = {
        'urm_names': np.array([r['name'] for r in all_results]),
        'seg_ok': np.array([r['seg_ok'] for r in all_results]),
        'sorted_means': np.array([r['sorted_means'] for r in all_results]),
        'n_highlight_px': np.array([r['n_highlight_px'] for r in all_results]),
        'n_shadow_px': np.array([r['n_shadow_px'] for r in all_results]),
    }

    # Weibull参数：每张URM存3个值（C, loc, scale），拟合失败存NaN
    wb_hl_array = np.array([
        [r['wb_highlight'][0] or np.nan, r['wb_highlight'][1] or np.nan, r['wb_highlight'][2] or np.nan]
        for r in all_results
    ])
    wb_sh_array = np.array([
        [r['wb_shadow'][0] or np.nan, r['wb_shadow'][1] or np.nan, r['wb_shadow'][2] or np.nan]
        for r in all_results
    ])
    save_dict['wb_highlight'] = wb_hl_array  # shape: (N, 3) -> [C, loc, scale]
    save_dict['wb_shadow'] = wb_sh_array

    # 功率谱：每张URM存一个二维数组，尺寸可能不同（32x32或16x16），用列表存
    # 成功的存实际数组，失败的存全零占位
    psd_hl_list = []
    psd_sh_list = []
    psd_hl_sizes = []
    psd_sh_sizes = []
    for r in all_results:
        if r['psd_highlight'] is not None:
            psd_hl_list.append(r['psd_highlight'])
            psd_hl_sizes.append(r['psd_highlight_size'])
        else:
            psd_hl_list.append(np.zeros((1, 1)))  # 占位
            psd_hl_sizes.append(0)
        if r['psd_shadow'] is not None:
            psd_sh_list.append(r['psd_shadow'])
            psd_sh_sizes.append(r['psd_shadow_size'])
        else:
            psd_sh_list.append(np.zeros((1, 1)))
            psd_sh_sizes.append(0)

    save_dict['psd_highlight_sizes'] = np.array(psd_hl_sizes)
    save_dict['psd_shadow_sizes'] = np.array(psd_sh_sizes)

    # 功率谱用单独的key逐个存（因为尺寸可能不一致）
    for idx, (phl, psh) in enumerate(zip(psd_hl_list, psd_sh_list)):
        save_dict[f'psd_hl_{idx}'] = phl
        save_dict[f'psd_sh_{idx}'] = psh

    param_path = os.path.join(OUTPUT_DATA_DIR, "urm_params.npz")
    np.savez_compressed(param_path, **save_dict)
    print(f"  Saved: {param_path}")

    # ---- Phase 6: 自检 ----
    print(f"\n[Self-Check]")
    check_results = run_self_check(all_results)
    for line in check_results:
        print(f"  {line}")

    # ---- Phase 7: 保存统计报告 ----
    print(f"\n[Saving] Statistics report...")
    report_path = os.path.join(OUTPUT_STATS_DIR, "stage2_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Stage 2: URM Parameter Extraction Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")

        f.write(f"[Parameters]\n")
        f.write(f"  Denoise h: {DENOISE_H}\n")
        f.write(f"  Median kernel: {MEDIAN_KERNEL}\n")
        f.write(f"  GMM components: {GMM_N_COMPONENTS}, covariance: {GMM_COVARIANCE}\n")
        f.write(f"  Morphology kernel: {MORPH_KERNEL_SIZE}\n")
        f.write(f"  PSD patch size: {PSD_PATCH_SIZE_PRIMARY} (fallback: {PSD_PATCH_SIZE_FALLBACK})\n")
        f.write(f"  PSD min patches: {PSD_MIN_PATCHES}\n\n")

        f.write(f"[Input]\n")
        f.write(f"  URM images: {len(img_paths)}\n\n")

        f.write(f"[Per-image Results]\n")
        f.write(f"  {'Name':<12} {'Seg':>3} {'HL_px':>8} {'SH_px':>8} "
                f"{'WB_HL_C':>8} {'WB_HL_loc':>9} {'WB_HL_sc':>9} "
                f"{'WB_SH_C':>8} {'WB_SH_loc':>9} {'WB_SH_sc':>9} "
                f"{'PSD_HL':>7} {'PSD_SH':>7}\n")
        f.write(f"  {'-'*110}\n")
        for r in all_results:
            wh = r['wb_highlight']
            ws = r['wb_shadow']
            f.write(f"  {r['name']:<12} {'OK' if r['seg_ok'] else 'NO':>3} "
                    f"{r['n_highlight_px']:>8} {r['n_shadow_px']:>8} "
                    f"{wh[0]:>8.2f} {wh[1]:>9.1f} {wh[2]:>9.1f} " if wh[0] is not None else
                    f"  {r['name']:<12} {'OK' if r['seg_ok'] else 'NO':>3} "
                    f"{r['n_highlight_px']:>8} {r['n_shadow_px']:>8} "
                    f"{'FAIL':>8} {'':>9} {'':>9} ")
            if ws[0] is not None:
                f.write(f"{ws[0]:>8.2f} {ws[1]:>9.1f} {ws[2]:>9.1f} ")
            else:
                f.write(f"{'FAIL':>8} {'':>9} {'':>9} ")
            psd_hl_str = f"{r['psd_highlight_n']}x{r['psd_highlight_size']}" if r['psd_highlight'] is not None else "FAIL"
            psd_sh_str = f"{r['psd_shadow_n']}x{r['psd_shadow_size']}" if r['psd_shadow'] is not None else "FAIL"
            f.write(f"{psd_hl_str:>7} {psd_sh_str:>7}\n")

        f.write(f"\n[Self-Check]\n")
        for line in check_results:
            f.write(f"  {line}\n")

    print(f"  Saved: {report_path}")

    # ---- 完成 ----
    print(f"\n{'='*60}")
    print(f"STAGE 2 COMPLETE")
    print(f"  Params: {param_path}")
    print(f"  Report: {report_path}")
    print(f"  QC images: {OUTPUT_STATS_DIR}/qc_*.png ({len(qc_indices)} files)")
    print(f"{'='*60}")
    print(f"\nPlease visually inspect:")
    print(f"  1. qc_*.png - Is segmentation reasonable (3 clear regions)?")
    print(f"  2. qc_*.png - Does Weibull curve roughly fit the histogram?")
    print(f"  3. stage2_report.txt - Are all self-checks PASS?")


if __name__ == "__main__":
    main()
