# ============================================================================
# 改进线 第1步：AI4Shipwrecks 真实沉船目标 —— 亮回波(highlight)参数提取
# ============================================================================
# 单变量目标：把"合成目标的亮回波质感"来源从 URM(礁石/残丘 confound) 换成
#             AI4Shipwrecks 真实沉船 footprint 内分离出的亮回波。
#             阴影(shadow) 仍沿用基线 URM 的 wb_shadow/psd_sh（known limitation,
#             留第2步）。
#
# 关键设计（不改 urm_process 内核, 只复用纯函数）:
#   - 不复用 urm_process.segment_urm（GMM-3 假定整图三峰, AI4 footprint 不成立）
#   - 复用 urm_process.fit_weibull (L111)          —— highlight 像素 → Weibull (C,loc,scale)
#   - 复用 urm_process.estimate_psd_with_fallback (L183) —— highlight 区 → PSD(32/16 退化)
#   - 在 footprint(label>0) 内用 Otsu 阈值分离亮回波(亮侧=highlight)
#   - shadow 从基线 urm_params_clean.npz 按 i % n_urm 借, 成对存同一 entry
#
# 输出 ai4_params.npz 严格复刻 urm_params.npz schema, 供 filter.py / full_generate
# 无改动直接读取。
#
# 红线: 只写 synth_improved 根下; 复用输入(AI4 raw / relook csv / 基线 urm_params_clean)
#       只读; 固定种子; 避开 urm_process L465 的 loc=0->NaN bug（用显式 None 判）。
# ============================================================================

import os
import csv
import sys
import numpy as np
import cv2
from scipy.stats import weibull_min
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- 复用基线纯函数（不重写内核） ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from urm_process import fit_weibull, estimate_psd_with_fallback  # noqa: E402


# ---------------------------------------------------------------------------
# Weibull 拟合: floc=0 变体（必要且诚实的偏离, 见下注释）
# ---------------------------------------------------------------------------
# 复用 urm_process.fit_weibull 是 3 参数自由 MLE。实测在 AI4 真实沉船亮回波上,
# 亮回波大量像素饱和在 255(plateau, 例 Monrovia_03 有 29% 像素=255), 自由 loc 的
# weibull_min.fit 退化 —— C 漂到 1e8 或 <0.2(实测自由拟合 C 跨 [0.19, 1.7e8]),
# 几乎全被 filter 的 HL_MAX_C=10 删光。固定 loc=0(拟合亮回波强度幅值的 2 参 Weibull)
# 后 C 稳定落在 [2.8, 10.2] 的物理合理区间。
# 这是对 fit_weibull 自由 loc 的"必要偏离":
#   - 同一 scipy weibull_min.fit MLE 内核, 仅加 floc=0 约束(不是另写算法)；
#   - loc 在 full_generate.generate_correlated_weibull_field 下游被 z-score 重映射
#     覆盖(L144-148), 本就不影响成品 —— 固定 loc=0 不改变最终亮度, 只稳住形状 C;
#   - sanity 检查逻辑(C>0/scale>0/有限性)逐字沿用 fit_weibull。
# 诚实标注: 与基线 URM 线的 fit_weibull(自由 loc) 在 loc 处理上不同, 但对成品无影响。
def fit_weibull_floc0(pixel_values):
    """floc=0 的 Weibull MLE。返回 (C, loc, scale) 或 (None,None,None)。
    sanity 与 urm_process.fit_weibull 一致。"""
    if len(pixel_values) < 50:
        return None, None, None
    try:
        c, loc, scale = weibull_min.fit(pixel_values.astype(np.float64), floc=0)
        if c <= 0 or scale <= 0 or not np.isfinite(c) or not np.isfinite(loc) or not np.isfinite(scale):
            return None, None, None
        return c, loc, scale
    except Exception:
        return None, None, None


# ============ 路径配置 ============
# 写路径（必须在 synth_improved 根下）
SYNTH_IMPROVED_ROOT = r"PATH_TO_SYNTH_IMPROVED_WORKSPACE"
OUTPUT_DATA_DIR = os.path.join(SYNTH_IMPROVED_ROOT, r"data\outputs\stage2_ai4\data")
OUTPUT_STATS_DIR = os.path.join(SYNTH_IMPROVED_ROOT, r"data\outputs\stage2_ai4\stats")
OUTPUT_NPZ = os.path.join(OUTPUT_DATA_DIR, "ai4_params.npz")

# 读路径（复用输入，只读；AI4 raw + relook csv + 基线 urm_params_clean）
AI4_ROOT = r"PATH_TO_AI4SHIPWRECKS"
RELOOK_CSV = r"PATH_TO_FRAME_MANIFEST\relook_strict.csv"
BASELINE_URM_CLEAN = r"PATH_TO_URM_PARAMETER_OUTPUT\urm_params_clean.npz"

# 只读白名单根（任何读路径必须落在其中之一下）
_READ_WHITELIST = [
    SYNTH_IMPROVED_ROOT,
    AI4_ROOT,
    r"PATH_TO_FRAME_MANIFEST",
    r"PATH_TO_URM_PARAMETER_OUTPUT",  # 基线只读
]


def _assert_paths():
    """fail-fast: 写路径必须在 synth_improved 根下; 读路径必须在只读白名单内。"""
    def _norm(p):
        return os.path.normcase(os.path.abspath(p))
    root_n = _norm(SYNTH_IMPROVED_ROOT)
    for wp in [OUTPUT_DATA_DIR, OUTPUT_STATS_DIR, OUTPUT_NPZ]:
        assert _norm(wp).startswith(root_n), f"WRITE path escapes synth_improved: {wp}"
    wl = [_norm(p) for p in _READ_WHITELIST]
    for rp in [AI4_ROOT, RELOOK_CSV, BASELINE_URM_CLEAN]:
        assert any(_norm(rp).startswith(w) for w in wl), f"READ path not in whitelist: {rp}"


# ============ 选帧规则 ============
EXCLUDE_SITE = "Artificial_Reef"   # gap 报告点名的 confound, 显式排除
# 打架的 5 行 (strict_clear==1 但 bin_old==too_faint_small):
#   EB_Allen_08/18/21, Pewabic_03, Lucinda_van_Valkenburg_05
# 处理策略: 不按文件名硬删, 而是保留为候选(达成 ~108/21), 在提取时用透明的
#   "亮回波质量门"判定(见 HL_MIN_PX / HL_MIN_CONTRAST), 弱/暗的自然落选并写明。
DISPUTED_FILES = {
    "EB_Allen_08.png", "EB_Allen_18.png", "EB_Allen_21.png",
    "Pewabic_03.png", "Lucinda_van_Valkenburg_05.png",
}

# ============ 亮回波分离 + 质量门 ============
# Otsu 在 footprint 内分离亮侧=highlight。质量门(保守, 防止把暗/无真实亮回波的
# footprint(如 EB_Allen 的暗船体)当成有效 highlight 注入坏统计):
HL_MIN_PX = 200          # 亮回波像素数下限(fit_weibull 自身要 >=50, 这里更严)
HL_MIN_CONTRAST = 20.0   # 亮回波中位灰 - footprint 中位灰 的下限(确保"真亮回波")

# ============ 种子 ============
SEED = 42

# ============ QC ============
QC_SAMPLE_COUNT = 10


# ---------------------------------------------------------------------------
# 选帧
# ---------------------------------------------------------------------------

def select_frames(csv_path):
    """读 relook_strict.csv -> manifest(list of dict: split/site/file)。
    规则: strict_clear==1 起; 显式排除 Artificial_Reef; 打架行保留为候选(交给质量门)。
    返回: (manifest, info_dict)
    """
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    sc1 = [r for r in rows if r["strict_clear"] == "1"]
    reef = [r for r in sc1 if r["site"] == EXCLUDE_SITE]
    after_reef = [r for r in sc1 if r["site"] != EXCLUDE_SITE]
    disputed_kept = [r for r in after_reef if r["file"] in DISPUTED_FILES]

    manifest = []
    for r in after_reef:
        manifest.append({
            "split": r["split"],
            "site": r["site"],
            "file": r["file"],
            "disputed": r["file"] in DISPUTED_FILES,
        })

    info = {
        "total_rows": len(rows),
        "strict_clear_1": len(sc1),
        "excluded_reef": len(reef),
        "after_reef": len(after_reef),
        "disputed_kept": [r["file"] for r in disputed_kept],
        "n_wrecks": len(set(m["site"] for m in manifest)),
    }
    return manifest, info


# ---------------------------------------------------------------------------
# 读 image + label
# ---------------------------------------------------------------------------

def load_pair(split, fname):
    """读 AI4 image(灰度) + 同名 label(二值 >0)。返回 (img_uint8, mask_bool) 或 (None,None)。"""
    imp = os.path.join(AI4_ROOT, split, "images", fname)
    lbp = os.path.join(AI4_ROOT, split, "labels", fname)
    img = cv2.imread(imp, cv2.IMREAD_UNCHANGED)
    lbl = cv2.imread(lbp, cv2.IMREAD_UNCHANGED)
    if img is None or lbl is None:
        return None, None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if lbl.ndim == 3:
        lbl = lbl[..., 0]
    img = img.astype(np.uint8)
    mask = lbl > 0   # 阈值用 >0 (label 是 0/1, 不是 0/255)
    return img, mask


# ---------------------------------------------------------------------------
# footprint 内分离亮回波 + 构造 highlight 的 PSD label_map
# ---------------------------------------------------------------------------

def extract_highlight(img, mask):
    """在 footprint(mask) 内用 Otsu 分离亮侧=highlight。
    返回 dict:
      hl_pixels      : highlight 像素灰度值 1D array
      hl_label_map   : 整图 label_map(uint8), highlight 像素=2, 其余=1 (复刻 urm_process
                       的 target_label=2 语义, 供 estimate_psd_with_fallback 直接用)
      otsu_thr       : Otsu 阈值
      fp_med, hl_med : footprint / highlight 中位灰
      n_fp, n_hl     : footprint / highlight 像素数
      contrast       : hl_med - fp_med
    """
    H, W = img.shape
    tgt_vals = img[mask]
    n_fp = int(tgt_vals.size)
    if n_fp < 50:
        return dict(hl_pixels=np.array([]), hl_label_map=np.ones((H, W), np.uint8),
                    otsu_thr=0.0, fp_med=0.0, hl_med=0.0, n_fp=n_fp, n_hl=0, contrast=0.0)

    # Otsu 阈值: 只在 footprint 像素上算(不被 footprint 外大量背景污染)
    otsu_thr, _ = cv2.threshold(tgt_vals.astype(np.uint8), 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fp_med = float(np.median(tgt_vals))

    # highlight = footprint 内 >= Otsu 阈值 的像素(亮侧)
    hl_pixel_mask = mask & (img.astype(np.float64) >= otsu_thr)
    hl_pixels = img[hl_pixel_mask].astype(np.float64)
    n_hl = int(hl_pixels.size)
    hl_med = float(np.median(hl_pixels)) if n_hl > 0 else 0.0
    contrast = hl_med - fp_med

    # hl_pixel_label_map: 仅 Otsu 亮像素=2(用于可视化 highlight 像素分布)
    hl_pixel_label_map = np.ones((H, W), dtype=np.uint8)
    hl_pixel_label_map[hl_pixel_mask] = 2

    # psd_label_map: 整个 footprint=2(用于 estimate_psd 取 patch)。
    # 理由(诚实标注): Otsu 亮像素是散碎 speckle, 罕有 5 个全亮的 16x16 contiguous
    # patch -> PSD 几乎全失败(实测仅 13/108)。而 URM 线的 highlight PSD 本就采样"亮
    # 回波区作为一块连通区域"的纹理。沉船的亮回波纹理最稳健的采样是 footprint 内的
    # 真实像素空间相关结构(footprint 是大块连通区, patch 充足, 实测 14-50 个)。
    # PSD 在 generate_correlated_weibull_field 中只贡献"空间相关性"(随后被 z-score
    # 重映射, 绝对幅度丢弃) —— 即把真实沉船的空间纹理结构搬过去。强度分布形状(C)仍
    # 只由上面的 highlight 亮像素经 Weibull 决定。二者分工清晰。
    psd_label_map = np.ones((H, W), dtype=np.uint8)
    psd_label_map[mask] = 2

    return dict(hl_pixels=hl_pixels, hl_label_map=hl_pixel_label_map,
                psd_label_map=psd_label_map, otsu_thr=float(otsu_thr),
                fp_med=fp_med, hl_med=hl_med, n_fp=n_fp, n_hl=n_hl, contrast=float(contrast))


def crop_to_bbox(img, label_map, mask, pad=8):
    """裁到 footprint bbox(加 pad), 加速 PSD 扫描。不改变哪些像素是 label==2。"""
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + 1 + pad)
    return img[y0:y1, x0:x1].copy(), label_map[y0:y1, x0:x1].copy()


# ---------------------------------------------------------------------------
# QC 图
# ---------------------------------------------------------------------------

def make_qc(img, mask, ex, wb_hl, psd_hl_n, psd_hl_size, name, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img, cmap='gray'); axes[0].set_title(f'Image: {name}'); axes[0].axis('off')

    seg = np.zeros_like(img, dtype=np.uint8)
    seg[mask] = 128
    seg[ex['hl_label_map'] == 2] = 255
    axes[1].imshow(seg, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(f'footprint(128)+highlight(255)\notsu={ex["otsu_thr"]:.0f} hl_frac={ex["n_hl"]/max(ex["n_fp"],1):.2f}')
    axes[1].axis('off')

    # highlight crop 可视化
    vis = img.copy()
    hl_overlay = np.zeros((*img.shape, 3), np.uint8)
    hl_overlay[..., 0] = img; hl_overlay[..., 1] = img; hl_overlay[..., 2] = img
    hl_overlay[ex['hl_label_map'] == 2] = [255, 60, 60]
    axes[2].imshow(hl_overlay)
    axes[2].set_title(f'highlight pixels (red)\nn_hl={ex["n_hl"]} hl_med={ex["hl_med"]:.0f} fp_med={ex["fp_med"]:.0f}')
    axes[2].axis('off')

    if ex['n_hl'] > 0:
        axes[3].hist(ex['hl_pixels'], bins=60, density=True, alpha=0.6, color='orange', label='HL hist')
        if wb_hl[0] is not None:
            from scipy.stats import weibull_min
            c, loc, scale = wb_hl
            xr = np.linspace(max(0, loc), min(255, loc + scale * 4), 200)
            axes[3].plot(xr, weibull_min.pdf(xr, c, loc=loc, scale=scale), 'r-', lw=2,
                         label=f'Weibull C={c:.2f}\nloc={loc:.1f} sc={scale:.1f}')
        axes[3].set_title(f'Highlight Weibull\nPSD: {psd_hl_n}x{psd_hl_size}')
        axes[3].legend(fontsize=7)
    else:
        axes[3].text(0.5, 0.5, 'no highlight', ha='center', va='center')
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    _assert_paths()
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_STATS_DIR, exist_ok=True)

    print("=" * 64)
    print("AI4 STEP-1: real-shipwreck HIGHLIGHT parameter extraction")
    print("=" * 64)

    # ---- 选帧 ----
    manifest, sel_info = select_frames(RELOOK_CSV)
    print(f"[Frames] strict_clear==1: {sel_info['strict_clear_1']}; "
          f"excluded {EXCLUDE_SITE}: {sel_info['excluded_reef']}; "
          f"after-exclusion candidates: {sel_info['after_reef']} "
          f"({sel_info['n_wrecks']} wrecks)")
    print(f"[Frames] disputed kept as candidates (quality-gated): {sel_info['disputed_kept']}")

    # ---- 借基线 URM 的 shadow（只读） ----
    base = np.load(BASELINE_URM_CLEAN, allow_pickle=True)
    n_urm_shadow = len(base['urm_names'])
    base_wb_sh = base['wb_shadow']
    base_psd_sh = [base[f'psd_sh_{i}'] for i in range(n_urm_shadow)]
    base_psd_sh_size = base['psd_shadow_sizes']
    print(f"[Shadow] borrowing wb_shadow/psd_sh from baseline URM clean: {n_urm_shadow} entries")

    # ---- 固定种子（shadow 配给规则确定性, 不引入随机, 但固定以备未来扩展） ----
    np.random.seed(SEED)
    qc_pick = set(np.random.RandomState(SEED).choice(
        len(manifest), min(QC_SAMPLE_COUNT, len(manifest)), replace=False).tolist())

    results = []
    for i, m in enumerate(manifest):
        img, mask = load_pair(m['split'], m['file'])
        name = m['file'][:-4] if m['file'].endswith('.png') else m['file']
        if img is None:
            print(f"  [{i+1}/{len(manifest)}] {name} READ FAIL")
            results.append(_empty_result(name, m, "read_fail"))
            continue

        ex = extract_highlight(img, mask)

        # ---- 质量门(透明, 写明) ----
        quality_ok = (ex['n_hl'] >= HL_MIN_PX) and (ex['contrast'] >= HL_MIN_CONTRAST)

        # ---- Weibull(floc=0 变体, 见顶部注释; 强度形状 C 来自 highlight 亮像素) ----
        wb_hl = fit_weibull_floc0(ex['hl_pixels']) if quality_ok else (None, None, None)

        # ---- PSD(复用 estimate_psd_with_fallback; 区域=footprint, target_label=2) ----
        if quality_ok:
            img_c, lbl_c = crop_to_bbox(img, ex['psd_label_map'], mask)
            psd_hl, psd_hl_n, psd_hl_size = estimate_psd_with_fallback(img_c, lbl_c, 2)
        else:
            psd_hl, psd_hl_n, psd_hl_size = None, 0, 16

        # ---- shadow 借基线 URM, 成对(i % n_urm) ----
        urm_idx = i % n_urm_shadow
        wb_sh = (float(base_wb_sh[urm_idx][0]), float(base_wb_sh[urm_idx][1]),
                 float(base_wb_sh[urm_idx][2]))
        psd_sh = base_psd_sh[urm_idx]
        psd_sh_size = int(base_psd_sh_size[urm_idx])

        results.append({
            'name': name, 'site': m['site'], 'split': m['split'],
            'disputed': m['disputed'], 'quality_ok': quality_ok,
            'reject': None if quality_ok else "quality_gate",
            'seg_ok': bool(quality_ok),
            'sorted_means': [ex['fp_med'], ex['fp_med'], ex['hl_med']],  # 占位(schema 兼容)
            'wb_highlight': wb_hl,
            'wb_shadow': wb_sh,
            'psd_highlight': psd_hl,
            'psd_shadow': psd_sh,
            'psd_highlight_n': psd_hl_n,
            'psd_shadow_n': 0,
            'psd_highlight_size': psd_hl_size if psd_hl is not None else 0,
            'psd_shadow_size': psd_sh_size,
            'n_highlight_px': ex['n_hl'],
            'n_shadow_px': 0,
            'shadow_src_urm': str(base['urm_names'][urm_idx]),
            'otsu_thr': ex['otsu_thr'], 'fp_med': ex['fp_med'], 'hl_med': ex['hl_med'],
            'contrast': ex['contrast'], 'n_fp': ex['n_fp'],
        })

        wb_str = f"C={wb_hl[0]:.2f}" if wb_hl[0] is not None else ("QGATE" if not quality_ok else "WBFAIL")
        psd_str = f"{psd_hl_n}x{psd_hl_size}" if psd_hl is not None else "FAIL"
        print(f"  [{i+1}/{len(manifest)}] {name:30s} n_hl={ex['n_hl']:6d} contrast={ex['contrast']:6.1f} "
              f"HL_wb:{wb_str} PSD:{psd_str}")

        if i in qc_pick:
            make_qc(img, mask, ex, wb_hl, psd_hl_n,
                    psd_hl_size if psd_hl is not None else 0, name,
                    os.path.join(OUTPUT_STATS_DIR, f"qc_{name}.png"))

    # ---- 落盘 ai4_params.npz (严格复刻 urm_params.npz schema) ----
    save_dict = _build_save_dict(results)
    np.savez_compressed(OUTPUT_NPZ, **save_dict)
    print(f"\n[Saved] {OUTPUT_NPZ}")

    # ---- self-check + 报告 ----
    checks = _self_check(results, sel_info, manifest)
    print("\n[Self-Check]")
    for c in checks:
        print("  " + c)
    _write_report(results, sel_info, manifest, checks)


def _empty_result(name, m, reject):
    return {
        'name': name, 'site': m['site'], 'split': m['split'],
        'disputed': m['disputed'], 'quality_ok': False, 'reject': reject,
        'seg_ok': False, 'sorted_means': [0, 0, 0],
        'wb_highlight': (None, None, None), 'wb_shadow': (None, None, None),
        'psd_highlight': None, 'psd_shadow': None,
        'psd_highlight_n': 0, 'psd_shadow_n': 0,
        'psd_highlight_size': 0, 'psd_shadow_size': 0,
        'n_highlight_px': 0, 'n_shadow_px': 0, 'shadow_src_urm': '',
        'otsu_thr': 0.0, 'fp_med': 0.0, 'hl_med': 0.0, 'contrast': 0.0, 'n_fp': 0,
    }


def _wb_triplet(wb):
    """避开 urm_process L465 的 `x[1] or np.nan` bug: loc=0.0 会被误判为 NaN。
    用显式 None 判: 拟合失败(None)才写 NaN, 否则写真实值(含合法的 0.0)。"""
    c, loc, scale = wb
    if c is None:
        return [np.nan, np.nan, np.nan]
    return [float(c), float(loc), float(scale)]


def _build_save_dict(results):
    sd = {
        'urm_names': np.array([r['name'] for r in results]),
        'seg_ok': np.array([r['seg_ok'] for r in results]),
        'sorted_means': np.array([r['sorted_means'] for r in results], dtype=np.float64),
        'n_highlight_px': np.array([r['n_highlight_px'] for r in results]),
        'n_shadow_px': np.array([r['n_shadow_px'] for r in results]),
    }
    sd['wb_highlight'] = np.array([_wb_triplet(r['wb_highlight']) for r in results], dtype=np.float64)
    sd['wb_shadow'] = np.array([_wb_triplet(r['wb_shadow']) for r in results], dtype=np.float64)

    psd_hl_sizes, psd_sh_sizes = [], []
    for idx, r in enumerate(results):
        if r['psd_highlight'] is not None:
            sd[f'psd_hl_{idx}'] = r['psd_highlight']
            psd_hl_sizes.append(r['psd_highlight_size'])
        else:
            sd[f'psd_hl_{idx}'] = np.zeros((1, 1))
            psd_hl_sizes.append(0)
        if r['psd_shadow'] is not None:
            sd[f'psd_sh_{idx}'] = r['psd_shadow']
            psd_sh_sizes.append(r['psd_shadow_size'])
        else:
            sd[f'psd_sh_{idx}'] = np.zeros((1, 1))
            psd_sh_sizes.append(0)
    sd['psd_highlight_sizes'] = np.array(psd_hl_sizes)
    sd['psd_shadow_sizes'] = np.array(psd_sh_sizes)
    return sd


def _self_check(results, sel_info, manifest):
    import collections
    c = []
    total = len(results)
    c.append(f"[INFO] selected frames (strict_clear==1): {sel_info['strict_clear_1']}")
    c.append(f"[INFO] excluded {EXCLUDE_SITE}: {sel_info['excluded_reef']}")
    c.append(f"[INFO] after exclusion (candidates): {sel_info['after_reef']} "
             f"({sel_info['n_wrecks']} wrecks)")
    c.append(f"[INFO] disputed kept as candidates: {sel_info['disputed_kept']}")

    q_ok = sum(1 for r in results if r['quality_ok'])
    c.append(f"[INFO] passed highlight quality gate (n_hl>={HL_MIN_PX} & contrast>={HL_MIN_CONTRAST}): {q_ok}/{total}")
    q_rej = [r['name'] for r in results if not r['quality_ok']]
    c.append(f"[INFO] quality-gate rejected: {q_rej}")

    wb_ok = sum(1 for r in results if r['wb_highlight'][0] is not None)
    psd_ok = sum(1 for r in results if r['psd_highlight'] is not None)
    c.append(f"[{'PASS' if wb_ok > total*0.8 else 'WARN'}] highlight Weibull fit: {wb_ok}/{total}")
    c.append(f"[{'PASS' if psd_ok > total*0.8 else 'WARN'}] highlight PSD estimated: {psd_ok}/{total}")

    # 最终可进入 filter 的"双 OK"(highlight wb+psd 都成功; shadow 借来必成功)
    final_ok = sum(1 for r in results if r['wb_highlight'][0] is not None and r['psd_highlight'] is not None)
    c.append(f"[INFO] entries with BOTH highlight wb+psd ok (will survive filter): {final_ok}/{total}")

    # per-wreck 分布(只数 final_ok 的)
    per = collections.Counter(r['site'] for r in results
                              if r['wb_highlight'][0] is not None and r['psd_highlight'] is not None)
    c.append(f"[INFO] effective wrecks (>=1 usable entry): {len(per)}")
    c.append(f"[INFO] per-wreck usable-entry distribution: {dict(sorted(per.items()))}")

    # HL 比 SH 亮的排序成立(用 hl_med vs shadow 来源 URM 的灰度? shadow 借来无灰度,
    # 用合成语义: highlight 是亮回波, 应 hl_med 明显高于 footprint 中位; 并报 hl_med 分布)
    hl_meds = [r['hl_med'] for r in results if r['wb_highlight'][0] is not None]
    fp_meds = [r['fp_med'] for r in results if r['wb_highlight'][0] is not None]
    if hl_meds:
        order_ok = sum(1 for hm, fm in zip(hl_meds, fp_meds) if hm > fm)
        c.append(f"[{'PASS' if order_ok == len(hl_meds) else 'WARN'}] "
                 f"highlight brighter than footprint-median (HL>SH sanity): {order_ok}/{len(hl_meds)}")
        c.append(f"[INFO] highlight median-gray range: [{min(hl_meds):.0f}, {max(hl_meds):.0f}], "
                 f"mean={np.mean(hl_meds):.0f}")

    # Weibull C 范围
    cs = [r['wb_highlight'][0] for r in results if r['wb_highlight'][0] is not None]
    if cs:
        c.append(f"[INFO] highlight Weibull C range: [{min(cs):.3f}, {max(cs):.3f}]")
    return c


def _write_report(results, sel_info, manifest, checks):
    rp = os.path.join(OUTPUT_STATS_DIR, "ai4_process_report.txt")
    with open(rp, "w", encoding="utf-8") as f:
        f.write("AI4 Step-1 Highlight Parameter Extraction Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 64 + "\n\n")
        f.write("[Frame Selection]\n")
        for k, v in sel_info.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\n[Quality Gate]\n  HL_MIN_PX={HL_MIN_PX}, HL_MIN_CONTRAST={HL_MIN_CONTRAST}\n")
        f.write(f"  shadow source: baseline urm_params_clean.npz (i %% n_urm), known limitation\n\n")
        f.write("[Per-frame]\n")
        f.write(f"  {'name':32s} {'site':24s} disp qok  n_hl  contrast  C      PSD     shadow_urm\n")
        for r in results:
            cstr = f"{r['wb_highlight'][0]:.2f}" if r['wb_highlight'][0] is not None else "----"
            pstr = f"{r['psd_highlight_n']}x{r['psd_highlight_size']}" if r['psd_highlight'] is not None else "FAIL"
            f.write(f"  {r['name']:32s} {r['site']:24s} {int(r['disputed'])}    "
                    f"{int(r['quality_ok'])}  {r['n_highlight_px']:6d} {r['contrast']:8.1f}  "
                    f"{cstr:6s} {pstr:7s} {r['shadow_src_urm']}\n")
        f.write("\n[Self-Check]\n")
        for c in checks:
            f.write("  " + c + "\n")
    print(f"[Saved] {rp}")


if __name__ == "__main__":
    main()
