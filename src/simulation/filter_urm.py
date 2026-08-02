# ============================================================================
# 阶段2补丁：过滤异常URM参数
# ============================================================================
# 功能：读取urm_params.npz，按合理性标准剔除异常URM，输出干净的参数文件
#
# 输入：outputs\stage2_params\data\urm_params.npz
# 输出：outputs\stage2_params\data\urm_params_clean.npz
#       outputs\stage2_params\stats\filter_report.txt
# ============================================================================

import numpy as np
import os
from datetime import datetime


# ============ 路径配置 ============
INPUT_PATH = r"PATH_TO_URM_PARAMETER_OUTPUT\urm_params.npz"
OUTPUT_PATH = r"PATH_TO_URM_PARAMETER_OUTPUT\urm_params_clean.npz"
REPORT_PATH = r"PATH_TO_URM_STATS_OUTPUT\filter_report.txt"

# ============ 过滤标准 ============
# 高亮区
HL_MAX_C = 10
HL_MAX_ABS_LOC = 500
HL_MAX_SCALE = 500

# 阴影区
SH_MAX_C = 50
SH_MAX_ABS_LOC = 3000
SH_MAX_SCALE = 3000


def main():
    # 读取原始参数文件
    data = np.load(INPUT_PATH, allow_pickle=True)

    names = data['urm_names']
    seg_ok = data['seg_ok']
    sorted_means = data['sorted_means']
    n_highlight_px = data['n_highlight_px']
    n_shadow_px = data['n_shadow_px']
    wb_highlight = data['wb_highlight']   # (N, 3): [C, loc, scale]
    wb_shadow = data['wb_shadow']         # (N, 3): [C, loc, scale]
    psd_hl_sizes = data['psd_highlight_sizes']
    psd_sh_sizes = data['psd_shadow_sizes']

    total = len(names)
    print(f"Loaded {total} URM entries from {INPUT_PATH}")
    print(f"Filter criteria:")
    print(f"  Highlight: C <= {HL_MAX_C}, |loc| <= {HL_MAX_ABS_LOC}, scale <= {HL_MAX_SCALE}")
    print(f"  Shadow:    C <= {SH_MAX_C}, |loc| <= {SH_MAX_ABS_LOC}, scale <= {SH_MAX_SCALE}")
    print(f"  PSD:       both highlight and shadow must succeed (size > 0)")
    print(f"{'='*60}")

    # 逐条检查
    keep_mask = np.ones(total, dtype=bool)
    reject_reasons = {}

    for i in range(total):
        reasons = []

        # Weibull高亮区检查
        c_hl, loc_hl, sc_hl = wb_highlight[i]
        if np.isnan(c_hl):
            reasons.append("HL Weibull fit failed")
        else:
            if c_hl > HL_MAX_C:
                reasons.append(f"HL C={c_hl:.2f} > {HL_MAX_C}")
            if abs(loc_hl) > HL_MAX_ABS_LOC:
                reasons.append(f"HL |loc|={abs(loc_hl):.1f} > {HL_MAX_ABS_LOC}")
            if sc_hl > HL_MAX_SCALE:
                reasons.append(f"HL scale={sc_hl:.1f} > {HL_MAX_SCALE}")

        # Weibull阴影区检查
        c_sh, loc_sh, sc_sh = wb_shadow[i]
        if np.isnan(c_sh):
            reasons.append("SH Weibull fit failed")
        else:
            if c_sh > SH_MAX_C:
                reasons.append(f"SH C={c_sh:.2f} > {SH_MAX_C}")
            if abs(loc_sh) > SH_MAX_ABS_LOC:
                reasons.append(f"SH |loc|={abs(loc_sh):.1f} > {SH_MAX_ABS_LOC}")
            if sc_sh > SH_MAX_SCALE:
                reasons.append(f"SH scale={sc_sh:.1f} > {SH_MAX_SCALE}")

        # PSD检查
        if psd_hl_sizes[i] == 0:
            reasons.append("HL PSD failed")
        if psd_sh_sizes[i] == 0:
            reasons.append("SH PSD failed")

        if reasons:
            keep_mask[i] = False
            reject_reasons[str(names[i])] = reasons

    n_keep = np.sum(keep_mask)
    n_reject = total - n_keep
    keep_indices = np.where(keep_mask)[0]

    print(f"\nResults: {n_keep} kept, {n_reject} rejected")
    print(f"\nRejected URM images:")
    for name, reasons in reject_reasons.items():
        print(f"  {name}: {'; '.join(reasons)}")

    # 构建干净的参数文件
    clean_dict = {
        'urm_names': names[keep_indices],
        'seg_ok': seg_ok[keep_indices],
        'sorted_means': sorted_means[keep_indices],
        'n_highlight_px': n_highlight_px[keep_indices],
        'n_shadow_px': n_shadow_px[keep_indices],
        'wb_highlight': wb_highlight[keep_indices],
        'wb_shadow': wb_shadow[keep_indices],
        'psd_highlight_sizes': psd_hl_sizes[keep_indices],
        'psd_shadow_sizes': psd_sh_sizes[keep_indices],
    }

    # 功率谱重新编号：新索引从0开始
    for new_idx, old_idx in enumerate(keep_indices):
        psd_hl_key = f'psd_hl_{old_idx}'
        psd_sh_key = f'psd_sh_{old_idx}'
        clean_dict[f'psd_hl_{new_idx}'] = data[psd_hl_key]
        clean_dict[f'psd_sh_{new_idx}'] = data[psd_sh_key]

    np.savez_compressed(OUTPUT_PATH, **clean_dict)
    print(f"\nSaved clean params: {OUTPUT_PATH}")
    print(f"  {n_keep} URM entries, re-indexed 0 ~ {n_keep - 1}")

    # 保存过滤报告
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Stage 2 Filter Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")

        f.write(f"[Filter Criteria]\n")
        f.write(f"  Highlight: C <= {HL_MAX_C}, |loc| <= {HL_MAX_ABS_LOC}, scale <= {HL_MAX_SCALE}\n")
        f.write(f"  Shadow:    C <= {SH_MAX_C}, |loc| <= {SH_MAX_ABS_LOC}, scale <= {SH_MAX_SCALE}\n")
        f.write(f"  PSD:       both highlight and shadow must succeed\n\n")

        f.write(f"[Summary]\n")
        f.write(f"  Total input:  {total}\n")
        f.write(f"  Kept:         {n_keep}\n")
        f.write(f"  Rejected:     {n_reject}\n\n")

        f.write(f"[Rejected URM Images]\n")
        if reject_reasons:
            for name, reasons in reject_reasons.items():
                f.write(f"  {name}: {'; '.join(reasons)}\n")
        else:
            f.write(f"  (none)\n")

        f.write(f"\n[Kept URM Parameter Ranges]\n")
        kept_wb_hl = wb_highlight[keep_indices]
        kept_wb_sh = wb_shadow[keep_indices]
        f.write(f"  Highlight C:     [{kept_wb_hl[:,0].min():.3f}, {kept_wb_hl[:,0].max():.3f}]\n")
        f.write(f"  Highlight loc:   [{kept_wb_hl[:,1].min():.1f}, {kept_wb_hl[:,1].max():.1f}]\n")
        f.write(f"  Highlight scale: [{kept_wb_hl[:,2].min():.1f}, {kept_wb_hl[:,2].max():.1f}]\n")
        f.write(f"  Shadow C:        [{kept_wb_sh[:,0].min():.3f}, {kept_wb_sh[:,0].max():.3f}]\n")
        f.write(f"  Shadow loc:      [{kept_wb_sh[:,1].min():.1f}, {kept_wb_sh[:,1].max():.1f}]\n")
        f.write(f"  Shadow scale:    [{kept_wb_sh[:,2].min():.1f}, {kept_wb_sh[:,2].max():.1f}]\n")

        f.write(f"\n[Output]\n")
        f.write(f"  Clean params file: {OUTPUT_PATH}\n")
        f.write(f"  Entries: {n_keep}, indexed 0 ~ {n_keep - 1}\n")

    print(f"Saved filter report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
