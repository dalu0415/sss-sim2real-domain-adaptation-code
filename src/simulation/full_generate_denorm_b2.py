# ============================================================================
# Step 1.5b' (de-normalization, JOINT mean+std) + 软膝 soft-knee + 脏块重抽
#   == DENORM_B2 变体 (= full_generate_denorm_b.py 的逐字拷贝 + 两件新增) ==
# ============================================================================
# 本文件 = synth_improved\src\full_generate_denorm_b.py (1.5b) 的【逐字拷贝】,
# 仅相对 1.5b 新增两件事来杀掉"死黑纯0平板" artifact:
#   (Knee) 软膝: bg 层平移后那行 `np.clip(bg_layer,0,255)` 的【下界硬切】换成
#          指数软膝 _soft_floor: 拐点 KNEE_K 以上原样、以下平滑(C1连续)单调指数
#          渐近到 0 —— 暗区"渐渐"降到 0 而非被一刀拍成同一个精确 0。上界对称
#          _soft_ceil 防纯白板。**纯确定性函数, 不碰任何 rng** (单变量铁律)。
#          compat15 仍走 1.5 硬切 (软膝 gated 在非 compat 路径), 保字节锚 1。
#   (Reject) 脏块重抽: bg crop 若 exact-0 占比 > BLOCK_ZERO_REJECT (stage1 已死
#          的 nadir-skirt 脏块, 任何曲线救不回), 用【独立 reject_rng】重抽到干净
#          块。第一次抽永远用主 rng (与 1.5b 逐字一致) -> 主纹理 rng 流字节不变;
#          只有触发重抽那几张的 bg 像素被换。compat15 不重抽, 保字节锚 1/2。
#
#   两个新参数 (KNEE_K, BLOCK_ZERO_REJECT) 均在【看任何 format-probe 之前】用
#   clip/纹理/nadir-lift 证据扫定、写死成常量, 绝不朝 KLSG 拧 (KLSG 路径/常数
#   一律不出现)。诚实: 软膝只【部分】恢复纹理 (faint、非丰富真实暗纹理; 上游
#   仿射+8位量化把深浅压到只剩 2-5 灰阶); stage1 已压成精确 0 的像素软膝救不回
#   -> 由脏块重抽处理。本步目标 = 让 1.5b 干净可交付(无黑洞/白板 artifact),
#   非 format-probe 奇迹。
#
#   两个字节锚 (单变量证明):
#     (1) --compat15        -> 字节复现 1.5  step1p5_denorm  (软膝/重抽全 gated off)
#     (2) --plain_b2 (软膝关+重抽关) -> 字节复现 1.5b step1p5b_denorm (证只多了软膝+重抽)
#
# ---- 以下为继承自 1.5b 的原始说明 (相对 1.5 的外科改动 A-E) ----
# 本文件 = synth_improved\src\full_generate_denorm.py (1.5) 的【逐字拷贝】,
# 仅做以下外科改动, 便于与 full_generate_denorm.py 直接 diff:
#   (A) 输出路径 _STEP1P5_ROOT -> _STEP1P5B_ROOT (step1p5b_denorm);
#       POOL_PATH -> bg_brightness_pool_joint.npz; assert 白名单更新到 step1p5b。
#   (B) _load_brightness_pool -> 返回成对 (mean_band, std_band, lo, p95):
#       按 mean ∈ [lo, p95] 过滤、std 同步取对应元素 (成对, 不打乱配对)。
#       lo = pool_p5 (★丢掉 1.5 的 MU_TARGET_FLOOR=40, 由 §4 clip sweep 跑前冻结)。
#   (C) synthesize_single 内【唯一】合成逻辑改动 = 把 1.5 的纯平移
#         bg_layer = bg_layer + (mu_target - bg_raw_mean)              # 只定均值
#       改成仿射 (同时定均值 AND std):
#         bg_layer = (bg_layer - bg_raw_mean)*(sigma_target/bg_raw_std) + mu_target
#       (mu_target, sigma_target) 是同一个 AI4 块的成对值, 由主循环用独立第二
#       rng 一次 draw 出一对 (见 (D)), 不触主纹理 rng。
#   (D) 主循环 "抽 mu" -> "抽一个成对索引": pair_idx = mu_rng.randint(...);
#       mu_target=mean_band[pair_idx], sigma_target=std_band[pair_idx]。
#   (E) --compat15 单变量证明开关: 开启时 mu 用 1.5 方式抽 (mean-only band
#       [max(p5,40),p95]、mu_rng.choice)、sigma_target := bg_raw_std (仿射退化
#       为恒等) -> 输出必须与 1.5 step1p5_denorm 对应产物逐字节一致。
#
# 单变量保证: 除 (A)-(E) 外, 所有合成常数 (PERTURB / OFFSET / std_ratio /
#   HIGHLIGHT_STD_RATIO / BLEND / DOWNSAMPLE / WEIBULL_C_FLOOR / 主 rng seed=42 /
#   纹理 rng 调用序列 / CSV schema) 与 1.5 逐字冻结。纹理/几何/speckle 一律不动。
#   bg_std 现在 per-image 变化会自然传导到 HL/SH target 对比度 —— 这是"对比度
#   放开"的预期效果, 不是另一个旋钮 (HIGHLIGHT_STD_RATIO 等逐字冻结)。
#
# 诚实: 目标 (mu,sigma) 来自 AI4 terrain 源天然联合分布 (mean≈58.5/std≈35.6,
#   比 KLSG 的 80/30 还暗/异质), 非 KLSG。绝无 KLSG 路径/数值。
#   叙事 = de-artifacting: 恢复 pipeline (normalize_blocks + 常数90) 抹掉的
#   逐图亮度 AND 对比度起伏。1.5b 相对 1.5 唯一新增 = std 放开 + 丢 floor。
# ============================================================================

import os
import glob
import json
import csv
import argparse
import importlib.util
import numpy as np
import cv2
from scipy.stats import weibull_min
from scipy.ndimage import distance_transform_edt
from scipy import ndimage
from pathlib import Path
from datetime import datetime
import shutil

# ★ STEP2: import the LOCKED v3 shadow geometry from masks.py (whole-silhouette
#   sweep + far-edge undulation). Loaded by file path so the import works
#   regardless of cwd / package layout. masks.py top level = def + constants,
#   no side effects (its main() aborts via SystemExit but is never called here).
_MASKS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "masks.py")
_spec_masks = importlib.util.spec_from_file_location("masks_step2", _MASKS_PATH)
MASKS = importlib.util.module_from_spec(_spec_masks)
_spec_masks.loader.exec_module(MASKS)


# ============ 路径配置 ============
# 读路径 (与 full_generate.py / full_generate_denorm.py 完全相同, 复用只读):
URM_PARAMS_PATH = r"PATH_TO_SYNTH_IMPROVED_WORKSPACE\data\outputs\stage2_ai4\data\ai4_params_clean.npz"
MASK_DATA_DIR = r"PATH_TO_BASELINE_SYNTH_OUTPUTS\stage3_masks\data"
MASK_META_PATH = r"PATH_TO_BASELINE_SYNTH_OUTPUTS\stage3_masks\data\mask_metadata.json"
BG_BLOCKS_DIR = r"PATH_TO_BASELINE_SYNTH_OUTPUTS\stage1_bg\blocks"

# (B) 只读: 复用 1.5b 已建的 AI4 天然 JOINT (mean,std) 池 (1.5b' 不重建, 只读)
POOL_PATH = r"PATH_TO_SYNTH_IMPROVED_WORKSPACE\data\outputs\step1p5b_denorm\bg_brightness_pool_joint.npz"

# (A) 输出根: Step1.5b' 子树 step1p5b2_denorm。trial -> stage4_trial; full -> stage5_dataset。
#     注意: 写路径全在 step1p5b2_denorm 下, 绝不写 1.5b/1.5/Step1/基线 (路径隔离铁律)。
_STEP1P5B2_ROOT = r"PATH_TO_SYNTH_IMPROVED_WORKSPACE\data\outputs\step1p5b2_denorm"
# 兼容原有引用名 (本文件内部仍以 _STEP1P5B_ROOT 这个名字使用根, 但指向 b2 子树)
_STEP1P5B_ROOT = _STEP1P5B2_ROOT

# ============================================================================
# ★ STEP2 阴影: 新输出根 step2_shadow (NEW shadow 写这里; 1.5b' 留作"加阴影前"对照)。
#   SHADOW_MODE 选码路:
#     'new' (default) = v3 整剪影扫掠 + 远边起伏 + 暗透镜 alpha 渲染 (Step2 产物);
#     'old'           = pre-Step2 = 读 stage3 *_shadow.png + assemble_image ramp
#                       (byte-anchor: 复现 1.5b-prime stage5 BYTE-EXACT, 证单变量)。
#   *关键*: 整条亮度线 (highlight / bg affine / soft-knee / mu-sigma / KNEE_K /
#   HIGHLIGHT_OFFSET / 主 rng / mu_rng / reject_rng / downsample) 两模式逐字节相同;
#   唯一差异 = shadow_region 填充 + obj_and_shadow 边缘 (NEW shadow geometry)。
#   OLD 输出仍落 step1p5b2_denorm (用于 byte-anchor 自比); NEW 输出落 step2_shadow。
# ============================================================================
_STEP2_ROOT = r"PATH_TO_SYNTH_IMPROVED_WORKSPACE\data\outputs\step2_shadow"

OUTPUT_BASE_FULL = os.path.join(_STEP1P5B_ROOT, "stage5_dataset")
OUTPUT_BASE_TRIAL = os.path.join(_STEP1P5B_ROOT, "stage4_trial")

# 以下 5 个 OUTPUT_*_DIR 由 main() 据 --mode 在运行时绑定 (保持同名以便 diff)。
OUTPUT_BASE = OUTPUT_BASE_FULL
OUTPUT_PLANE_DIR = os.path.join(OUTPUT_BASE, "airplane")
OUTPUT_SHIP_DIR = os.path.join(OUTPUT_BASE, "ship")
OUTPUT_ALL_DIR = os.path.join(OUTPUT_BASE, "all")
OUTPUT_STATS_DIR = os.path.join(OUTPUT_BASE, "stats")

# (C/D) (mu_target,sigma_target) 抽样: 独立第二 RandomState 固定种子 (不扰主纹理 rng)。
#   1.5b 一次 draw 出一个【成对索引】, 同时决定 mu 和 sigma (保持配对)。沿用 1.5 种子。
MU_SAMPLE_SEED = 1234

# (B) mean-band 下限 lo。【跑前由 clip-vs-baseline 经验扫描冻结, _qc_floor_sweep_joint.py】
#   ★实测纠正 spec 的乐观假设: 原以为"联合放开 std 后暗块 sigma 变小 -> lo 可降到 p5
#   不爆 clip"。但 sweep 实测相反 —— AI4 块的【within-block std 偏大 (mean≈46、p50≈34)】,
#   暗均值 (mu~25-37) 配上这种 sigma 后, 左尾仍被推到 0 以下大片 clip。故 JOINT 抽样下
#   clip0 不降反升, 没有任何 lo 能压回 Step1 基线 (clip0 max≈1.8%)。
#   sweep 证据 (15-trial, per-image clip0):
#     lo=p5(22.43): clip0 max 21.0%  | lo=25: 22.6%  | lo=30: 7.8%(最低) |
#     lo=35:17.6% lo=40:19.5% lo=50:12.7% lo=60:19.5% ... (>=35 不再单调改善、spread 反降)
#   clip0 max 由 N=15 里 1 个暗图撞宽 sigma 的极端 case 主导 (lo=30 时仅 1 张 >5%、
#   p90=1.3%、mean=0.9%)。lo=30 是【证据支持的最小 clamp】: clip0 max/mean/p90 三项全最低,
#   同时 mean-spread=25.1 (> 1.5 的 20.35)、std-spread=11.5 (1.5≈0)。再低 (p5/25) clip 翻 3x、
#   再高 clip 不降而 spread 单调掉。
#   依 spec §4 fallback "若仍超基线 -> 取证据支持的最小 clamp" -> 冻结 lo=30。
#   注意: lo=30【非贴任何外部数据集 (KLSG mean=80, 离 30 极远)】, 是 AI4 池自身近 p25
#   区, 纯按 clip 证据选。比 1.5 的 floor=40 更低 -> 朝 AI4 源更暗端走了一步 (mu 能到 30),
#   只是没到 p5。clip 残留是诚实代价、honest_caveats 已记。**全程未看 format-probe。**
MEAN_BAND_LO_MODE = "value"      # 'p5'=丢floor; 'value'=用下方 clamp 数值当下限
MEAN_BAND_LO_CLAMP = 30.0        # 冻结值 (clip-vs-baseline sweep 跑前定死, 非贴 KLSG)

# 1.5 的 floor (仅 --compat15 模式复刻 1.5 band 时使用; 1.5b 正常路径不用它)
MU_TARGET_FLOOR_COMPAT15 = 40.0

# ============================================================================
# 1.5b' 新增常量 (软膝 + 脏块重抽) —— 均在看任何 format-probe 之前用 clip/纹理/
#   nadir-lift 证据扫定后冻结, 写死成常量。绝非朝任何外部数据集拧 (clip/纹理证据
#   冻结、非贴外部). sweep 证据见 §C / 报告 knee_sweep / reject_sweep 字段。
# ============================================================================
# 软膝拐点 k: bg 层下界软膝 _soft_floor 在 x>=k 原样、x<k 指数渐近 0 的拐点。
#   k 越大 -> 拐点上移、former-0 暗区恢复越多纹理, 但 genuine-black(原图 raw==0
#   且被硬切拍成 0)像素的 nadir-lift(灰阶提升) 也越大。
#   选"能去掉死黑平板、且 genuine-black nadir-lift <= 2 灰阶"的【最大】k。
#   §C sweep (trial15 + worst {0126/0143/0383/0453/0470/0215/0214}) 实测:
#     k= 6: former0 redux 61.4% | recoverStd(7x7)=13.14 | worst-img-mean nadir-lift=1.0 (OK)
#     k= 8: former0 redux 69.9% | recoverStd=13.12      | nadir-lift=2.0 (OK, 边界)
#     k=10: former0 redux 91.5% | recoverStd=13.06      | nadir-lift=2.0 (OK, 边界) <== 选
#     k=12: former0 redux 100%  | recoverStd=13.00      | nadir-lift=3.0 (FAIL >2)
#   k=10 = 满足 nadir-lift<=2 的最大候选 -> 去死黑最多 (91.5%) 而真黑像素仍 <=2 灰阶。
#   former-0 区软膝后是 2-5 个不同灰阶的 faint 渐变 (非单值死平板; ship_0126: 灰阶 [2,3,4]),
#   非 "新平暗灰 plate"。**全程未看 format-probe。**
KNEE_K = 10.0   # FROZEN (clip/纹理/nadir-lift 证据冻结, 非贴外部数据集; sweep 选最大且 lift<=2)

# 脏块重抽: bg crop 的 exact-0 占比 > 此阈值 -> 判为 stage1-dead nadir-skirt 脏块,
#   用独立 reject_rng 重抽。§C sweep 实测 first-draw crop exact-0 占比分布:
#     最脏 = ship_0126(16.2%)/ship_0383(12.1%)/ship_0453(11.6%)/ship_0215(11.0%)/ship_0149(10.2%),
#     次干净 = ship_0470(2.6%) -> 0.10 与 0.026 间有明显 gap。
#   thr=0.10 恰好 flag 这 5 张最脏 (= spec 预测的 ship_0126/0383/0453/0215 类), 不误伤干净块。
BLOCK_ZERO_REJECT = 0.10   # FROZEN (bg-crop exact-0 占比分布扫定, 落在 0.026-0.102 的 gap 上沿, 非贴外部)
REJECT_SEED = 4321         # 独立 reject_rng 种子 (与主 rng seed=42 / mu_rng seed=1234 并列, 互不干扰)
MAX_REDRAW = 30            # 重抽上限; 仍超阈值则取这些次里最干净的一块

# ---- fail-fast: 写路径必在 step1p5b2_denorm 下; 读路径在 synth_improved(自产)/基线只读/池 白名单内 ----
_IMPROVED_ROOT = r"PATH_TO_SYNTH_IMPROVED_WORKSPACE"
_BASELINE_RO = r"PATH_TO_BASELINE_SYNTH_OUTPUTS"
# 1.5 只读对照根 (compat15 字节比对用, 只读不写)
_STEP1P5_RO = r"PATH_TO_SYNTH_IMPROVED_WORKSPACE\data\outputs\step1p5_denorm"
# 1.5b 只读对照根 (plain_b2 字节比对 + 复用池/不重建, 只读不写)
_STEP1P5B_RO = r"PATH_TO_SYNTH_IMPROVED_WORKSPACE\data\outputs\step1p5b_denorm"
def _assert_paths():
    sr = os.path.normcase(os.path.abspath(_STEP1P5B2_ROOT))   # OLD byte-anchor 写根
    s2 = os.path.normcase(os.path.abspath(_STEP2_ROOT))       # NEW Step2 写根
    rn = os.path.normcase(os.path.abspath(_IMPROVED_ROOT))
    bn = os.path.normcase(os.path.abspath(_BASELINE_RO))
    # 写路径必在 step1p5b2_denorm (OLD anchor) 或 step2_shadow (NEW) 子树下 (绝不写 1.5b/1.5/Step1/基线 synth)
    for p in [OUTPUT_BASE, OUTPUT_PLANE_DIR, OUTPUT_SHIP_DIR, OUTPUT_ALL_DIR, OUTPUT_STATS_DIR]:
        pn = os.path.normcase(os.path.abspath(p))
        assert pn.startswith(sr) or pn.startswith(s2), \
            f"WRITE must be under step1p5b2_denorm or step2_shadow: {p}"
    # 读路径: URM 与 池 在 synth_improved 内; mask/bg 在基线只读
    for p in [URM_PARAMS_PATH, POOL_PATH]:
        assert os.path.normcase(os.path.abspath(p)).startswith(rn), f"read must be in synth_improved: {p}"
    for p in [MASK_DATA_DIR, MASK_META_PATH, BG_BLOCKS_DIR]:
        assert os.path.normcase(os.path.abspath(p)).startswith(bn), f"baseline RO input not whitelisted: {p}"
    # 诚实自检: 绝无 KLSG / SeabedObjects 路径
    for p in [URM_PARAMS_PATH, POOL_PATH, MASK_DATA_DIR, BG_BLOCKS_DIR, OUTPUT_BASE, _STEP2_ROOT]:
        low = p.lower()
        assert "klsg" not in low and "seabedobject" not in low, f"FORBIDDEN KLSG path: {p}"


def _load_brightness_pool(path, compat15=False):
    """(B) 读 AI4 天然 JOINT 池, 返回成对 (mean_band, std_band, lo, p95)。
       1.5b: 按 mean ∈ [lo, p95] 过滤, std 同步取对应元素 (成对, 不打乱配对)。
       lo 由 MEAN_BAND_LO_MODE/CLAMP 决定 (默认 = pool p5, 丢 floor)。
       compat15=True: 复刻 1.5 的 mean-only band lo=max(p5,40)。"""
    assert os.path.exists(path), (
        f"joint brightness pool missing: {path}\n"
        f"  -> run ai4_bg_brightness_pool_joint.py first")
    d = np.load(path, allow_pickle=True)
    block_means = np.asarray(d['block_means'], dtype=np.float64)
    block_stds = np.asarray(d['block_stds'], dtype=np.float64)
    assert block_means.size == block_stds.size, "pool mean/std count mismatch (broken pairing)"
    p5 = float(d['p5']); p95 = float(d['p95'])

    if compat15:
        # --compat15: 复刻 1.5 band 下限 = max(p5, MU_TARGET_FLOOR=40)
        lo = max(p5, MU_TARGET_FLOOR_COMPAT15)
    elif MEAN_BAND_LO_MODE == "p5":
        lo = p5 if MEAN_BAND_LO_CLAMP is None else max(p5, float(MEAN_BAND_LO_CLAMP))
    elif MEAN_BAND_LO_MODE == "value":
        assert MEAN_BAND_LO_CLAMP is not None, "MEAN_BAND_LO_MODE='value' requires MEAN_BAND_LO_CLAMP"
        lo = float(MEAN_BAND_LO_CLAMP)
    else:
        lo = float(MEAN_BAND_LO_MODE)

    # mean ∈ [lo, p95] 过滤, std 用【同一 boolean mask】取对应元素 -> 配对不打乱
    sel = (block_means >= lo) & (block_means <= p95)
    mean_band = block_means[sel]
    std_band = block_stds[sel]
    assert mean_band.size > 0, f"empty band [{lo},{p95}] in pool"
    assert mean_band.size == std_band.size, "band pairing broken after filter"
    return mean_band, std_band, lo, p95


# ---------------------------------------------------------------------------
# 1.5b' 软膝 (soft-knee) —— 纯确定性, 不碰任何 rng (单变量铁律)
# ---------------------------------------------------------------------------
def _soft_floor(x, k):
    """下界软膝: x>=k 原样; x<k 指数渐近到 0。
       C1 连续: 在 x=k 处 值 = k * exp(0) = k (连续)、导数 = exp((x-k)/k) 在 x=k 为
       1 (与上半段斜率 1 接合, 无折角)、单调递增、x->-inf 渐近 0 (永不直接砸 0)。
       => 暗区"渐渐"降到 0 而非一刀拍成同一个精确 0 = 杀掉死黑平板 tell。"""
    return np.where(x >= k, x, k * np.exp((x - k) / k))


def _soft_ceil(x, k, hi=255.0):
    """上界软膝: 对称镜像 _soft_floor。x<=hi-k 原样; 否则渐近到 hi (防纯白板)。
       C1 连续同理 (在 x=hi-k 处 值=hi-k、斜率=1、单调、x->+inf 渐近 hi)。"""
    return np.where(x <= hi - k, x, hi - k * np.exp(((hi - k) - x) / k))


# ============ 合成参数（经阶段4调优确定，与 full_generate.py / 1.5 逐字冻结） ============
# Weibull扰动
PERTURB_MIN = 0.90
PERTURB_MAX = 1.10

# 背景亮度平移 (名义保留; 1.5/1.5b 不再作绝对锚)
TARGET_BG_MEAN = 90

# 灰度相对定位
HIGHLIGHT_OFFSET = 2.5
SHADOW_OFFSET = 3.0
HIGHLIGHT_STD_RATIO = 1.0
SHADOW_STD_RATIO = 0.7

# Weibull形状参数下限
WEIBULL_C_FLOOR = 1.5

# 边界混合
BLEND_WIDTH = 10

# 降采样
DOWNSAMPLE_FACTOR_MIN = 1.0
DOWNSAMPLE_FACTOR_MAX = 1.5


# ============================================================================
# ★ STEP2 暗透镜渲染常量 (FROZEN, LOCKED 自 _poc_step2_shadow_v3.py)
#   绝不重标定 / 不朝 4.0 调 / 不对任何真实数据 (KLSG) 调。
#   仅 NEW shadow 用; OLD shadow 仍走 SHADOW_OFFSET=3.0 + assemble_image ramp。
# ============================================================================
DARK_OFFSET_SIGMA = 3.75   # NEW 暗透镜暗度 (替代 OLD 的 SHADOW_OFFSET=3.0; 仅 NEW 用)
DARK_FLOOR        = 12.0    # post-composite floor clamp (zero pure-0 shadow px)
TAPER_FAR_FRAC    = 0.55    # 沿程 taper 起淡分数
FEATHER_NEAR_PX   = 3.0
FEATHER_FAR_PX    = 18.0
FEATHER_SIDE_PX   = 11.0
# (SHADOW_STD_RATIO=0.7 复用上面已定义的同名常量; NEW dark layer 也用它)
# NEW shadow 几何 rng 基种子 (独立于主纹理 rng / mu_rng / reject_rng, 互不干扰;
#   per-image 用 base + i 派生 -> shadow 几何不消耗主 rng 一个 draw)。
STEP2_SHADOW_SEED = 20260604


# ---------------------------------------------------------------------------
# ★ STEP2 暗透镜 alpha 场 (LOCKED 自 POC v3 shadow_alpha_field; FIX-B 局部半宽 cap)
# ---------------------------------------------------------------------------
def step2_shadow_alpha_field(M_obj, M_shadow_only, theta_deg):
    """alpha (0=全背景, 1=全暗核), 各向异性 taper + 羽化, FIX-B 局部半宽 cap。"""
    h, w = M_obj.shape
    alpha = np.zeros((h, w), np.float64)
    if M_shadow_only.sum() == 0:
        return alpha

    th = np.radians(theta_deg)
    ux, uy = np.cos(th), np.sin(th)        # 投影方向 (远离声源)
    px, py = -uy, ux                       # 侧向

    ys_o, xs_o = np.where(M_obj > 0)
    cx, cy = xs_o.mean(), ys_o.mean()
    ys_s, xs_s = np.where(M_shadow_only > 0)
    t_s = (xs_s - cx) * ux + (ys_s - cy) * uy
    t_min, t_max = t_s.min(), t_s.max()
    span = max(t_max - t_min, 1.0)
    yy, xx = np.mgrid[0:h, 0:w]
    t_norm = ((xx - cx) * ux + (yy - cy) * uy - t_min) / span     # 0=近边,1=远端

    # ---- (a) 沿程 taper: 近边 alpha=1, TAPER_FAR_FRAC 之后线性淡出 ----
    far = TAPER_FAR_FRAC
    fade = (t_norm - far) / max(1e-6, (1.0 - far))
    taper = np.where(t_norm > far, 1.0 - np.clip(fade, 0, 1), 1.0)
    taper = np.clip(taper, 0.0, 1.0)

    # ---- (b) 各向异性羽化 + FIX-B 局部半宽 cap ----
    sh = (M_shadow_only > 0).astype(np.uint8)
    edt_in = ndimage.distance_transform_edt(sh).astype(np.float64)  # 到阴影外缘距离 (内部正)

    s_coord = (xx - cx) * px + (yy - cy) * py
    s_in = s_coord[sh > 0]
    s_min, s_max = s_in.min(), s_in.max()
    s_span = max(s_max - s_min, 1.0)
    s_norm = (s_coord - (s_min + s_max) / 2.0) / (s_span / 2.0)     # -1..1 居中

    fn, ff, fs = FEATHER_NEAR_PX, FEATHER_FAR_PX, FEATHER_SIDE_PX
    feather_along = fn + (ff - fn) * np.clip(t_norm, 0, 1)
    side_weight = np.clip(np.abs(s_norm), 0, 1)
    feather_r = feather_along * (1 - side_weight) + np.maximum(feather_along, fs) * side_weight
    feather_r = np.clip(feather_r, 1.0, None)

    # ★FIX B: 局部阴影半宽 cap
    max_feather = float(max(fn, ff, fs))
    rad = int(max(1, round(max_feather)))
    ksz = 2 * rad + 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    local_halfwidth = cv2.dilate(edt_in.astype(np.float32), se).astype(np.float64)
    local_halfwidth = np.maximum(local_halfwidth, 1.0)
    feather_eff = np.minimum(feather_r, local_halfwidth)
    feather_eff = np.clip(feather_eff, 1.0, None)

    edge_soft = np.clip(edt_in / feather_eff, 0.0, 1.0)            # 0 外缘, 1 内部深处

    alpha = taper * edge_soft
    alpha = np.where(sh > 0, alpha, 0.0)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.2)                   # 轻微去锯齿
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha = np.where(sh > 0, alpha, 0.0)
    return alpha


def step2_make_shadow_dark_layer(bg_layer, urm_assigned, tex_rng, offset_sigma):
    """NEW 暗层 = 背景驱动 Weibull 场, dark_mean=bg_mean-offset_sigma*bg_std,
       dark_std=bg_std*SHADOW_STD_RATIO, post-clamp >= DARK_FLOOR。
       LOCKED 自 POC v3 make_shadow_dark_layer。用独立 shadow tex rng (不动主 rng)。"""
    h, w = bg_layer.shape
    bg_mean = float(np.mean(bg_layer))
    bg_std  = max(float(np.std(bg_layer)), 5.0)
    dark_mean = max(bg_mean - offset_sigma * bg_std, DARK_FLOOR)
    dark_std  = bg_std * SHADOW_STD_RATIO
    dark = np.clip(generate_correlated_weibull_field(
        h, w, urm_assigned['psd_sh'], urm_assigned['wb_shadow'],
        tex_rng, dark_mean, dark_std), 0, 255)
    dark = np.maximum(dark, DARK_FLOOR)
    return dark


def step2_assemble_image_new(M_obj, M_shadow_full, highlight_layer, bg_layer,
                             shadow_dark_layer, shadow_alpha):
    """★STEP2 NEW 装配 (LOCKED 自 POC v3 assemble_faithful, shadow_mode='lens'):
      shadow_region = (M_shadow==1)&(M_obj==0) -> I = a*dark + (1-a)*bg (暗透镜)
      obj_and_shadow = (M_obj==1)&(M_shadow==1) -> HARD highlight (无 blend)
      obj_only       = (M_obj==1)&(M_shadow==0) -> BW=10 soft ramp
    FIX-C: shadow_region 在 round/cast 前 clamp >= DARK_FLOOR (zero pure-0)。
    *目标-边缘逻辑 (obj_and_shadow HARD / obj_only BW10) 与 assemble_image 字节同语义。*"""
    h, w = M_obj.shape
    dist_obj = distance_transform_edt(M_obj).astype(np.float64)
    w_obj = np.clip(dist_obj / BLEND_WIDTH, 0, 1)

    I = bg_layer.copy()

    M_shadow = (M_shadow_full > 0).astype(np.uint8)
    shadow_region = (M_shadow == 1) & (M_obj == 0)
    a = shadow_alpha
    I[shadow_region] = (a[shadow_region] * shadow_dark_layer[shadow_region]
                        + (1 - a[shadow_region]) * bg_layer[shadow_region])

    # ---- 目标层: 复刻 assemble_image 的 hard/soft 决策 ----
    obj_and_shadow = (M_obj == 1) & (M_shadow == 1)
    obj_only       = (M_obj == 1) & (M_shadow == 0)
    I[obj_and_shadow] = highlight_layer[obj_and_shadow]
    I[obj_only] = (w_obj[obj_only] * highlight_layer[obj_only]
                   + (1 - w_obj[obj_only]) * I[obj_only])

    # ---- FIX-C: shadow_region floor clamp (round/cast 前) ----
    if shadow_region.any():
        I[shadow_region] = np.maximum(I[shadow_region], DARK_FLOOR)

    I = np.clip(np.round(I), 0, 255).astype(np.uint8)
    n_pure0 = int((I[shadow_region] == 0).sum()) if shadow_region.any() else 0
    return I, n_pure0


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_urm_params(path):
    """加载干净的URM参数文件"""
    data = np.load(path, allow_pickle=True)
    n = len(data['urm_names'])
    params_list = []
    for i in range(n):
        entry = {
            'name': str(data['urm_names'][i]),
            'wb_highlight': data['wb_highlight'][i],
            'wb_shadow': data['wb_shadow'][i],
            'psd_hl': data[f'psd_hl_{i}'],
            'psd_sh': data[f'psd_sh_{i}'],
        }
        params_list.append(entry)
    return params_list


def load_bg_block_paths(bg_dir):
    """获取所有背景块文件路径"""
    return sorted(glob.glob(os.path.join(bg_dir, "bg_*.png")))


def load_mask_pair(tag, mask_dir):
    """加载一对掩膜（obj + shadow），转为0/1二值"""
    obj_path = os.path.join(mask_dir, f"{tag}_obj.png")
    shadow_path = os.path.join(mask_dir, f"{tag}_shadow.png")

    M_obj = cv2.imread(obj_path, cv2.IMREAD_GRAYSCALE)
    M_shadow = cv2.imread(shadow_path, cv2.IMREAD_GRAYSCALE)

    if M_obj is None or M_shadow is None:
        return None, None

    M_obj = (M_obj > 127).astype(np.uint8)
    M_shadow = (M_shadow > 127).astype(np.uint8)
    return M_obj, M_shadow


# ============================================================================
# ★ STEP3 (E2 ablation ④): per-pixel T offset map -> structure rides AI4 grain.
#   step3=off  -> 完全走原标量 highlight 路径 (回退分支, 一字不改; flat-T byte anchor)。
#   step3=on   -> 从 step3_arm_templates\{arm} 读 {tag}_obj.png + {tag}_T.npy:
#                 NEW M_obj = arm obj.png; 逐像素 hl mean map = min(bg_mean+bg_std*T,240);
#                 T<0 暗格掺 wb_shadow 颗粒 (独立 step3_dark_rng, 绝不动主 rng);
#                 ship shadow 用 MASKS.generate_shadow 从 NEW M_obj 生 (与 OLD/NEW 同路)。
#   单变量铁律: arm1 复用现成 step2_shadow (走 step3=off 原路); 最终arm2只改变T map。
#   背景/亮度池/Step2 暗透镜/downsample/主 rng/mu_rng/reject_rng 全同。
# ============================================================================
_STEP3_ARM_ROOT = r"PATH_TO_SYNTH_IMPROVED_WORKSPACE\data\outputs\step3_arm_templates"
STEP3_DARK_SEED = 70260606   # 独立 dark-cell blend rng 基种子 (与主 42 / mu 1234 / reject 4321 / shadow 20260604 并列)


def load_step3_template(tag, arm, arm_root=_STEP3_ARM_ROOT):
    """STEP3: 读 arm 模板 (NEW M_obj + 逐像素 T map)。
       返回 (M_obj uint8 0/1, T float32, canvas_h, canvas_w) 或 (None,None,None,None)。
       T: float32 offset map, M 外 = 0 (自然退化到 bg_mean); 范围约 [-0.69,+3.60]。"""
    obj_path = os.path.join(arm_root, arm, f"{tag}_obj.png")
    T_path = os.path.join(arm_root, arm, f"{tag}_T.npy")
    if not (os.path.exists(obj_path) and os.path.exists(T_path)):
        return None, None, None, None
    M_obj = cv2.imread(obj_path, cv2.IMREAD_GRAYSCALE)
    if M_obj is None:
        return None, None, None, None
    M_obj = (M_obj > 127).astype(np.uint8)
    T = np.load(T_path).astype(np.float64)
    assert T.shape == M_obj.shape, f"STEP3 T/obj shape mismatch for {tag}: T={T.shape} obj={M_obj.shape}"
    h, w = M_obj.shape
    return M_obj, T, h, w


# ---------------------------------------------------------------------------
# 灰度场生成
# ---------------------------------------------------------------------------

def generate_correlated_weibull_field(canvas_h, canvas_w, psd, wb_params, rng,
                                      target_mean, target_std):
    """
    生成具有空间相关性的灰度场。
    PSD着色（空间相关性）→ Weibull CDF变换（分布形状）→ 线性重映射（亮度定位）
    """
    C, loc, scale = wb_params
    C = max(C, WEIBULL_C_FLOOR)

    # PSD插值到画布尺寸
    psd_full = cv2.resize(psd.astype(np.float64), (canvas_w, canvas_h),
                           interpolation=cv2.INTER_LINEAR)
    psd_full = np.maximum(psd_full, 0)

    # 频域着色白噪声
    noise = rng.randn(canvas_h, canvas_w)
    NOISE = np.fft.fft2(noise)
    COLORED = NOISE * np.sqrt(psd_full)
    colored_noise = np.real(np.fft.ifft2(COLORED))

    # Weibull CDF变换
    flat = colored_noise.flatten()
    n_pixels = len(flat)
    ranks = np.argsort(np.argsort(flat))
    p = (ranks + 0.5) / n_pixels
    gray_values = weibull_min.ppf(p, C, loc=loc, scale=scale)

    # 线性重映射到目标均值和标准差
    wb_mean = np.mean(gray_values)
    wb_std = np.std(gray_values)
    if wb_std < 1e-6:
        wb_std = 1.0
    gray_values = (gray_values - wb_mean) / wb_std * target_std + target_mean

    gray_values = np.round(gray_values)
    gray_values = np.clip(gray_values, 0, 255)
    return gray_values.reshape(canvas_h, canvas_w)


# ---------------------------------------------------------------------------
# 背景获取
# ---------------------------------------------------------------------------

def get_background_crop(bg_paths, canvas_h, canvas_w, rng):
    """从背景池随机取一块，随机裁剪到画布尺寸"""
    idx = rng.randint(0, len(bg_paths))
    bg_block = cv2.imread(bg_paths[idx], cv2.IMREAD_GRAYSCALE)

    if bg_block is None:
        return np.full((canvas_h, canvas_w), 128, dtype=np.uint8)

    bh, bw = bg_block.shape
    if bh < canvas_h or bw < canvas_w:
        bg_block = cv2.resize(bg_block, (max(bw, canvas_w), max(bh, canvas_h)),
                               interpolation=cv2.INTER_LINEAR)
        bh, bw = bg_block.shape

    max_y = bh - canvas_h
    max_x = bw - canvas_w
    y0 = rng.randint(0, max_y + 1)
    x0 = rng.randint(0, max_x + 1)

    crop = bg_block[y0:y0+canvas_h, x0:x0+canvas_w]
    return crop.astype(np.float64)


# ---------------------------------------------------------------------------
# 层叠装配
# ---------------------------------------------------------------------------

def assemble_image(M_obj, M_shadow, highlight_layer, shadow_layer, bg_layer):
    """三层装配：背景→阴影（混合过渡）→目标（混合过渡，硬切阴影）"""
    dist_obj = distance_transform_edt(M_obj).astype(np.float64)
    dist_shadow = distance_transform_edt(M_shadow).astype(np.float64)

    w_obj = np.clip(dist_obj / BLEND_WIDTH, 0, 1)
    w_shadow = np.clip(dist_shadow / BLEND_WIDTH, 0, 1)

    I_synth = bg_layer.copy()

    # 阴影层（M_shadow=1 且 M_obj=0）
    shadow_region = (M_shadow == 1) & (M_obj == 0)
    I_synth[shadow_region] = (
        w_shadow[shadow_region] * shadow_layer[shadow_region] +
        (1 - w_shadow[shadow_region]) * bg_layer[shadow_region]
    )

    # 目标层（M_obj=1）
    obj_and_shadow = (M_obj == 1) & (M_shadow == 1)
    obj_only = (M_obj == 1) & (M_shadow == 0)

    # 目标-阴影：硬边界
    I_synth[obj_and_shadow] = highlight_layer[obj_and_shadow]
    # 目标-背景：混合过渡
    I_synth[obj_only] = (
        w_obj[obj_only] * highlight_layer[obj_only] +
        (1 - w_obj[obj_only]) * bg_layer[obj_only]
    )

    I_synth = np.round(I_synth)
    I_synth = np.clip(I_synth, 0, 255).astype(np.uint8)
    return I_synth


# ---------------------------------------------------------------------------
# 降采样
# ---------------------------------------------------------------------------

def azimuth_downsample(image, rng):
    """随机方向、随机因子的方位向降采样"""
    factor = rng.uniform(DOWNSAMPLE_FACTOR_MIN, DOWNSAMPLE_FACTOR_MAX)
    direction = rng.choice(['vertical', 'horizontal'])
    h, w = image.shape

    if direction == 'vertical':
        new_h = max(1, int(round(h / factor)))
        result = cv2.resize(image, (w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        new_w = max(1, int(round(w / factor)))
        result = cv2.resize(image, (new_w, h), interpolation=cv2.INTER_LINEAR)

    return result, direction, float(factor)


# ---------------------------------------------------------------------------
# URM分配与参数扰动
# ---------------------------------------------------------------------------

def assign_urm_and_perturb(urm_params, sample_idx, rng):
    """循环分配URM参数，C和scale施加±10%扰动，loc不变"""
    n_urm = len(urm_params)
    urm_idx = sample_idx % n_urm
    urm = urm_params[urm_idx]

    wb_hl = [
        urm['wb_highlight'][0] * rng.uniform(PERTURB_MIN, PERTURB_MAX),
        urm['wb_highlight'][1],
        urm['wb_highlight'][2] * rng.uniform(PERTURB_MIN, PERTURB_MAX)
    ]
    wb_sh = [
        urm['wb_shadow'][0] * rng.uniform(PERTURB_MIN, PERTURB_MAX),
        urm['wb_shadow'][1],
        urm['wb_shadow'][2] * rng.uniform(PERTURB_MIN, PERTURB_MAX)
    ]

    return {
        'urm_name': urm['name'],
        'wb_highlight': wb_hl,
        'wb_shadow': wb_sh,
        'psd_hl': urm['psd_hl'],
        'psd_sh': urm['psd_sh'],
    }


# ---------------------------------------------------------------------------
# 单张合成
# ---------------------------------------------------------------------------

def synthesize_single(meta, urm_assigned, bg_paths, rng, mu_target, sigma_target,
                      compat15=False, reject_rng=None, plain_b2=False,
                      shadow_mode="new", shadow_geom_rng=None, shadow_tex_rng=None,
                      step3=False, step3_arm=None, step3_dark_rng=None):
    """生成一张完整的合成声呐图像。
    (C/D) (mu_target,sigma_target) 由调用方用【独立第二 RandomState】从联合池
          一次 draw 出一个成对值后传入, 不触主 rng。
    compat15=True: 背景变换严格复刻 1.5 的纯平移表达式 (sigma_target 不参与),
          保证与 1.5 step1p5_denorm 产物逐字节一致 -> 证明纹理 rng 流/均值路径未动。
    1.5b' 新增 (compat15/plain_b2 下均 gated off):
      - 脏块重抽 (reject_rng): 第一次抽永远用主 rng (与 1.5b 字节一致), 仅当 crop
        exact-0 占比 > BLOCK_ZERO_REJECT 时用独立 reject_rng 重抽 -> 主 rng 流不变。
      - 软膝 (_soft_floor/_soft_ceil): 替换 bg 层硬切下界, 纯确定性不碰 rng。
    plain_b2=True: 软膝关 + 重抽关 (退化为 1.5b), 用于字节锚 2 (== 1.5b)。
    ★STEP2 shadow_mode:
      'old' = pre-Step2: 用 stage3 *_shadow.png + assemble_image ramp (3.0σ shadow_layer)
              -> 与 1.5b-prime 逐字节一致 (byte-anchor)。
      'new' = v3: 从 M_obj 算整剪影扫掠+远边起伏掩膜 (独立 shadow_geom_rng) +
              暗透镜 alpha 渲染 (3.75σ dark layer, 独立 shadow_tex_rng)。
              亮度线 (highlight/bg/shadow_layer 主 rng 消耗/downsample) 与 'old' 逐字节相同,
              唯一差异 = shadow_region 填充 + obj_and_shadow 边缘 -> 单变量。
      shadow_geom_rng / shadow_tex_rng = 独立 per-image RandomState, 绝不动主 rng。
    返回 (I_final, qc); qc 多带 bg_redrawn / bg_crop_zerofrac / shadow_* 备查。"""
    tag = meta['tag']
    canvas_h = meta['canvas_h']
    canvas_w = meta['canvas_w']

    # ---- STEP3 (off): 原始码路 = 读 stage3 obj+shadow PNG, T_map 未定义 (不进任何新分支) ----
    # ---- STEP3 (on):  读 arm 模板 NEW M_obj + 逐像素 T map; M_shadow 由 NEW path 从 M_obj 生 ----
    T_map = None
    if step3:
        M_obj, T_map, canvas_h, canvas_w = load_step3_template(tag, step3_arm)
        if M_obj is None:
            return None
        M_shadow = None   # step3 强制走 NEW shadow (从 M_obj 生); 不读 stage3 *_shadow.png
    else:
        M_obj, M_shadow = load_mask_pair(tag, MASK_DATA_DIR)
        if M_obj is None:
            return None

    # 获取背景并重映射
    # (C) 1.5b 唯一合成逻辑改动:
    #   1.5 (纯平移定均值): bg_layer = bg_layer + (mu_target - bg_raw_mean)
    #   1.5b (仿射定均值+std): bg_layer = (bg_layer - bg_raw_mean)*(sigma_target/bg_raw_std) + mu_target
    #   (mu_target,sigma_target) = 同一 AI4 块的成对值; sigma/std 放开 = 攻击"对比度异常一致"指纹。
    # ---- 第一次抽: 永远用主 rng (消耗 idx/y0/x0 与 1.5b 逐字一致, 保主 rng 流不变) ----
    bg_layer = get_background_crop(bg_paths, canvas_h, canvas_w, rng)
    bg_redrawn = 0                              # 重抽次数 (0 = 没触发)
    bg_crop_zerofrac0 = float((bg_layer == 0).mean())   # 第一次抽的 crop exact-0 占比 (备查)
    # ---- 1.5b' 脏块重抽: 仅非 compat15 且非 plain_b2; 用独立 reject_rng (绝不动主 rng) ----
    if (not compat15) and (not plain_b2) and reject_rng is not None:
        tries = 0
        best = bg_layer
        best_zf = (best == 0).mean()
        while (bg_layer == 0).mean() > BLOCK_ZERO_REJECT and tries < MAX_REDRAW:
            # 独立 reject_rng: 消耗的是 reject_rng 的 idx/y0/x0, 主纹理 rng 一个 draw 都不动
            bg_layer = get_background_crop(bg_paths, canvas_h, canvas_w, reject_rng)
            cur_zf = (bg_layer == 0).mean()
            if cur_zf < best_zf:
                best = bg_layer
                best_zf = cur_zf
            tries += 1
        if (bg_layer == 0).mean() > BLOCK_ZERO_REJECT:
            bg_layer = best     # MAX 次仍超 -> 取这些次里最干净的那块
        bg_redrawn = tries
    bg_crop_zerofrac = float((bg_layer == 0).mean())    # 最终采用的 crop exact-0 占比

    bg_raw_mean = np.mean(bg_layer)
    bg_raw_std = max(np.std(bg_layer), 1e-6)
    if compat15:
        # --compat15: 逐字复刻 1.5 表达式 (浮点运算顺序也必须一致 -> 字节复现)。
        #   sigma_target 在此模式下不参与 (退化为恒等); 这里仅为 QC 记一个名义值。
        bg_layer = bg_layer + (mu_target - bg_raw_mean)
        sigma_target = bg_raw_std
    else:
        bg_layer = (bg_layer - bg_raw_mean) * (sigma_target / bg_raw_std) + mu_target
    # ---- 1.5b' 软膝: 替换硬切下界 (纯确定性, 不碰 rng); compat15/plain_b2 仍走 1.5 硬切 ----
    if compat15 or plain_b2:
        bg_layer = np.clip(bg_layer, 0, 255)              # 字节锚: 保 1.5 / 1.5b 硬切, 不动
    else:
        bg_layer = _soft_ceil(_soft_floor(bg_layer, KNEE_K), KNEE_K)
        bg_layer = np.clip(bg_layer, 0, 255)              # 兜底(渐近本在界内、量化后无负, 此 clip 实际不咬)

    bg_mean = np.mean(bg_layer)
    bg_std = max(np.std(bg_layer), 5.0)

    # 灰度目标值 (HL/SH 推导逐字不动; bg_std per-image 变化自然传导 = 对比度放开的预期效果)
    hl_target_mean = min(bg_mean + HIGHLIGHT_OFFSET * bg_std, 240)
    hl_target_std = bg_std * HIGHLIGHT_STD_RATIO
    sh_target_mean = max(bg_mean - SHADOW_OFFSET * bg_std, 10)
    sh_target_std = bg_std * SHADOW_STD_RATIO

    if not step3:
        # ===== STEP3 OFF = 原始标量码路 (回退分支, 一字不改; flat-T byte anchor) =====
        # 生成Weibull灰度场（不透明，不掺背景）
        highlight_layer = np.clip(generate_correlated_weibull_field(
            canvas_h, canvas_w, urm_assigned['psd_hl'],
            urm_assigned['wb_highlight'], rng,
            hl_target_mean, hl_target_std
        ), 0, 255)
    else:
        # ===== STEP3 ON = 逐像素 T-map 路径 (新分支, 仅 step3 进入) =====
        # 结构骑在 AI4 真实颗粒上: AI4 颗粒(PSD/Weibull/主 rng 消耗)与标量路径【逐字相同】——
        #   把整块的【标量】均值换成【逐像素】均值 map, 颗粒纹理完全不动。
        #
        # ★关键 (修过的坑): generate_correlated_weibull_field 末段自带 round+clip(0,255)。
        #   若用 target_mean=0 抽"零均值颗粒"再加 map, 内部 clip 会把负半轴颗粒砸成 0 ->
        #   纹理被毁、且 uniform-T 退不回标量 (code-path delta 爆掉, 实测 maxΔ=240)。
        #   正确做法: 用【与标量路径完全相同的参数】(hl_target_mean, hl_target_std) 抽 grain
        #   (-> 与标量路径逐字节相同的 highlight_layer, 含其内部 round+clip), 再逐像素叠加
        #   【相对标量参考的偏移】 (hl_mean_map - hl_target_mean)。
        #   uniform T=HIGHLIGHT_OFFSET -> hl_mean_map == hl_target_mean -> 偏移=0 -> 字节退回标量
        #   (单变量铁律: arm2 vs arm1 唯一差异 = T 结构, 非 code-path artifact)。
        highlight_layer = np.clip(generate_correlated_weibull_field(
            canvas_h, canvas_w, urm_assigned['psd_hl'],
            urm_assigned['wb_highlight'], rng,
            hl_target_mean, hl_target_std
        ), 0, 255)
        T_pos = np.clip(T_map, 0, None)                      # 只有 T>=0 (亮结构) 骑 highlight 均值
        hl_mean_map = np.minimum(bg_mean + bg_std * T_pos, 240.0)  # 逐像素绝对目标均值 (cap 240)
        # 仅 M_obj 内按 (逐像素目标 - 标量参考) 平移; M 外偏移≈0 由下方 cell-mask 之外不动保证
        hl_shift = (hl_mean_map - hl_target_mean)
        in_obj = (M_obj == 1)
        highlight_layer[in_obj] = np.clip(highlight_layer[in_obj] + hl_shift[in_obj], 0, 255)

        # ---- 暗格 (T<0): 掺 wb_shadow 颗粒, 用独立 step3_dark_rng (绝不消耗主 rng) ----
        #   T<0 越深 -> 越偏 shadow 颗粒; T≈0 -> 保 highlight/bg; 仅 M_obj 内生效。
        if step3_dark_rng is not None and (T_map < 0).any():
            grain_dark = generate_correlated_weibull_field(
                canvas_h, canvas_w, urm_assigned['psd_sh'],
                urm_assigned['wb_shadow'], step3_dark_rng,
                sh_target_mean, sh_target_std)
            T_dark_ref = float(SHADOW_OFFSET)                # |T| 达到此值 -> 全 shadow 颗粒
            w_dark = np.clip(-T_map / T_dark_ref, 0.0, 1.0)
            cell = (M_obj == 1) & (T_map < 0)
            highlight_layer[cell] = ((1.0 - w_dark[cell]) * highlight_layer[cell]
                                     + w_dark[cell] * grain_dark[cell])
            highlight_layer = np.clip(highlight_layer, 0, 255)

    # shadow_layer (3.0σ) 在两模式下都生成 -> 主 rng 消耗逐字节相同 (单变量铁律)。
    # OLD 模式它喂进 assemble_image; NEW 模式它不参与渲染 (NEW 用 3.75σ dark layer),
    #   但仍生成以保主 rng 流不变 = OLD/NEW 单变量。
    shadow_layer = np.clip(generate_correlated_weibull_field(
        canvas_h, canvas_w, urm_assigned['psd_sh'],
        urm_assigned['wb_shadow'], rng,
        sh_target_mean, sh_target_std
    ), 0, 255)

    # ---- 层叠装配 (shadow_mode 选码路; 主 rng 在此之后只剩 downsample, 两模式相同) ----
    n_pure0_shadow = -1   # 仅 NEW 模式有意义 (floor clamp 后阴影纯0像素计数)
    shadow_theta = float('nan')
    shadow_L_px = float('nan')
    # STEP3 强制走 NEW shadow (从 arm NEW M_obj 生; arm 模板无 stage3 *_shadow.png)。
    assert not (step3 and shadow_mode == "old"), "STEP3 requires shadow_mode='new' (arm M_obj has no stage3 shadow PNG)"
    if shadow_mode == "old":
        # OLD = pre-Step2: stage3 *_shadow.png + assemble_image ramp -> byte-anchor。
        I_synth = assemble_image(M_obj, M_shadow, highlight_layer, shadow_layer, bg_layer)
        M_shadow_used = M_shadow
        shadow_dark_for_qc = shadow_layer
    else:
        # NEW = v3: 整剪影扫掠+远边起伏 (独立 shadow_geom_rng) + 暗透镜 (3.75σ, 独立 shadow_tex_rng)
        M_shadow_new, _sh_params = MASKS.generate_shadow(M_obj, shadow_geom_rng)
        shadow_theta = float(_sh_params.get('theta', float('nan')))
        shadow_L_px = float(_sh_params.get('L_px', float('nan')))
        M_shadow_only_new = ((M_shadow_new > 0) & (M_obj == 0)).astype(np.uint8)
        shadow_alpha = step2_shadow_alpha_field(M_obj, M_shadow_only_new, shadow_theta)
        dark_layer = step2_make_shadow_dark_layer(bg_layer, urm_assigned, shadow_tex_rng,
                                                  DARK_OFFSET_SIGMA)
        I_synth, n_pure0_shadow = step2_assemble_image_new(
            M_obj, M_shadow_new, highlight_layer, bg_layer, dark_layer, shadow_alpha)
        M_shadow_used = (M_shadow_new > 0).astype(np.uint8)
        shadow_dark_for_qc = dark_layer

    # ---- QC 统计 (在降采样前, 用已在 scope 的层/掩膜算, 不消耗任何 rng) ----
    hl_region_mean = float(np.mean(highlight_layer[M_obj == 1])) if np.sum(M_obj) > 0 else float('nan')
    sh_only = (M_shadow_used == 1) & (M_obj == 0)
    sh_region_mean = float(np.mean(shadow_dark_for_qc[sh_only])) if np.sum(sh_only) > 0 else float('nan')
    qc = {
        'mu_target': float(mu_target),
        'sigma_target': float(sigma_target),
        'bg_raw_mean': float(bg_raw_mean),
        'bg_raw_std': float(bg_raw_std),
        'bg_shifted_mean': float(bg_mean),
        'bg_shifted_std': float(bg_std),   # = 仿射后 bg_std (per-image)
        'bg_std': float(bg_std),
        'hl_region_mean': hl_region_mean,
        'sh_region_mean': sh_region_mean,
        'bg_redrawn': int(bg_redrawn),               # 1.5b': 重抽次数 (0=没触发)
        'bg_crop_zerofrac': float(bg_crop_zerofrac),  # 1.5b': 最终采用 crop 的 exact-0 占比
        'bg_crop_zerofrac0': float(bg_crop_zerofrac0),# 1.5b': 第一次抽 crop 的 exact-0 占比 (备查)
        'shadow_mode': shadow_mode,
        'shadow_theta': shadow_theta,
        'shadow_L_px': shadow_L_px,
        'shadow_n_pure0': int(n_pure0_shadow),       # NEW: floor clamp 后阴影纯0像素数 (应=0)
        'shadow_only_px': int(sh_only.sum()),
    }

    # 降采样
    I_final, ds_dir, ds_factor = azimuth_downsample(I_synth, rng)

    return I_final, qc


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _bind_output_dirs(mode, shadow_mode="new", step3=False, step3_arm="arm2"):
    """据 (mode, shadow_mode, step3) 绑定模块级 OUTPUT_*。
       step3=True  -> step2_shadow\_step3_{arm}\{stage4_trial|stage5_dataset}
         (STEP3 arm 产物; 绝不覆盖冻结的 step2_shadow\stage5_dataset)。
       shadow_mode='new' -> step2_shadow\{stage4_trial|stage5_dataset} (Step2 产物);
       shadow_mode='old' -> step2_shadow\_byte_anchor_old\{stage4_trial|stage5_dataset}
         (byte-anchor scratch; *绝不* 覆盖冻结的 1.5b-prime step1p5b2_denorm\stage5_dataset)。
       NEW/OLD/STEP3 都落 step2_shadow 子树 -> 冻结产物永不被本脚本覆写。"""
    global OUTPUT_BASE, OUTPUT_PLANE_DIR, OUTPUT_SHIP_DIR, OUTPUT_ALL_DIR, OUTPUT_STATS_DIR
    sub = "stage4_trial" if mode == "trial" else "stage5_dataset"
    if step3:
        root = os.path.join(_STEP2_ROOT, f"_step3_{step3_arm}")
    elif shadow_mode == "old":
        root = os.path.join(_STEP2_ROOT, "_byte_anchor_old")
    else:
        root = _STEP2_ROOT
    OUTPUT_BASE = os.path.join(root, sub)
    OUTPUT_PLANE_DIR = os.path.join(OUTPUT_BASE, "airplane")
    OUTPUT_SHIP_DIR = os.path.join(OUTPUT_BASE, "ship")
    OUTPUT_ALL_DIR = os.path.join(OUTPUT_BASE, "all")
    OUTPUT_STATS_DIR = os.path.join(OUTPUT_BASE, "stats")


def main(mode="full", trial_count=15, compat15=False, plain_b2=False,
         shadow_mode="new", max_images=None, step3=False, step3_arm="arm2"):
    if step3:
        # STEP3 强制 NEW shadow (arm M_obj 无 stage3 *_shadow.png)。compat15/plain_b2 不兼容。
        assert not (compat15 or plain_b2), "STEP3 incompatible with --compat15/--plain_b2"
        shadow_mode = "new"
    _bind_output_dirs(mode, shadow_mode=shadow_mode, step3=step3, step3_arm=step3_arm)
    _assert_paths()
    # 创建输出目录
    for d in [OUTPUT_PLANE_DIR, OUTPUT_SHIP_DIR, OUTPUT_ALL_DIR, OUTPUT_STATS_DIR]:
        os.makedirs(d, exist_ok=True)

    _modetag = '  [--compat15]' if compat15 else ('  [--plain_b2]' if plain_b2 else '  [soft-knee+reject ON]')
    print(f"{'='*60}")
    print(f"STEP 2 SHADOW (shadow_mode={shadow_mode.upper()}) on 1.5b' base{_modetag}: "
          f"{'TRIAL ('+str(trial_count)+')' if mode=='trial' else 'FULL Dataset'} Generation")
    print(f"  output -> {OUTPUT_BASE}")
    if shadow_mode == "new":
        print(f"  [NEW shadow] v3 whole-silhouette sweep + far-edge undulation + dark-lens")
        print(f"  DARK_OFFSET_SIGMA={DARK_OFFSET_SIGMA}  DARK_FLOOR={DARK_FLOOR}  "
              f"TAPER_FAR_FRAC={TAPER_FAR_FRAC}  FEATHER(near/far/side)="
              f"{FEATHER_NEAR_PX}/{FEATHER_FAR_PX}/{FEATHER_SIDE_PX}  STEP2_SHADOW_SEED={STEP2_SHADOW_SEED}")
        print(f"  length: lognormal(median={MASKS.STEP2_LENGTH_RATIO_MEDIAN},sigma={MASKS.STEP2_LENGTH_RATIO_SIGMA}) "
              f"clip{MASKS.STEP2_LENGTH_RATIO_CLIP}; undul A1/A2={MASKS.STEP2_UNDULATION_AMP_A1}/{MASKS.STEP2_UNDULATION_AMP_A2} "
              f"f1/f2mult={MASKS.STEP2_UNDULATION_F1}/{MASKS.STEP2_UNDULATION_F2_MULT}")
    else:
        print(f"  [OLD shadow] BYTE-ANCHOR: pre-Step2 *_shadow.png + assemble_image ramp "
              f"(must reproduce 1.5b-prime byte-exact)")
    if max_images is not None:
        print(f"  [max_images={max_images}] (subset, full-mode ordering preserved)")
    if not (compat15 or plain_b2):
        print(f"  KNEE_K={KNEE_K}  BLOCK_ZERO_REJECT={BLOCK_ZERO_REJECT}  REJECT_SEED={REJECT_SEED}  MAX_REDRAW={MAX_REDRAW}")
    print(f"{'='*60}")

    # 加载输入
    print("[Loading] URM parameters...")
    urm_params = load_urm_params(URM_PARAMS_PATH)
    print(f"  {len(urm_params)} clean URM entries")

    print("[Loading] Mask metadata...")
    with open(MASK_META_PATH, 'r') as f:
        all_meta = json.load(f)
    print(f"  {len(all_meta)} mask entries")

    print("[Loading] Background blocks...")
    bg_paths = load_bg_block_paths(BG_BLOCKS_DIR)
    print(f"  {len(bg_paths)} blocks")

    # (B) 读 AI4 天然 JOINT 池
    print("[Loading] AI4 JOINT (mean,std) pool (denorm_b target source)...")
    mean_band, std_band, mu_lo, mu_p95 = _load_brightness_pool(POOL_PATH, compat15=compat15)
    print(f"  sampling band: mean in [lo,p95]=[{mu_lo:.2f},{mu_p95:.2f}], paired n={mean_band.size}")
    print(f"  mean_band mean/std = {np.mean(mean_band):.2f}/{np.std(mean_band):.2f}")
    if compat15:
        print(f"  [--compat15] mu via 1.5 mean-only choice; sigma_target := bg_raw_std (affine = identity)")
    else:
        print(f"  std_band  mean/std = {np.mean(std_band):.2f}/{np.std(std_band):.2f}  (1.5b: std放开)")

    print(f"\n[Parameters]")
    print(f"  HL offset: {HIGHLIGHT_OFFSET}, SH offset: {SHADOW_OFFSET}")
    print(f"  HL std ratio: {HIGHLIGHT_STD_RATIO}, SH std ratio: {SHADOW_STD_RATIO}")
    print(f"  Weibull C floor: {WEIBULL_C_FLOOR}, Blend width: {BLEND_WIDTH}")
    print(f"  Downsample: [{DOWNSAMPLE_FACTOR_MIN}, {DOWNSAMPLE_FACTOR_MAX}]")
    print(f"  mean-band lo mode: {MEAN_BAND_LO_MODE} (clamp={MEAN_BAND_LO_CLAMP}); compat15={compat15}")

    # ---- STEP3: arm 模板只覆盖一部分 tag (ship_0091...) -> 把 all_meta 过滤到 arm 有的 tag。
    #   过滤后保持 all_meta 的原始顺序 (mask_metadata.json 顺序 = full-mode 顺序), 用 arm 模板
    #   自带的 canvas_h/canvas_w (synthesize_single 在 step3 下从 T.npy/obj.png 重新取 canvas)。
    if step3:
        arm_meta_path = os.path.join(_STEP3_ARM_ROOT, step3_arm, "mask_metadata.json")
        with open(arm_meta_path, 'r') as f:
            arm_meta_list = json.load(f)
        arm_tags = set(m['tag'] for m in arm_meta_list)
        all_meta = [m for m in all_meta if m['tag'] in arm_tags]
        print(f"  [STEP3 {step3_arm}] filtered to {len(all_meta)} tags present in arm template")

    # ---- 选取要处理的 meta 列表 (与 1.5 逐字相同的 trial 选择, 保 RNG 流一致) ----
    if mode == "trial":
        sel_rng = np.random.RandomState(42)
        airplane_meta = [m for m in all_meta if m['label'] == 'airplane']
        ship_meta = [m for m in all_meta if m['label'] == 'ship']
        n_plane_trial = min(trial_count // 3, len(airplane_meta))
        n_ship_trial = trial_count - n_plane_trial
        plane_sample = list(sel_rng.choice(len(airplane_meta), n_plane_trial, replace=False)) if len(airplane_meta) else []
        ship_sample = list(sel_rng.choice(len(ship_meta), n_ship_trial, replace=False)) if len(ship_meta) else []
        process_meta = [airplane_meta[i] for i in plane_sample] + [ship_meta[i] for i in ship_sample]
        sel_rng.shuffle(process_meta)
    else:
        process_meta = all_meta

    # ★ max_images: 截断到前 N 张 (full-mode 顺序保持 -> 主/mu/reject rng 流与全量逐字一致,
    #   仅生成前 N 张; byte-anchor 用它只跑 >=10 张验 OLD==1.5b-prime 而不跑满 490)。
    if max_images is not None:
        process_meta = process_meta[:max_images]

    # CSV标签表头 (1.5b 基础上加 1.5b' 的 bg_redrawn / bg_crop_zerofrac 两列)
    csv_header = ['filename', 'label', 'variant', 'damage', 'stratify_key',
                  'source_silhouette', 'canvas_w', 'canvas_h',
                  'final_w', 'final_h', 'mu_target', 'sigma_target',
                  'img_mean', 'img_std', 'clip0_frac', 'clip255_frac',
                  'bg_redrawn', 'bg_crop_zerofrac']

    # 标签记录
    plane_rows = []
    ship_rows = []
    all_rows = []
    qc_records = []

    rng = np.random.RandomState(42)                 # 主纹理 rng (与 Step1/1.5 一致)
    mu_rng = np.random.RandomState(MU_SAMPLE_SEED)   # (C/D) 独立第二 rng, 仅抽 (mu,sigma) 对
    reject_rng = np.random.RandomState(REJECT_SEED)  # (1.5b') 独立第三 rng, 仅脏块重抽用 (不动主 rng)
    n_success = 0
    n_fail = 0
    n_planes = 0
    n_ships = 0

    print(f"\n{'='*60}")
    print(f"[Generating {len(process_meta)} images...]")

    for i, meta in enumerate(process_meta):
        tag = meta['tag']
        label = meta['label']

        # 分配URM参数 (消耗主 rng, 顺序与 Step1/1.5 一致)
        urm_assigned = assign_urm_and_perturb(urm_params, i, rng)

        # (C/D) 从联合池抽 (mu_target, sigma_target) (独立 mu_rng, 不扰主 rng)
        if compat15:
            # --compat15: 严格复刻 1.5 = 用 mu_rng.choice(mean_band) 一次 draw 抽 mu;
            #             sigma_target 不参与 (synthesize_single 在 compat15 下退化为 1.5 纯平移)。
            #             mu_rng 调用形式 (.choice) 与 1.5 逐字一致 -> 抽样序列与 1.5 同。
            mu_target = float(mu_rng.choice(mean_band))
            sigma_target = -1.0   # 占位 (compat15 下 synthesize_single 会用 bg_raw_std 覆盖)
        else:
            # 1.5b: 一次 draw 出一个成对索引 -> 同时决定 mu 和 sigma (保持配对)
            pair_idx = mu_rng.randint(0, mean_band.size)
            mu_target = float(mean_band[pair_idx])
            sigma_target = float(std_band[pair_idx])

        # ★STEP2 NEW: per-image 独立 shadow rng (绝不动主纹理 rng / mu_rng / reject_rng)。
        #   geom_rng 决定方向/长度/起伏相位; tex_rng 生成 3.75σ 暗层。两者从
        #   STEP2_SHADOW_SEED + i 派生 -> 可复现且与图像位置绑定; OLD 模式不创建/不使用。
        if shadow_mode == "new":
            shadow_geom_rng = np.random.RandomState(STEP2_SHADOW_SEED + i)
            shadow_tex_rng = np.random.RandomState(STEP2_SHADOW_SEED + 100000 + i)
        else:
            shadow_geom_rng = None
            shadow_tex_rng = None

        # ★STEP3: per-image 独立 dark-cell blend rng (绝不动主纹理 rng / mu_rng / reject_rng /
        #   shadow_geom/tex_rng)。从 STEP3_DARK_SEED + i 派生 -> 可复现且与图像位置绑定。
        step3_dark_rng = np.random.RandomState(STEP3_DARK_SEED + i) if step3 else None

        # 合成
        I_final, qc = synthesize_single(meta, urm_assigned, bg_paths, rng,
                                        mu_target, sigma_target, compat15=compat15,
                                        reject_rng=reject_rng, plain_b2=plain_b2,
                                        shadow_mode=shadow_mode,
                                        shadow_geom_rng=shadow_geom_rng,
                                        shadow_tex_rng=shadow_tex_rng,
                                        step3=step3, step3_arm=step3_arm,
                                        step3_dark_rng=step3_dark_rng)

        if I_final is None:
            n_fail += 1
            print(f"  [{i+1}/{len(process_meta)}] {tag} - FAILED")
            continue

        n_success += 1
        final_h, final_w = I_final.shape

        # ---- per-image 亮度 QC (从最终图算, 只读) ----
        img_mean = float(np.mean(I_final))
        img_std = float(np.std(I_final))
        npx = I_final.size
        clip0_frac = float(np.sum(I_final == 0)) / npx
        clip255_frac = float(np.sum(I_final == 255)) / npx
        qc_records.append({
            'filename': f"{tag}.png", 'label': label,
            'mu_target': qc['mu_target'], 'sigma_target': qc['sigma_target'],
            'bg_shifted_mean': qc['bg_shifted_mean'], 'bg_shifted_std': qc['bg_shifted_std'],
            'img_mean': img_mean, 'img_std': img_std,
            'clip0_frac': clip0_frac, 'clip255_frac': clip255_frac,
            'hl_region_mean': qc['hl_region_mean'], 'sh_region_mean': qc['sh_region_mean'],
            'bg_redrawn': qc['bg_redrawn'], 'bg_crop_zerofrac': qc['bg_crop_zerofrac'],
            'bg_crop_zerofrac0': qc['bg_crop_zerofrac0'],
        })

        # 文件名
        filename = f"{tag}.png"

        # 保存到类别文件夹
        if label == 'airplane':
            cv2.imwrite(os.path.join(OUTPUT_PLANE_DIR, filename), I_final)
            n_planes += 1
        else:
            cv2.imwrite(os.path.join(OUTPUT_SHIP_DIR, filename), I_final)
            n_ships += 1

        # 保存到all文件夹
        cv2.imwrite(os.path.join(OUTPUT_ALL_DIR, filename), I_final)

        # 标签记录
        row = {
            'filename': filename,
            'label': label,
            'variant': meta['variant'],
            'damage': meta['damage'],
            'stratify_key': meta.get('stratify_key', f"{label}_{meta['damage']}"),
            'source_silhouette': meta['source'],
            'canvas_w': meta['canvas_w'],
            'canvas_h': meta['canvas_h'],
            'final_w': final_w,
            'final_h': final_h,
            'mu_target': round(qc['mu_target'], 3),
            'sigma_target': round(qc['sigma_target'], 3),
            'img_mean': round(img_mean, 3),
            'img_std': round(img_std, 3),
            'clip0_frac': round(clip0_frac, 6),
            'clip255_frac': round(clip255_frac, 6),
            'bg_redrawn': qc['bg_redrawn'],
            'bg_crop_zerofrac': round(qc['bg_crop_zerofrac'], 6),
        }

        all_rows.append(row)
        if label == 'airplane':
            plane_rows.append(row)
        else:
            ship_rows.append(row)

        # 进度
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(process_meta)}] Generated ({n_planes} airplane + {n_ships} ship)")

    print(f"\n  Done: {n_success} success, {n_fail} failed")

    # ---- 保存CSV标签 ----
    print(f"\n[Saving labels...]")

    def write_csv(filepath, rows, header):
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)

    plane_csv = os.path.join(OUTPUT_PLANE_DIR, "airplane_labels.csv")
    ship_csv = os.path.join(OUTPUT_SHIP_DIR, "ship_labels.csv")
    all_csv = os.path.join(OUTPUT_ALL_DIR, "dataset_labels.csv")

    write_csv(plane_csv, plane_rows, csv_header)
    write_csv(ship_csv, ship_rows, csv_header)
    write_csv(all_csv, all_rows, csv_header)

    print(f"  {plane_csv} ({len(plane_rows)} rows)")
    print(f"  {ship_csv} ({len(ship_rows)} rows)")
    print(f"  {all_csv} ({len(all_rows)} rows)")

    # ---- 自检 ----
    print(f"\n[Self-Check]")
    checks = []

    checks.append(f"[{'PASS' if n_success == len(process_meta) else 'WARN'}] "
                   f"Generated: {n_success}/{len(process_meta)}")

    # 标签分布
    stratify_dist = {}
    for row in all_rows:
        k = row['stratify_key']
        stratify_dist[k] = stratify_dist.get(k, 0) + 1
    checks.append(f"[INFO] Stratify distribution: {stratify_dist}")

    # 文件数一致性
    plane_files = len(glob.glob(os.path.join(OUTPUT_PLANE_DIR, "*.png")))
    ship_files = len(glob.glob(os.path.join(OUTPUT_SHIP_DIR, "*.png")))
    all_files = len(glob.glob(os.path.join(OUTPUT_ALL_DIR, "*.png")))
    checks.append(f"[{'PASS' if plane_files == len(plane_rows) else 'FAIL'}] "
                   f"Plane folder files: {plane_files} == CSV rows: {len(plane_rows)}")
    checks.append(f"[{'PASS' if ship_files == len(ship_rows) else 'FAIL'}] "
                   f"Ship folder files: {ship_files} == CSV rows: {len(ship_rows)}")
    checks.append(f"[{'PASS' if all_files == len(all_rows) else 'FAIL'}] "
                   f"All folder files: {all_files} == CSV rows: {len(all_rows)}")

    # ---- ★ 亮度+对比度去归一化 QC (1.5b 核心验证) ----
    img_means = np.array([r['img_mean'] for r in qc_records])
    img_stds = np.array([r['img_std'] for r in qc_records])
    bg_shifted_stds = np.array([r['bg_shifted_std'] for r in qc_records])
    mu_targets = np.array([r['mu_target'] for r in qc_records])
    sigma_targets = np.array([r['sigma_target'] for r in qc_records])
    clip0 = np.array([r['clip0_frac'] for r in qc_records])
    clip255 = np.array([r['clip255_frac'] for r in qc_records])
    img_mean_spread = float(np.std(img_means))    # ★ per-image 亮度 std
    img_std_spread = float(np.std(img_stds))       # ★ NEW: per-image 对比度 spread
    bg_shifted_std_spread = float(np.std(bg_shifted_stds))
    # HL > BG > SH 层次
    order_ok = sum(1 for r in qc_records
                   if r['hl_region_mean'] > r['bg_shifted_mean'] > r['sh_region_mean'])
    checks.append(f"[INFO] per-image img-mean spread (std across images) = {img_mean_spread:.2f} "
                  f"(Step1 ~3.8; 1.5 ~21.6; AI4 source ~35; grow toward source, NOT pin to KLSG 30)")
    checks.append(f"[INFO] per-image img-STD spread (NEW 1.5b) = {img_std_spread:.2f} "
                  f"(1.5 ~0 [std被统一]; 1.5b should be clearly >0)")
    checks.append(f"[INFO] bg_shifted_std spread = {bg_shifted_std_spread:.2f}")
    checks.append(f"[INFO] mu_target sampled mean/std = {np.mean(mu_targets):.2f}/{np.std(mu_targets):.2f}")
    checks.append(f"[INFO] sigma_target sampled mean/std = {np.mean(sigma_targets):.2f}/{np.std(sigma_targets):.2f}")
    checks.append(f"[INFO] img-mean range = [{img_means.min():.1f}, {img_means.max():.1f}]")
    checks.append(f"[{'PASS' if clip0.max() < 0.05 else 'WARN'}] "
                  f"clip-to-0 frac: max={clip0.max():.4f}, mean={clip0.mean():.4f} (expect <= Step1 baseline ~1.8%)")
    checks.append(f"[{'PASS' if clip255.max() < 0.05 else 'WARN'}] "
                  f"clip-to-255 frac: max={clip255.max():.4f}, mean={clip255.mean():.4f}")
    checks.append(f"[{'PASS' if order_ok == len(qc_records) else 'WARN'}] "
                  f"HL > BG > SH ordering: {order_ok}/{len(qc_records)}")

    # ---- ★ 1.5b' 脏块重抽 QC ----
    redrawn_n = sum(1 for r in qc_records if r['bg_redrawn'] > 0)
    crop_zf0 = np.array([r['bg_crop_zerofrac0'] for r in qc_records])
    crop_zf = np.array([r['bg_crop_zerofrac'] for r in qc_records])
    checks.append(f"[INFO] (1.5b') dirty-block redraw: {redrawn_n}/{len(qc_records)} images triggered "
                  f"(BLOCK_ZERO_REJECT={BLOCK_ZERO_REJECT}, indep reject_rng seed={REJECT_SEED})")
    checks.append(f"[INFO] (1.5b') bg-crop exact-0 frac: first-draw max={crop_zf0.max():.4f} "
                  f"-> final(after redraw) max={crop_zf.max():.4f}")
    checks.append(f"[INFO] (1.5b') KNEE_K={KNEE_K} (soft-knee replaces hard lower clip; pure-deterministic, no rng)")

    all_pass = all('FAIL' not in c and 'WARN' not in c for c in checks)
    checks.append(f"\n{'='*40}")
    checks.append(f"SELF-CHECK RESULT: {'ALL PASSED' if all_pass else 'ISSUES DETECTED'}")

    for line in checks:
        print(f"  {line}")

    # ---- 生成亮度+对比度对照 montage ----
    try:
        n_m = min(15, len(qc_records))
        if n_m > 0:
            cols = 5
            rows_m = (n_m + cols - 1) // cols
            cell = 256
            gap = 4
            canvas = np.full((rows_m * (cell + gap) - gap, cols * (cell + gap) - gap),
                             200, dtype=np.uint8)
            sel = qc_records[:n_m]
            for k, r in enumerate(sel):
                im = cv2.imread(os.path.join(OUTPUT_ALL_DIR, r['filename']), cv2.IMREAD_GRAYSCALE)
                if im is None:
                    continue
                im = cv2.resize(im, (cell, cell), interpolation=cv2.INTER_AREA)
                rr, cc = k // cols, k % cols
                y, x = rr * (cell + gap), cc * (cell + gap)
                canvas[y:y+cell, x:x+cell] = im
                cv2.putText(canvas, f"mu{r['mu_target']:.0f} si{r['sigma_target']:.0f}",
                            (x+4, y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,), 1)
                cv2.putText(canvas, f"m{r['img_mean']:.0f} s{r['img_std']:.0f}",
                            (x+4, y+38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,), 1)
            montage_path = os.path.join(OUTPUT_STATS_DIR, "brightness_montage.png")
            cv2.imwrite(montage_path, canvas)
            print(f"  [montage] {montage_path}")
    except Exception as e:
        print(f"  [montage] skipped: {e}")

    # ---- 保存报告 ----
    report_name = "stage4_trial_report.txt" if mode == "trial" else "stage5_report.txt"
    report_path = os.path.join(OUTPUT_STATS_DIR, report_name)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Step 1.5b' DENORM (JOINT mean+std) + soft-knee + dirty-block redraw: "
                f"{'Trial' if mode=='trial' else 'Full Dataset'} Generation Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"compat15: {compat15}  plain_b2: {plain_b2}\n")
        f.write(f"{'='*60}\n\n")

        f.write(f"[1.5b' NEW vs 1.5b - two additions, both gated OFF under compat15/plain_b2]\n")
        f.write(f"  (Knee) soft-knee replaces bg-layer hard lower clip:\n")
        f.write(f"         _soft_floor: x>=k identity; x<k = k*exp((x-k)/k) -> C1, monotone, asymptote->0\n")
        f.write(f"         _soft_ceil : symmetric upper knee toward 255 (anti white-plate)\n")
        f.write(f"         KNEE_K = {KNEE_K}  (FROZEN pre-format-probe via clip/texture/nadir-lift evidence)\n")
        f.write(f"         pure-deterministic: touches NO rng (single-variable law)\n")
        f.write(f"  (Reject) dirty-block redraw via INDEPENDENT reject_rng (main rng stream byte-unchanged):\n")
        f.write(f"         BLOCK_ZERO_REJECT = {BLOCK_ZERO_REJECT}  REJECT_SEED = {REJECT_SEED}  MAX_REDRAW = {MAX_REDRAW}\n")
        f.write(f"         first draw ALWAYS uses main rng (byte-identical to 1.5b idx/y0/x0)\n\n")

        f.write(f"[The ONLY synthesis-logic diff vs full_generate_denorm.py (1.5)]\n")
        f.write(f"  1.5  (mean-only shift): bg_layer = bg_layer + (mu_target - bg_raw_mean)\n")
        f.write(f"  1.5b (affine mean+std): bg_layer = (bg_layer - bg_raw_mean)*(sigma_target/bg_raw_std) + mu_target\n")
        f.write(f"  (mu_target,sigma_target) = paired AI4 block values, independent RandomState({MU_SAMPLE_SEED})\n")
        f.write(f"  mean band [lo,p95] = [{mu_lo:.2f},{mu_p95:.2f}], lo mode = {MEAN_BAND_LO_MODE} (clamp={MEAN_BAND_LO_CLAMP})\n")
        f.write(f"    (lo frozen pre-run via joint clip-vs-baseline sweep; 1.5b drops the 1.5 MU_TARGET_FLOOR=40\n")
        f.write(f"     because per-image std is now from the source too -> dark means no longer clip; NOT pinned to KLSG)\n")
        f.write(f"  pool source: {POOL_PATH}\n")
        f.write(f"  (pool built ONLY from AI4 terrain; NO KLSG/SeabedObjects data)\n\n")

        f.write(f"[Parameters - frozen verbatim from full_generate_denorm.py (1.5)]\n")
        f.write(f"  HIGHLIGHT_OFFSET: {HIGHLIGHT_OFFSET}\n")
        f.write(f"  SHADOW_OFFSET: {SHADOW_OFFSET}\n")
        f.write(f"  HIGHLIGHT_STD_RATIO: {HIGHLIGHT_STD_RATIO}\n")
        f.write(f"  SHADOW_STD_RATIO: {SHADOW_STD_RATIO}\n")
        f.write(f"  WEIBULL_C_FLOOR: {WEIBULL_C_FLOOR}\n")
        f.write(f"  BLEND_WIDTH: {BLEND_WIDTH}\n")
        f.write(f"  DOWNSAMPLE: [{DOWNSAMPLE_FACTOR_MIN}, {DOWNSAMPLE_FACTOR_MAX}]\n")
        f.write(f"  PERTURB: [{PERTURB_MIN}, {PERTURB_MAX}]\n\n")

        f.write(f"[Output]\n")
        f.write(f"  Total: {n_success}\n")
        f.write(f"  Airplanes: {n_planes}\n")
        f.write(f"  Ships: {n_ships}\n")
        f.write(f"  Failed: {n_fail}\n\n")

        f.write(f"[Brightness + Contrast QC]\n")
        f.write(f"  per-image img-mean spread (std) = {img_mean_spread:.4f}\n")
        f.write(f"  per-image img-STD spread (NEW 1.5b) = {img_std_spread:.4f}\n")
        f.write(f"  bg_shifted_std spread = {bg_shifted_std_spread:.4f}\n")
        f.write(f"  img-mean range = [{img_means.min():.2f}, {img_means.max():.2f}]\n")
        f.write(f"  mu_target sampled mean/std = {np.mean(mu_targets):.4f}/{np.std(mu_targets):.4f}\n")
        f.write(f"  sigma_target sampled mean/std = {np.mean(sigma_targets):.4f}/{np.std(sigma_targets):.4f}\n")
        f.write(f"  clip-to-0 frac max/mean = {clip0.max():.6f}/{clip0.mean():.6f}\n")
        f.write(f"  clip-to-255 frac max/mean = {clip255.max():.6f}/{clip255.mean():.6f}\n")
        f.write(f"  HL>BG>SH ordering: {order_ok}/{len(qc_records)}\n\n")

        f.write(f"[Stratify Distribution]\n")
        for k, v in sorted(stratify_dist.items()):
            f.write(f"  {k}: {v}\n")

        f.write(f"\n[Output Paths]\n")
        f.write(f"  Airplane: {OUTPUT_PLANE_DIR}\n")
        f.write(f"  Ship:     {OUTPUT_SHIP_DIR}\n")
        f.write(f"  All:      {OUTPUT_ALL_DIR}\n")
        f.write(f"  Labels:   airplane_labels.csv, ship_labels.csv, dataset_labels.csv\n\n")

        f.write(f"[Self-Check]\n")
        for line in checks:
            f.write(f"  {line}\n")

    print(f"\n{'='*60}")
    print(f"STEP 1.5b DENORM {'TRIAL' if mode=='trial' else 'FULL'} COMPLETE")
    print(f"  Airplane: {OUTPUT_PLANE_DIR} ({n_planes} images)")
    print(f"  Ship:     {OUTPUT_SHIP_DIR} ({n_ships} images)")
    print(f"  All:      {OUTPUT_ALL_DIR} ({n_success} images)")
    print(f"  Report:   {report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Step 1.5b joint de-normalized synthetic dataset generator")
    ap.add_argument("--mode", choices=["trial", "full"], default="full",
                    help="trial: 15-sample QC -> stage4_trial; full: all 490 -> stage5_dataset")
    ap.add_argument("--trial-count", type=int, default=15)
    ap.add_argument("--compat15", action="store_true",
                    help="1.5-compat mode: mean-only 1.5 band + sigma=bg_raw_std (affine=identity); "
                         "output must be byte-identical to 1.5 step1p5_denorm. (软膝/重抽全 gated off)")
    ap.add_argument("--plain_b2", action="store_true",
                    help="1.5b' 字节锚 2: 软膝关 + 脏块重抽关 (退化为 1.5b); "
                         "output must be byte-identical to 1.5b step1p5b_denorm. 证明相对 1.5b 只多了软膝+重抽。")
    ap.add_argument("--shadow_mode", choices=["old", "new"], default="new",
                    help="new (default): v3 整剪影扫掠+远边起伏+暗透镜 -> step2_shadow (Step2 产物); "
                         "old: pre-Step2 *_shadow.png + assemble_image ramp -> step2_shadow/_byte_anchor_old "
                         "(byte-anchor: 复现 1.5b-prime BYTE-EXACT, 证单变量; 绝不覆盖冻结的 step1p5b2_denorm)")
    ap.add_argument("--max_images", type=int, default=None,
                    help="只生成前 N 张 (full-mode 顺序保持; 主/mu/reject rng 流与全量逐字一致)。"
                         "byte-anchor 用它跑 >=10 张验 OLD==1.5b-prime 而不跑满 490。")
    ap.add_argument("--step3", choices=["off", "on"], default="off",
                    help="off (default): 原始标量 highlight 路径 (flat-T byte anchor); "
                         "on: 逐像素 T-map 路径 (结构骑 AI4 颗粒), 从 step3_arm_templates\\{arm} 读 "
                         "obj.png + T.npy; 强制 NEW shadow (从 arm M_obj 生); 输出落 step2_shadow\\_step3_{arm}。")
    ap.add_argument("--step3_arm", choices=["arm2"], default="arm2",
                    help="最终STEP3模板: arm2 (真船形+rim+亮内, 无内部细结构)。")
    args = ap.parse_args()
    assert not (args.compat15 and args.plain_b2), "--compat15 and --plain_b2 are mutually exclusive"
    _step3 = (args.step3 == "on")
    assert not (_step3 and (args.compat15 or args.plain_b2)), "--step3 on incompatible with --compat15/--plain_b2"
    main(mode=args.mode, trial_count=args.trial_count, compat15=args.compat15, plain_b2=args.plain_b2,
         shadow_mode=args.shadow_mode, max_images=args.max_images,
         step3=_step3, step3_arm=args.step3_arm)
