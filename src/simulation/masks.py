# ============================================================================
# 阶段3：路径A掩膜生成
# ============================================================================
# 功能：从飞机和船舶剪影生成全部合成图像的 M_obj + M_shadow 掩膜对
# 流程：掩膜提取 → 损坏处理 → 几何变换 → 画布放置 → 阴影生成
#
# 输入：
#   飞机剪影：PATH_TO_AIRPLANE_SILHOUETTES\*.png
#   船舶剪影：PATH_TO_BASELINE_SHIP_SILHOUETTES\*.png
#
# 输出：
#   掩膜文件：outputs\stage3_masks\data\airplane_XXXX_obj.png / _shadow.png
#             outputs\stage3_masks\data\ship_XXXX_obj.png / _shadow.png
#   元数据：  outputs\stage3_masks\data\mask_metadata.npz
#   抽检图：  outputs\stage3_masks\stats\qc_*.png
#   报告：    outputs\stage3_masks\stats\stage3_report.txt
# ============================================================================

import os
import glob
import numpy as np
import cv2
from scipy import ndimage
from pathlib import Path
from datetime import datetime
import json


# ============ 路径配置 ============
PLANE_DIR = r"PATH_TO_AIRPLANE_SILHOUETTES"
SHIP_DIR = r"PATH_TO_BASELINE_SHIP_SILHOUETTES"
OUTPUT_DATA_DIR = r"PATH_TO_BASELINE_MASK_OUTPUT"
OUTPUT_STATS_DIR = r"PATH_TO_BASELINE_MASK_STATS"

# ============ 目标数量 ============
# 飞机：v0=仅形变, v1=断裂+形变, v2=掩埋+形变
PLANE_VARIANT_TARGETS = {0: 45, 1: 22, 2: 23}   # 总计90
# 船舶：v0=完整, v1=断裂+形变, v2=掩埋+形变, v3=断裂+掩埋+形变
SHIP_VARIANT_TARGETS = {0: 190, 1: 70, 2: 70, 3: 70}   # 总计400
TARGET_PLANES = sum(PLANE_VARIANT_TARGETS.values())   # 90
TARGET_SHIPS = sum(SHIP_VARIANT_TARGETS.values())      # 400

# ============ 画布尺寸参数 ============
CANVAS_MIN_PLANE = 200
CANVAS_MAX = 700           # 背景块硬约束上限
CANVAS_MIN_SHIP = 350
ASPECT_RATIO_MIN = 0.67    # 长宽比下限
ASPECT_RATIO_MAX = 1.50    # 长宽比上限

# ============ 目标尺寸参数 ============
SIZE_RATIO_MIN = 0.30      # 目标占画布较长边的最小比例
SIZE_RATIO_MAX = 0.80      # 最大比例
SIZE_RATIO_BIAS = 0.50     # 多数取值应 > 此值

# ============ 阴影参数 (OLD = pre-Step2 平移阴影; 仅 byte-anchor / 历史用) ============
SHADOW_DISTANCE_RATIO = 0.10  # 平移距离占目标厚度的固定比例（10%）
SHADOW_ANGLE_JITTER = 20    # 角度扰动范围（±度）
SHADOW_BLUR_SIGMA = 1.0     # 阴影边缘高斯模糊sigma
SHADOW_THRESHOLD = 0.3      # 模糊后二值化阈值（低于0.5使边缘外扩）

# ============================================================================
# ★ STEP2 NEW 阴影几何 (FROZEN, ported from _poc_step2_shadow_v3.py LOCKED v3)
# ----------------------------------------------------------------------------
# v3 设计 (用户眼门验收 + reviewer PASS, 设计已锁、参数死冻、不重调):
#   CHANGE 1 整剪影扫掠 (飞机==船, 无 is_airplane 特例 / 无 fuselage_core / 无机翼机器)
#   CHANGE 2 远边平滑起伏 (per-column 扫掠长度 L*(1+a1 sin 2pi f1 s + a2 sin 2pi f2 s),
#            A=a1+a2=0.08, a1=0.0572 a2=0.0228, f1=1 f2=2, 相位取 per-target geom RNG,
#            近边固定/远边波动, 每列长度 clip 进 [0.2,2.0]*target_size 包络)
#   方向 = full-360 随机 per-image; 长度 = lognormal(median=0.5, sigma=0.55)*target_size,
#            clip ratio [0.2,2.0]。
# IRON RULES: 纯 2D 几何; 参数 FROZEN 自 POC v3 (绝不重调 darkness/length/undulation,
#   绝不读 KLSG, 绝不对任何真实数据调参)。本几何只产 *扫掠+起伏二值掩膜* (含 footprint
#   重叠), 暗透镜渲染在 full_generate 端。
# ============================================================================
STEP2_DIRECTION_MODE       = "full_360_per_image"
STEP2_LENGTH_RATIO_MEDIAN  = 0.50
STEP2_LENGTH_RATIO_SIGMA   = 0.55
STEP2_LENGTH_RATIO_CLIP    = (0.20, 2.00)
STEP2_UNDULATION_AMP_A1    = 0.0572   # a1
STEP2_UNDULATION_AMP_A2    = 0.0228   # a2/a1=0.399 (<=0.4) -> A=a1+a2=0.080 in [0.06,0.10]
STEP2_UNDULATION_F1        = 1.0      # cycles across shadow width, in [0.5,1.5]
STEP2_UNDULATION_F2_MULT   = 2.0      # f2 = 2*f1
STEP2_PER_TARGET_SEED_BASE = 20260604

# ============ 断裂参数（统一，不区分轻重） ============
FRAC_ROTATION = 20            # 碎片旋转范围（±度）
FRAC_GAP_MIN = 0.00           # 碎片间隔占L_max最小比例
FRAC_GAP_MAX = 0.20           # 最大比例
FRAC_CUT_MIN = 0.40           # 断裂位置：长轴40%-60%（集中在中部）
FRAC_CUT_MAX = 0.60

# ============ 弹性形变参数 ============
DEFORM_GRID_SIZE = 6          # 位移场网格尺寸（越小越平滑）
DEFORM_STRENGTH = (0.01, 0.02)  # 统一微小形变幅度（占L_max比例）

# ============ 掩埋参数（飞机和船舶均可） ============
BURY_RATIO_MIN = 0.10
BURY_RATIO_MAX = 0.25

# ============ 抽检参数 ============
QC_SAMPLE_COUNT = 20


# ---------------------------------------------------------------------------
# Phase 1: 掩膜提取
# ---------------------------------------------------------------------------

def extract_mask(img_path):
    """从剪影图像提取干净的二值掩膜"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # 二值化：黑色目标=1, 白色背景=0
    mask = (img < 128).astype(np.uint8)

    # 闭运算填充内部孔洞
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 保留最大连通区域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None

    # 跳过label 0（背景）
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1
    mask = (labels == largest_label).astype(np.uint8)

    return mask


# ---------------------------------------------------------------------------
# Phase 2: 损坏处理
# ---------------------------------------------------------------------------

def get_long_axis(mask):
    """计算掩膜bounding box的长轴方向和尺寸"""
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return 'horizontal', 0, 0
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    h = y_max - y_min + 1
    w = x_max - x_min + 1
    if w >= h:
        return 'horizontal', w, h
    else:
        return 'vertical', h, w


def fracture_mask(mask, rng):
    """对掩膜执行断裂处理（统一参数，不区分飞机/船舶）"""
    axis, long_len, short_len = get_long_axis(mask)
    if long_len < 20:
        return mask  # 太小不处理

    coords = np.where(mask > 0)
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    # 断裂位置：长轴40%-60%处（集中在中部）
    cut_ratio = rng.uniform(FRAC_CUT_MIN, FRAC_CUT_MAX)

    if axis == 'horizontal':
        cut_x = int(x_min + cut_ratio * (x_max - x_min))
        # 用随机折线切割（简化为带抖动的垂直线）
        jitter = max(1, int(0.05 * (x_max - x_min)))
        frag1 = mask.copy()
        frag2 = mask.copy()
        for y in range(mask.shape[0]):
            offset = rng.randint(-jitter, jitter + 1)
            cx = cut_x + offset
            frag1[y, max(0, cx):] = 0
            frag2[y, :max(0, cx)] = 0
    else:
        cut_y = int(y_min + cut_ratio * (y_max - y_min))
        jitter = max(1, int(0.05 * (y_max - y_min)))
        frag1 = mask.copy()
        frag2 = mask.copy()
        for x in range(mask.shape[1]):
            offset = rng.randint(-jitter, jitter + 1)
            cy = cut_y + offset
            frag1[max(0, cy):, x] = 0
            frag2[:max(0, cy), x] = 0

    # 清除可能产生的小碎片
    frag1 = keep_largest_component(frag1)
    frag2 = keep_largest_component(frag2)

    if frag1 is None or frag2 is None:
        return mask  # 切割失败，返回原始

    # 碎片独立旋转（统一±20°）
    angle1 = rng.uniform(-FRAC_ROTATION, FRAC_ROTATION)
    angle2 = rng.uniform(-FRAC_ROTATION, FRAC_ROTATION)
    frag1 = rotate_fragment(frag1, angle1)
    frag2 = rotate_fragment(frag2, angle2)

    # 碎片间隔（统一0-20%）
    gap_pixels = int(rng.uniform(FRAC_GAP_MIN, FRAC_GAP_MAX) * long_len)

    # 在大画布上合并两个碎片（留出间隔）
    result = merge_fragments(frag1, frag2, axis, gap_pixels)
    return result


def keep_largest_component(mask):
    """保留掩膜中最大的连通区域"""
    if mask is None or np.sum(mask) == 0:
        return None
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return None
    largest = np.argmax(areas) + 1
    return (labels == largest).astype(np.uint8)


def rotate_fragment(frag, angle):
    """绕碎片质心旋转"""
    if frag is None or np.sum(frag) == 0:
        return frag
    coords = np.where(frag > 0)
    cy = int(np.mean(coords[0]))
    cx = int(np.mean(coords[1]))
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(frag, M, (frag.shape[1], frag.shape[0]),
                              flags=cv2.INTER_LINEAR, borderValue=0)
    return (rotated > 0.5).astype(np.uint8)


def merge_fragments(frag1, frag2, axis, gap):
    """将两个碎片在指定方向上拉开间隔后合并"""
    # 获取两个碎片的有效区域
    bbox1 = get_bbox(frag1)
    bbox2 = get_bbox(frag2)
    if bbox1 is None or bbox2 is None:
        if bbox1 is not None:
            return crop_to_content(frag1)
        if bbox2 is not None:
            return crop_to_content(frag2)
        return np.zeros((50, 50), dtype=np.uint8)

    c1 = crop_to_content(frag1)
    c2 = crop_to_content(frag2)

    if axis == 'horizontal':
        # 水平排列：frag1 | gap | frag2
        h = max(c1.shape[0], c2.shape[0])
        w = c1.shape[1] + gap + c2.shape[1]
        result = np.zeros((h, w), dtype=np.uint8)
        y_off1 = (h - c1.shape[0]) // 2
        y_off2 = (h - c2.shape[0]) // 2
        result[y_off1:y_off1+c1.shape[0], 0:c1.shape[1]] = c1
        result[y_off2:y_off2+c2.shape[0], c1.shape[1]+gap:] = c2
    else:
        # 垂直排列
        w = max(c1.shape[1], c2.shape[1])
        h = c1.shape[0] + gap + c2.shape[0]
        result = np.zeros((h, w), dtype=np.uint8)
        x_off1 = (w - c1.shape[1]) // 2
        x_off2 = (w - c2.shape[1]) // 2
        result[0:c1.shape[0], x_off1:x_off1+c1.shape[1]] = c1
        result[c1.shape[0]+gap:, x_off2:x_off2+c2.shape[1]] = c2

    return result


def bury_mask(mask, rng):
    """对掩膜执行部分掩埋处理（仅船舶）"""
    axis, long_len, short_len = get_long_axis(mask)
    if long_len < 30:
        return mask

    bury_ratio = rng.uniform(BURY_RATIO_MIN, BURY_RATIO_MAX)
    coords = np.where(mask > 0)
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    end = rng.choice(['head', 'tail'])
    result = mask.copy()

    if axis == 'horizontal':
        bury_len = int(bury_ratio * (x_max - x_min))
        if end == 'head':
            # 裁掉左端
            cut_x = x_min + bury_len
            # 不规则边缘
            for y in range(mask.shape[0]):
                jitter = rng.randint(-3, 4)
                result[y, :max(0, cut_x + jitter)] = 0
        else:
            cut_x = x_max - bury_len
            for y in range(mask.shape[0]):
                jitter = rng.randint(-3, 4)
                result[y, min(mask.shape[1], cut_x + jitter):] = 0
    else:
        bury_len = int(bury_ratio * (y_max - y_min))
        if end == 'head':
            cut_y = y_min + bury_len
            for x in range(mask.shape[1]):
                jitter = rng.randint(-3, 4)
                result[:max(0, cut_y + jitter), x] = 0
        else:
            cut_y = y_max - bury_len
            for x in range(mask.shape[1]):
                jitter = rng.randint(-3, 4)
                result[min(mask.shape[0], cut_y + jitter):, x] = 0

    return result


def elastic_deform(mask, rng):
    """
    对掩膜施加弹性形变，模拟冲击/水压导致的结构扭曲。
    用低频随机位移场做非刚性变形。
    统一微小形变幅度：1-2% L_max
    """
    if mask is None or np.sum(mask) == 0:
        return mask

    h, w = mask.shape
    L_max = max(h, w)

    # 位移幅度
    strength = rng.uniform(DEFORM_STRENGTH[0], DEFORM_STRENGTH[1]) * L_max

    # 生成低频随机位移场：小网格 → 插值到掩膜尺寸
    grid_h = DEFORM_GRID_SIZE
    grid_w = DEFORM_GRID_SIZE

    # 随机位移（在小网格上）
    dx_grid = rng.uniform(-1, 1, (grid_h, grid_w)).astype(np.float32) * strength
    dy_grid = rng.uniform(-1, 1, (grid_h, grid_w)).astype(np.float32) * strength

    # 双线性插值到掩膜尺寸，形成平滑位移场
    dx_field = cv2.resize(dx_grid, (w, h), interpolation=cv2.INTER_LINEAR)
    dy_field = cv2.resize(dy_grid, (w, h), interpolation=cv2.INTER_LINEAR)

    # 构建映射坐标
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + dx_field).astype(np.float32)
    map_y = (grid_y + dy_field).astype(np.float32)

    # remap
    deformed = cv2.remap(mask.astype(np.float32), map_x, map_y,
                          interpolation=cv2.INTER_LINEAR, borderValue=0)
    deformed = (deformed > 0.5).astype(np.uint8)

    return deformed


# ---------------------------------------------------------------------------
# Phase 3: 几何变换
# ---------------------------------------------------------------------------

def get_bbox(mask):
    """获取掩膜的bounding box，返回(y_min, y_max, x_min, x_max)或None"""
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return None
    return (coords[0].min(), coords[0].max(), coords[1].min(), coords[1].max())


def crop_to_content(mask):
    """裁剪掩膜到内容区域"""
    bbox = get_bbox(mask)
    if bbox is None:
        return mask
    y_min, y_max, x_min, x_max = bbox
    return mask[y_min:y_max+1, x_min:x_max+1].copy()


def scale_and_rotate(mask, canvas_h, canvas_w, rng):
    """缩放到目标比例并随机旋转"""
    mask = crop_to_content(mask)
    if mask is None or mask.shape[0] == 0 or mask.shape[1] == 0:
        return None

    # 目标尺寸比例（偏向 > 0.50）
    if rng.random() < 0.7:
        size_ratio = rng.uniform(SIZE_RATIO_BIAS, SIZE_RATIO_MAX)
    else:
        size_ratio = rng.uniform(SIZE_RATIO_MIN, SIZE_RATIO_BIAS)

    canvas_long = max(canvas_h, canvas_w)
    target_L = size_ratio * canvas_long

    L_max = max(mask.shape[0], mask.shape[1])
    if L_max == 0:
        return None
    scale = target_L / L_max

    # 缩放
    new_h = max(1, int(mask.shape[0] * scale))
    new_w = max(1, int(mask.shape[1] * scale))
    scaled = cv2.resize(mask.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    scaled = (scaled > 0.5).astype(np.uint8)

    # 旋转（需要更大画布以防裁切）
    angle = rng.uniform(0, 360)
    diag = int(np.ceil(np.sqrt(new_h**2 + new_w**2))) + 10
    pad_y = (diag - new_h) // 2
    pad_x = (diag - new_w) // 2
    padded = np.zeros((diag, diag), dtype=np.uint8)
    padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = scaled

    center = (diag // 2, diag // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(padded, M, (diag, diag),
                              flags=cv2.INTER_LINEAR, borderValue=0)
    rotated = (rotated > 0.5).astype(np.uint8)

    # 裁剪回内容区域
    result = crop_to_content(rotated)
    return result, angle, size_ratio


# ---------------------------------------------------------------------------
# Phase 4: 画布放置
# ---------------------------------------------------------------------------

def determine_canvas_size(label, rng):
    """根据类别确定画布尺寸"""
    min_size = CANVAS_MIN_PLANE if label == 'airplane' else CANVAS_MIN_SHIP

    # 反复采样直到满足长宽比约束
    for _ in range(100):
        w = rng.randint(min_size, CANVAS_MAX + 1)
        h = rng.randint(min_size, CANVAS_MAX + 1)
        ratio = max(w, h) / min(w, h)
        if ASPECT_RATIO_MIN <= ratio <= ASPECT_RATIO_MAX:
            return h, w

    # 保底：正方形
    s = rng.randint(min_size, CANVAS_MAX + 1)
    return s, s


def place_on_canvas(mask, canvas_h, canvas_w, rng):
    """将掩膜放置到画布上"""
    obj_h, obj_w = mask.shape

    # 如果目标比画布大，需要缩小
    if obj_h > canvas_h or obj_w > canvas_w:
        scale = min(canvas_h / obj_h, canvas_w / obj_w) * 0.95
        new_h = max(1, int(obj_h * scale))
        new_w = max(1, int(obj_w * scale))
        mask = cv2.resize(mask.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = (mask > 0.5).astype(np.uint8)
        obj_h, obj_w = mask.shape

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    # 可用放置范围
    max_x = canvas_w - obj_w
    max_y = canvas_h - obj_h
    if max_x < 0 or max_y < 0:
        return canvas  # 不应该发生

    # 默认居中，加随机偏移（不超过可用范围的50%）
    center_x = max_x // 2
    center_y = max_y // 2
    jitter_x = max(0, max_x // 4)
    jitter_y = max(0, max_y // 4)

    place_x = center_x + rng.randint(-jitter_x, jitter_x + 1) if jitter_x > 0 else center_x
    place_y = center_y + rng.randint(-jitter_y, jitter_y + 1) if jitter_y > 0 else center_y

    place_x = np.clip(place_x, 0, max_x)
    place_y = np.clip(place_y, 0, max_y)

    canvas[place_y:place_y+obj_h, place_x:place_x+obj_w] = mask
    return canvas


# ---------------------------------------------------------------------------
# Phase 5: 阴影生成
# ---------------------------------------------------------------------------
# 两条码路:
#   generate_shadow (NEW, Step2 默认) = v3 整剪影扫掠 + 远边平滑起伏 (飞机==船一路)。
#   generate_shadow_old (OLD, pre-Step2) = 0.10x 平移 + choice[0,180]+-20 + 高斯模糊。
#     仅 byte-anchor (复现 1.5b-prime) / 历史对照用; 不再是生产默认。
# 二者 *几何参数* 全冻 (NEW 自 POC v3, OLD 自 baseline)。
# ---------------------------------------------------------------------------

# ===== STEP2 NEW 几何工具 (LOCKED 自 _poc_step2_shadow_v3.py, 字节对齐) =====

def _step2_target_size_px(mask):
    """目标尺寸 = bbox 较长边 (px)。"""
    bb = get_bbox(mask)
    if bb is None:
        return 0
    y0, y1, x0, x1 = bb
    return max(x1 - x0 + 1, y1 - y0 + 1)


def step2_sample_direction(rng):
    """方向 = full-360 随机 (per-image; 多碎片共享一方向)。"""
    return float(rng.uniform(0.0, 360.0))


def step2_sample_length_ratio(rng):
    """长度 ratio = lognormal(median=0.5, sigma=0.55), clip [0.2,2.0]。"""
    med = STEP2_LENGTH_RATIO_MEDIAN
    sigma = STEP2_LENGTH_RATIO_SIGMA
    lo, hi = STEP2_LENGTH_RATIO_CLIP
    r = med * np.exp(rng.normal(0.0, sigma))
    return float(np.clip(r, lo, hi))


def step2_phases_from_rng(geom_seed):
    """CHANGE 2 相位取自既有 per-target geom RNG (不新开噪声流) —— 与 POC v3 一致。"""
    rng = np.random.RandomState(geom_seed + 321)
    p1 = float(rng.uniform(0, 2 * np.pi))
    p2 = float(rng.uniform(0, 2 * np.pi))
    return (p1, p2)


def _step2_undulation_factor(s_norm, phases):
    """C1-smooth 低频起伏因子 m(s)=1+a1 sin(2pi f1 s+p1)+a2 sin(2pi f2 s+p2)。
       smooth low-freq ONLY (非噪声/speckle/ripple); 两正弦谐波。"""
    a1 = STEP2_UNDULATION_AMP_A1
    a2 = STEP2_UNDULATION_AMP_A2
    f1 = STEP2_UNDULATION_F1
    f2 = STEP2_UNDULATION_F2_MULT * f1
    p1, p2 = phases
    m = (1.0
         + a1 * np.sin(2 * np.pi * f1 * s_norm + p1)
         + a2 * np.sin(2 * np.pi * f2 * s_norm + p2))
    return m


def step2_sweep_silhouette(seed_mask, theta_deg, L_px,
                           undulate=False, phases=(0.0, 0.0),
                           L_envelope=None):
    """剪影沿 theta 方向 Minkowski 加长 L_px 线段 (近边贴目标不动)。

    undulate=False -> 均匀长度 (POC undulate OFF, 字节等价 v2)。
    undulate=True  -> ★CHANGE 2: 每列扫掠长度 L_local(s)=L_px*m(s_norm) 起伏。
       近/目标贴边 (s 小) 永远保留, 只远端终止随横向位置波动 = 远边 soft scallop。
       L_local clip 进 L_envelope=[L_lo,L_hi] (冻结的 [0.2,2.0]*target_size 包络)。
    LOCKED 自 _poc_step2_shadow_v3.py sweep_silhouette。
    """
    if L_px < 1 or seed_mask.sum() == 0:
        return seed_mask.copy().astype(np.uint8)
    h, w = seed_mask.shape
    th = np.radians(theta_deg)
    ux, uy = np.cos(th), np.sin(th)        # 投影方向 (远离声源, along-track)
    px, py = -uy, ux                       # 侧向 (cross-track)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    s_coord = xx * px + yy * py
    ys_m, xs_m = np.where(seed_mask > 0)
    s_seed = xs_m * px + ys_m * py
    s_lo, s_hi = float(s_seed.min()), float(s_seed.max())
    s_span = max(s_hi - s_lo, 1.0)
    s_norm_field = (s_coord - s_lo) / s_span           # 0..1 across cross-track

    if undulate:
        m_field = _step2_undulation_factor(s_norm_field, phases)   # per-pixel length mult
        L_local_field = L_px * m_field
        if L_envelope is not None:
            L_lo, L_hi = L_envelope
            L_local_field = np.clip(L_local_field, L_lo, L_hi)     # 守 [0.2,2.0] 包络
        L_max = float(L_local_field.max())
    else:
        L_local_field = np.full((h, w), float(L_px))
        L_max = float(L_px)

    swept = seed_mask.copy().astype(np.uint8)
    n_steps = int(np.ceil(L_max))
    for s in range(1, n_steps + 1):
        dx = int(round(s * ux))
        dy = int(round(s * uy))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(seed_mask, M, (w, h),
                                 flags=cv2.INTER_NEAREST, borderValue=0)
        if undulate:
            keep = (L_local_field >= s).astype(np.uint8)
            shifted = (shifted & keep).astype(np.uint8)
        swept = np.maximum(swept, shifted)
    return swept.astype(np.uint8)


def step2_build_new_shadow_mask(M_obj, theta_deg, L_px, phases, L_envelope):
    """构建 NEW 阴影 *全* 掩膜 (扫掠投影, 含 footprint 重叠)。
    ★CHANGE 1 飞机==船 (无 is_airplane 特例); ★CHANGE 2 远边平滑起伏。
    LOCKED 自 _poc_step2_shadow_v3.py build_new_shadow_mask。"""
    return step2_sweep_silhouette(M_obj, theta_deg, L_px,
                                  undulate=True, phases=phases,
                                  L_envelope=L_envelope)


def generate_shadow(M_obj, rng, base_direction=None):
    """STEP2 NEW 阴影掩膜 (默认生产码路, LOCKED 自 POC v3)。

    整剪影沿 full-360 随机方向扫掠 (飞机==船一路, 无机翼机器) + 远边平滑起伏。
    返回 (M_shadow_full[swept+undulated, uint8, 含 footprint 重叠], params)。
    含 footprint 重叠 -> 下游 assemble 的目标-边缘逻辑 (obj_and_shadow=HARD /
      obj_only=BW10) 才能正确工作。

    direction/length 从传入 rng 采样 (full_generate 用 per-target geom RNG 调用,
      保证可复现 + 多碎片共享方向)。base_direction 若指定则复用 (断裂碎片共享方向)。
    """
    bb = get_bbox(M_obj)
    if bb is None:
        return np.zeros_like(M_obj), {'mode': 'step2_new', 'theta': 0.0,
                                      'L_px': 0.0, 'length_ratio': 0.0,
                                      'target_size_px': 0, 'phases': [0.0, 0.0]}

    # 方向: full-360 随机 (base_direction 给定则复用, 供多碎片同方向)
    if base_direction is None:
        theta = step2_sample_direction(rng)
    else:
        theta = float(base_direction)

    tsize = _step2_target_size_px(M_obj)
    lratio = step2_sample_length_ratio(rng)
    L_px = lratio * tsize

    # CHANGE 2 起伏后每列长度守冻结包络 [0.2,2.0]*target_size
    lo_r, hi_r = STEP2_LENGTH_RATIO_CLIP
    L_envelope = (lo_r * tsize, hi_r * tsize)

    # 相位: 取自既有 per-target geom RNG (用 rng 再抽一个整数种子, 不新开噪声流)
    geom_seed = int(rng.randint(0, 2**31 - 1))
    phases = step2_phases_from_rng(geom_seed)

    M_shadow_full = step2_build_new_shadow_mask(M_obj, theta, L_px, phases, L_envelope)

    params = {
        'mode': 'step2_new',
        'theta': float(theta),
        'length_ratio': float(lratio),
        'target_size_px': int(tsize),
        'L_px': float(L_px),
        'phases': [float(phases[0]), float(phases[1])],
    }
    return M_shadow_full.astype(np.uint8), params


def generate_shadow_old(M_obj, rng, base_direction=None):
    """
    OLD 阴影掩膜 (pre-Step2; 仅 byte-anchor 复现 1.5b-prime / 历史对照用)。
    平移距离基于目标在垂直于平移方向上的投影宽度（即目标"厚度"），保证阴影与目标部分重叠。
    base_direction: 如果指定则使用该基础方向（用于断裂碎片共享方向）
    """
    if base_direction is None:
        base_direction = rng.choice([0, 180])

    angle_jitter = rng.uniform(-SHADOW_ANGLE_JITTER, SHADOW_ANGLE_JITTER)
    theta = base_direction + angle_jitter
    theta_rad = np.radians(theta)

    bbox = get_bbox(M_obj)
    if bbox is None:
        return np.zeros_like(M_obj), {'base_dir': base_direction, 'jitter': angle_jitter,
                                       'theta': theta, 'distance': 0, 'dx': 0, 'dy': 0}

    # 计算目标在垂直于平移方向上的投影宽度（目标"厚度"）
    # 平移方向: (cos(theta), sin(theta))
    # 垂直方向: (-sin(theta), cos(theta))
    obj_coords = np.where(M_obj > 0)
    ys = obj_coords[0].astype(np.float64)
    xs = obj_coords[1].astype(np.float64)

    # 投影到垂直于平移方向的轴上
    perp_projections = xs * (-np.sin(theta_rad)) + ys * np.cos(theta_rad)
    proj_thickness = perp_projections.max() - perp_projections.min()

    # 保底：厚度至少10像素
    proj_thickness = max(proj_thickness, 10)

    dist_ratio = SHADOW_DISTANCE_RATIO
    d = dist_ratio * proj_thickness

    dx = int(round(d * np.cos(theta_rad)))
    dy = int(round(d * np.sin(theta_rad)))

    # 平移M_obj生成阴影
    h, w = M_obj.shape
    M_shadow = np.zeros_like(M_obj)
    for y in range(h):
        for x in range(w):
            if M_obj[y, x] == 1:
                new_y = y + dy
                new_x = x + dx
                if 0 <= new_y < h and 0 <= new_x < w:
                    M_shadow[new_y, new_x] = 1

    # 高斯模糊 + 低阈值二值化（边缘外扩）
    blurred = cv2.GaussianBlur(M_shadow.astype(np.float32), (0, 0), SHADOW_BLUR_SIGMA)
    M_shadow = (blurred > SHADOW_THRESHOLD).astype(np.uint8)

    params = {
        'base_dir': int(base_direction),
        'jitter': float(angle_jitter),
        'theta': float(theta),
        'proj_thickness': float(proj_thickness),
        'distance_ratio': float(dist_ratio),
        'dx': dx, 'dy': dy
    }
    return M_shadow, params


def generate_shadow_multi_fragment(M_obj, rng):
    """OLD: 对包含多个连通区域的目标生成独立阴影（断裂船舶用; pre-Step2 历史码路）"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(M_obj, connectivity=8)

    if num_labels <= 2:
        # 只有一个连通区域（或为空），整体生成
        return generate_shadow_old(M_obj, rng)

    # 多碎片：共享基础方向
    base_direction = rng.choice([0, 180])
    M_shadow_combined = np.zeros_like(M_obj)

    for i in range(1, num_labels):
        frag_mask = (labels == i).astype(np.uint8)
        frag_shadow, _ = generate_shadow_old(frag_mask, rng, base_direction=base_direction)
        M_shadow_combined = np.maximum(M_shadow_combined, frag_shadow)

    # 统一再做一次模糊+二值化
    blurred = cv2.GaussianBlur(M_shadow_combined.astype(np.float32), (0, 0), SHADOW_BLUR_SIGMA)
    M_shadow_combined = (blurred > SHADOW_THRESHOLD).astype(np.uint8)

    params = {'base_dir': int(base_direction), 'multi_fragment': True, 'n_fragments': num_labels - 1}
    return M_shadow_combined, params


# ---------------------------------------------------------------------------
# 单张合成样本的完整生成
# ---------------------------------------------------------------------------

def generate_single_sample(silhouette_path, label, variant_idx, rng):
    """
    从一张剪影生成一组 (M_obj, M_shadow, metadata)。
    label: 'airplane' 或 'ship'
    variant_idx: 当前变体编号（0-indexed）
    """
    # Phase 1: 掩膜提取
    mask = extract_mask(silhouette_path)
    if mask is None:
        return None

    is_ship = (label == 'ship')
    meta = {
        'source': Path(silhouette_path).name,
        'label': label,
        'variant': variant_idx
    }

    # Phase 2: 损坏处理
    # 飞机：v0=形变, v1=断裂+形变, v2=掩埋+形变
    # 船舶：v0=完整, v1=断裂+形变, v2=掩埋+形变, v3=断裂+掩埋+形变
    if label == 'airplane':
        if variant_idx == 0:
            # 仅弹性形变（微小，模拟坠海冲击）
            mask = elastic_deform(mask, rng)
            meta['damage'] = 'deform_only'
        elif variant_idx == 1:
            # 断裂 + 形变
            mask = fracture_mask(mask, rng)
            mask = elastic_deform(mask, rng)
            meta['damage'] = 'fracture+deform'
        elif variant_idx == 2:
            # 掩埋 + 形变
            mask = bury_mask(mask, rng)
            mask = elastic_deform(mask, rng)
            meta['damage'] = 'burial+deform'
    else:  # ship
        if variant_idx == 0:
            # 完整无损（漏水/倾覆沉没）
            meta['damage'] = 'none'
        elif variant_idx == 1:
            # 断裂 + 形变
            mask = fracture_mask(mask, rng)
            mask = elastic_deform(mask, rng)
            meta['damage'] = 'fracture+deform'
        elif variant_idx == 2:
            # 掩埋 + 形变
            mask = bury_mask(mask, rng)
            mask = elastic_deform(mask, rng)
            meta['damage'] = 'burial+deform'
        elif variant_idx == 3:
            # 断裂 + 掩埋 + 形变
            mask = fracture_mask(mask, rng)
            mask = bury_mask(mask, rng)
            mask = elastic_deform(mask, rng)
            meta['damage'] = 'fracture+burial+deform'

    # 分层抽样标识：label + damage组合，用于后续划分训练/验证/测试集时按比例分配
    meta['stratify_key'] = f"{label}_{meta['damage']}"

    # Phase 4先行：确定画布尺寸
    canvas_h, canvas_w = determine_canvas_size(label, rng)
    meta['canvas_h'] = int(canvas_h)
    meta['canvas_w'] = int(canvas_w)

    # Phase 3: 几何变换
    result = scale_and_rotate(mask, canvas_h, canvas_w, rng)
    if result is None:
        return None
    mask_transformed, angle, size_ratio = result
    meta['rotation'] = float(angle)
    meta['size_ratio'] = float(size_ratio)

    # Phase 4: 画布放置
    M_obj = place_on_canvas(mask_transformed, canvas_h, canvas_w, rng)

    if np.sum(M_obj) == 0:
        return None

    # Phase 5: 阴影生成
    # 检查是否为多碎片（断裂的船）
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(M_obj, connectivity=8)
    if num_labels > 2 and is_ship:
        M_shadow, shadow_params = generate_shadow_multi_fragment(M_obj, rng)
    else:
        M_shadow, shadow_params = generate_shadow(M_obj, rng)

    meta['shadow'] = shadow_params

    return M_obj, M_shadow, meta


# ---------------------------------------------------------------------------
# QC可视化
# ---------------------------------------------------------------------------

def generate_qc_image(M_obj, M_shadow, meta):
    """生成单张合成样本的三列可视化：M_obj | M_shadow | 叠加"""
    h, w = M_obj.shape

    # M_obj可视化（白色目标在灰色画布上）
    vis_obj = np.full((h, w), 180, dtype=np.uint8)
    vis_obj[M_obj == 1] = 255

    # M_shadow可视化
    vis_shadow = np.full((h, w), 180, dtype=np.uint8)
    vis_shadow[M_shadow == 1] = 80

    # 叠加彩色图（目标红，阴影蓝，重叠紫）
    vis_overlay = np.full((h, w, 3), 180, dtype=np.uint8)
    # 仅阴影：蓝色
    shadow_only = (M_shadow == 1) & (M_obj == 0)
    vis_overlay[shadow_only] = [200, 100, 50]     # BGR: 蓝色调
    # 仅目标：红色
    obj_only = (M_obj == 1) & (M_shadow == 0)
    vis_overlay[obj_only] = [50, 50, 220]          # BGR: 红色调
    # 重叠：紫色
    overlap = (M_obj == 1) & (M_shadow == 1)
    vis_overlay[overlap] = [180, 50, 180]          # BGR: 紫色

    # 拼接（统一转BGR）
    vis_obj_bgr = cv2.cvtColor(vis_obj, cv2.COLOR_GRAY2BGR)
    vis_shadow_bgr = cv2.cvtColor(vis_shadow, cv2.COLOR_GRAY2BGR)

    # 加标注
    font = cv2.FONT_HERSHEY_SIMPLEX
    sc = max(0.35, h / 1000)
    th = max(1, int(h / 500))
    label_text = f"{meta['label']} v{meta['variant']} [{meta['damage']}]"
    cv2.putText(vis_obj_bgr, "M_obj", (4, 18), font, sc, (0, 200, 0), th)
    cv2.putText(vis_shadow_bgr, "M_shadow", (4, 18), font, sc, (0, 200, 0), th)
    cv2.putText(vis_overlay, label_text, (4, 18), font, sc, (0, 200, 0), th)
    size_text = f"{w}x{h}"
    cv2.putText(vis_overlay, size_text, (4, h - 8), font, sc, (0, 200, 0), th)

    combined = np.hstack([vis_obj_bgr, vis_shadow_bgr, vis_overlay])
    return combined


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

def run_self_check(all_meta, n_planes, n_ships):
    checks = []
    all_pass = True

    total = len(all_meta)
    checks.append(f"[INFO] Total masks generated: {total}")
    checks.append(f"[INFO] Airplanes: {n_planes}, Ships: {n_ships}")

    # 目标数量检查
    plane_ok = n_planes >= TARGET_PLANES * 0.9
    ship_ok = n_ships >= TARGET_SHIPS * 0.9
    checks.append(f"[{'PASS' if plane_ok else 'WARN'}] Airplane count: {n_planes} (target: {TARGET_PLANES})")
    checks.append(f"[{'PASS' if ship_ok else 'WARN'}] Ship count: {n_ships} (target: {TARGET_SHIPS})")

    # 画布尺寸检查
    canvas_sizes = [(m['canvas_w'], m['canvas_h']) for m in all_meta]
    widths = [s[0] for s in canvas_sizes]
    heights = [s[1] for s in canvas_sizes]
    checks.append(f"[INFO] Canvas width range: [{min(widths)}, {max(widths)}]")
    checks.append(f"[INFO] Canvas height range: [{min(heights)}, {max(heights)}]")

    max_dim = max(max(widths), max(heights))
    checks.append(f"[{'PASS' if max_dim <= CANVAS_MAX else 'FAIL'}] Max dimension <= {CANVAS_MAX}: {max_dim}")
    if max_dim > CANVAS_MAX:
        all_pass = False

    # 掩膜非空检查（已在生成时过滤，这里再确认）
    checks.append(f"[PASS] All masks non-empty (filtered during generation)")

    # 损坏类型分布
    damage_types = {}
    for m in all_meta:
        d = m['damage']
        damage_types[d] = damage_types.get(d, 0) + 1
    checks.append(f"[INFO] Damage distribution: {damage_types}")

    checks.append("")
    checks.append(f"{'='*40}")
    checks.append(f"SELF-CHECK RESULT: {'ALL PASSED' if all_pass else 'ISSUES DETECTED'}")

    return checks


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    # PUBLIC SNAPSHOT GUARD: the final arm2 pipeline reuses the prepared baseline masks read-only.
    # Keep standalone generation disabled until all input/output placeholders have been reviewed.
    raise SystemExit("Standalone baseline-mask generation is disabled in this public snapshot. "
                     "Review the configured input/output paths and this guard before intentionally "
                     "regenerating the baseline masks.")
    os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_STATS_DIR, exist_ok=True)

    # 获取所有剪影
    plane_files = sorted(glob.glob(os.path.join(PLANE_DIR, "*.png")))
    ship_files = sorted(glob.glob(os.path.join(SHIP_DIR, "*.png")))

    print(f"{'='*60}")
    print(f"STAGE 3: Path A Mask Generation")
    print(f"{'='*60}")
    print(f"Plane silhouettes: {len(plane_files)} (from {PLANE_DIR})")
    print(f"Ship silhouettes:  {len(ship_files)} (from {SHIP_DIR})")
    print(f"Plane variant targets: {PLANE_VARIANT_TARGETS} = {TARGET_PLANES}")
    print(f"Ship variant targets:  {SHIP_VARIANT_TARGETS} = {TARGET_SHIPS}")
    print(f"{'='*60}")

    rng = np.random.RandomState(42)
    all_meta = []
    n_planes = 0
    n_ships = 0
    sample_idx = 0
    qc_collected = 0

    # ---- 生成飞机掩膜（按变体类型分别生成指定数量） ----
    print(f"\n[Generating airplane masks...]")
    for var_idx, var_target in PLANE_VARIANT_TARGETS.items():
        var_count = 0
        sil_cycle = 0  # 循环使用剪影
        while var_count < var_target:
            sil_path = plane_files[sil_cycle % len(plane_files)]
            sil_cycle += 1

            result = generate_single_sample(sil_path, 'airplane', var_idx, rng)
            if result is None:
                continue

            M_obj, M_shadow, meta = result
            sample_idx += 1
            n_planes += 1
            var_count += 1
            tag = f"airplane_{sample_idx:04d}"
            meta['tag'] = tag

            cv2.imwrite(os.path.join(OUTPUT_DATA_DIR, f"{tag}_obj.png"), M_obj * 255)
            cv2.imwrite(os.path.join(OUTPUT_DATA_DIR, f"{tag}_shadow.png"), M_shadow * 255)
            all_meta.append(meta)

            # 抽检图
            if qc_collected < QC_SAMPLE_COUNT // 2:
                if rng.random() < 0.15:
                    qc_img = generate_qc_image(M_obj, M_shadow, meta)
                    cv2.imwrite(os.path.join(OUTPUT_STATS_DIR, f"qc_{tag}.png"), qc_img)
                    qc_collected += 1

        print(f"  Airplane v{var_idx} ({meta['damage']}): {var_count}")

    print(f"  Airplanes total: {n_planes}")

    # ---- 生成船舶掩膜（按变体类型分别生成指定数量） ----
    print(f"\n[Generating ship masks...]")
    for var_idx, var_target in SHIP_VARIANT_TARGETS.items():
        var_count = 0
        sil_cycle = 0
        while var_count < var_target:
            sil_path = ship_files[sil_cycle % len(ship_files)]
            sil_cycle += 1

            result = generate_single_sample(sil_path, 'ship', var_idx, rng)
            if result is None:
                continue

            M_obj, M_shadow, meta = result
            sample_idx += 1
            n_ships += 1
            var_count += 1
            tag = f"ship_{sample_idx:04d}"
            meta['tag'] = tag

            cv2.imwrite(os.path.join(OUTPUT_DATA_DIR, f"{tag}_obj.png"), M_obj * 255)
            cv2.imwrite(os.path.join(OUTPUT_DATA_DIR, f"{tag}_shadow.png"), M_shadow * 255)
            all_meta.append(meta)

            if qc_collected < QC_SAMPLE_COUNT:
                if rng.random() < 0.08:
                    qc_img = generate_qc_image(M_obj, M_shadow, meta)
                    cv2.imwrite(os.path.join(OUTPUT_STATS_DIR, f"qc_{tag}.png"), qc_img)
                    qc_collected += 1

            if var_count % 50 == 0:
                print(f"    v{var_idx} progress: {var_count}/{var_target}")

        print(f"  Ship v{var_idx} ({meta['damage']}): {var_count}")

    print(f"  Ships total: {n_ships}")

    # ---- 保存元数据 ----
    print(f"\n[Saving metadata...]")
    meta_path = os.path.join(OUTPUT_DATA_DIR, "mask_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {meta_path}")

    # ---- 自检 ----
    print(f"\n[Self-Check]")
    check_results = run_self_check(all_meta, n_planes, n_ships)
    for line in check_results:
        print(f"  {line}")

    # ---- 保存报告 ----
    report_path = os.path.join(OUTPUT_STATS_DIR, "stage3_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Stage 3: Path A Mask Generation Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")

        f.write(f"[Input]\n")
        f.write(f"  Plane silhouettes: {len(plane_files)}\n")
        f.write(f"  Ship silhouettes: {len(ship_files)}\n\n")

        f.write(f"[Parameters]\n")
        f.write(f"  Plane variant targets: {PLANE_VARIANT_TARGETS}\n")
        f.write(f"  Ship variant targets:  {SHIP_VARIANT_TARGETS}\n")
        f.write(f"  Canvas: Plane [{CANVAS_MIN_PLANE}-{CANVAS_MAX}], Ship [{CANVAS_MIN_SHIP}-{CANVAS_MAX}]\n")
        f.write(f"  Aspect ratio: [{ASPECT_RATIO_MIN}, {ASPECT_RATIO_MAX}]\n")
        f.write(f"  Size ratio: [{SIZE_RATIO_MIN}, {SIZE_RATIO_MAX}] (bias > {SIZE_RATIO_BIAS})\n")
        f.write(f"  Fracture cut position: [{FRAC_CUT_MIN}, {FRAC_CUT_MAX}]\n")
        f.write(f"  Fracture rotation: +/-{FRAC_ROTATION} deg, gap: [{FRAC_GAP_MIN}, {FRAC_GAP_MAX}] x L_max\n")
        f.write(f"  Elastic deform: unified {DEFORM_STRENGTH} (1-2% L_max)\n")
        f.write(f"  Shadow distance: {SHADOW_DISTANCE_RATIO} x thickness (fixed)\n")
        f.write(f"  Shadow angle jitter: +/-{SHADOW_ANGLE_JITTER} deg\n\n")

        f.write(f"[Output]\n")
        f.write(f"  Airplanes: {n_planes}\n")
        f.write(f"  Ships: {n_ships}\n")
        f.write(f"  Total: {n_planes + n_ships}\n")
        f.write(f"  Masks dir: {OUTPUT_DATA_DIR}\n\n")

        f.write(f"[Self-Check]\n")
        for line in check_results:
            f.write(f"  {line}\n")

    print(f"\n{'='*60}")
    print(f"STAGE 3 COMPLETE")
    print(f"  Total: {n_planes + n_ships} mask pairs ({n_planes} airplane + {n_ships} ship)")
    print(f"  Data:  {OUTPUT_DATA_DIR}")
    print(f"  Report: {report_path}")
    print(f"  QC: {OUTPUT_STATS_DIR}/qc_*.png ({qc_collected} files)")
    print(f"{'='*60}")
    print(f"\nPlease visually inspect:")
    print(f"  1. qc_*.png - Target shapes reasonable?")
    print(f"  2. qc_*.png - Shadow direction and size natural?")
    print(f"  3. qc_*.png - Fractured pieces have independent shadows?")
    print(f"  4. qc_*.png - Canvas sizes varied?")


if __name__ == "__main__":
    main()
