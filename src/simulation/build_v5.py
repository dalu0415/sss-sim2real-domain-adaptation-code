"""Step3 沉船目标 v5 POC 生成器 —— (M_obj 二值 + T 模板) 对接架构版.

迭代自 v4 (build_v4.py)。实现 UNIFIED_SPEC_v1.md「★v4 更新 (v5 build)」节 (authoritative)。
v5 在 v4 上一次改 4 块 (★v4 节)。v3->v4 的 3 块 (扭曲/耦合/bury) 全保留为底子。

  ★v5 #1 去"可乐瓶/掐腰" (hull_polygon_local):
     v4 舷边 `+0.05*sin(π t*U(2,4)+φ)` 是把整条 w(t) 半宽包络掐出腰+鼓 = 葫芦/可乐瓶。
     FIX: 砍掉这条低频半宽起伏。船宽保持单调 (艏->max beam->平行中体->艉、中间不掐腰)。
     仍要边缘毛糙 -> 只在 RASTER 后给轮廓加极小高频边界小锯齿 (jitter_edge, 贴边 ±0.5-1px、
     不动半宽包络) -> 边缘不像 CAD 直线、但船宽包络单调。

  ★v5 #2 去"尖刺船头" (sample_hull_params + hull_polygon_local):
     v4 bow_sharp 可到 ~2.4(凹) + bow_frac 偏长 = 又长又凹的针刺艏。
     FIX: bow_sharp 全档压低 (cargo/warship/barge <=~1.3, 入水线偏直不凹); bow_frac 缩短。
     只 ancient(撞角) 仍可尖 (bow_sharp 高 + 细 stem); skiff 渔船钝(果核, bow_sharp≈1.0-1.1)。

  ★v5 #3 亮内 + 窄亮边 (paint_T_local + 常量) —— 实测纠正 (_brightness_probe, 109 frame):
     实测: interior_z 中位 -0.22 (内部≈海床、不暗)、rim_z 中位 -0.08 但 rim_peak_z 中位 +3.38
       (rim 均值低、峰值高 = 一条窄边里有稀疏的极亮峰)、真正黑的 shadow_z -1.82 是外部(Step2)。
     FIX:
       (a) 内部填充 = 亮 (略高于海床的恒定底色 T_FILL~0.35)、★砍掉 v4 的 `interior=T_FILL_LO*(1-s)
           +0.25*s` 随 s 扣暗 —— 内部底色不随 s 变 (s 只管结构丰富度)。
       (b) rim = 一条窄亮边: 带更窄 (0.10 半宽, 回 v2/实测窄边)、峰值高 (T_RIM 提到 3.4 锚 rim_peak_z)、
           但用高方差调制 (大部分 pixel 中等亮、稀疏 spike 到峰) -> 均值差小峰值差大 (锚实测)。
       (c) 暗格只在结构少数派、局部、不铺满: inter-rib 暗格仅在 has ribs 且 s 较高时画、且只挑
           少数格 (非全部相邻格)、深度浅 (T_CELL_DARK 抬到 -0.5)。

  ★v5 #4 增大渔船比例 (ARCHETYPES weights): skiff w 0.30 -> 0.45 (~45%), 其余按比例分剩下。

T 的语义 (offset, 喂给 port-preview 的 hl_target_mean_map):
    T(x,y) = offset, 大致 [-0.7, +3.6]:
      T ~ 0.35  : 内部底色 (亮于海床的填充, 不随 s 变)  -> hl_mean = bg_mean + bg_std*T
      T -> +    : 亮回波 (rim / 上层建筑块 / 断面 / 肋)
      T -> 负   : 暗格(浅) / 掩埋  -> 用 wb_shadow Weibull 暗场 (真黑=外部声影=Step2)
    与 live 的 HIGHLIGHT_OFFSET=2.5 对齐: rim 峰 T~3.4 锚 AI4 实测 rim_peak_z≈+3.4。

铁律: 纯 2D 几何/图像启发式, 绝无 3D/声学/掠射角; provenance 清白 (参数=船舶常识 +
  亮度锚=AI4 实测, 不读 KLSG)。只写 _build_v5/; 绝不改 live full_generate_denorm_b2.py / masks.py。

Environment: Python 3 with NumPy 2.x (np.ptp).
"""
import os, math, glob
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt

OUT = r"PATH_TO_BUILD_V5_OUTPUT"
SIL = r"PATH_TO_OPTIONAL_SILHOUETTES"

TILE = 300                    # tile size (px)
GLOBAL_SEED = 20260606

# T offset constants (aligned with live HIGHLIGHT_OFFSET=2.5 convention) -----------
# ★v5 #3: anchored to AI4 _brightness_probe (109 strict_clear frames):
#   interior_z med -0.22 (≈seabed, NOT dark) ; rim_z med -0.08 but rim_peak_z med +3.38
#   (rim = NARROW band whose MEAN is modest but whose PEAK spikes high) ; shadow_z -1.82
#   (the truly-black region is EXTERIOR acoustic shadow = Step2, not the target interior).
T_BG        = 0.0            # interior baseline ~ bg
T_RIM       = 3.4           # ★v5 #3b rim PEAK offset, anchored to AI4 rim_peak_z≈+3.4. The
#   rim is modulated to high VARIANCE (most rim px moderate, sparse px spike to this peak) so
#   the rim MEAN stays modest (~seabed+) while the PEAK is high — matches measured rim profile.
T_BLOCK     = 3.4           # superstructure deckhouse (bright, ~rim peak)
T_RIB       = 2.0           # transverse rib bright bar
T_KEEL      = 1.6           # keel centerline
T_FACE      = 3.6           # torn break face (hard sonar reflection, brightest)
T_DEBRIS    = 2.6           # debris blob
T_FILL      = 0.35          # ★v5 #3a interior fill = BRIGHT (slightly above seabed, ≈ measured
#   interior_z, but on the >0 side so the filled hull reads brighter-than-seabed, NOT a hollow
#   ring). CONSTANT — does NOT vary with s (砍掉 v4 的 interior=T_FILL_LO*(1-s)+0.25*s 扣暗).
T_CELL_DARK = -0.5          # ★v5 #3c inter-rib dark bay = SHALLOW (was -0.9). Dark bays are a
#   MINORITY/local cue, not a floor that darkens the whole interior; only a few bays get one.
T_HOLE      = -1.2          # skeleton bay punched to seabed (label=bg handled separately)


# =============================================================================
# L1  OUTLINE — archetype library + 3-segment hull as a SMOOTH SPLINE polygon
#
# ★v4 #3: each archetype carries a STRUCTURE PROFILE (s_band, has_superstructure prob,
#   rib density, block count) so archetype<->structure are COUPLED. Bias is continuous
#   (not a hard binary): skiff still gets ~0.10 chance of a block as an exception, etc.
#   `s` is drawn from each archetype's s_band so the coupling shows up statistically.
#
# Profile fields:
#   lb, mid, transom, bow_frac, bow_sharp : outline shape centers
#   w        : archetype sampling weight (skiff biased high = real small-craft prevalence)
#   s_band   : (lo,hi) structure-richness band for THIS archetype (skiff low, cargo/warship hi)
#   sup_p    : P(has_superstructure block) BASELINE (continuous bias, gated again by size)
#   nblk     : (min,max) block count WHEN has_superstructure fires
#   rib_a, rib_b : n_rib = rib_a + rib_b*s   (barge = strong regular deck ribs, no blocks)
#   pit      : True -> "fruit-pit" outline mode (blunt round bow+stern, full midbody)
#
# ★STRUCTURE-COUPLING FIX (2026-06-06, route-B arm3 re-bake): the v5 rib coupling was
#   PHYSICALLY BACKWARDS — it gave the decked big ships (cargo/warship) dense internal ribs
#   while leaving most skiffs smooth. In a top-down SSS, an internal "rib/frame" texture is the
#   signature of an OPEN/exposed structure:
#     OPEN  (deck up / rotted / open-bay) ->露肋: skiff (open fishing boat), ancient (rotted
#            deck exposes the frame = "鱼骨" / fishbone), barge (open compartments, regular bulkheads).
#     DECKED (intact steel deck)          -> 看不到肋: cargo, warship. The deck hides the frame;
#            from above you see only上层建筑块 (deckhouse / bridge / turret) + a flat deck face.
#   So arm3 internal structure is now: ribs ONLY for skiff/ancient/barge, blocks ONLY for
#   cargo/warship. Two NEW gate fields drive this WITHOUT touching hull shape / weights / block gate:
#     rib_on : bool — does THIS archetype expose internal ribs at all (open vs decked)?
#              decked cargo/warship = False -> arm3 internal structure = blocks, NEVER ribs.
#     rib_p  : float — P(an eligible ship of this archetype actually shows ribs), drawn against
#              a structure-rng. skiff is a small OPEN boat: it shows 1-2 sparse ribs MOST of the
#              time but a fraction stay smooth (hull-up / degraded / too small to read), so rib_p<1.
#              ancient/barge are skeleton/compartment archetypes -> rib_p=1.0 (always露肋 when eligible).
#   NOTE these fields ONLY change which T pixels get the rib bars; they do NOT touch Mlocal -> M_obj
#   stays byte-identical across arms (paint_T_local never writes Mlocal; verified).
# =============================================================================
# ★v5 #2 去尖刺: bow_sharp 全档压低 + bow_frac 缩短 (见下每档注释)。bow_sharp<=1.3 时
#   w=(t/bf)^bs 在 [0,1] 上是 ~线性偏外凸(入水线偏直、不向内凹成针), 配短 bow_frac -> 钝/尖而不刺。
#   只 ancient(撞角铁甲) 留 bow_sharp 高 + 细 stem = 唯一允许的尖艏 (规格 §1.2 撞角例外)。
# ★v5 #4 增大渔船比例: skiff w 0.30 -> 0.45 (~45%), 其余 (cargo/warship/ancient/barge) 分剩下 0.55。
ARCHETYPES = {
    # skiff 渔船 (OPEN 小渔船, 甲板朝上敞口): 果核形, L/B 2.5-3.0, rim 主导, 无块(块 p≈0.10 例外), s 低。
    #   ★v5 #2: 钝头 bow_sharp≈1.0-1.1, bow_frac 短。 ★v5 #4: 权重 0.45 (~45%)。
    #   ★STRUCTURE FIX 2026-06-06: skiff = 开敞船 -> 露肋. rib_on=True. SMALL boat = FEW sparse ribs
    #     (rib_a/rib_b kept at LEGACY 1.0/2.0 -> typically 1-2 frames). The "preserved smooth" skiffs
    #     are the LOW-s ones that the frozen v5 s<0.30 early gate already floors smooth (= hull-up /
    #     degraded / too small to resolve). rib_a/rib_b UNCHANGED from v5 (changing them would shift
    #     the rib-loop rng draw count and move baseline arm2's damaged ships — see RNG-STABILITY note
    #     in paint_T_local; the v5 eligible-skiff露肋 rate was already ~96%, physically correct).
    "skiff":   dict(lb=(2.5, 3.0), mid=0.55, transom=0.45, bow_frac=0.26, bow_sharp=1.1, w=0.45,
                    s_band=(0.10, 0.40), sup_p=0.10, nblk=(1, 1), rib_a=1.0, rib_b=2.0, pit=True,
                    rib_on=True),
    # cargo 货轮 (DECKED 钢甲板货轮): 长+偏方+钝头, s 高。 ★v5 #2: bow_sharp 1.25, bow_frac 0.18.
    #   ★STRUCTURE FIX 2026-06-06: cargo = 完整钢甲板 -> 看不到肋. rib_on=False -> arm3 internal
    #     structure = ONLY the superstructure block (bridge/deckhouse). sup_p=0.95, nblk=(1,3) UNCHANGED.
    #     rib_a/rib_b KEPT at v5 values (NOT moot): the rib/dark-bay/keel draws are still PERFORMED to
    #     keep the master rng byte-stable (baseline arm2 preserved); only the PAINTING is suppressed.
    "cargo":   dict(lb=(4.5, 6.0), mid=0.65, transom=0.60, bow_frac=0.18, bow_sharp=1.25, w=0.16,
                    s_band=(0.50, 0.90), sup_p=0.95, nblk=(1, 3), rib_a=2.0, rib_b=3.5, pit=False,
                    rib_on=False),
    # warship 军舰 (DECKED 钢甲板军舰): 长 slim 钝头, s 中高。 ★v5 #2: bow_sharp 1.3, bow_frac 0.20.
    #   ★STRUCTURE FIX 2026-06-06: warship = 完整钢甲板 -> 看不到肋. rib_on=False -> arm3 internal
    #     structure = ONLY blocks (bridge / turret). sup_p=0.90, nblk=(1,2) + rib_a/rib_b UNCHANGED
    #     (draws performed-then-discarded for rng stability; only blocks painted).
    "warship": dict(lb=(5.5, 6.5), mid=0.55, transom=0.35, bow_frac=0.20, bow_sharp=1.3, w=0.14,
                    s_band=(0.50, 0.80), sup_p=0.90, nblk=(1, 2), rib_a=1.0, rib_b=2.5, pit=False,
                    rib_on=False),
    # ancient 古船 (ROTTED 朽甲板古木船, 露肋骨=鱼骨相): ★v5 #2 唯一允许尖艏(撞角)。
    #   ★STRUCTURE FIX 2026-06-06: ancient = 朽甲板露肋骨 -> 露肋 (fishbone). rib_on=True (kept as the
    #     skeleton archetype). rib_a/rib_b UNCHANGED from v5 (1.0+2.0*s): v5 ancient already露肋 100%
    #     of eligible at this density = a clear rib ladder; raising it would shift rng draws & move
    #     baseline arm2. NO block (sup_p=0.05).
    "ancient": dict(lb=(4.0, 5.0), mid=0.50, transom=0.10, bow_frac=0.24, bow_sharp=2.3, w=0.13,
                    s_band=(0.30, 0.55), sup_p=0.05, nblk=(1, 1), rib_a=1.0, rib_b=2.0, pit=False,
                    rib_on=True),
    # barge 驳船 (OPEN 开敞驳船, 规则分舱隔板): 近矩形, 近零块, 强规则肋(甲板格)=真"肋无块"代表, s 中。
    #   ★STRUCTURE FIX 2026-06-06: barge = 开敞分舱 -> 强规则肋. rib_on=True. rib_a/rib_b KEPT strong
    #     v5 (3.0+4.0*s) = clean regular bulkhead ladder. NO block (sup_p=0.03).
    "barge":   dict(lb=(3.0, 4.0), mid=0.70, transom=0.65, bow_frac=0.16, bow_sharp=1.2, w=0.12,
                    s_band=(0.35, 0.70), sup_p=0.03, nblk=(1, 1), rib_a=3.0, rib_b=4.0, pit=False,
                    rib_on=True),
}


def sample_archetype(rng):
    keys = list(ARCHETYPES.keys())
    ws = np.array([ARCHETYPES[k]["w"] for k in keys], float)
    return keys[int(rng.choice(len(keys), p=ws / ws.sum()))]


def sample_s_for(arche, rng):
    """★v4 #3.2: s is DRIVEN BY archetype (skiff low, cargo/warship high) so a batch shows
       the archetype<->structure coupling statistically. Used when a spec leaves s unset."""
    lo, hi = ARCHETYPES[arche]["s_band"]
    return float(rng.uniform(lo, hi))


def decide_superstructure(arche, params, s, rng):
    """★v4 #3.3: BLOCK GATE = has_superstructure(archetype × size), NOT an s threshold.
       Continuous bias (not hard binary): each archetype has a baseline P(block); bigger /
       longer ships nudge it up a touch. Returns (has_block, n_block)."""
    a = ARCHETYPES[arche]
    p = a["sup_p"]
    # size nudge: longer hull -> slightly more likely to carry a deckhouse (gentle, ±0.10)
    lb = params["L"] / max(params["BW"], 1e-6)
    p = float(np.clip(p + 0.10 * (lb - 4.0) / 3.0, 0.0, 0.98))
    has = rng.random() < p
    if not has:
        return False, 0
    nlo, nhi = a["nblk"]
    # cargo: even at mid s keep >=1 block; high s can reach the max. warship 1-2.
    n = nlo + int(rng.random() < (0.4 + 0.6 * s)) * (nhi - nlo)
    n = int(np.clip(n, nlo, nhi))
    return True, n


def sample_hull_params(arche, rng):
    a = ARCHETYPES[arche]
    pit = a.get("pit", False)
    lb = rng.uniform(*a["lb"])
    L = rng.uniform(150, 240)
    BW = L / lb
    mid_frac = float(np.clip(a["mid"] + rng.uniform(-0.05, 0.05), 0.30, 0.72))
    # ★v5 #2 去尖刺: bow_frac 上限收到 0.28 (was 0.34) 防"船头拉长"。下限保 0.14 (钝箱头)。
    bow_frac = float(np.clip(a["bow_frac"] + rng.uniform(-0.03, 0.03), 0.14, 0.28))
    # ★v5 #2: fruit-pit skiff bow_sharp ≈ 1.0-1.15 (blunt round). non-ancient others keep their
    #   (already lowered, <=1.3) center ±0.25 but are CAPPED at 1.4 so cargo/warship = 尖而不刺,
    #   入水线偏直不凹. Only ancient (ram bow) center 2.3 stays high (uncapped, the尖艏 exception).
    if pit:
        bs_lo, bs_hi = 1.0, 1.15
    else:
        bs_lo, bs_hi = a["bow_sharp"] - 0.25, a["bow_sharp"] + 0.25
    cap_hi = 2.4 if arche == "ancient" else 1.4       # ★v5 #2: only ancient may be pointy/凹
    bow_sharp = float(np.clip(rng.uniform(bs_lo, bs_hi), 1.0, cap_hi))
    transom_w = float(np.clip(a["transom"] + rng.uniform(-0.06, 0.06), 0.0, 0.65))
    asym = rng.uniform(0.03, 0.08)
    keel_bend = rng.uniform(-0.04, 0.04)
    # #1.1 stem-post half-width (fraction of half-beam). ancient = ram bow -> tiny stem
    #   (allowed pointy per §1.2). skiff PIT = BLUNT ROUND bow -> FAT stem (~0.45-0.60) so
    #   both ends are round, reading like a peach/olive pit, NOT a needle/leaf. others blunt-ish.
    if arche == "ancient":
        stem_w = 0.05
    elif pit:
        stem_w = float(rng.uniform(0.45, 0.62))     # fat blunt stem -> rounded bow
    else:
        stem_w = float(rng.uniform(0.14, 0.22))
    # ★v4 #3: pit skiff = blunt ROUND stern too (transom large) + fuller midbody bulge below.
    sheer_f = rng.uniform(2, 4); sheer_phi = rng.uniform(0, 6.28)
    # #5 small LOCAL dent (one mild local beam collapse, NOT a global sine bend)
    dent_at = rng.uniform(0.30, 0.75); dent_w = rng.uniform(0.05, 0.12)
    dent_depth = rng.uniform(0.0, 0.18)   # 0 = no dent on many ships
    return dict(L=L, BW=BW, mid_frac=mid_frac, bow_frac=bow_frac, bow_sharp=bow_sharp,
                transom_w=transom_w, asym=asym, keel_bend=keel_bend, stem_w=stem_w,
                sheer_f=sheer_f, sheer_phi=sheer_phi, arche=arche, pit=pit,
                dent_at=dent_at, dent_w=dent_w, dent_depth=dent_depth)


def _smooth_periodic(pts, sigma=2.5):
    """#2: smooth a closed polygon by periodic Gaussian filtering of coords -> spline-like
       curve, kills the polygon vertices that the v2 label-image exposed."""
    n = len(pts)
    k = int(max(3, round(sigma * 3)) | 1)
    g = cv2.getGaussianKernel(k, sigma).ravel()
    out = np.empty_like(pts)
    ext = np.concatenate([pts[-k:], pts, pts[:k]], 0)   # wrap (periodic)
    for d in range(2):
        out[:, d] = np.convolve(ext[:, d], g, mode="same")[k:k + n]
    return out


def hull_polygon_local(p, n=400):
    """Half-beam w(t), t∈[0,1] bow->stern. bow taper -> PARALLEL MIDBODY (w≈1) ->
       stern run-in to flat TRANSOM (>0). + keel bend + low-freq sheer ripple +
       #5 one small LOCAL beam dent. Returned as a HULL-LOCAL polygon, long axis = +x,
       centered at origin, then #2 SMOOTHED into a spline-like closed curve."""
    L, BW = p["L"], p["BW"]
    bf, mf, bs, tw = p["bow_frac"], p["mid_frac"], p["bow_sharp"], p["transom_w"]
    stem_w = p.get("stem_w", 0.16)        # #1.1 bow STEM-POST floor: bow ends in a small
    #   finite width (a stem), NOT a zero-width needle -> ships don't read as leaves/pencils.
    #   ancient ram-bow archetype gets a tiny stem (can stay pointy, spec §1.2 allowance).
    pit = p.get("pit", False)             # ★v4 #3: fruit-pit skiff = blunt round ends + FULL midbody
    # midbody profile: cargo/warship/barge stay ~flat (parallel midbody, tiny -0.04 dip);
    #   pit skiff SWELLS the midbody (+bulge) so it reads as a fat peach/olive pit, not a plank.
    mid_bulge = 0.14 if pit else -0.04
    # pit stern: blunt round run-in (exponent ~1.0, so the stern fairs out roundly to a wide
    #   transom) instead of the cargo square-ish 1.4 run-in.
    stern_exp = 1.0 if pit else 1.4
    stern_frac = max(1.0 - bf - mf, 1e-6)
    top, bot = [], []
    for i in range(n + 1):
        t = i / n
        if t < bf:
            # taper from the stem (finite half-width at t=0) up to full beam at t=bf.
            # pit bow uses a fat stem + low exponent -> a ROUNDED blunt bow (no needle).
            w = stem_w + (1.0 - stem_w) * (t / bf) ** bs
        elif t < bf + mf:
            u = (t - bf) / mf
            w = 1.0 + mid_bulge * math.sin(u * math.pi)
        else:
            u = (t - (bf + mf)) / stern_frac
            w = 1.0 - (1.0 - tw) * (u ** stern_exp)
        w = max(0.0, w)
        # ★v5 #1 去"可乐瓶/掐腰": 砍掉 v4 的低频半宽起伏
        #   `+0.05*sin(π t*U(2,4)+φ)` —— 那条沿全船的低频正弦把整条半宽包络掐出腰+鼓 = 葫芦。
        #   现在 w_rail = w 单调 (艏->max beam->平行中体->艉、中间不掐腰)。边缘毛糙改到 raster
        #   后用极小高频边界小锯齿 (见 _hull_local_canvas 的 jitter_edge), 不动半宽包络。
        w_rail = w
        # #5 small local dent (gaussian notch in beam at ONE spot, mild). 这是"局部单点小凹陷"
        #   (一处塌陷段), 不是沿全船的周期起伏 -> 不会掐出可乐瓶腰; 保留 (规格 §3 D4 局部塌陷)。
        if p["dent_depth"] > 0:
            d = math.exp(-((t - p["dent_at"]) ** 2) / (2 * p["dent_w"] ** 2))
            w_rail = w_rail * (1.0 - p["dent_depth"] * d)
        w_rail = max(0.0, w_rail)
        x = (t - 0.5) * L
        bend = p["keel_bend"] * math.sin(t * math.pi) * BW
        wl = (BW / 2) * w_rail * (1 + p["asym"])
        wr = (BW / 2) * w_rail * (1 - p["asym"])
        top.append((x, -wl + bend))
        bot.append((x,  wr + bend))
    poly = np.array(top + bot[::-1], np.float32)
    poly = _smooth_periodic(poly, sigma=2.5)            # #2: spline-ify
    return poly


# =============================================================================
# L2  T-TEMPLATE PAINTING in HULL-LOCAL coords (s = structure richness)
#   Output: T_local (float32 offset map, hull-local frame) + Mlocal (uint8 binary)
#   ribs ⊥ keel (=local x), so ribs are VERTICAL lines in hull-local; dark cells =
#   real bays between two consecutive ribs ∩ hull; blocks aligned to ship axis.
# =============================================================================
def _jitter_edge(M, rng, amp_px=1.0):
    """★v5 #1: tiny HIGH-FREQUENCY edge roughness applied at RASTER time (NOT on the half-width
       envelope). We perturb only the 1px boundary band by ±amp_px so the contour reads granular
       (not a CAD-clean curve) WITHOUT touching the monotone beam envelope -> no cola-bottle waist.
       Implementation: add a small high-freq noise to the binary edge then re-threshold; net effect
       is the boundary gains a ~±1px ragged texture only. Interior/envelope untouched."""
    if amp_px <= 0 or M.sum() < 40:
        return M
    er = cv2.erode(M, np.ones((3, 3), np.uint8))
    dil = cv2.dilate(M, np.ones((3, 3), np.uint8))
    edge_band = (dil > 0) & (er == 0)              # 1-2px ring straddling the boundary
    f = M.astype(np.float32) / 255.0
    noise = rng.normal(0, 1, M.shape).astype(np.float32)
    # high-freq: blur with a tiny kernel so noise has ~1-2px correlation length (fine teeth)
    noise = cv2.GaussianBlur(noise, (0, 0), 0.8)
    f2 = f.copy()
    f2[edge_band] = f[edge_band] + 0.35 * amp_px * noise[edge_band]   # nudge boundary in/out
    Mj = (f2 > 0.5).astype(np.uint8) * 255
    # keep it from disconnecting: union with the eroded core so we never punch interior holes
    Mj = cv2.bitwise_or(Mj, er)
    return Mj


def _hull_local_canvas(poly_local, pad=24, rng=None, edge_jitter=1.0):
    """Rasterize the hull-local polygon into its own tight canvas (no global rot yet).
       ★v5 #1: optionally add tiny high-freq EDGE jitter (raster-time, boundary-only) so the
       contour is granular without a cola-bottle waist. Returns (Mlocal uint8, origin offset)."""
    xs = poly_local[:, 0]; ys = poly_local[:, 1]
    x0 = math.floor(xs.min()) - pad; y0 = math.floor(ys.min()) - pad
    W = int(math.ceil(xs.max()) - x0 + pad)
    H = int(math.ceil(ys.max()) - y0 + pad)
    pts = (poly_local - [x0, y0]).astype(np.int32)
    M = np.zeros((H, W), np.uint8)
    cv2.fillPoly(M, [pts], 255)
    if rng is not None and edge_jitter > 0:
        M = _jitter_edge(M, rng, amp_px=edge_jitter)
    return M, (x0, y0)


def paint_T_local(Mlocal, s, rng, arche="cargo", params=None, internal_structure=True):
    """Paint the hull-local T offset map. Long axis = +x (keel). Ribs are VERTICAL
       (⊥ keel). Returns (T float32, info).
       #3 rim thicker + more bright structure;  #6 ribs ⊥ keel (vertical here);
       #7 inter-rib dark cells = real bays (two ribs ∩ hull, filled, no leftover gaps).
       ★v4 #3: ARCHETYPE-COUPLED structure. arche+params decide:
         - rib density (rib_a + rib_b*s per archetype; barge = strong regular ribs)
         - WHETHER blocks exist (decide_superstructure gate: archetype×size, NOT s threshold)
           skiff/barge/ancient -> usually NO block; cargo/warship -> blocks.
       Blocks are clipped to the hull interior (never探出船舷 / no hammerhead overhang).

       ★STEP③ (route-B baker) internal_structure switch (ADDITIVE, default True = legacy v5):
         internal_structure=True  -> arm3: narrow bright rim + bright interior fill (T_FILL) +
             ALL internal fine structure (ribs / inter-rib dark bays / keel centerline / blocks).
         internal_structure=False -> arm2: narrow bright rim + bright interior fill (T_FILL) ONLY.
             NO ribs / NO dark bays / NO keel / NO blocks. The hull interior is filled FLAT at
             T_FILL (honest seabed-ish brightness, 岔路E: NOT pulled to arm1's fake +2.5σ).
       Crucially: when False the function returns at the SAME early-return point the low-s smooth
       variant uses (right after rim+fill), so for a given (Mlocal, s, rng, arche, params) the
       arm2 T and arm3 T are PIXEL-IDENTICAL everywhere except where arm3 adds internal structure,
       and consume rng IDENTICALLY up to that return -> downstream damage rng stays in lockstep."""
    prof = ARCHETYPES.get(arche, ARCHETYPES["cargo"])
    inside = Mlocal > 0
    H, W = Mlocal.shape
    T = np.zeros((H, W), np.float32)
    info = dict(blocks=[], n_rib=0, ribs=[])
    if inside.sum() < 40:
        return T, info

    dist = cv2.distanceTransform(Mlocal, cv2.DIST_L2, 5)
    maxd = dist.max() + 1e-6
    # ★v5 #3b NARROW bright rim (0.10 of half-beam, back from v4's 0.16). AI4 measures a NARROW
    #   bright edge whose MEAN is modest but PEAK is high — a wide ring would over-brighten the
    #   mean. Narrow band + high-variance modulation = the measured "窄而很亮的边".
    rim_w = max(2.5, 0.10 * maxd)

    # ★v5 #3a interior baseline = BRIGHT, CONSTANT (does NOT vary with s). v4 had
    #   `T_FILL_LO*(1-s)+0.25*s` which darkened the interior as s rose (the high-s "散块" tell:
    #   a dark interior speckled with bright bits). AI4 interior_z med ≈ -0.22 (≈seabed, NOT
    #   dark); we sit it just on the >0 side (T_FILL=0.35) so the filled hull reads brighter than
    #   seabed regardless of s. s now ONLY controls structure richness (ribs/blocks/bays), not
    #   interior darkness.
    interior_T = T_FILL
    T[inside] = interior_T
    rim = inside & (dist <= rim_w)
    # ★v5 #3b rim modulation = HIGH VARIANCE so MEAN stays modest but PEAK spikes (anchors AI4
    #   rim_z med -0.08 vs rim_peak_z med +3.38). Most rim pixels sit at a moderate fraction of
    #   the peak; a sparse subset spikes to ~full T_RIM. Implementation: a base low-freq term
    #   (plating variation) pushed LOW (mean ~0.42 of peak) + sparse high multiplicative spikes.
    ys_r, xs_r = np.where(rim)
    a = np.arctan2(ys_r - ys_r.mean(), xs_r - xs_r.mean())
    base = (0.42 + 0.14 * np.sin(a * rng.uniform(2, 4) + rng.uniform(0, 6.28))
            + 0.08 * np.sin(a * rng.uniform(5, 9) + rng.uniform(0, 6.28)))   # mean ~0.42 of peak
    # sparse bright spikes: ~22% of rim pixels jump toward the peak (variable-plating hot returns)
    spike = (rng.random(len(ys_r)) < 0.22) * rng.uniform(0.45, 0.58, len(ys_r))
    rmod = np.clip(base + spike, 0.22, 1.0)
    T[ys_r, xs_r] = T_RIM * rmod

    ys, xs = np.where(inside)
    lo, hi = int(xs.min()), int(xs.max())          # keel extent (x)
    span = hi - lo
    cyc = float(ys.mean())
    info.update(lo=lo, hi=hi, span=span, interior_T=interior_T)

    if (not internal_structure) or s < 0.30 or span < 24:
        # ★STEP③ arm2 (internal_structure=False) OR low-s smooth variant: narrow bright rim kept,
        #   BRIGHT interior fill (★v5 #3a: bright at T_FILL regardless of s), NO ribs/blocks/bays/
        #   keel. This is the EXACT pixel state arm3 starts from before adding internal structure.
        return T, info

    # ---- #6 ribs ⊥ keel (vertical bars here); ★STRUCTURE-COUPLING FIX 2026-06-06 ----
    #   PHYSICS: an internal rib/frame texture is the signature of an OPEN/exposed structure. Only
    #   OPEN archetypes露肋 (skiff = open boat, ancient = rotted deck/fishbone, barge = open
    #   bulkheads); the DECKED big ships (cargo/warship) have an intact steel deck that HIDES the
    #   frame, so they show NO internal ribs — only上层建筑块 (handled below by the block gate).
    #
    #   ★RNG-STABILITY DISCIPLINE (load-bearing): the route-B baker advances the SHARED master rng by
    #   re-running this paint with internal_structure=True (the arm3 superset). To keep that advance —
    #   and therefore the baseline arm2 templates + the damage geometry — BYTE-STABLE while flipping
    #   which archetypes露肋, we MUST consume rng IDENTICALLY to the legacy v5 logic for EVERY ship.
    #   So we COMPUTE the ribs (and below the dark bays / rib-paint / keel) with the exact same rng
    #   draws as before for ALL archetypes, and only the FINAL T-PAINTING is gated by rib_on:
    #     rib_on=True  (skiff/ancient/barge): ribs/keel/dark-bays painted (open structure露肋).
    #     rib_on=False (cargo/warship)      : same rng drawn, but painting DISCARDED (decked -> no
    #                                         ribs/keel; only blocks below). n_rib recorded as 0.
    rib_on = prof.get("rib_on", True)
    # compute rib positions with the LEGACY draw sequence (always, every archetype) ----
    n_rib = int(round(prof["rib_a"] + prof["rib_b"] * s))
    n_rib = int(np.clip(n_rib, 1, 7))
    barge_like = (arche == "barge")
    drop_p = 0.0 if barge_like else 0.15
    rib_jit = 0.06 if barge_like else 0.18       # barge ribs are REGULAR (low jitter)
    bw = span / (n_rib + 1)
    ribs = []
    for k in range(1, n_rib + 1):
        if rng.random() < drop_p:
            continue
        x = int(lo + bw * k + rng.uniform(-rib_jit, rib_jit) * bw)
        if not (lo < x < hi):
            continue
        ribs.append(x)
    ribs = sorted(ribs)
    # rib_on gate: DECKED ships keep the rng draws above but expose NO ribs (painting suppressed).
    ribs_paint = ribs if rib_on else []
    info["n_rib"] = len(ribs_paint); info["ribs"] = ribs_paint

    # ---- ★v5 #3c inter-rib DARK BAYS = MINORITY / LOCAL cue (NOT a floor that fills interior) ----
    #   v4 darkened EVERY inter-rib cell -> the interior read as a dark grid (the high-s "散块" tell
    #   + contradicts AI4 interior_z≈seabed). v5: keep the BRIGHT interior fill as the default; only
    #   a FEW bays (a minority) get a SHALLOW dark patch (a flooded/silted compartment), and even
    #   those darken only ~half the bay (a local pocket, eroded off the rim) — so dark格 is a
    #   structural少数派 cue, not a wall-to-wall darkening. Higher s -> a few more bays may darken.
    #   ★RNG-STABILITY: the bays are formed from the COMPUTED `ribs` (legacy, pre-gate) so the
    #   rng.choice + per-bay uniform draws below are byte-identical to legacy for every archetype.
    #   For DECKED ships (rib_on=False) the dark painting is suppressed (no ribs -> no inter-rib bays
    #   to silt) by `paint_dark`, but the draws still happen -> master rng stays byte-stable.
    paint_dark = rib_on
    inner = cv2.erode(Mlocal, np.ones((3, 3), np.uint8))   # stay just inside the rim
    inner_b = inner > 0
    edges = [lo] + ribs + [hi]
    bays = list(zip(edges[:-1], edges[1:]))
    # number of dark bays: a MINORITY of all bays, scaled mildly by s but CAPPED at ~1/3 of bays
    #   (and at most 2) so dark格 stays a structural少数派 cue, never铺满内部 (★v5 #3c). Many ships
    #   (esp. low s) get ZERO dark bays.
    n_dark = int(np.clip(round((0.12 + 0.22 * s) * len(bays)), 0, min(2, max(0, len(bays) - 1))))
    dark_idx = set(rng.choice(len(bays), size=n_dark, replace=False).tolist()) if n_dark > 0 else set()
    for bi, (a, b) in enumerate(bays):
        if bi not in dark_idx or b - a < 10:
            continue
        # darken only a LOCAL pocket inside the bay (not the whole cell): a small sub-span AND a
        # vertical sub-band so it reads as a silted pocket, leaving bright fill around it.
        cw = b - a
        px0 = int(a + cw * rng.uniform(0.20, 0.40)); px1 = int(b - cw * rng.uniform(0.20, 0.40))
        if px1 <= px0 + 2:
            continue
        # vertical sub-band: darken only a central beam-fraction of the bay's height
        col_rows = np.where(inner_b[:, (px0 + px1) // 2])[0]
        cell = np.zeros_like(inside)
        cell[:, px0:px1] = True
        if len(col_rows) > 6:
            ry0 = col_rows.min(); ry1 = col_rows.max(); rh = ry1 - ry0
            vy0 = int(ry0 + rh * rng.uniform(0.12, 0.28)); vy1 = int(ry1 - rh * rng.uniform(0.12, 0.28))
            vband = np.zeros_like(inside); vband[max(0, vy0):vy1, :] = True
            cell &= vband
        cell &= inner_b & (T < T_RIM - 0.3)        # don't overwrite the rim
        # shallow dark (T_CELL_DARK=-0.5) with mild jitter; this is a faint silted bay, not black.
        dval = T_CELL_DARK * rng.uniform(0.7, 1.0)  # draw ALWAYS (rng stability); paint only if open
        if paint_dark:
            T[cell] = dval

    # ---- ribs drawn ON TOP of the dark bays (bright bars between dark cells) ----
    #   ★RNG-STABILITY: iterate the COMPUTED `ribs` (legacy count) so the per-rib val/thick draws are
    #   byte-identical for every archetype; paint only when rib_on (DECKED ships draw-but-discard).
    for x in ribs:
        val = T_RIB * rng.uniform(0.55, 1.0)        # uneven brightness (drawn always)
        thick = 1 + int(rng.random() < 0.4)         # drawn always
        if not rib_on:
            continue                                 # decked: consume draws, paint nothing
        for d in range(thick):
            xp = x + d
            if lo < xp < hi:
                col = inside[:, xp]
                T[col, xp] = np.maximum(T[col, xp], val)

    # ---- keel centerline p~0.5 (longitudinal bright line along x) ----
    #   ★STRUCTURE-COUPLING FIX 2026-06-06: the keel is an INTERNAL longitudinal frame, same
    #   open-structure family as the ribs — you only see it on an OPEN/exposed hull, NOT through a
    #   decked ship's intact steel deck. So gate it on rib_on too (cargo/warship = decked = no keel),
    #   consistent with "cargo/warship 只块无肋". We STILL draw the rng decision (uniform stream),
    #   then paint only when rib_on.
    draw_keel = rng.random() < 0.5
    if rib_on and draw_keel:
        cy = int(round(cyc))
        for cyo in (cy, cy + 1):
            if 0 <= cyo < H:
                row = inside[cyo, lo:hi]
                seg = np.arange(lo, hi)[row]
                T[cyo, seg] = np.maximum(T[cyo, seg], T_KEEL)

    # ---- ★v4 #3: superstructure blocks GATED by archetype×size, NOT s threshold ----
    #   skiff/barge/ancient -> usually has_block=False (skiff keeps ~0.10 exception).
    #   cargo -> ≥1 even at mid s; warship 1-2. n_block from decide_superstructure().
    if params is None:
        # fallback for callers without params (e.g. real-profile seed): use s-driven legacy.
        has_block = s > 0.55; n_block = 1 + int(s > 0.55) + int(s > 0.8 and rng.random() < 0.6)
    else:
        has_block, n_block = decide_superstructure(arche, params, s, rng)
    info["has_superstructure"] = bool(has_block and n_block > 0)
    used = []
    for _ in range(n_block if has_block else 0):
        for _try in range(8):
            c0f = rng.uniform(0.20, 0.78)
            ln = rng.uniform(0.10, 0.22)            # #3 a bit bigger blocks
            if any(abs(c0f - u) < 0.13 for u in used):
                continue
            used.append(c0f)
            c0 = int(lo + span * c0f); c1 = int(min(hi, c0 + span * ln))
            if c1 - c0 < 5:
                break
            seg = inside[:, c0:c1]
            rr = np.where(seg.any(1))[0]
            if not len(rr):
                break
            pad = int(np.ptp(rr) * rng.uniform(0.20, 0.36))
            r0, r1 = rr.min() + pad, rr.max() - pad
            if r1 <= r0:
                break
            blk = np.zeros_like(inside)
            blk[r0:r1 + 1, c0:c1] = True
            blk &= inside
            T[blk] = T_BLOCK + rng.uniform(-0.15, 0.2)
            info["blocks"].append((c0f, ln, int(blk.sum())))
            # p~0.3: a couple of faint dark window seams (mullions), NOT a full grid
            if rng.random() < 0.3 and blk.sum() > 70:
                for gx in np.linspace(c0, c1, rng.integers(2, 4) + 1)[1:-1]:
                    col = blk[:, int(gx)]
                    T[col, int(gx)] = T_BLOCK * 0.55
            break
    return T, info


# =============================================================================
# L3  DAMAGE OPERATORS  — edit BOTH Mlocal and T_local in HULL-LOCAL coords
#   (break cut in hull-local BEFORE global rotation, per spec §3).
# =============================================================================
def _coarse_blocky_edge(n, amp, rng):
    """#4 coarse, low-freq, BLOCKY irregular breakout edge (NO fine teeth, NO clean cut):
       sum of a couple very-low harmonics + a few big blocky step chunks."""
    base = np.zeros(n)
    for _ in range(int(rng.integers(2, 4))):
        k = rng.uniform(0.6, 2.0)
        ph = rng.uniform(0, 6.28)
        base += rng.uniform(0.4, 1.0) * np.sin(np.arange(n) / max(n, 1) * math.pi * k + ph)
    m = np.abs(base).max() + 1e-6
    base = base / m * amp
    for _ in range(int(rng.integers(1, 4))):    # big blocky chunks (wide step offsets)
        c = int(rng.integers(0, n)); bw = int(rng.integers(max(2, n // 12), max(3, n // 5)))
        base[max(0, c - bw):c + bw] += rng.uniform(-1, 1) * amp * 0.9
    return base


def _ragged_chunk_field(H, W, xc, off_row, rng, depth, period):
    """#4 build a per-row CRUMBLED edge mask: starting from the blocky cut line, eat
       further into the hull by a low-frequency, big-amplitude, BLOCKY amount that differs
       per band -> the half-end looks like big chunks fell off (崩口), not a clean slice.
       Returns a per-row extra-eat (in px, >=0) on the given side (sign in `side`)."""
    eat = np.zeros(H)
    r = 0
    while r < H:
        band = int(rng.integers(period // 2, period + 1))
        e = max(0.0, rng.uniform(-0.25, 1.0)) * depth      # many bands eat 0 (intact), some eat a lot
        eat[r:r + band] = e
        r += band
    # light smooth so band steps are chunky but not 1px aliased
    eat = cv2.GaussianBlur(eat.reshape(-1, 1), (1, 5), 0).ravel()
    return eat


def dmg_break_local(Mlocal, T, info, rng):
    """#4 transverse break in hull-local: split at x in [0.35,0.65]*span. Carve a coarse
       BLOCKY OPEN gap, then eat RAGGED BIG CHUNKS off BOTH half-ends (崩口, low-freq,
       big-amplitude, blocky — NO fine teeth, NO clean perpendicular slice). Brighten the
       crumbled faces (~T_FACE). Translate+hinge the stern half (stays adjacent). Edits
       BOTH Mlocal and T."""
    inside = Mlocal > 0
    ys, xs = np.where(inside)
    lo, hi = int(xs.min()), int(xs.max()); span = hi - lo
    H, W = Mlocal.shape
    xc = int(lo + rng.uniform(0.38, 0.62) * span)
    gap_half = rng.uniform(4.0, 8.0)
    off = _coarse_blocky_edge(H, amp=14.0, rng=rng)          # per-row blocky cut centerline
    chunk_depth = rng.uniform(8.0, 18.0)                     # how deep chunks can crumble
    period = max(8, int(span * 0.10))
    eat_bow = _ragged_chunk_field(H, W, xc, off, rng, chunk_depth, period)   # eats LEFT half end
    eat_stern = _ragged_chunk_field(H, W, xc, off, rng, chunk_depth, period) # eats RIGHT half end
    xx = np.arange(W)[None, :]
    cut_center = (xc + off)[:, None]
    # bow half kept where x < cut_center - eat_bow ; stern half kept where x > cut_center + gap + eat_stern
    bow_keep = inside & (xx < (cut_center - gap_half - eat_bow[:, None]))
    stern_keep = inside & (xx > (cut_center + gap_half + eat_stern[:, None]))
    Mnew = Mlocal.copy(); Tnew = T.copy()
    removed = inside & (~bow_keep) & (~stern_keep)
    Mnew[removed] = 0; Tnew[removed] = 0.0                   # open water between/at the崩口

    # translate+hinge the STERN half a bit about its centroid (halves stay adjacent)
    if stern_keep.sum() > 50:
        sy, sx = np.where(stern_keep)
        ctr = (float(sx.mean()), float(sy.mean()))
        dx = rng.uniform(-0.03, 0.07) * span
        dy = rng.uniform(-0.06, 0.06) * (ys.max() - ys.min() + 1)
        hinge = rng.uniform(-16, 16)
        R = cv2.getRotationMatrix2D(ctr, hinge, 1.0)
        R[0, 2] += dx; R[1, 2] += dy
        sm = cv2.warpAffine((stern_keep.astype(np.uint8) * 255), R, (W, H), flags=cv2.INTER_NEAREST)
        st = cv2.warpAffine(T * stern_keep, R, (W, H), flags=cv2.INTER_LINEAR)
        Mnew = np.zeros((H, W), np.uint8)
        Mnew[bow_keep] = 255; Mnew[sm > 0] = 255
        Tnew = np.zeros((H, W), np.float32)
        Tnew[bow_keep] = T[bow_keep]; Tnew[sm > 0] = st[sm > 0]

    # crumbled faces = THIN edge of the new mask near the seam -> bright hard reflection.
    #   (thin rim following the ragged contour, NOT a fat clean white block.)
    seam_band = np.zeros((H, W), np.uint8)
    seam_lo = int(xc - chunk_depth - gap_half - 4); seam_hi = int(xc + chunk_depth + gap_half + 4)
    seam_band[:, max(0, seam_lo):min(W, seam_hi)] = 1
    er = cv2.erode(Mnew, np.ones((3, 3), np.uint8))
    edge = (Mnew > 0) & (er == 0) & (seam_band > 0)
    Tnew[edge] = T_FACE
    return Mnew, Tnew


def dmg_bury_local(Mlocal, T, info, rng):
    """★v4 #3 BURY REWRITE (replaces v3 end-fade; fixes #2 rim-dims-to-vanishing).

    One whole END (bow OR stern, randomly chosen — NEVER both) sinks into the seabed:
      - pick a side along the keel (x): bow = low-x end, stern = high-x end.
      - draw a ROUGH IRREGULAR SEDIMENT LINE across the ship's WIDTH (a near-transverse
        boundary in hull-local that wobbles low-freq + blocky, same crumble style as the
        break崩口 — NOT a straight cut).
      - EVERYTHING on the buried side of that line — contour, rim, ribs, blocks alike —
        is REMOVED from M_obj (label -> background, it has merged into the sediment) and
        T set to a deep-negative sediment value over a thin transition collar so the port
        stage blends it into the wb_shadow seabed grain rather than leaving a bright cliff.
      - The far end keeps its full bright rim/structure (this is why it fixes #2: we no longer
        ramp the rim down to nothing along the whole hull — only the buried end vanishes).

    Edits BOTH Mlocal and T. Returns (Mnew, Tnew)."""
    inside = Mlocal > 0
    ys, xs = np.where(inside)
    lo, hi = int(xs.min()), int(xs.max())
    span = max(hi - lo, 1)
    H, W = Mlocal.shape
    Mnew = Mlocal.copy(); Tnew = T.copy()
    if inside.sum() < 60:
        return Mnew, Tnew

    bury_bow = rng.random() < 0.5                       # one end only, never both
    frac = rng.uniform(0.22, 0.42)                      # how much of the length is buried
    base = (lo + span * frac) if bury_bow else (hi - span * frac)

    # rough irregular sediment line: per-ROW x-position of the boundary (the line runs across
    # the ship's WIDTH = y), wobbling along y with the SAME low-freq blocky crumble as a break.
    wob = _coarse_blocky_edge(H, amp=0.10 * span, rng=rng)     # ±~10% of length, blocky
    bound_x = base + wob                                       # per-row boundary x (length-wise)

    xx = np.arange(W)[None, :]
    bx = bound_x[:, None]
    if bury_bow:
        buried = inside & (xx < bx)                     # everything toward the low-x (bow) end
        # signed distance INTO the hull from the line (px), >0 = deeper under sediment
        depth = (bx - xx)
    else:
        buried = inside & (xx > bx)                     # toward the high-x (stern) end
        depth = (xx - bx)

    # transition collar: a thin band right at the sediment line stays partly visible (the
    # structure half-emerging from the sand), then fully gone deeper in. collar ~ 0.05*span.
    collar = max(4.0, 0.05 * span)
    # fully buried (label -> background): deeper than the collar -> remove from M and set T deep
    gone = buried & (depth > collar)
    Mnew[gone] = 0; Tnew[gone] = T_CELL_DARK * 1.4
    # collar band: ramp T from its current value down toward sediment as depth -> collar.
    band = buried & (depth >= 0) & (depth <= collar)
    if band.any():
        w_s = np.clip(depth / collar, 0, 1).astype(np.float32)   # 0 at line, 1 at full burial
        Tnew[band] = T[band] * (1 - w_s[band]) + (T_CELL_DARK * 1.3) * w_s[band]
    return Mnew, Tnew


def dmg_gunwale_gap_local(Mlocal, T, info, rng):
    """#L3 D3 breach the bright rim at 1-2 spots (collapsed gunwale). T -> interior at breach."""
    inside = Mlocal > 0
    ys, xs = np.where(inside)
    lo, hi = int(xs.min()), int(xs.max()); span = hi - lo
    H, W = Mlocal.shape
    dist = cv2.distanceTransform(Mlocal, cv2.DIST_L2, 5)
    rim = (dist <= max(2.5, 0.10 * dist.max())) & inside   # ★v5 #3b match narrower rim
    interior_T = info.get("interior_T", T_FILL)            # breach reveals bright interior fill
    Tnew = T.copy()
    for _ in range(int(rng.integers(1, 3))):
        x = int(lo + span * rng.uniform(0.15, 0.85))
        w = max(3, int(span * rng.uniform(0.03, 0.07)))
        seg = np.zeros((H, W), bool); seg[:, max(0, x - w):x + w] = True
        breach = rim & seg
        Tnew[breach] = interior_T
    return Mlocal.copy(), Tnew


def dmg_internal_distort_local(Mlocal, T, info, rng, amp=None):
    """★v4 #1 SMALL + LOCAL internal distortion (replaces v3 noodle-bend).

    Root cause of v3 noodle look: coarse field (H//16 = low-freq -> whole-ship bend) +
    amp=3.5 (peak ~10px ≈ 50-65% of half-beam) + warping the CONTOUR M with the full field.

    v4 fix (per spec ★v3 #1):
      - amp 0.8-1.2 (peak ~2.5-3.5px ≈ 12-18% of half-beam) -> small.
      - FINER field grid (H//5 ~ H//7) so the warp is fine-grain, not a single global wave.
      - LOCAL ENVELOPE: multiply the field by 1-2 gaussian bumps at random positions so the
        distortion is confined to 1-2 small patches (a local interior collapse), not all over.
      - Warp ONLY the interior T at full amp; the CONTOUR M is warped with a 0.3x field so the
        HULL SHELL barely flexes (no rubber-yield). Interior structure slumps; shell stays put.
    Edits BOTH Mlocal and T (M only slightly). Returns (Mnew, Tnew)."""
    H, W = Mlocal.shape
    if amp is None:
        amp = float(rng.uniform(0.8, 1.2))
    # finer field: grid at H//5 ~ H//7 (was H//16). higher grid -> higher spatial frequency.
    div = int(rng.integers(5, 8))                       # 5,6,7
    gh = max(3, H // div); gw = max(3, W // div)
    nx = rng.normal(0, 1, (gh, gw)).astype(np.float32)
    ny = rng.normal(0, 1, (gh, gw)).astype(np.float32)
    dx = cv2.resize(nx, (W, H), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(ny, (W, H), interpolation=cv2.INTER_CUBIC)

    # LOCAL envelope: 1-2 gaussian bumps at random interior positions -> confine the warp.
    ys, xs = np.where(Mlocal > 0)
    env = np.zeros((H, W), np.float32)
    if len(xs):
        n_bump = int(rng.integers(1, 3))               # 1 or 2 small patches
        for _ in range(n_bump):
            j = int(rng.integers(0, len(xs)))
            cx, cy = float(xs[j]), float(ys[j])
            sig = rng.uniform(0.10, 0.20) * max(np.ptp(xs), np.ptp(ys))   # small patch
            yy, xxg = np.mgrid[0:H, 0:W]
            env += np.exp(-(((xxg - cx) ** 2 + (yy - cy) ** 2) / (2 * sig * sig))).astype(np.float32)
        env = np.clip(env, 0, 1)
    dx *= amp * env; dy *= amp * env                    # field only acts inside the bump(s)

    mx, my = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    # interior T: full-amplitude warp (ribs/blocks slump locally)
    map_x = np.ascontiguousarray(mx + dx); map_y = np.ascontiguousarray(my + dy)
    Tnew = cv2.remap(T, map_x, map_y, cv2.INTER_LINEAR)
    # contour M: 0.3x field only -> shell barely moves (no whole-ship rubber bend)
    mxs = np.ascontiguousarray(mx + 0.3 * dx); mys = np.ascontiguousarray(my + 0.3 * dy)
    Mnew = cv2.remap(Mlocal, mxs, mys, cv2.INTER_NEAREST)
    # keep T zero outside the (barely moved) hull so we don't leak warped interior past the shell
    Tnew[Mnew == 0] = 0.0
    return (Mnew > 0).astype(np.uint8) * 255, Tnew


def dmg_skeleton_local(Mlocal, T, info, rng):
    """#L3 D7 heavy decay: keep rim + thin ribs; inter-rib bays punch through to seabed.
       Returns (Mlocal, T, holes_local) where holes = label background (seabed shows)."""
    inside = Mlocal > 0
    dist = cv2.distanceTransform(Mlocal, cv2.DIST_L2, 5)
    rim = inside & (dist <= max(2.5, 0.10 * dist.max()))   # ★v5 #3b match narrower rim
    bright = (T > 1.0) & inside                            # ★v5 catch ribs/blocks (T_RIB=2.0×0.55..)
    bu = (bright).astype(np.uint8) * 255
    er = cv2.erode(bu, np.ones((3, 3), np.uint8))
    block_like = er > 0
    rib_like = bright & (~block_like)
    keep = rim | rib_like
    holes = inside & (~keep)                # open bays + rotted blocks punch through
    Tnew = T.copy()
    Tnew[holes] = T_HOLE
    return Mlocal.copy(), Tnew, (holes.astype(np.uint8) * 255)


def dmg_debris_local(Mlocal, T, info, rng):
    """#L3 D6 structured bright debris blobs near the hull (<0.6L), NOT flung, NOT noise.
       Returns a separate (debris_local_mask, debris_T) painted in hull-local frame."""
    inside = Mlocal > 0
    ys, xs = np.where(inside)
    if not len(xs):
        return np.zeros_like(Mlocal), np.zeros_like(T)
    cx, cy = int(xs.mean()), int(ys.mean())
    L = max(np.ptp(xs), np.ptp(ys))
    H, W = Mlocal.shape
    dmask = np.zeros_like(Mlocal); dT = np.zeros_like(T)
    for _ in range(int(rng.integers(3, 9))):
        r = rng.uniform(0.15, 0.55) * L; th = rng.uniform(0, 6.28)
        px = int(np.clip(cx + r * math.cos(th), 4, W - 5))
        py = int(np.clip(cy + r * math.sin(th), 4, H - 5))
        ax = int(rng.integers(2, 6)); ay = int(rng.integers(2, 6)); rot = rng.uniform(0, 180)
        cv2.ellipse(dmask, (px, py), (ax, ay), rot, 0, 360, 255, -1)
        cv2.ellipse(dT, (px, py), (ax, ay), rot, 0, 360, float(T_DEBRIS * rng.uniform(0.8, 1.1)), -1)
    return dmask, dT


# =============================================================================
# GEOMETRY: warp BOTH M_obj and T (same rotation/scale/placement) into the TILE.
#   T uses INTER_LINEAR (NOT thresholded); M_obj uses INTER_NEAREST. (spec A/B)
# =============================================================================
def place_local_on_tile(Mlocal, Tlocal, ang_deg, cx=TILE // 2, cy=TILE // 2,
                        extra_layers=None):
    """Rotate+translate the hull-local (M,T) into the global TILE. T warped with
       INTER_LINEAR, M with INTER_NEAREST. extra_layers = list of (name, local_arr,
       interp) warped identically (e.g. holes, debris)."""
    H, W = Mlocal.shape
    lcx, lcy = W / 2.0, H / 2.0
    R = cv2.getRotationMatrix2D((lcx, lcy), ang_deg, 1.0)
    R[0, 2] += cx - lcx; R[1, 2] += cy - lcy
    Mg = cv2.warpAffine(Mlocal, R, (TILE, TILE), flags=cv2.INTER_NEAREST)
    Tg = cv2.warpAffine(Tlocal, R, (TILE, TILE), flags=cv2.INTER_LINEAR)
    Mg = (Mg > 127).astype(np.uint8) * 255
    out = {}
    if extra_layers:
        for name, arr, interp in extra_layers:
            out[name] = cv2.warpAffine(arr, R, (TILE, TILE), flags=interp)
    return Mg, Tg, out


# =============================================================================
# ASSEMBLY: build one wreck -> (M_obj TILE binary, T TILE offset map, meta)
# =============================================================================
def build_one(spec, rng):
    """Returns dict with M_obj (uint8 0/255), T (float32 offset), holes (uint8 or None),
       debris_mask/debris_T (or None), view, phi, proud_flag, txt, info."""
    ang = rng.uniform(0, 360)
    view = spec["view"]; damages = spec.get("damages", [])
    real_path = spec.get("real_path")
    arche = spec.get("arche", "cargo")
    # ★v4 #3.2: s is archetype-driven. If a spec pins s explicitly we honor it (for systematic
    #   montage coverage), else draw from the archetype's s_band so coupling shows up.
    s = spec["s"] if spec.get("s") is not None else sample_s_for(arche, rng)

    if real_path:
        Mlocal, Tlocal, info = real_plan_local(real_path, s, rng)
    else:
        params = sample_hull_params(arche, rng)
        poly = hull_polygon_local(params)
        Mlocal, _ = _hull_local_canvas(poly, rng=rng, edge_jitter=1.0)   # ★v5 #1 edge jitter
        # ★v4 #3.1/#3.3: pass archetype + params so structure (ribs/blocks) couples to type.
        Tlocal, info = paint_T_local(Mlocal, s, rng, arche=arche, params=params)

    holes_local = None; debris_pair = None

    # --- damage ops in HULL-LOCAL (break first, before rotation, per spec §3) ---
    if "break" in damages:
        Mlocal, Tlocal = dmg_break_local(Mlocal, Tlocal, info, rng)
    if "internal_distort" in damages:
        Mlocal, Tlocal = dmg_internal_distort_local(Mlocal, Tlocal, info, rng)
    if "gunwale_gap" in damages:
        Mlocal, Tlocal = dmg_gunwale_gap_local(Mlocal, Tlocal, info, rng)
    if "bury" in damages:
        Mlocal, Tlocal = dmg_bury_local(Mlocal, Tlocal, info, rng)
    if "skeleton" in damages:
        Mlocal, Tlocal, holes_local = dmg_skeleton_local(Mlocal, Tlocal, info, rng)
    if "debris" in damages:
        debris_pair = dmg_debris_local(Mlocal, Tlocal, info, rng)

    # --- warp BOTH M and T (+ holes/debris) identically into the TILE ---
    extra = []
    if holes_local is not None:
        extra.append(("holes", holes_local, cv2.INTER_NEAREST))
    if debris_pair is not None:
        extra.append(("dmask", debris_pair[0], cv2.INTER_NEAREST))
        extra.append(("dT", debris_pair[1], cv2.INTER_LINEAR))
    Mg, Tg, out = place_local_on_tile(Mlocal, Tlocal, ang, extra_layers=extra)

    holes = out.get("holes")
    dmask = out.get("dmask"); dT = out.get("dT")
    if holes is not None:
        holes = (holes > 127).astype(np.uint8) * 255
    if dmask is not None:
        dmask = (dmask > 127).astype(np.uint8) * 255

    phi = rng.uniform(0, 2 * math.pi) if view in ("proud", "side") else None
    proud_flag = view in ("proud", "side")
    return dict(M_obj=Mg, T=Tg, holes=holes, dmask=dmask, dT=dT,
                view=view, phi=phi, proud_flag=proud_flag,
                txt=spec.get("txt", ""), info=info, s=s,
                arche=spec.get("arche", "real"), damages=damages)


# ---- 1.3 real-profile seed (optional): plan-view deck mask -> hull-local (M,T) ----
def real_plan_local(path, s, rng):
    g = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    if g is None:
        params = sample_hull_params("cargo", rng)
        Mlocal, _ = _hull_local_canvas(hull_polygon_local(params))
        return Mlocal, *paint_T_local(Mlocal, s, rng)[:1], paint_T_local(Mlocal, s, rng)[1]
    _, th = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY_INV)
    n, lab, st, _ = cv2.connectedComponentsWithStats(th)
    if n <= 1:
        params = sample_hull_params("cargo", rng)
        Mlocal, _ = _hull_local_canvas(hull_polygon_local(params))
        T, info = paint_T_local(Mlocal, s, rng); return Mlocal, T, info
    idx = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    blob = (lab == idx).astype(np.uint8) * 255
    ys, xs = np.where(blob); blob = blob[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # orient long-axis horizontal (keel = local x), pad
    h, w = blob.shape
    if h > w:
        blob = cv2.rotate(blob, cv2.ROTATE_90_CLOCKWISE); h, w = blob.shape
    sc = 180.0 / max(h, w)
    blob = cv2.resize(blob, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_NEAREST)
    h, w = blob.shape
    pad = 24
    Mlocal = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    Mlocal[pad:pad + h, pad:pad + w] = (blob > 110).astype(np.uint8) * 255
    # smooth contour edges (#2)
    cnts, _ = cv2.findContours(Mlocal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts:
        c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
        c = _smooth_periodic(c, sigma=2.0)
        Mlocal = np.zeros_like(Mlocal)
        cv2.fillPoly(Mlocal, [c.astype(np.int32)], 255)
    T, info = paint_T_local(Mlocal, s, rng)
    return Mlocal, T, info


# =============================================================================
# §6 SAMPLE GRID (>=24 systematic coverage)  +  TEMPLATE MONTAGE (M_obj + T)
# =============================================================================
def _lab(t, s, y=16):
    cv2.putText(t, s, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(t, s, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1, cv2.LINE_AA)


def _T_to_rgb(T, M_obj, holes=None, dmask=None):
    """Colorize the T offset map for human reading (BGR):
       blue = negative (dark cells / shadow / bury), grey = ~bg,
       orange->white = positive (rim / block / face). Outside hull = dark grey."""
    H, W = T.shape
    rgb = np.full((H, W, 3), 30, np.uint8)
    inside = M_obj > 0
    if dmask is not None:
        inside = inside | (dmask > 0)
    tv = T
    neg = (tv < -0.05) & inside
    pos = (tv > 0.05) & inside
    mid = inside & (~neg) & (~pos)
    nval = np.clip(-tv / 1.3, 0, 1)
    rgb[neg] = np.stack([(40 + 150 * nval[neg]).astype(np.uint8),    # B
                         (30 + 50 * nval[neg]).astype(np.uint8),     # G
                         (10 + 5 * nval[neg]).astype(np.uint8)], -1) # R
    rgb[mid] = (110, 110, 110)
    pval = np.clip(tv / 3.2, 0, 1)
    rgb[pos] = np.stack([(60 + 195 * pval[pos]).astype(np.uint8),    # B
                         (90 + 165 * pval[pos]).astype(np.uint8),    # G
                         (160 + 95 * pval[pos]).astype(np.uint8)], -1)  # R
    if holes is not None:
        rgb[holes > 0] = (90, 90, 30)
    cont, _ = cv2.findContours((M_obj > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgb, cont, -1, (255, 255, 255), 1)
    return rgb


def build_specs():
    specs = []
    # --- A. ★v4 #3 ARCHETYPE<->STRUCTURE COUPLING (s=None -> archetype's s_band drives it) ---
    #   skiff = fruit-pit, no block, rim-dominant;  cargo/warship = long + blocks;
    #   barge = strong regular ribs, no block;  ancient = pointy, no block.
    specs += [
        dict(arche="skiff",   s=None, damages=[], view="footprint", txt="skiff PIT noblk"),
        dict(arche="skiff",   s=None, damages=[], view="footprint", txt="skiff PIT 2"),
        dict(arche="cargo",   s=None, damages=[], view="footprint", txt="cargo blocks"),
        dict(arche="warship", s=None, damages=[], view="footprint", txt="warship blk slim"),
        dict(arche="barge",   s=None, damages=[], view="footprint", txt="barge RIBS noblk"),
        dict(arche="ancient", s=None, damages=[], view="footprint", txt="ancient pointy noblk"),
        # explicit s-axis demo (pinned s) so the s low->high fill gradient is visible too
        dict(arche="skiff",   s=0.10, damages=[], view="footprint", txt="skiff s.10 filled"),
        dict(arche="cargo",   s=0.85, damages=[], view="footprint", txt="cargo s.85 multiblk"),
        dict(arche="cargo",   s=0.50, damages=[], view="proud",     txt="cargo PROUD"),
        dict(arche="warship", s=0.80, damages=[], view="proud",     txt="warship PROUD"),
    ]
    # --- B. each damage operator alone (bury tested on multiple archetypes: one end only) ---
    specs += [
        dict(arche="cargo",   s=0.60, damages=["break"],            view="footprint", txt="BREAK alone"),
        dict(arche="warship", s=0.55, damages=["break"],            view="proud",     txt="BREAK PROUD"),
        dict(arche="cargo",   s=None, damages=["bury"],             view="footprint", txt="BURY 1end cargo"),
        dict(arche="skiff",   s=None, damages=["bury"],             view="footprint", txt="BURY 1end skiff"),
        dict(arche="barge",   s=None, damages=["bury"],             view="footprint", txt="BURY 1end barge"),
        dict(arche="cargo",   s=0.65, damages=["gunwale_gap"],      view="footprint", txt="GUNWALE gap"),
        dict(arche="cargo",   s=0.70, damages=["internal_distort"], view="footprint", txt="INTERNAL distort"),
        dict(arche="cargo",   s=0.60, damages=["debris"],           view="footprint", txt="DEBRIS field"),
        dict(arche="cargo",   s=0.85, damages=["skeleton"],         view="footprint", txt="SKELETON"),
    ]
    # --- C. combos ---
    specs += [
        dict(arche="cargo",   s=0.65, damages=["break", "bury"],            view="proud",     txt="BREAK+BURY PROUD"),
        dict(arche="warship", s=0.60, damages=["gunwale_gap", "bury"],      view="proud",     txt="gap+bury PROUD"),
        dict(arche="cargo",   s=0.70, damages=["break", "debris"],          view="footprint", txt="BREAK+DEBRIS"),
        dict(arche="skiff",   s=0.30, damages=["bury", "internal_distort"], view="footprint", txt="bury+distort skiff"),
    ]
    # --- D. side-lying ---
    specs += [
        dict(arche="warship", s=0.55, damages=["bury"],   view="side", txt="SIDE-LYING warship"),
        dict(arche="ancient", s=0.45, damages=["break"],  view="side", txt="SIDE-LYING ancient brk"),
    ]
    return specs


def build_all(specs):
    items = []
    for i, sp in enumerate(specs):
        rng = np.random.default_rng(GLOBAL_SEED + i * 17)
        try:
            it = build_one(sp, rng)
        except Exception as e:
            import traceback; traceback.print_exc()
            print("ERR", i, sp.get("txt"), repr(e))
            it = dict(M_obj=np.zeros((TILE, TILE), np.uint8), T=np.zeros((TILE, TILE), np.float32),
                      holes=None, dmask=None, dT=None, view="footprint", phi=None,
                      proud_flag=False, txt="ERR " + sp.get("txt", ""), info={}, s=0, arche="?", damages=[])
        items.append(it)
    return items


def montage(tiles, cols=5, pad=4, bg=20, channels=1):
    rows = (len(tiles) + cols - 1) // cols
    if channels == 3:
        canvas = np.full((rows * (TILE + pad) + pad, cols * (TILE + pad) + pad, 3), bg, np.uint8)
    else:
        canvas = np.full((rows * (TILE + pad) + pad, cols * (TILE + pad) + pad), bg, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y0 = pad + r * (TILE + pad); x0 = pad + c * (TILE + pad)
        canvas[y0:y0 + TILE, x0:x0 + TILE] = t
    return canvas


def make_template_montage(items):
    """Visualize M_obj (white outline) + T colormap -> v5_template_montage.png.
       Shows core shape / structure / damage / orientation / ribs / dark bays /
       ★v5: 去瓶(单调船宽) / 去刺(钝头) / 亮内(bright fill, 窄亮边) / fruit-pit skiff /
       archetype<->structure coupling (2nd label: arche s blk)."""
    tiles = []
    for it in items:
        rgb = _T_to_rgb(it["T"], it["M_obj"], it.get("holes"), it.get("dmask"))
        if it.get("dmask") is not None and it.get("dT") is not None:
            dT = it["dT"]; db = it["dmask"] > 0
            pv = np.clip(dT / 3.2, 0, 1)
            rgb[db] = np.stack([(60 + 195 * pv[db]).astype(np.uint8),
                                (90 + 165 * pv[db]).astype(np.uint8),
                                (160 + 95 * pv[db]).astype(np.uint8)], -1)
        _lab(rgb, it["txt"])
        # 2nd label line: archetype / s / block-count -> makes coupling auditable at a glance
        blk = len(it.get("info", {}).get("blocks", []))
        _lab(rgb, f"{it['arche']} s{it['s']:.2f} blk{blk}", y=34)
        ys, xs = np.where(it["M_obj"] > 0)
        if len(xs) > 20:
            pts = np.stack([xs, ys], 1).astype(np.float32)
            mean, ev = cv2.PCACompute(pts, mean=None)
            cxy = mean[0]; axis = ev[0]
            p2 = cxy + axis * 40
            cv2.arrowedLine(rgb, tuple(cxy.astype(int)), tuple(p2.astype(int)),
                            (0, 255, 255), 1, tipLength=0.3)
        tiles.append(rgb)
    canvas = montage(tiles, cols=5, channels=3)
    p = os.path.join(OUT, "v5_template_montage.png")
    cv2.imencode(".png", canvas)[1].tofile(p)
    return p


def save_templates_npz(items):
    """Decision ②: store T (and M_obj + extra layers) as .npz for the port stage."""
    d = {}
    for i, it in enumerate(items):
        d[f"M_{i}"] = it["M_obj"]
        d[f"T_{i}"] = it["T"]
        if it.get("holes") is not None:
            d[f"holes_{i}"] = it["holes"]
        if it.get("dmask") is not None:
            d[f"dmask_{i}"] = it["dmask"]; d[f"dT_{i}"] = it["dT"]
    meta = [dict(txt=it["txt"], view=it["view"], arche=it["arche"], s=it["s"],
                 damages=it["damages"], n_block=len(it.get("info", {}).get("blocks", [])),
                 has_sup=int(bool(it.get("info", {}).get("has_superstructure", False))),
                 phi=(float(it["phi"]) if it["phi"] is not None else -1.0),
                 proud=int(it["proud_flag"])) for it in items]
    p = os.path.join(OUT, "build_v5_templates.npz")
    np.savez_compressed(p, meta=np.array(meta, dtype=object), n=len(items), **d)
    return p


def _beam_envelope(params, n=200):
    """Return the half-width envelope w_rail(t) (BEFORE local dent, BEFORE edge jitter) so we can
       audit ★v5 #1: monotone bow->max->stern, no cola-bottle waist. We re-evaluate the same
       w(t) formula used in hull_polygon_local, with the dent and (removed) sheer ripple off."""
    bf, mf, bs, tw = params["bow_frac"], params["mid_frac"], params["bow_sharp"], params["transom_w"]
    stem_w = params.get("stem_w", 0.16); pit = params.get("pit", False)
    mid_bulge = 0.14 if pit else -0.04; stern_exp = 1.0 if pit else 1.4
    stern_frac = max(1.0 - bf - mf, 1e-6)
    ws = []
    for i in range(n + 1):
        t = i / n
        if t < bf:
            w = stem_w + (1.0 - stem_w) * (t / bf) ** bs
        elif t < bf + mf:
            u = (t - bf) / mf; w = 1.0 + mid_bulge * math.sin(u * math.pi)
        else:
            u = (t - (bf + mf)) / stern_frac; w = 1.0 - (1.0 - tw) * (u ** stern_exp)
        ws.append(max(0.0, w))
    return np.array(ws)


def selfcheck_distribution(n=2000):
    """★v5 #4: prove skiff sampling weight ≈ 45%. ★v5 #1/#2: over a random batch, audit that the
       beam envelope is monotone (no waist) and bow is blunt (no needle), per archetype."""
    rng = np.random.default_rng(424242)
    from collections import Counter
    cnt = Counter()
    waist_violations = 0; needle = 0; total = 0
    bow_sharp_by = {}
    for _ in range(n):
        a = sample_archetype(rng); cnt[a] += 1
        p = sample_hull_params(a, rng)
        bow_sharp_by.setdefault(a, []).append(p["bow_sharp"])
        env = _beam_envelope(p)
        # waist test: find the index of global max beam; before it should be ~non-decreasing,
        # after it ~non-increasing. A "cola-bottle" waist = a dip BETWEEN two higher lobes.
        imax = int(np.argmax(env))
        # tolerance 0.02 (raster/round noise); count a violation if either side breaks monotonic
        rise = env[:imax + 1]; fall = env[imax:]
        bad = (np.diff(rise) < -0.02).any() or (np.diff(fall) > 0.02).any()
        # but the local dent is applied SEPARATELY (not in env), and a single mid dip from the
        # removed-ripple would show here; env here excludes ripple+dent, so any bad = real waist.
        waist_violations += int(bad)
        # needle test: bow stem width tiny AND bow_sharp high = needle. only ancient allowed.
        if a != "ancient" and p["stem_w"] < 0.12 and p["bow_sharp"] > 1.5:
            needle += 1
        total += 1
    print("\n=== ★v5 #4 archetype distribution (n=%d) ===" % n)
    for k in ARCHETYPES:
        print(f"  {k:8s} {cnt[k]/n*100:5.1f}%  (weight {ARCHETYPES[k]['w']:.2f})  "
              f"bow_sharp mean {np.mean(bow_sharp_by.get(k,[0])):.2f}")
    print(f"  skiff share = {cnt['skiff']/n*100:.1f}%  (target ~40-50%)")
    print("=== ★v5 #1 waist check: %d/%d envelopes had a cola-bottle waist (target 0) ===" % (waist_violations, total))
    print("=== ★v5 #2 needle check: %d/%d non-ancient ships had a needle bow (target 0) ===" % (needle, total))
    return cnt, waist_violations, needle


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    specs = build_specs()
    items = build_all(specs)
    tp = make_template_montage(items)
    npp = save_templates_npz(items)
    print(f"{len(items)} items")
    print(f"template montage -> {tp}")
    print(f"templates npz    -> {npp}")
    # per-item T stats. ★v5 #3: report interior-fill mean (should be ~T_FILL, ~CONSTANT across s),
    #   rim mean vs rim peak (mean modest, peak high), dark%.
    print("\n  idx txt                    arche   s    blk  area   T[mn/mu/mx]      rimMU rimPK fill  bri% drk% view")
    for i, it in enumerate(items):
        T = it["T"]; M = it["M_obj"] > 0
        if M.sum() == 0:
            print(f"  [{i:2d}] {it['txt']:<22} EMPTY"); continue
        tin = T[M]
        bright = float((tin > T_RIM - 0.6).mean())
        dark = float((tin < -0.1).mean())
        # interior fill px = inside, near T_FILL, not rim/rib/block/dark
        fill_px = tin[(tin > T_FILL - 0.25) & (tin < T_FILL + 0.25)]
        fillmu = float(fill_px.mean()) if fill_px.size else 0.0
        rim_px = tin[tin > 1.5]                          # rim/rib/block bright population
        rimmu = float(rim_px.mean()) if rim_px.size else 0.0
        rimpk = float(np.percentile(tin, 99.5))
        nblk = len(it.get("info", {}).get("blocks", []))
        print(f"  [{i:2d}] {it['txt']:<22} {it['arche']:<7} s{it['s']:.2f} blk{nblk} "
              f"{int(M.sum()):6d} {tin.min():+.1f}/{tin.mean():+.1f}/{tin.max():+.1f}  "
              f"{rimmu:4.1f} {rimpk:4.1f} {fillmu:+.2f} "
              f"{bright*100:4.1f} {dark*100:4.1f} {it['view']}")
    selfcheck_distribution()
