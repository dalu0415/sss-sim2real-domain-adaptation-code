"""STEP ③ — offline (M_obj + T) baker, ROUTE B (per UNIFIED_SPEC ★v4 + ledger 末节 route-B裁决).

Bakes the 400 random ship templates used by the final arm2 dataset:
  arm2 = v5 real ship shape + narrow bright rim + BRIGHT FLAT interior fill (T_FILL≈seabed) +
         damage. NO internal fine structure (no ribs / dark bays / keel / blocks).
For exact reproducibility, the original arm3 superset is still evaluated in memory to preserve the
validated random-number stream and to run the M_obj single-variable self-check, but arm3 templates
and metadata are not written by this public final-version script.

Geometry control variable (ledger ★几何控制变量): we REUSE the baseline stage3 ship geometry
PER-TAG so arm2/arm3 land on the SAME canvas / same placement / same scale / same orientation as
arm1 (the baseline capsule line) tag-for-tag. Strategy chosen: PER-TAG REUSE from the baseline
mask_metadata.json (canvas_h/w + size_ratio) + the baseline {tag}_obj.png centroid (placement) and
its principal-axis orientation. See bake_geometry_for_tag() for the exact derivation + why.

ABSOLUTELY OFFLINE: imports build_v5 read-only as a primitive library; reads baseline masks
read-only; writes ONLY the final arm2 templates to a new output dir.

Environment: Python 3 with NumPy 2.x (np.ptp).
"""
import os, sys, math, json, hashlib
from collections import Counter
import numpy as np
import cv2

# import the FROZEN v5 generator as a primitive library (read-only use; v5 params NOT retuned)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import build_v5 as v5

# ---------------------------------------------------------------------------
# paths (NEW output dir only; baseline read-only)
# ---------------------------------------------------------------------------
BASE_MASK_DIR = r"PATH_TO_BASELINE_MASK_OUTPUT"
BASE_META = os.path.join(BASE_MASK_DIR, "mask_metadata.json")

OUT_ROOT = r"PATH_TO_STEP3_TEMPLATE_OUTPUT"
ARM_DIR = os.path.join(OUT_ROOT, "arm2")

SEED = 20260606            # deterministic master seed (reproducible)
N_SHIPS = 400

# damage mix (task spec): 70% intact / 15% break / 15% bury. Zero-fragment operators
#   (skeleton / gunwale_gap / internal_distort / debris) are DEFAULT-OFF (并掉不单出);
#   a broken ship MAY carry crumbled fragments via the break operator's own崩口 (built in).
DMG_INTACT, DMG_BREAK, DMG_BURY = 0.70, 0.15, 0.15


# ===========================================================================
# GEOMETRY: per-tag reuse of baseline arm1 geometry
# ===========================================================================
def load_ship_meta():
    with open(BASE_META, "r", encoding="utf-8") as f:
        meta = json.load(f)
    ships = [m for m in meta if m["label"] == "ship"]
    ships.sort(key=lambda m: m["tag"])     # stable tag order (ship_0091 .. ship_0490)
    return ships


def bake_geometry_for_tag(tag):
    """Per-tag geometry reuse. Returns (canvas_h, canvas_w, cx, cy, target_L, base_angle_deg).

    WHY this strategy (chosen over a fresh distribution-matched rng stream):
      The baseline mask_metadata.json carries canvas_h/w + size_ratio + rotation per ship, AND
      the baseline {tag}_obj.png exists for all 400 ships (verified). The PNG already bakes in the
      placement jitter (which is NOT separately stored in metadata). So the most faithful per-tag
      reuse reads geometry DIRECTLY from the baseline obj PNG:
        - canvas size      = PNG shape (== metadata canvas_h/w; verified equal for all 400).
        - placement (cx,cy)= centroid of the baseline mask (captures the placement jitter exactly).
        - target_L         = long side of the baseline mask's content bbox (== size_ratio*canvas_long
                             up to the v5-vs-baseline outline-fill difference; we use the bbox long
                             side so arm2/arm3 occupy the SAME footprint span as arm1 tag-for-tag).
        - base_angle       = baseline mask principal-axis (PCA) angle. We then rotate the hull-local
                             template (long axis = +x) by this angle so its keel aligns with arm1's.
      This makes arm2/arm3 share arm1's canvas distribution, occupy-ratio, placement, and orientation
      PER TAG -> the geometry confound between arms is eliminated (it is literally arm1's geometry)."""
    g = cv2.imread(os.path.join(BASE_MASK_DIR, tag + "_obj.png"), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(tag + "_obj.png")
    obj = (g > 127).astype(np.uint8)
    H, W = obj.shape
    # orientation + footprint span + PLACEMENT via the ORIENTED bounding box (minAreaRect) of the
    #   largest contour. minAreaRect is exact for elongated masks and far more stable than
    #   centroid-moment PCA on near-square / damaged blobs (verified: synthetic-bar recovery err
    #   0.0; on real masks minRect≈PCA within ~1° for aspect>2.5, disagreeing only on near-square
    #   masks where the long axis is genuinely ambiguous and orientation barely matters).
    #   - placement (cx,cy) = OBB CENTER (not the area centroid): for damaged/buried baseline masks
    #     the centroid is pulled toward the heavier surviving half, while the OBB center marks the
    #     region the ship OCCUPIES. Our symmetric hull's center == its OBB center, so matching OBB
    #     centers makes arm2/arm3 cover the same canvas region as arm1 (centroid would offset ~30px).
    #   - target_L = LONG side of the oriented box (the ship length span as rendered in arm1), so
    #     arm2/arm3 occupy the same keel-length footprint as arm1 tag-for-tag.
    cnts, _ = cv2.findContours(obj * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    (obx, oby), (rw, rh), rang = cv2.minAreaRect(c)
    cx, cy = float(obx), float(oby)
    long_side = max(rw, rh)
    base_angle = rang if rw >= rh else rang + 90.0   # angle of the LONG side, deg
    target_L = float(long_side)
    return H, W, cx, cy, target_L, base_angle


def place_local_on_canvas(Mlocal, Tlocal, canvas_h, canvas_w, cx, cy, ang_deg, scale,
                          extra_layers=None):
    """Rotate(ang)+scale(scale)+translate the hull-local (M,T) onto the FULL baseline canvas.
       Mirrors build_v5.place_local_on_tile but onto an arbitrary (canvas_h,canvas_w) at center
       (cx,cy) with an explicit scale. M -> INTER_NEAREST, T -> INTER_LINEAR (spec A/B).
       After warp, T is zeroed wherever M_obj==0 (no T leaking outside the hull)."""
    H, W = Mlocal.shape
    lcx, lcy = W / 2.0, H / 2.0
    R = cv2.getRotationMatrix2D((lcx, lcy), ang_deg, scale)
    R[0, 2] += cx - lcx
    R[1, 2] += cy - lcy
    Mg = cv2.warpAffine(Mlocal, R, (canvas_w, canvas_h), flags=cv2.INTER_NEAREST)
    Tg = cv2.warpAffine(Tlocal, R, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR)
    Mg = (Mg > 127).astype(np.uint8) * 255
    out = {}
    if extra_layers:
        for name, arr, interp in extra_layers:
            out[name] = cv2.warpAffine(arr, R, (canvas_w, canvas_h), flags=interp)
    Tg[Mg == 0] = 0.0          # T zeroed outside hull
    return Mg, Tg, out


# ===========================================================================
# PRODUCTION SAMPLER (replaces build_v5.build_specs() hand-placed demo)
# ===========================================================================
def build_specs_production(n=N_SHIPS, seed=SEED):
    """Deterministic random sampler of n ship specs. archetype ~ v5 ARCHETYPES weights
       (skiff≈46%); damage ~ 70 intact / 15 break / 15 bury. s left None -> archetype s_band
       drives it inside build_one (coupling). Returns list of dicts (no view/proud -> footprint:
       this is a TEMPLATE baker; proud/shadow geometry is Step2's job downstream)."""
    rng = np.random.default_rng(seed)
    ships = load_ship_meta()
    assert len(ships) >= n, f"baseline has {len(ships)} ships < {n}"
    specs = []
    for i in range(n):
        arche = v5.sample_archetype(rng)
        u = rng.random()
        if u < DMG_INTACT:
            damages = []
        elif u < DMG_INTACT + DMG_BREAK:
            damages = ["break"]
        else:
            damages = ["bury"]
        specs.append(dict(arche=arche, s=None, damages=damages, view="footprint",
                          tag=ships[i]["tag"], txt=f"{arche}|{'+'.join(damages) or 'intact'}"))
    return specs


# ===========================================================================
# BAKE ONE SHIP -> (M_obj shared, T_arm2, T_arm3) on the per-tag baseline canvas
# ===========================================================================
def _clone(rng):
    """Deep-clone a numpy Generator so two arms consume an IDENTICAL rng stream independently."""
    new = np.random.default_rng()
    new.bit_generator.state = rng.bit_generator.state
    return new


def bake_one(spec, master_rng):
    """Build ONE ship in both arms. arm2/arm3 share hull shape + geometry + damage + rng; differ
       ONLY by internal_structure. Returns dict with canvas-size M_obj (uint8 0/255, IDENTICAL for
       both arms), T_arm2 (float32), T_arm3 (float32), plus audit info.

       rng discipline (the crux of the single-variable guarantee):
         1) hull params + polygon + raster (edge jitter) consume master_rng ONCE -> shared Mlocal.
         2) s drawn ONCE from master_rng -> shared.
         3) paint arm2 and arm3 each on a CLONE of master_rng's CURRENT state -> the rim+fill draws
            are byte-identical for both arms; arm3 then consumes MORE (ribs/bays/keel/blocks).
         4) advance master_rng PAST the arm3 (superset) consumption by re-running paint on it,
            so the damage stage starts from a single well-defined state.
            ★STRUCTURE-COUPLING FIX 2026-06-06: to keep this advance — and therefore baseline arm2
            bytes + the damage geometry — UNCHANGED while re-coupling structure to the right
            archetypes, build_v5.paint_T_local's structure section is written to consume the master
            rng stream IDENTICALLY to the legacy v5 logic: for the now-rib-less DECKED archetypes
            (cargo/warship) the rib / dark-bay / rib-paint / keel draws are still PERFORMED (same
            count/order) but their painting is DISCARDED. So the master state after step 4 is
            byte-identical to baseline for every ship -> arm2 templates + damage stay byte-stable.
         5) run damage ops on BOTH arms using a SHARED damage-rng clone -> M_obj stays identical
            (damage edits M+T together with the same random decisions; the only T difference is the
            pre-existing interior grid, which does not change WHICH pixels are carved)."""
    arche = spec["arche"]
    damages = spec.get("damages", [])
    tag = spec["tag"]

    # ---- per-tag baseline geometry (reuse arm1 canvas/placement/scale/orientation) ----
    canvas_h, canvas_w, cx, cy, target_L, base_angle = bake_geometry_for_tag(tag)

    # ---- (1) shared hull shape (master_rng) ----
    params = v5.sample_hull_params(arche, master_rng)
    poly = v5.hull_polygon_local(params)
    Mlocal, _ = v5._hull_local_canvas(poly, rng=master_rng, edge_jitter=1.0)

    # scale so the hull-local long side maps to the baseline target_L (== arm1 footprint span)
    ys, xs = np.where(Mlocal > 0)
    local_L = float(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
    scale = target_L / max(local_L, 1.0)
    # hull-local long axis is +x (0deg, OBB convention). The baseline mask's long axis is at
    #   base_angle in the SAME OBB/image convention. cv2.getRotationMatrix2D rotates CCW for +angle
    #   in math convention = CW in image (y-down) coords, so to land the hull's 0deg axis onto the
    #   baseline's base_angle we apply -base_angle (verified end-to-end: err 0.00deg for elongated
    #   tags; see _diag3.py). This makes arm2/arm3 keel orientation = arm1's per tag.
    ang_deg = -base_angle

    # ---- (2) shared s ----
    s = v5.sample_s_for(arche, master_rng)

    # ---- (3) paint both arms on identical rng clones ----
    rng_a2 = _clone(master_rng)
    rng_a3 = _clone(master_rng)
    T2_local, info2 = v5.paint_T_local(Mlocal, s, rng_a2, arche=arche, params=params,
                                       internal_structure=False)
    T3_local, info3 = v5.paint_T_local(Mlocal, s, rng_a3, arche=arche, params=params,
                                       internal_structure=True)
    # ---- (4) advance master_rng past the arm3 (superset) consumption ----
    #   This re-runs the arm3 paint on master_rng so the damage stage starts from the SAME
    #   well-defined state the baseline used (preserving baseline arm2 bytes). The arm3 structure
    #   logic is written to consume rng IDENTICALLY to the legacy v5 structure section even when a
    #   given archetype's painting is suppressed (cargo/warship ribs/keel are computed-then-discarded
    #   to keep the rng stream byte-stable) — see build_v5.paint_T_local ★STRUCTURE-COUPLING FIX.
    _ = v5.paint_T_local(Mlocal, s, master_rng, arche=arche, params=params, internal_structure=True)

    # ---- (5) damage on BOTH arms with a SHARED damage rng (identical decisions) ----
    M2_local = Mlocal.copy(); M3_local = Mlocal.copy()
    if "break" in damages:
        dr2 = _clone(master_rng); dr3 = _clone(master_rng)
        M2_local, T2_local = v5.dmg_break_local(M2_local, T2_local, info2, dr2)
        M3_local, T3_local = v5.dmg_break_local(M3_local, T3_local, info3, dr3)
        _ = v5.dmg_break_local(Mlocal.copy(), T3_local.copy(), info3, master_rng)  # advance master
    if "bury" in damages:
        dr2 = _clone(master_rng); dr3 = _clone(master_rng)
        M2_local, T2_local = v5.dmg_bury_local(M2_local, T2_local, info2, dr2)
        M3_local, T3_local = v5.dmg_bury_local(M3_local, T3_local, info3, dr3)
        _ = v5.dmg_bury_local(Mlocal.copy(), T3_local.copy(), info3, master_rng)  # advance master

    # ---- warp both arms onto the SAME baseline canvas with identical geometry ----
    M2, T2, _ = place_local_on_canvas(M2_local, T2_local, canvas_h, canvas_w, cx, cy, ang_deg, scale)
    M3, T3, _ = place_local_on_canvas(M3_local, T3_local, canvas_h, canvas_w, cx, cy, ang_deg, scale)

    return dict(tag=tag, arche=arche, damages=damages, s=float(s),
                canvas_h=canvas_h, canvas_w=canvas_w,
                M_obj_arm2=M2, T_arm2=T2, M_obj_arm3=M3, T_arm3=T3,
                n_block_arm3=len(info3.get("blocks", [])),
                n_rib_arm3=info3.get("n_rib", 0))


# ===========================================================================
# DRIVE / SAVE / SELFCHECK / MONTAGE
# ===========================================================================
def _save_arm(arm, tag, M_obj, T, out_dir):
    cv2.imencode(".png", M_obj)[1].tofile(os.path.join(out_dir, f"{tag}_obj.png"))
    np.save(os.path.join(out_dir, f"{tag}_T.npy"), T.astype(np.float32))


def bake_all():
    os.makedirs(ARM_DIR, exist_ok=True)
    specs = build_specs_production()
    master = np.random.default_rng(SEED + 7)   # master generation rng (distinct from sampler rng)

    meta_arm2 = []
    arche_cnt = Counter()
    dmg_cnt = Counter()
    md5_match = 0
    t2_lo, t2_hi, t3_lo, t3_hi = [], [], [], []
    items_for_montage = []

    for i, spec in enumerate(specs):
        # per-ship deterministic sub-rng (reproducible, decoupled across ships)
        ship_rng = np.random.default_rng(SEED + 7 + i * 7919)
        it = bake_one(spec, ship_rng)
        tag = it["tag"]
        arche_cnt[it["arche"]] += 1
        dmg_cnt[("+".join(it["damages"]) or "intact")] += 1

        _save_arm("arm2", tag, it["M_obj_arm2"], it["T_arm2"], ARM_DIR)

        # md5 of the two M_obj (must be identical -> single-variable)
        m2 = hashlib.md5(it["M_obj_arm2"].tobytes()).hexdigest()
        m3 = hashlib.md5(it["M_obj_arm3"].tobytes()).hexdigest()
        if m2 == m3:
            md5_match += 1

        in2 = it["T_arm2"][it["M_obj_arm2"] > 0]
        in3 = it["T_arm3"][it["M_obj_arm3"] > 0]
        if in2.size:
            t2_lo.append(float(in2.min())); t2_hi.append(float(in2.max()))
        if in3.size:
            t3_lo.append(float(in3.min())); t3_hi.append(float(in3.max()))

        common = dict(tag=tag, label="ship", arche=it["arche"],
                      damages=it["damages"], damage=("+".join(it["damages"]) or "none"),
                      s=it["s"], canvas_h=it["canvas_h"], canvas_w=it["canvas_w"])
        meta_arm2.append(dict(common, arm="arm2", internal_structure=False, md5_obj=m2))

        # keep a light record for stratified montage sampling (don't hold all 400 heavy arrays:
        #   keep the small tiles we need by stashing tag+arche+damage+s and re-reading is overkill;
        #   instead keep the item dicts — 400 canvas pairs fit in memory fine for this batch).
        items_for_montage.append(it)

    # ---- write final arm2 metadata (format aligned with full_generate's consumption) ----
    with open(os.path.join(ARM_DIR, "mask_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta_arm2, f, ensure_ascii=False, indent=1)

    # ---- selfcheck numbers ----
    n = len(specs)
    print("=" * 70)
    print(f"STEP③ route-B baker: {n} final arm2 ships baked")
    print(f"  out arm2 -> {ARM_DIR}")
    print("\n--- damage mix (target 70/15/15 intact/break/bury) ---")
    for k in ("intact", "break", "bury"):
        print(f"  {k:8s} {dmg_cnt.get(k,0):4d}  {dmg_cnt.get(k,0)/n*100:5.1f}%")
    other = {k: v for k, v in dmg_cnt.items() if k not in ("intact", "break", "bury")}
    if other:
        print(f"  OTHER (unexpected): {other}")
    print("\n--- archetype dist (target skiff≈46%) ---")
    for k in v5.ARCHETYPES:
        print(f"  {k:8s} {arche_cnt.get(k,0):4d}  {arche_cnt.get(k,0)/n*100:5.1f}%  "
              f"(weight {v5.ARCHETYPES[k]['w']:.2f})")
    print("\n--- T range sanity (over hull interior) ---")
    print(f"  arm2 T: min {min(t2_lo):+.2f} .. max {max(t2_hi):+.2f}  "
          f"(expect ~[-1.7..+3.6], arm2 mostly fill T_FILL={v5.T_FILL}+rim)")
    print(f"  arm3 T: min {min(t3_lo):+.2f} .. max {max(t3_hi):+.2f}")
    print(f"\n--- internal control: M_obj(arm2) md5 == M_obj(arm3-control) md5 ---")
    print(f"  {md5_match}/{n} ships have BYTE-IDENTICAL M_obj across arms "
          f"({'ALL IDENTICAL — PASS' if md5_match == n else 'MISMATCH — FAIL'})")

    mp = make_montage(pick_montage_sample(items_for_montage))
    print(f"\n--- montage -> {mp}")
    return dict(n=n, dmg=dict(dmg_cnt), arche=dict(arche_cnt), md5_match=md5_match,
                t2=(min(t2_lo), max(t2_hi)), t3=(min(t3_lo), max(t3_hi)), montage=mp)


N_MONTAGE_SHIPS = 16


def pick_montage_sample(items):
    """Stratified pick so the eye-gate sees the full distribution: cover all 5 archetypes AND all
       damage types (intact/break/bury), biased to skiff (the 46% majority) but never all-skiff.
       Deterministic (sorted by tag)."""
    by = {}
    for it in items:
        key = (it["arche"], "+".join(it["damages"]) or "intact")
        by.setdefault(key, []).append(it)
    for k in by:
        by[k].sort(key=lambda it: it["tag"])
    picked, seen = [], set()
    # 1) guarantee one of every (archetype x damage) combo that exists
    for k in sorted(by):
        picked.append(by[k][0]); seen.add(by[k][0]["tag"])
        if len(picked) >= N_MONTAGE_SHIPS:
            return picked[:N_MONTAGE_SHIPS]
    # 2) fill remaining slots from the largest strata (skiff-heavy) for representativeness
    flat = sorted(items, key=lambda it: it["tag"])
    for it in flat:
        if it["tag"] in seen:
            continue
        picked.append(it); seen.add(it["tag"])
        if len(picked) >= N_MONTAGE_SHIPS:
            break
    return picked[:N_MONTAGE_SHIPS]


def _crop_tile(M, T, size=300):
    """Crop a square tile around the object and colorize T via build_v5._T_to_rgb, resized."""
    ys, xs = np.where(M > 0)
    if len(xs) == 0:
        return np.full((size, size, 3), 30, np.uint8)
    cy, cx = int(ys.mean()), int(xs.mean())
    half = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * 0.62) + 8
    H, W = M.shape
    y0, y1 = max(0, cy - half), min(H, cy + half)
    x0, x1 = max(0, cx - half), min(W, cx + half)
    rgb = v5._T_to_rgb(T[y0:y1, x0:x1], M[y0:y1, x0:x1])
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_NEAREST)


def make_montage(items):
    """Final arm2 montage, stratified by archetype and damage."""
    tiles = []
    for it in items:
        t2 = _crop_tile(it["M_obj_arm2"], it["T_arm2"])
        v5._lab(t2, f"arm2 {it['tag']}")
        v5._lab(t2, f"{it['arche']} {('+'.join(it['damages']) or 'intact')} s{it['s']:.2f}", y=34)
        tiles.append(t2)
    canvas = v5.montage(tiles, cols=4, channels=3)
    p = os.path.join(OUT_ROOT, "armtemplates_montage.png")
    cv2.imencode(".png", canvas)[1].tofile(p)
    return p


if __name__ == "__main__":
    bake_all()
