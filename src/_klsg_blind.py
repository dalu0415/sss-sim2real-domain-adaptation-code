#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
_klsg_blind.py -- KLSG 目标域【盲化】数据入口（全研究"绝不读真实标签"红线的机器级护栏单点）
================================================================================
站2 块2 第②批。run2-5 训练脚本、run1-3 的 eval_adabn、以及后续站5 / batch③ 都 **import 本模块**
来取 KLSG 目标域图，从而把"绝不读 KLSG 真实标签"这条红线收敛到 **一个文件、一处维护**。

★ 红线（污染事故定义）：
  - 绝不从 KLSG manifest 行 / filepath 反推类别（如 Path(p).parent.name）—— filepath 里虽含
    `airplane`/`ship` 目录名，但 **只许拿来开像素、绝不许解析成 label**。本模块只存 filepath 字符串。
  - KLSG 目标 label 一律替换为 **FORBIDDEN_LABEL 毒丸单例**：任何"取值 / 转换 / 索引 / 比较 / .to /
    转 numpy/torch"都会 raise RuntimeError。健康的 UDA 代码本就丢弃目标 label（`x_t, _ = ...`），
    碰不到毒丸；一旦有 bug 真去用目标 label，立刻在该处炸出，而不是静默泄漏。
  - **默认/生产路径**（read_klsg_unlabeled 毒丸 / 各训练器 read_split_rows 的 klsg 守卫）**绝不返回真实 KLSG label**。
  - **唯一例外 = 授权路径**（_read_labeled_authorized / read_klsg_labeled，站4 起生效）：ceiling[全研究唯一合法用真标签训练的跑法] +
    #1/描述轨的 KLSG 授权评估，经【显式令牌 KlsgLabelAuth】（由 CLI --authorize-klsg-labels + reason 构造）读真标签，
    每次追加写审计 _LABEL_ACCESS_LOG.txt；无令牌碰真标签即 raise。**第一条 domain=klsg 审计行 = 负向核查行作废时点（§3.2.7 揭盲时序锁）。**

★ manifest 不可变锚（防有人偷换 manifest 注入标签序）：
  read_klsg_unlabeled 先对 manifest 文件字节做 sha256 断言，再 split。manifest 任何改动都会让
  sha 断言失败（这是保护，不是 bug）。

本模块只载 filepath、开像素；**绝不**做任何 label 解析。可被任何站/批 import。
"""

import hashlib
from pathlib import Path

import torch
from PIL import Image

# ============================================================ 红线常量
# manifest 相对路径（相对 project_root）
_KLSG_MANIFEST_REL = ("data", "split", "klsg_unlabeled_manifest.txt")
# manifest 文件字节的 sha256（不可变锚；改 manifest 即 assert 失败）
KLSG_MANIFEST_SHA256 = "f7241c601543999f1cc14a7947df5962e25a94c44f687980b8184ea64b66bc23"
# KLSG transductive 全集张数（红线固定）
KLSG_N_EXPECTED = 553

# ---- churn 固定 100 张子集（站2块2第③批·纯追加；生成器 = src/_make_churn_subset.py）----
# churn manifest 相对路径（相对 project_root）
_KLSG_CHURN_MANIFEST_REL = ("data", "split", "klsg_churn100_manifest.txt")
# churn manifest 文件字节的 sha256（不可变锚；_make_churn_subset.py 跑出后回填，偷换即 assert 失败）
KLSG_CHURN100_SHA256 = "bf4d498594f5a2c7e009caad51670f230a32da9c974cc475cb24c7cbfe17451d"
# churn 子集张数（红线固定）
CHURN_N_EXPECTED = 100

_RED_LINE_MSG = "RED LINE: KLSG target labels are unavailable on the unlabeled path"


# ============================================================ 毒丸单例
def _boom(*args, **kwargs):
    """任何取值/转换/比较/索引方法都路由到这里 -> 立刻炸。*args 吞掉 self（作类方法用时）。"""
    raise RuntimeError(_RED_LINE_MSG)


class _ForbiddenLabel:
    """KLSG 目标 label 的毒丸占位单例。

    重载所有"把对象变成可用数值/张量/序列/布尔/哈希"的协议方法，一律 raise。设计目的：
    健康代码丢弃目标 label（不会触发任何方法）；任何 bug 真去【用】目标 label（取整 / 转 tensor /
    比较 / 索引 / .to(device) / 进 set/dict / numpy 化 / 迭代）都会在该处 raise，把红线违例变成
    显式崩溃而非静默泄漏。

    例外：__repr__/__str__ **不 raise**，返回安全标记串（不泄漏任何类别信息）——保证日志/traceback
    可读、不会因为打印这个对象而二次崩溃。identity（`is` / `x is FORBIDDEN_LABEL`）不调用任何方法，
    可安全用于自检。
    """
    __slots__ = ()

    # --- 数值转换 ---
    __int__ = _boom
    __index__ = _boom          # 用作序列索引 / range / np 索引时
    __float__ = _boom
    __complex__ = _boom
    __bool__ = _boom           # `if label:` / truthiness

    # --- 序列 / 迭代 ---
    __iter__ = _boom
    __getitem__ = _boom
    __len__ = _boom
    __contains__ = _boom

    # --- 比较 / 哈希（定义 __eq__ 会令 __hash__ 失效, 故显式给 __hash__ 也炸）---
    __eq__ = _boom
    __ne__ = _boom
    __lt__ = _boom
    __gt__ = _boom
    __le__ = _boom
    __ge__ = _boom
    __hash__ = _boom           # 进 set / dict key / Counter 时

    # --- numpy / 张量互转 ---
    __array__ = _boom          # np.array(label) / np.asarray
    __array_interface__ = property(_boom)

    # --- 张量/设备搬运 等具名方法（obj.to / obj.tolist ...）---
    to = _boom
    cpu = _boom
    cuda = _boom
    numpy = _boom
    item = _boom
    tolist = _boom
    detach = _boom
    long = _boom
    float = _boom
    int = _boom
    view = _boom
    reshape = _boom
    size = _boom
    dim = _boom

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """被任何 torch 函数（torch.tensor/stack/as_tensor/...）吃到时炸。"""
        raise RuntimeError(_RED_LINE_MSG)

    def __reduce__(self):
        """pickle 时复原为同一单例（多进程 DataLoader 防身；本研究统一 num_workers=0 不触发）。
        注：返回构造器引用而非任何 label 值，不构成取值路径。"""
        return (_get_forbidden_label, ())

    def __repr__(self):
        return "<FORBIDDEN_LABEL: KLSG target label unavailable>"

    __str__ = __repr__


# 红线毒丸单例（全研究共用同一个对象 -> identity 检查可靠）
FORBIDDEN_LABEL = _ForbiddenLabel()


def _get_forbidden_label():
    """pickle 复原入口（见 _ForbiddenLabel.__reduce__）。"""
    return FORBIDDEN_LABEL


# ============================================================ manifest -> 盲化 items
def _read_manifest_rel_paths(project_root):
    """读 KLSG manifest：sha256 断言（不可变锚）-> split LF -> 过滤空行 -> 断言 553 行。
    返回相对路径字符串列表。**只读 filepath 字符串，绝不解析类别。**"""
    project_root = Path(project_root)
    manifest_path = project_root.joinpath(*_KLSG_MANIFEST_REL)
    assert manifest_path.is_file(), f"KLSG manifest 不存在: {manifest_path}"
    raw = manifest_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == KLSG_MANIFEST_SHA256, (
        f"RED LINE: KLSG manifest sha256 不符（manifest 被改动？）\n"
        f"  期望 {KLSG_MANIFEST_SHA256}\n  实得 {digest}")
    rel_paths = [ln.strip() for ln in raw.decode("utf-8").split("\n") if ln.strip()]
    assert len(rel_paths) == KLSG_N_EXPECTED, (
        f"RED LINE: KLSG manifest 行数={len(rel_paths)} != {KLSG_N_EXPECTED}（应为 transductive 全集）")
    return rel_paths


def read_klsg_unlabeled(project_root):
    """KLSG 目标域【全 553 张、无标签】入口（transductive，fold 无关）。

    流程：read manifest bytes -> sha256 断言 -> split LF / 过滤空行 -> 断言 len==553 ->
    返回 [(abspath, FORBIDDEN_LABEL), ...]。

    ★ label 位一律是毒丸 FORBIDDEN_LABEL（不是 int、不是 None）—— 任何真去【用】它的代码都会炸。
    ★ 只把 manifest 行当 **filepath 字符串**拼成绝对路径供开像素；**绝不**从路径反推类别（红线）。
    """
    project_root = Path(project_root)
    rel_paths = _read_manifest_rel_paths(project_root)
    items = []
    for rel in rel_paths:
        abspath = (project_root / rel).resolve()       # 仅为开像素拼路径；不解析任何目录名为 label
        items.append((str(abspath), FORBIDDEN_LABEL))  # label 位 = 毒丸单例
    return items


def read_klsg_churn_subset(project_root):
    """KLSG churn 固定 100 张子集入口（站2块2第③批 · transductive，fold/seed 无关，跨档恒同一份）。

    churn 标量 = 相邻 epoch 对【这固定 100 张】的 argmax 预测不一致比例。子集由 src/_make_churn_subset.py
    一次性盲采落盘（**仅对行索引抽样、绝不解析类别**），下游 5 个训练脚本都经本函数恒同入场。

    流程（镜像 read_klsg_unlabeled）：read churn manifest bytes -> sha256 不可变锚断言（偷换即 raise）->
    split LF / 过滤空行 -> 断言 len==100 -> 返回 [(abspath, FORBIDDEN_LABEL), ...]。

    ★ label 位一律毒丸 FORBIDDEN_LABEL（churn_predict 只取 batch[0] 图像、绝不索引 label 位）。
    ★ 只把 manifest 行当 **filepath 字符串**拼绝对路径供开像素；**绝不**从路径反推类别（红线）。
    """
    project_root = Path(project_root)
    manifest_path = project_root.joinpath(*_KLSG_CHURN_MANIFEST_REL)
    assert manifest_path.is_file(), (
        f"KLSG churn 子集 manifest 不存在: {manifest_path}（先跑 src/_make_churn_subset.py）")
    raw = manifest_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == KLSG_CHURN100_SHA256, (
        f"RED LINE: KLSG churn 子集 manifest sha256 不符（manifest 被改动/偷换？）\n"
        f"  期望 {KLSG_CHURN100_SHA256}\n  实得 {digest}")
    rel_paths = [ln.strip() for ln in raw.decode("utf-8").split("\n") if ln.strip()]
    assert len(rel_paths) == CHURN_N_EXPECTED, (
        f"RED LINE: KLSG churn 子集行数={len(rel_paths)} != {CHURN_N_EXPECTED}")
    items = []
    for rel in rel_paths:
        abspath = (project_root / rel).resolve()       # 仅为开像素拼路径；不解析任何目录名为 label
        items.append((str(abspath), FORBIDDEN_LABEL))  # label 位 = 毒丸单例
    return items


# ============================================================ 授权读取（站4 起 · ceiling + 授权评估 · gated 单门）
# 站4 起：ceiling（全研究唯一合法用真标签训练的跑法）+ #1/描述轨的 KLSG 授权评估，经【此唯一门】读真标签。
# 默认/生产路径（read_klsg_unlabeled 毒丸、各训练器 read_split_rows 的 klsg 守卫）不变、仍 deny。
# 门控 = 显式授权令牌 KlsgLabelAuth（由 CLI --authorize-klsg-labels + reason 构造；非 bool 旗标、难误触）。
# 每次授权读取追加写审计 _LABEL_ACCESS_LOG.txt（= §3.2.7 揭盲时序锁证据 + 负向核查行作废时间戳）。

# 类别索引（与各训练器 CLASSES 一致：airplane=0, ship=1）；改此即与训练口径不符
CLASSES = ["airplane", "ship"]
LABEL2IDX = {c: i for i, c in enumerate(CLASSES)}

_SPLITS_CSV_REL = ("data", "split", "splits.csv")                       # 含 klsg 真标签明文的总表
_LABEL_ACCESS_LOG_REL = ("results", "run1_s4_baseline_ceiling", "_LABEL_ACCESS_LOG.txt")


_AUTH_CTOR_TOKEN = object()   # 私有哨兵：仅 klsg_label_auth_from_cli 持有；裸构造 KlsgLabelAuth 即 raise（机制锁死单一绑定点、非仅约定）


class KlsgLabelAuth:
    """KLSG 真标签授权令牌（显式对象、非 bool 旗标、难误触）。
    ★ 只能经 klsg_label_auth_from_cli() 构造（它在显式 CLI 开关 --authorize-klsg-labels 置位时持哨兵构造）；
    裸构造（无哨兵）即 raise。必带非空 reason + caller（写入审计日志）。"""
    __slots__ = ("reason", "caller")

    def __init__(self, reason, caller, _token=None):
        if _token is not _AUTH_CTOR_TOKEN:
            raise RuntimeError(
                "RED LINE: KlsgLabelAuth 不可裸构造——须经 klsg_label_auth_from_cli("
                "由 CLI --authorize-klsg-labels 置位时)构造")
        if not (isinstance(reason, str) and reason.strip()):
            raise ValueError("KlsgLabelAuth 须带非空 reason（写入审计日志）")
        if not (isinstance(caller, str) and caller.strip()):
            raise ValueError("KlsgLabelAuth 须带非空 caller（写入审计日志）")
        self.reason = reason.strip()
        self.caller = caller.strip()


def klsg_label_auth_from_cli(authorize_flag, reason, caller):
    """把显式 CLI 开关 --authorize-klsg-labels 绑成授权令牌：开关未置 -> 返回 None（碰真标签即 raise）。
    ★ 全研究唯一构造 KlsgLabelAuth 的入口（持私有哨兵）；别处无法裸构造（机制锁死）。"""
    if not authorize_flag:
        return None
    return KlsgLabelAuth(reason=reason, caller=caller, _token=_AUTH_CTOR_TOKEN)


def _append_label_access_log(project_root, caller, reason, csv_rel, domain, fold, split, n_rows):
    """授权读取审计（追加写、不覆盖）。时间戳由**脚本自取真实 wall-clock**——审计日志不进决定论哈希集，
    故自取真时间防伪（决定论敏感产物 = curve/ckpt sha、与本日志解耦）。"""
    import datetime
    project_root = Path(project_root)
    log_path = project_root.joinpath(*_LABEL_ACCESS_LOG_REL)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().astimezone().isoformat()
    line = (f"{ts}\tcaller={caller}\treason={reason}\tcsv={'/'.join(csv_rel)}"
            f"\tdomain={domain}\tfold={fold}\tsplit={split}\tn_labels_read={n_rows}\n")
    with open(log_path, "a", encoding="utf-8", newline="\n") as f:
        f.write(line)


def _assert_klsg_fold_consistency(project_root, fold):
    """按折锚：splits.csv 中该 fold 的 (train∪test) klsg 路径集 == manifest 553 路径集。
    ★ 只校 **filepath 路径**、绝不读 label（与红线一致：读路径非读标签）。"""
    project_root = Path(project_root)
    manifest_abs = {str((project_root / rel).resolve()) for rel in _read_manifest_rel_paths(project_root)}
    import csv as _csv
    csv_path = project_root.joinpath(*_SPLITS_CSV_REL)
    fold_paths = set()
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for r in _csv.DictReader(f):
            if r["domain"] != "klsg":
                continue
            if fold is not None and int(r["fold"]) != int(fold):
                continue
            fold_paths.add(str((project_root / r["filepath"]).resolve()))   # 只取 filepath、不碰 r["label"]
    assert fold_paths == manifest_abs, (
        f"RED LINE: splits.csv klsg fold={fold} 路径集({len(fold_paths)}) != manifest 553 路径集({len(manifest_abs)})")


def _read_labeled_authorized(auth, project_root, csv_rel, domain, fold, split):
    """【授权】读 csv 的 (domain,fold,split) 行 -> [(abspath, label_idx), ...] 带**真标签** + 写审计。

    ★ 红线门控：auth 必须是 KlsgLabelAuth 实例（经 CLI --authorize-klsg-labels 构造）；否则 raise。
       这是全研究【唯一】返回 KLSG 真标签的口；默认/生产路径（毒丸、read_split_rows 的 klsg 守卫）不经此。
    ★ klsg 域额外按折校一致性（路径集==manifest 553，只校路径不读标签）。
    """
    if not isinstance(auth, KlsgLabelAuth):
        raise RuntimeError(
            _RED_LINE_MSG + "：_read_labeled_authorized 需 KlsgLabelAuth 授权令牌"
            f"（CLI --authorize-klsg-labels + reason），实得 {type(auth).__name__}")
    project_root = Path(project_root)
    csv_path = project_root.joinpath(*csv_rel)
    assert csv_path.is_file(), f"授权读取的 csv 不存在: {csv_path}"
    import csv as _csv
    if domain == "klsg":
        _assert_klsg_fold_consistency(project_root, fold)   # ★先校路径锚、再读标签（任何碰 klsg 标签前先验，失败则未读标签）
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for r in _csv.DictReader(f):
            if r["domain"] != domain:
                continue
            if fold is not None and int(r["fold"]) != int(fold):
                continue
            if split is not None and r["split"] != split:
                continue
            abspath = (project_root / r["filepath"]).resolve()
            rows.append((str(abspath), LABEL2IDX[r["label"]]))
    _append_label_access_log(project_root, auth.caller, auth.reason, csv_rel, domain, fold, split, len(rows))
    return rows


def read_klsg_labeled(auth, project_root, fold, split):
    """【授权】读 KLSG 真标签（splits.csv 的 domain==klsg 行）= _read_labeled_authorized 的薄包装。
    站4 起 ceiling 训练 / #1·描述轨授权评估专用；无 KlsgLabelAuth 令牌即 raise。
    ★ 调用此函数（在真 KLSG 上）= 跨红线动作、负向核查行自此作废（§3.2.7）。"""
    return _read_labeled_authorized(auth, project_root, _SPLITS_CSV_REL, "klsg", fold, split)


# ============================================================ 盲化 Dataset / collate
class UnlabeledImageDataset(torch.utils.data.Dataset):
    """KLSG 无标签数据集：预载 PIL(RGB) 进内存（小数据集，与 host ImageListDataset 同口径、确定性）。
    __getitem__ 返回 (transform(img), FORBIDDEN_LABEL)。**只用 path 开图，绝不碰 / 不推 label。**"""

    def __init__(self, items, transform):
        self.transform = transform
        self.images = []
        for path, _label in items:                     # _label 一律是毒丸，直接丢弃、绝不解析
            with Image.open(path) as im:
                self.images.append(im.convert("RGB").copy())   # 灰度声呐图 -> 3 通道

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(self.images[i]), FORBIDDEN_LABEL   # label 位永远毒丸


def collate_unlabeled(batch):
    """KLSG 无标签 batch 的 collate：只 stack 图像，label 位返回**同一个毒丸单例**。
    ★ 绝不 stack 标签、绝不走 default_collate（default_collate 会试图把 label 转 tensor -> 触发毒丸）。"""
    images = torch.stack([b[0] for b in batch], dim=0)
    return images, FORBIDDEN_LABEL


# ============================================================ 自测
def _selftest(project_root=None):
    """红线护栏自测：毒丸所有取值路径都炸 + read_klsg_unlabeled 返 553 + sha 过。"""
    print("=" * 78)
    print("★ _klsg_blind 自测（红线护栏）★")
    ok = True

    def _expect_raise(desc, fn):
        nonlocal ok
        try:
            fn()
            print(f"  [毒丸] {desc:<34} -> 未 raise！(FAIL)")
            ok = False
        except RuntimeError as e:
            good = (str(e) == _RED_LINE_MSG)
            print(f"  [毒丸] {desc:<34} -> raise RuntimeError {'(PASS)' if good else '(消息不符 FAIL)'}")
            ok = ok and good
        except Exception as e:  # noqa
            print(f"  [毒丸] {desc:<34} -> raise {type(e).__name__}（非 RuntimeError, FAIL）")
            ok = False

    L = FORBIDDEN_LABEL
    _expect_raise("int(FORBIDDEN_LABEL)", lambda: int(L))
    _expect_raise("float(FORBIDDEN_LABEL)", lambda: float(L))
    _expect_raise("bool(FORBIDDEN_LABEL)", lambda: bool(L))
    _expect_raise("operator.index(L)", lambda: __import__("operator").index(L))
    _expect_raise("FORBIDDEN_LABEL.to('cpu')", lambda: L.to("cpu"))
    _expect_raise("FORBIDDEN_LABEL[0]", lambda: L[0])
    _expect_raise("iter(FORBIDDEN_LABEL)", lambda: iter(L))
    _expect_raise("len(FORBIDDEN_LABEL)", lambda: len(L))
    _expect_raise("FORBIDDEN_LABEL == 0", lambda: (L == 0))
    _expect_raise("FORBIDDEN_LABEL < 1", lambda: (L < 1))
    _expect_raise("hash(FORBIDDEN_LABEL)", lambda: hash(L))
    _expect_raise("FORBIDDEN_LABEL.tolist()", lambda: L.tolist())
    _expect_raise("FORBIDDEN_LABEL.item()", lambda: L.item())
    _expect_raise("torch.tensor([L])", lambda: torch.tensor([L]))
    _expect_raise("torch.as_tensor(L)", lambda: torch.as_tensor(L))
    _expect_raise("np.array(L)", lambda: __import__("numpy").array(L).astype("int64"))
    _expect_raise("{L: 1} (dict key)", lambda: {L: 1})

    # identity 安全（不触发毒丸）
    id_ok = (L is FORBIDDEN_LABEL) and (L is _get_forbidden_label())
    print(f"  [identity] L is FORBIDDEN_LABEL and pickle-restore-singleton -> {'PASS' if id_ok else 'FAIL'}")
    ok = ok and id_ok
    # repr 安全（不 raise、不泄漏类别）
    r = repr(L)
    repr_ok = ("FORBIDDEN" in r) and ("airplane" not in r) and ("ship" not in r)
    print(f"  [repr] repr(L)={r!r} 安全(不raise/不泄漏) -> {'PASS' if repr_ok else 'FAIL'}")
    ok = ok and repr_ok

    # read_klsg_unlabeled：553 + sha
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    try:
        items = read_klsg_unlabeled(project_root)
        n_ok = (len(items) == KLSG_N_EXPECTED)
        # 每条 label 位都是毒丸单例 + path 是字符串（非空）
        label_ok = all(it[1] is FORBIDDEN_LABEL for it in items)
        path_ok = all(isinstance(it[0], str) and it[0] for it in items)
        print(f"  [manifest] read_klsg_unlabeled -> {len(items)} 条 (期望 {KLSG_N_EXPECTED}) "
              f"sha过 | label全毒丸={label_ok} | path全字符串={path_ok} "
              f"-> {'PASS' if (n_ok and label_ok and path_ok) else 'FAIL'}")
        ok = ok and n_ok and label_ok and path_ok
        # collate 不炸（只 stack 图位；这里用假 1x1 张量模拟，验 label 位返回毒丸单例）
        fake = [(torch.zeros(3, 2, 2), FORBIDDEN_LABEL) for _ in range(4)]
        imgs, lab = collate_unlabeled(fake)
        collate_ok = (tuple(imgs.shape) == (4, 3, 2, 2)) and (lab is FORBIDDEN_LABEL)
        print(f"  [collate] collate_unlabeled stack图={tuple(imgs.shape)} label位是毒丸单例={lab is FORBIDDEN_LABEL} "
              f"-> {'PASS' if collate_ok else 'FAIL'}")
        ok = ok and collate_ok
    except AssertionError as e:
        print(f"  [manifest] read_klsg_unlabeled 断言失败: {e}  (FAIL)")
        ok = False

    # churn 子集（站2块2第③批）：sha 不可变锚 + len==100（纯追加，不改原有 selftest 项）
    try:
        churn_items = read_klsg_churn_subset(project_root)
        cn_ok = (len(churn_items) == CHURN_N_EXPECTED)
        c_label_ok = all(it[1] is FORBIDDEN_LABEL for it in churn_items)
        c_path_ok = all(isinstance(it[0], str) and it[0] for it in churn_items)
        print(f"  [churn] read_klsg_churn_subset -> {len(churn_items)} 条 (期望 {CHURN_N_EXPECTED}) "
              f"sha过 | label全毒丸={c_label_ok} | path全字符串={c_path_ok} "
              f"-> {'PASS' if (cn_ok and c_label_ok and c_path_ok) else 'FAIL'}")
        ok = ok and cn_ok and c_label_ok and c_path_ok
    except AssertionError as e:
        print(f"  [churn] read_klsg_churn_subset 断言失败: {e}  (FAIL)")
        ok = False

    print(f"  >>> _klsg_blind 自测总判: {'全部 PASS' if ok else '存在 FAIL'}")
    print("=" * 78)
    assert ok, "_klsg_blind 自测失败（红线护栏异常）"
    return ok


if __name__ == "__main__":
    _selftest()
