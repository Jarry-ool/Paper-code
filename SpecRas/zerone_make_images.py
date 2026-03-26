# -*- coding: utf-8 -*-
"""
zerone_make_images.py  
==================================

把“原始振动 JSON/JSONL 数据” → “2144(+6) 特征向量” → “PNG 图像”，并生成 manifest / meta / scores。

本版关键点：
1) 新增 Raster-Stripe 无插值布局：
   - 读取 zerone_config 中的 LAYOUT_MODE / RASTER_STRIPE；
   - 当 LAYOUT_MODE=="raster" 时，使用逐点栅格图（每个特征点→一个像素块），
     面板之间可选插空白列；**面板内部绝不插空白**；
     支持 wrap_panels 控制是否折行：你设置了 stft/psd=False，则它们单行铺满；
   - 当 LAYOUT_MODE!="raster" 时，回退到原 150×150 条带图（vector_to_image）。

2) 频谱墙 PSD Wall：
   - 与旧版一致，不再重算 PSD、不做二次去直流；频谱墙直接从 T_kj 的 PSD 段获取（通过 split_feature_vector）。
   - 若开启 APPEND_PSD_WALL，则右侧追加 PSD 墙（会自动与左图同高）。

3) 归一化策略：
   - 左图（特征向量）与 PSD 均用 train 统计的 min/max 进行归一化（USE_GLOBAL_NORM=True）。
"""

from __future__ import annotations
import csv, json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

# ====== 读取项目配置 ======
try:
    from zerone_config import (
        SPLIT_DIRS, CLASSES, IMG_OUT_ROOT, SCORES_OUT_ROOT,
        IMAGE_SIZE, CHANNELS, PSD_FS,
        # 布局配置
        LAYOUT_MODE, RASTER_STRIPE,
    )
    HAS_CFG = True
except Exception as e:
    print("[WARN] 导入 zerone_config 失败，使用默认配置：", e)
    HAS_CFG = False
    SPLIT_DIRS = {"train":{"正常":[],"故障":[]}, "val":{"正常":[],"故障":[]}, "test":{"正常":[],"故障":[]}}
    CLASSES = ["正常","故障"]
    IMG_OUT_ROOT   = Path("./zerone_images").resolve()
    SCORES_OUT_ROOT= Path("./zerone_scores").resolve()
    IMAGE_SIZE = (150,150)  # 仅 legacy 模式使用
    CHANNELS = 3
    PSD_FS = 8192.0
    LAYOUT_MODE = "raster"
    RASTER_STRIPE = {
        "W_UNIT": 2, "H_UNIT": 2, "panel_order": ["time","stft","psd","hf"],
        "colormap": "jet",
        "canvas": {"width": 150, "height": 150, "allow_grow_w": False, "allow_grow_h": True,
                   "max_w": 2048, "max_h": 65535, "gap_rows": 1}
    }

# ====== 特征/编码函数 ======
from zerone_features import (
    build_sample_from_multichannel,
    normalize_features,
    vector_to_image,           # legacy 条带图
    split_feature_vector,      # 从 T_kj 中取 "psd" 段
    square_pad_np,             # 用于正方形补边
)

# 渲染参数
USE_EMPIRICAL_IN_FEATURES: bool = False    # 是否把经验 6 维拼到特征尾部（左侧图）
AGG_METHOD: str = "energy"                 # 多通道聚合方式
DEFAULT_COLORMAP: Optional[str] = "jet" if CHANNELS == 3 else None

# PSD 墙（右侧）
APPEND_PSD_WALL: bool = False
PSD_WALL_WIDTH: int = 0

# 归一化模式：train_minmax / split_minmax / local_minmax
NORM_MODE = "train_minmax"


# ===========================
# Raster-Stripe 无插值渲染器
# ===========================
def _apply_colormap(gray: np.ndarray, cmap_name: Optional[str]) -> np.ndarray:
    """把 [0,1] 的灰度图转为 (C,H,W)；cmap_name=None 时输出 1 通道，否则输出 3 通道。"""
    gray = np.clip(gray, 0.0, 1.0).astype(np.float32)
    if cmap_name is None:
        return gray[None, ...]  # (1,H,W)
    try:
        from matplotlib import colormaps as cmaps 
        cmap = cmaps.get_cmap(cmap_name)
    except Exception:
        import matplotlib.cm as cm
        cmap = cm.get_cmap("jet")
    rgb = cmap(gray)[:, :, :3]  # (H,W,3)
    return np.transpose(rgb, (2, 0, 1)).astype(np.float32)  # (3,H,W)

def _render_panel_raster(
    vec: np.ndarray,
    *,
    wrap: bool,
    tile_width: int,
    W_UNIT: int,
    H_UNIT: int,
) -> np.ndarray:
    """
    把一段 1D 特征向量渲染为“逐点栅格”灰度面板（不做插值）。
    - wrap=False：单行铺满，宽度 = len(vec) * W_UNIT，高度 = H_UNIT。
    - wrap=True ：按 tile_width 折行（列数=tile_width），行数=ceil(len/col)，
                  面板宽=tile_width*W_UNIT，高=rows*H_UNIT。
    - 面板内部绝不插空白。
    返回：float32 灰度 (H, W) ∈ [0,1]
    """
    v = np.asarray(vec, dtype=np.float32).ravel()
    n = v.size
    if n == 0:
        return np.zeros((H_UNIT, W_UNIT), dtype=np.float32)

    if not wrap:
        rows, cols = 1, n
    else:
        cols = max(1, int(tile_width))
        rows = int(np.ceil(n / float(cols)))

    H = rows * H_UNIT
    W = cols * W_UNIT
    canvas = np.zeros((H, W), dtype=np.float32)

    # 逐点填充：每个特征点→一个 W_UNIT×H_UNIT 色块
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= n:
                break
            x0, y0 = c * W_UNIT, r * H_UNIT
            canvas[y0:y0 + H_UNIT, x0:x0 + W_UNIT] = v[idx]
            idx += 1

    return np.clip(canvas, 0.0, 1.0)

def vector_to_image_raster(
    vec_norm01: np.ndarray,
    *,
    layout: Dict,
    channels: int = 3,
) -> np.ndarray:
    """
    固定基准画布，触边换行；若溢出且允许增长，则直接增长到不超过 max_* 的尺寸；
    不做任何下采样压缩。面板之间空 gap_rows 像素行。
    """
    # === 拆段 ===
    segs = split_feature_vector(np.asarray(vec_norm01, dtype=np.float32))
    time_seg = np.asarray(segs.get("time", np.zeros(0, dtype=np.float32)), dtype=np.float32)
    stft_seg = np.asarray(segs.get("stft", np.zeros(0, dtype=np.float32)), dtype=np.float32)
    psd_seg  = np.asarray(segs.get("psd",  np.zeros(0, dtype=np.float32)), dtype=np.float32)
    hf_seg   = np.asarray(segs.get("hf",   np.zeros(0, dtype=np.float32)), dtype=np.float32)

    # === 读取布局/画布参数 ===
    # === 优先使用 tile_widths 策略（垂直拼接各面板），否则回退到全局画布策略 ===
    tile_widths: Optional[Dict[str, int]] = layout.get("tile_widths", None)
    if tile_widths is not None:
        # 新的折行布局：按面板独立渲染，统一宽度后纵向堆叠
        W_UNIT = int(layout.get("W_UNIT", 2))
        H_UNIT = int(layout.get("H_UNIT", 2))
        panel_order = layout.get("panel_order", ["time", "stft", "psd", "hf"])
        # 颜色映射
        cmap_name = layout.get("colormap", "jet") if channels == 3 else None
        wrap_panels: Dict[str, bool] = layout.get("wrap_panels", {}) or {}
        insert_gap_after: Dict[str, bool] = layout.get("insert_gap_after", {}) or {}
        gap_tile_width = int(layout.get("gap_tile_width", 0))
        # 垂直 gap 行数（像素）
        canvas_cfg = layout.get("canvas", {}) or {}
        GAP_ROWS = int(canvas_cfg.get("gap_rows", 1))
        name2seg = {"time": time_seg, "stft": stft_seg, "psd": psd_seg, "hf": hf_seg}
        panel_arrays: List[np.ndarray] = []
        panel_widths: List[int] = []
        # 1) 渲染每个面板
        for name in panel_order:
            seg_vals = name2seg.get(name, np.zeros(0, dtype=np.float32))
            v = np.asarray(seg_vals, dtype=np.float32).ravel()
            if v.size == 0:
                # 空面板占位，避免缺失
                panel_arrays.append(np.zeros((H_UNIT, W_UNIT), dtype=np.float32))
                panel_widths.append(W_UNIT)
                continue
            wrap = bool(wrap_panels.get(name, True))
            tile_width = int(tile_widths.get(name, 0))
            # 若 tile_width <=0，表示不折行
            arr_panel = _render_panel_raster(v, wrap=wrap, tile_width=tile_width, W_UNIT=W_UNIT, H_UNIT=H_UNIT)
            # 若需要在面板右侧插入空白列，则在数组右侧拼接空白列
            if bool(insert_gap_after.get(name, False)) and gap_tile_width > 0:
                # gap_tile_width 是以特征点列为单位，需要乘以 W_UNIT
                gap_cols = gap_tile_width * W_UNIT
                gap_arr = np.zeros((arr_panel.shape[0], gap_cols), dtype=np.float32)
                arr_panel = np.hstack([arr_panel, gap_arr])
            panel_arrays.append(arr_panel)
            panel_widths.append(arr_panel.shape[1])
        # 2) 统一宽度
        max_w = max(panel_widths) if panel_widths else 1
        # 3) 拼接
        total_h = 0
        for idx, arr in enumerate(panel_arrays):
            total_h += arr.shape[0]
            if idx < len(panel_arrays) - 1:
                total_h += GAP_ROWS
        gray = np.zeros((total_h, max_w), dtype=np.float32)
        y0 = 0
        for idx, arr in enumerate(panel_arrays):
            h, w = arr.shape
            gray[y0:y0 + h, :w] = arr
            y0 += h
            if idx < len(panel_arrays) - 1:
                y0 += GAP_ROWS
        # 应用 colormap
        img = _apply_colormap(gray, cmap_name)
        if channels == 1 and img.shape[0] == 3:
            img = img[:1, :, :]
        if channels == 3 and img.shape[0] == 1:
            img = np.repeat(img, 3, axis=0)
        return img
    # ---- 以下为旧版全局画布策略 ----
    W_UNIT = int(layout.get("W_UNIT", 2))
    H_UNIT = int(layout.get("H_UNIT", 2))
    panel_order = layout.get("panel_order", ["time", "stft", "psd", "hf"])
    cmap_name = layout.get("colormap", "jet") if channels == 3 else None

    canvas_cfg = layout.get("canvas", {}) or {}
    CANVAS_W = int(canvas_cfg.get("width", 150))
    CANVAS_H = int(canvas_cfg.get("height", 150))
    ALLOW_GROW_W = bool(canvas_cfg.get("allow_grow_w", False))
    ALLOW_GROW_H = bool(canvas_cfg.get("allow_grow_h", True))
    MAX_W = int(canvas_cfg.get("max_w", CANVAS_W))
    MAX_H = int(canvas_cfg.get("max_h", CANVAS_H))
    GAP_ROWS = int(canvas_cfg.get("gap_rows", 1))

    # === 先用基准尺寸画，必要时动态增大 ===
    max_rows_hint = 20000
    gray = np.zeros((max_rows_hint, CANVAS_W), dtype=np.float32)
    cur_y, cur_x = 0, 0

    def ensure_capacity(target_h: int, target_w: int):
        nonlocal gray, CANVAS_W, CANVAS_H
        grow_h = target_h > gray.shape[0]
        grow_w = target_w > gray.shape[1]
        if grow_w:
            if not ALLOW_GROW_W and target_w > CANVAS_W:
                pass
            else:
                new_w = min(max(target_w, gray.shape[1] * 2), MAX_W)
                pad_w = np.zeros((gray.shape[0], new_w - gray.shape[1]), dtype=np.float32)
                gray = np.hstack([gray, pad_w])
                CANVAS_W = new_w
        if grow_h:
            if ALLOW_GROW_H:
                new_h = min(max(target_h, gray.shape[0] * 2), MAX_H)
                pad_h = np.zeros((new_h - gray.shape[0], gray.shape[1]), dtype=np.float32)
                gray = np.vstack([gray, pad_h])

    def paint_segment(seg_vals: np.ndarray):
        nonlocal cur_x, cur_y, gray
        v = np.asarray(seg_vals, dtype=np.float32).ravel()
        for val in v:
            if cur_x + W_UNIT > CANVAS_W:
                cur_x = 0
                cur_y += H_UNIT
            ensure_capacity(cur_y + H_UNIT, cur_x + W_UNIT)
            y0, y1 = cur_y, cur_y + H_UNIT
            x0, x1 = cur_x, cur_x + W_UNIT
            gray[y0:y1, x0:x1] = val
            cur_x += W_UNIT
        cur_x = 0
        cur_y += H_UNIT + GAP_ROWS

    name2seg = {"time": time_seg, "stft": stft_seg, "psd": psd_seg, "hf": hf_seg}
    for name in panel_order:
        paint_segment(name2seg.get(name, np.zeros(0, dtype=np.float32)))

    used_h = max(cur_y, 1)
    used_w = gray.shape[1]
    target_h = min(used_h, MAX_H) if ALLOW_GROW_H else min(CANVAS_H, gray.shape[0])
    target_w = min(used_w, MAX_W) if ALLOW_GROW_W else min(CANVAS_W, gray.shape[1])
    gray = gray[:target_h, :target_w]
    img = _apply_colormap(gray, cmap_name)
    if channels == 1 and img.shape[0] == 3:
        img = img[:1, :, :]
    if channels == 3 and img.shape[0] == 1:
        img = np.repeat(img, 3, axis=0)
    return img


# ====== I/O 工具 ======
def ensure_dirs():
    IMG_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SCORES_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for split in ["train","val","test"]:
        for cls in CLASSES:
            (IMG_OUT_ROOT / split / cls).mkdir(parents=True, exist_ok=True)

def load_records(file_path: Path) -> List[dict]:
    suf = file_path.suffix.lower()
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    if suf == ".jsonl":
        recs: List[dict] = []
        for line in text.splitlines():
            s = line.strip()
            if not s: continue
            try: recs.append(json.loads(s))
            except Exception: continue
        return recs
    try:
        data = json.loads(text)
    except Exception:
        return []
    if isinstance(data, list): return data
    if isinstance(data, dict) and data:
        v = next(iter(data.values()))
        if isinstance(v, list): return v
    return []

def parse_signal_csv(s: str, T: int = 8192) -> np.ndarray:
    xs: List[float] = []
    for p in s.replace("\\n","").replace("\\r","").split(","):
        p = p.strip()
        if not p: continue
        try: xs.append(float(p))
        except Exception: continue
    a = np.asarray(xs, dtype=np.float64)
    if a.size >= T: return a[:T]
    z = np.zeros((T,), dtype=np.float64); z[:a.size] = a; return z

def group_records_to_signals(records: List[dict], T: int = 8192) -> Tuple[List[List[np.ndarray]], List[str]]:
    from collections import defaultdict
    groups: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        t = r.get("data_time")
        if t is not None: groups[str(t)].append(r)
    signals_list: List[List[np.ndarray]] = []; time_list: List[str] = []
    for t, grp in groups.items():
        chans: List[np.ndarray] = []
        for r in grp:
            v = r.get("signal_value")
            if not v: continue
            sig = parse_signal_csv(str(v), T=T)
            chans.append(sig)  # 保留原始幅值，去直流在特征提取内部做
        if len(chans) >= 1:
            signals_list.append(chans); time_list.append(t)
    return signals_list, time_list

def class_to_label(cls: str) -> int:
    return CLASSES.index(cls) if cls in CLASSES else 0

# ====== PSD 墙渲染辅助 ======
def _compress_1d(vec: np.ndarray, target_len: int) -> np.ndarray:
    if vec is None or target_len <= 0:
        return np.zeros((target_len,), dtype=np.float32)
    v = np.asarray(vec, dtype=np.float32).ravel()
    if v.size == target_len: return v
    x_old = np.arange(v.size, dtype=np.float32)
    x_new = np.linspace(0, v.size - 1, target_len, dtype=np.float32)
    return np.interp(x_new, x_old, v).astype(np.float32)

def _psd_wall_to_image_block(psd_norm: np.ndarray, H: int, W: int) -> np.ndarray:
    if W <= 0: return np.zeros((H, 0), dtype=np.float32)
    col = _compress_1d(psd_norm, W)
    wall = np.tile(col.reshape(1, -1), (H, 1))
    return wall  # [0,1]

# ====== 主流程：提取 / 归一化 / 保存 ======
@dataclass
class SampleMeta:
    split: str; cls: str; label: int; img_path: str; src_json: str; data_time: str
    sample_idx: int; num_channels: int
    H_val: Optional[float] = None; C_val: Optional[float] = None
    V_val: Optional[float] = None; HF_val: Optional[float] = None
    health_index: Optional[float] = None; severity: Optional[int] = None
    def to_dict(self) -> Dict:
        d = {
            "split": self.split,
            "class": self.cls,
            "label": self.label,
            "img_path": self.img_path,
            "src_json": self.src_json,
            "data_time": self.data_time,
            "sample_idx": int(self.sample_idx),
            "num_channels": int(self.num_channels),
        }
        if self.H_val is not None:
            d.update(dict(
                H_val=float(self.H_val), C_val=float(self.C_val), V_val=float(self.V_val),
                HF_val=float(self.HF_val), health_index=float(self.health_index), severity=int(self.severity)
            ))
        return d

def process_split(split_name: str,
                  *, agg_method: str = AGG_METHOD,
                  use_empirical_in_features: bool = USE_EMPIRICAL_IN_FEATURES
                  ) -> Tuple[List[np.ndarray], List[int], List['SampleMeta'], List[np.ndarray]]:
    features: List[np.ndarray] = []
    labels: List[int] = []
    metas: List['SampleMeta'] = []
    psd_list: List[np.ndarray] = []

    split_dirs = SPLIT_DIRS.get(split_name, {})
    if not split_dirs:
        print(f"[WARN] SPLIT_DIRS['{split_name}'] 为空，跳过该划分")
        return features, labels, metas, psd_list

    for cls in CLASSES:
        files = split_dirs.get(cls, [])
        if not files:
            print(f"[INFO] split={split_name}, class={cls} 无文件，跳过")
            continue
        label = class_to_label(cls)
        print(f"[INFO] 处理 split={split_name}, class={cls}, 文件数={len(files)}")

        for fp_str in files:
            fp = Path(fp_str)
            if not fp.exists():
                print(f"[WARN] 文件不存在: {fp}")
                continue

            records = load_records(fp)
            if not records:
                print(f"[WARN] 文件中无有效记录: {fp}")
                continue
            signals_list, time_list = group_records_to_signals(records, T=8192)
            if not signals_list:
                print(f"[WARN] 文件中按 data_time 分组后没有样本: {fp}")
                continue

            for k, (signals, data_time) in enumerate(zip(signals_list, time_list)):
                # 2144(+6) 聚合特征 + 经验指标
                T_kj, emp_agg = build_sample_from_multichannel(
                    signals=signals, fs=float(PSD_FS),
                    agg_method=agg_method, include_empirical=True, return_empirical=True
                )
                if T_kj is None:
                    continue

                # === 频谱墙：直接从特征向量拆段拿到 PSD（不重算/不过度去直流） ===
                segments = split_feature_vector(T_kj)   # {'time','stft','psd','hf','_tail'(可选)}
                psd_from_feat = segments.get("psd", None)
                if psd_from_feat is None or np.size(psd_from_feat) == 0:
                    psd_from_feat = np.zeros((1,), dtype=np.float32)
                else:
                    psd_from_feat = np.asarray(psd_from_feat, dtype=np.float32).ravel()
                psd_list.append(psd_from_feat)

                # === 左侧特征：可选把经验 6 维拼接到尾部 ===
                feat_vec = np.asarray(T_kj, dtype=np.float32)
                H_val = C_val = V_val = HF_val = health_index = None; severity = None
                if emp_agg is not None and np.size(emp_agg) >= 6:
                    H_val, C_val, V_val, HF_val, health_index, severity = np.asarray(emp_agg).tolist()
                    if use_empirical_in_features:
                        feat_vec = np.concatenate([feat_vec, np.asarray(emp_agg, dtype=np.float32)], axis=0)

                features.append(feat_vec)
                labels.append(label)

                metas.append(SampleMeta(
                    split=split_name, cls=cls, label=label, img_path="",
                    src_json=str(fp.resolve().as_posix()), data_time=str(data_time),
                    sample_idx=k, num_channels=len(signals),
                    H_val=H_val, C_val=C_val, V_val=V_val, HF_val=HF_val,
                    health_index=health_index, severity=int(severity) if severity is not None else None
                ))

    return features, labels, metas, psd_list


def save_split_outputs(split_name: str,
                       features: List[np.ndarray],
                       labels: List[int],
                       metas: List[SampleMeta],
                       psd_list: List[np.ndarray],
                       feat_min: Optional[np.ndarray] = None,
                       feat_max: Optional[np.ndarray] = None,
                       psd_min: Optional[np.ndarray] = None,
                       psd_max: Optional[np.ndarray] = None) -> None:
    if not features:
        print(f"[WARN] split={split_name} 无特征，跳过保存。"); return

    # ==== 特征归一化 ====
    feat_mat = np.stack(features, axis=0)  # (N,D)
    
    if NORM_MODE == "train_minmax":
        # 期望由 main() 传入 train 的 min/max
        if feat_min is None or feat_max is None:
            raise ValueError("NORM_MODE=train_minmax 需要提供 feat_min/feat_max（来自 train）")
        feat_norm = normalize_features(feat_mat, minv=feat_min, maxv=feat_max)

    elif NORM_MODE == "split_minmax":
        # 用当前 split 的 min/max
        fm = feat_mat
        fm_min = fm.min(axis=0)
        fm_max = fm.max(axis=0)
        # 避免全常量维度除零
        same = (fm_max - fm_min) <= 1e-12
        fm_max[same] = fm_min[same] + 1.0
        feat_norm = (fm - fm_min) / (fm_max - fm_min)

    elif NORM_MODE == "local_minmax":
        # 逐样本单独 min-max（对比度最强，但跨样本不可比）
        eps = 1e-12
        fmin = feat_mat.min(axis=1, keepdims=True)
        fmax = feat_mat.max(axis=1, keepdims=True)
        feat_norm = (feat_mat - fmin) / (np.maximum(fmax - fmin, eps))

    elif NORM_MODE == "zscore_train":
        # 若你在 main() 里把 train 的均值/方差也算好传进来，可改函数签名或做成全局
        # 这里给一个占位写法（需要你在 main() 里计算 mean/std 并作为参数传入）
        raise NotImplementedError("请在 main() 中计算 train 的 mean/std 并传入后在此实现。")

    else:
        raise ValueError(f"未知 NORM_MODE: {NORM_MODE}")

    # ==== PSD 归一化 ====
    psd_arr = np.stack(psd_list, axis=0) if len(psd_list) else np.zeros((len(features), 1), dtype=np.float32)

    if NORM_MODE == "train_minmax":
        if (psd_min is None) or (psd_max is None):
            # 容错：若没传，退化为 split_minmax
            psd_min = np.nanmin(psd_arr, axis=0)
            psd_max = np.nanmax(psd_arr, axis=0)
        same = (psd_max - psd_min) <= 1e-12
        psd_max[same] = psd_min[same] + 1.0
        # 在循环里用这两个数组做归一化

    elif NORM_MODE == "split_minmax":
        psd_min = np.nanmin(psd_arr, axis=0)
        psd_max = np.nanmax(psd_arr, axis=0)
        same = (psd_max - psd_min) <= 1e-12
        psd_max[same] = psd_min[same] + 1.0
        # 在循环里用这两个数组做归一化

    elif NORM_MODE == "local_minmax":
        # 不提前求全局；在逐样本循环里对每个 psd 单独 min-max
        pass

    else:
        raise ValueError(f"未知 NORM_MODE: {NORM_MODE}")


    # ==== 输出目录 / 文件 ====
    (SCORES_OUT_ROOT / split_name).mkdir(parents=True, exist_ok=True)
    csv_path   = SCORES_OUT_ROOT / split_name / f"{split_name}_manifest.csv"
    meta_path  = SCORES_OUT_ROOT / split_name / f"{split_name}_meta.jsonl"
    score_path = SCORES_OUT_ROOT / split_name / f"{split_name}_scores.csv"

    # ==== 逐样本渲染并保存 ====
    with csv_path.open("w", encoding="utf-8", newline="") as f_csv, \
         meta_path.open("w", encoding="utf-8") as f_meta, \
         score_path.open("w", encoding="utf-8", newline="") as f_score:

        writer = csv.writer(f_csv); writer.writerow(["image_path","label"])
        score_writer = csv.writer(f_score); score_writer.writerow(["split","class","H","C","V","HF","HI","severity"])

        for idx, (vec01, label, meta, psd) in enumerate(zip(feat_norm, labels, metas, psd_list)):

            # ========== 左侧：特征图 ==========
            if LAYOUT_MODE == "raster":
                left_img_arr = vector_to_image_raster(
                    vec01,
                    layout=RASTER_STRIPE,
                    channels=CHANNELS
                )  # (C,H_left,W_left_var)
                H_left = left_img_arr.shape[1]
            else:
                # legacy：150×150 条带图（保持你的旧逻辑）
                H_target, W_total = IMAGE_SIZE
                W_wall = PSD_WALL_WIDTH if APPEND_PSD_WALL else 0
                W_left = max(1, W_total - W_wall)
                left_img_arr = vector_to_image(
                    vec01, img_size=(H_target, W_left),
                    replicate_channels=(CHANNELS==3),
                    colormap=(DEFAULT_COLORMAP if CHANNELS==3 else None)
                )
                H_left = left_img_arr.shape[1]

            # ========== 右侧：频谱墙（可选） ==========
            if APPEND_PSD_WALL and PSD_WALL_WIDTH > 0:
                psd = np.asarray(psd, dtype=np.float32)

                if NORM_MODE in ("train_minmax", "split_minmax"):
                    psd_norm = np.clip((psd - psd_min) / (psd_max - psd_min), 0.0, 1.0)
                elif NORM_MODE == "local_minmax":
                    pmin = np.nanmin(psd)
                    pmax = np.nanmax(psd)
                    if not np.isfinite(pmin) or not np.isfinite(pmax) or (pmax - pmin) <= 1e-12:
                        psd_norm = np.zeros_like(psd, dtype=np.float32)
                    else:
                        psd_norm = np.clip((psd - pmin) / (pmax - pmin), 0.0, 1.0)
                else:
                    raise ValueError(f"未知 NORM_MODE: {NORM_MODE}")

                wall_gray = _psd_wall_to_image_block(psd_norm, H_left, PSD_WALL_WIDTH)  # (H_left, W_wall)
                wall_uint8 = (np.clip(wall_gray, 0.0, 1.0) * 255).astype(np.uint8)
                # 与左图通道数一致
                if CHANNELS == 3:
                    wall_rgb = np.stack([wall_uint8] * 3, axis=0)  # (3,H,W_wall)
                else:
                    wall_rgb = np.expand_dims(wall_uint8, axis=0)  # (1,H,W_wall)
                combined = np.concatenate([left_img_arr, wall_rgb.astype(np.float32)/255.0], axis=2)
            else:
                combined = left_img_arr  # (C,H,W)

            # 保存 PNG
            # 将组合后的图像裁剪/补齐为正方形，以避免过宽或过窄
            arr = np.transpose(np.clip(combined, 0.0, 1.0), (1, 2, 0))  # (H,W,C)
            # 使用 square_pad_np 进行正方形补边。
            # 传入 target=0 表示最终边长等于 max(H, W)，即只补齐短边，不强制拉伸到固定尺寸。
            arr_sq = square_pad_np(arr, target=0, fill=0)
            img_uint8 = (arr_sq * 255).astype(np.uint8)
            src_stem = Path(meta.src_json).stem
            cls_dir = IMG_OUT_ROOT / split_name / meta.cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            img_name = f"{src_stem}_{meta.sample_idx:04d}.png"
            img_path = cls_dir / img_name
            Image.fromarray(img_uint8).save(img_path)

            # manifest
            writer.writerow([str(img_path.resolve().as_posix()), int(label)])

            # meta
            meta.img_path = str(img_path.resolve().as_posix())
            f_meta.write(json.dumps(meta.to_dict(), ensure_ascii=False) + "\n")

            # scores
            h = meta.H_val if meta.H_val is not None else float("nan")
            c = meta.C_val if meta.C_val is not None else float("nan")
            v = meta.V_val if meta.V_val is not None else float("nan")
            hf= meta.HF_val if meta.HF_val is not None else float("nan")
            hi= meta.health_index if meta.health_index is not None else float("nan")
            sev = meta.severity if meta.severity is not None else ""
            score_writer.writerow([split_name, meta.cls, h, c, v, hf, hi, sev])

    print(f"[INFO] split={split_name}: 保存图像 {len(features)} 张, CSV={csv_path.name}, META={meta_path.name}, SCORES={score_path.name}")

def ensure_dirs_all(): ensure_dirs()

def main():
    ensure_dirs_all()

    # === 1) 先处理 train，拿到所有特征和 PSD，以便计算全局 min/max ===
    tr_feats, tr_labels, tr_metas, tr_psd = process_split(split_name="train")
    if len(tr_feats) == 0:
        print("[ERROR] train 集没有样本，无法计算归一化统计量"); return

    feat_mat_train = np.stack(tr_feats, axis=0)          # (N_train, D)
    feat_min_train = feat_mat_train.min(axis=0)          # (D,)
    feat_max_train = feat_mat_train.max(axis=0)          # (D,)

    if len(tr_psd) > 0:
        psd_arr_train = np.stack(tr_psd, axis=0)         # (N_train, F_psd)
        psd_min_train = np.nanmin(psd_arr_train, axis=0)
        psd_max_train = np.nanmax(psd_arr_train, axis=0)
        same = (psd_max_train - psd_min_train) <= 1e-12
        psd_max_train[same] = psd_min_train[same] + 1.0
    else:
        psd_min_train = psd_max_train = None

    save_split_outputs(
        "train", tr_feats, tr_labels, tr_metas, tr_psd,
        feat_min=feat_min_train if NORM_MODE=="train_minmax" else None,
        feat_max=feat_max_train if NORM_MODE=="train_minmax" else None,
        psd_min=psd_min_train  if NORM_MODE=="train_minmax" else None,
        psd_max=psd_max_train  if NORM_MODE=="train_minmax" else None,
    )

    for split in ["val", "test"]:
        feats, labels, metas, psd_list = process_split(split_name=split)
        save_split_outputs(
            split, feats, labels, metas, psd_list,
            feat_min=feat_min_train if NORM_MODE=="train_minmax" else None,
            feat_max=feat_max_train if NORM_MODE=="train_minmax" else None,
            psd_min=psd_min_train  if NORM_MODE=="train_minmax" else None,
            psd_max=psd_max_train  if NORM_MODE=="train_minmax" else None,
        )

if __name__ == "__main__":
    main()
