# -*- coding: utf-8 -*-
"""
zerone_features.py
================================

本模块提供针对交流变压器振动数据的特征提取函数。实现了时域特征、
短时傅里叶段均值、功率谱密度以及高频指标等 4 组特征，默认总维度为
``TOTAL_FEAT_DIM``（当前为 1200）。本实现不再固定为 2144 维，而是根
据 ``FEAT_SCHEMA`` 自动决定维度。模块同时提供经验指标（H/C/V/HF/HI/Severity）
与多通道聚合接口。

目标：
1) 计算/变换 PSD（raw / log / log1p / octave 聚合）
2) 不改变数据的“行序列”与“列序列”，仅在最终拼图阶段做“列复制”（视觉加宽）
3) 提供 SquarePad 工具函数（numpy 版），保持原始宽高比，最后补成正方形（仅填充，不放大）
4) split_feature_vector：根据 ``FEAT_SCHEMA`` 切分一维特征向量为若干段，返回 {name: ndarray}，兼容训练脚本对 "psd" 的读取
5) 保留你原先用于特征提取的函数（时域 / STFT / PSD / HF 指标 / 经验评分 / RQA 简化实现）

注意：
- “不压缩不抽样”的约束适用于生成阶段（PNG 制作）。训练阶段的 SquarePad(150) 如需严格遵守，也可改为仅 pad 不缩放。
- 本文件不做任何图像缩放；如果最终拼图尺寸超过 150×150，请在保存 PNG 时保持原尺寸，训练时再按需处理。
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

# ===== 从配置读取策略开关 =====
from zerone_config import (
    PSD_MODE, OCTAVE_NUM_BANDS,
    IMG_TARGET_SIDE, PAD_FILL,
    STRIP_VISUAL_WEIGHTS,  # 用于列复制的视觉宽度权重
    # 新增引入特征开关和 PSD 参数，用于灵活控制特征提取
    USE_TIME_FEATURES, USE_PSD_FEATURES, USE_STFT_FEATURES,
    PSD_NPERSEG, PSD_FMAX,
)

__all__ = [
    # 核心特征
    "compute_time_features", "stft_segment_means", "compute_psd",
    "compute_high_frequency_ratios",
    # 单/多通道
    "extract_Sij", "aggregate_features", "build_sample_from_multichannel", "build_multi",
    # 经验指标与健康评分
    "score_power_entropy", "score_certainty", "score_volatility", "score_highfreq",
    "compute_health_index", "compute_violation_flags", "classify_severity", "classify_health_level",
    "rqa_det", "compute_metrics",
    # 编码/归一化
    "normalize_features", "vector_to_image",
    # 视觉拼图 / 工具
    "square_pad_np", "repeat_to_visual_width", "hstack_by_visual_weights",
    "FEAT_SCHEMA", "split_feature_vector", "build_named_strips_from_2144",
    "assemble_image_from_strips",
]

# =============================================================================
# A. SquarePad（numpy 版，仅填充，不拉伸）
# =============================================================================

def square_pad_np(arr: np.ndarray,
                  target: int = IMG_TARGET_SIDE,
                  fill: float | int = PAD_FILL) -> np.ndarray:
    """
    对 numpy 图阵进行正方形补边：
      - 仅在四周填充，不拉伸、不上采样；
      - target > 0 时：如果原图的长边已经 ≥ target，就直接返回原图（不缩放）；
      - target <= 0 或 None 时：自动按当前长边补成正方形。
    入参：
      arr   : (H,W) 或 (H,W,C)
      target: 目标边长；<=0 表示“自动方形化”
      fill  : 填充值（灰度或每通道常数）
    """
    if arr.ndim == 2:
        H, W = arr.shape
        C = None
    elif arr.ndim == 3:
        H, W, C = arr.shape
    else:
        raise ValueError("square_pad_np: 输入必须为 2D 或 3D 数组")

    # --- 判定是否自动方形化 ---
    auto_square = (target is None) or (target <= 0)
    if auto_square:
        # 以当前长边为目标边长
        target_side = int(max(H, W))
    else:
        target_side = int(target)

    # 非自动模式：如果原始尺寸已经不小于 target，就原样返回
    if (not auto_square) and (max(H, W) >= target_side):
        return arr

    pad_h = target_side - H
    pad_w = target_side - W
    top = pad_h // 2
    left = pad_w // 2
    bottom = pad_h - top
    right = pad_w - left

    if C is None:
        out = np.pad(
            arr,
            ((top, bottom), (left, right)),
            mode="constant",
            constant_values=fill,
        )
    else:
        out = np.pad(
            arr,
            ((top, bottom), (left, right), (0, 0)),
            mode="constant",
            constant_values=fill,
        )
    return out

# =============================================================================
# B. PSD 变换（raw/log/log1p/octave）
# =============================================================================
@dataclass
class PSDPack:
    freqs_idx: np.ndarray  # 横轴索引（octave 下为 0..n_bands-1）
    psd: np.ndarray        # 变换后 PSD
    mode: str              # "raw"/"log"/"log1p"/"octave"

def _safe_log10(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log10(np.maximum(x, eps))

def _octave_bins(n_freqs: int, n_bands: int) -> np.ndarray:
    """
    近似倍频分桶：将 0..n_freqs-1 指派到 n_bands 个桶（几何等比分割的简化实现）
    返回 shape=(n_freqs,) 的 band id
    """
    if n_bands <= 1:
        return np.zeros(n_freqs, dtype=int)
    edges = np.geomspace(1, n_freqs, n_bands + 1)
    edges = np.round(edges).astype(int)
    edges[0] = 0
    edges[-1] = n_freqs
    band_id = np.zeros(n_freqs, dtype=int)
    for b in range(n_bands):
        band_id[edges[b]:edges[b + 1]] = b
    return band_id

def transform_psd(psd_1d: np.ndarray, mode: str = PSD_MODE) -> PSDPack:
    """
    输入：原始 PSD (F,)
    输出：PSDPack(freqs_idx, psd, mode)
    """
    x = np.asarray(psd_1d, dtype=np.float32).ravel()
    F = x.size
    if F == 0:
        return PSDPack(freqs_idx=np.arange(0), psd=x, mode=mode)

    if mode == "raw":
        return PSDPack(freqs_idx=np.arange(F), psd=x, mode=mode)
    if mode == "log":
        return PSDPack(freqs_idx=np.arange(F), psd=_safe_log10(x), mode=mode)
    if mode == "log1p":
        return PSDPack(freqs_idx=np.arange(F), psd=_safe_log10(1.0 + x), mode=mode)
    if mode == "octave":
        band_id = _octave_bins(F, OCTAVE_NUM_BANDS)
        out = np.zeros(OCTAVE_NUM_BANDS, dtype=np.float32)
        for b in range(OCTAVE_NUM_BANDS):
            m = (band_id == b)
            if m.any():
                out[b] = float(np.mean(x[m]))
        return PSDPack(freqs_idx=np.arange(OCTAVE_NUM_BANDS), psd=out, mode=mode)
    # 兜底
    return PSDPack(freqs_idx=np.arange(F), psd=x, mode="raw")


# =============================================================================
# C. 列复制（视觉加宽）与横向拼接
# =============================================================================
def repeat_to_visual_width(strip: np.ndarray, target_cols: int) -> np.ndarray:
    """按列索引复制，保证“列序列”不变；不会插值。"""
    H, W = strip.shape[:2]
    if W == 0 or target_cols <= 0:
        return strip
    idx = np.linspace(0, W - 1, target_cols).round().astype(int)
    return strip[:, idx] if strip.ndim == 2 else strip[:, idx, :]

def hstack_by_visual_weights(named_strips: Dict[str, np.ndarray],
                             weights: Dict[str, float],
                             target_cols: int) -> Optional[np.ndarray]:
    """按权重为每个条带分配目标列数，并横向拼接（高用边缘复制对齐）。"""
    names = [k for k in named_strips.keys() if k in weights]
    if not names:
        return None
    w = np.array([weights[k] for k in names], dtype=np.float64)
    w = w / max(w.sum(), 1e-12)
    alloc = np.floor(w * target_cols).astype(int)
    rest = target_cols - int(alloc.sum())
    order = np.argsort(-w)
    for i in range(rest):
        alloc[order[i % len(order)]] += 1

    # 统一高度
    Hmax = 0
    for name in names:
        s = np.asarray(named_strips[name])
        Hmax = max(Hmax, s.shape[0])

    resized = []
    for name, cols in zip(names, alloc):
        s = np.asarray(named_strips[name])
        if s.ndim == 2:
            if s.shape[0] < Hmax:
                s = np.pad(s, ((0, Hmax - s.shape[0]), (0, 0)), mode="edge")
        else:
            if s.shape[0] < Hmax:
                s = np.pad(s, ((0, Hmax - s.shape[0]), (0, 0), (0, 0)), mode="edge")
        s2 = repeat_to_visual_width(s, cols)
        resized.append(s2)
    return np.concatenate(resized, axis=1)


# =============================================================================
# D. 固定 Schema 与分段工具
# =============================================================================
#
# 新版设计的特征维度如下：
#   - time : 15 维时域统计量。
#   - stft : 127 维短时傅里叶段均值（去除 DC 分量）。
#   - psd  : 1050 维功率谱密度，其中 1–1000 Hz 以 1 Hz 栅格共 1000 维，
#            1001–2000 Hz 以每 20 Hz 一段聚合为 50 维，合计 1050 维。
#   - hf   : 8 维高频指标，分别为在阈值 1000、2000、3000、4000 Hz 上的
#            幅值比和功率占比（两种比值×4 个阈值）。
#
FEAT_SCHEMA = [
    ("time", 15),      # 时域 15
    ("stft", 127),     # STFT 段均值 127
    ("psd", 1050),     # PSD 1..1000 每 1Hz + 1001..2000 每 20Hz ×50
    ("hf", 8),         # 高频 8 维
]

# 计算总特征维度，供零向量填充与完整性校验
TOTAL_FEAT_DIM: int = int(sum(length for _, length in FEAT_SCHEMA))

def split_feature_vector(vec: np.ndarray, schema=FEAT_SCHEMA) -> Dict[str, np.ndarray]:
    """
    将 2144(或扩展后)的一维特征向量按 schema 切分为若干段，返回 {name: ndarray}。
    若特征尾部还有其它字段（如经验 6 维），自动放入 '_tail' 段。
    """
    v = np.asarray(vec).ravel()
    out: Dict[str, np.ndarray] = {}
    start = 0
    for name, length in schema:
        end = start + int(length)
        if end > v.size:
            out[name] = v[start:].copy()
            start = v.size
            break
        out[name] = v[start:end].copy()
        start = end
    if start < v.size:
        out["_tail"] = v[start:].copy()
    return out


# =============================================================================
# E. 你原来的特征提取函数（整合并修复一致性：确保 PSD 为 1..2000，共 2000 维）
# =============================================================================
# 注：这里保留轻量版本，避免重度依赖（如 scipy）时的冲突；如需你项目中的完整版本，可在此基础上继续替换实现。

try:
    import scipy.signal as signal
except Exception as _e_scipy:
    signal = None  # 若环境无 scipy，相关函数会报错提示

def compute_time_features(x_raw: np.ndarray) -> np.ndarray:
    """15 维时域统计（基于原始信号）"""
    x = np.asarray(x_raw, dtype=float).ravel()
    N = x.size
    if N == 0:
        return np.zeros(15, dtype=float)

    mean_val = float(np.mean(x))
    rms_val  = float(np.sqrt(np.mean(x ** 2)))
    var_val  = float(np.var(x, ddof=0))
    std_val  = float(np.std(x, ddof=0))
    max_val  = float(np.max(x))
    min_val  = float(np.min(x))
    p2p_val  = max_val - min_val

    xc = x - mean_val
    m2 = np.mean(xc ** 2); m3 = np.mean(xc ** 3); m4 = np.mean(xc ** 4)
    kurtosis = float(m4 / (m2 ** 2)) if m2 > 0 else 0.0
    skewness = float(m3 / (std_val ** 3)) if std_val > 0 else 0.0

    sign_changes   = np.sum(np.signbit(x[1:]) != np.signbit(x[:-1]))
    zero_cross_rate= float(sign_changes) / (N - 1) if N > 1 else 0.0

    mean_abs   = float(np.mean(np.abs(x)))
    root4_mean = float(np.mean(np.abs(x) ** 4) ** 0.25)

    crest_factor   = max_val / rms_val    if rms_val  != 0 else 0.0
    impulse_factor = max_val / mean_abs   if mean_abs != 0 else 0.0
    margin_factor  = max_val / root4_mean if root4_mean != 0 else 0.0
    waveform_factor= rms_val / mean_abs   if mean_abs != 0 else 0.0

    return np.array([
        mean_val, rms_val, var_val, std_val, max_val, min_val, p2p_val,
        kurtosis, skewness, zero_cross_rate, mean_abs,
        crest_factor, impulse_factor, margin_factor, waveform_factor
    ], dtype=float)

# ---------------------------------------------------------------------
# 2) STFT 段均值（基于去直流信号）
# ---------------------------------------------------------------------
def stft_segment_means(x_dc: np.ndarray, fs: float, nperseg: int = 128, noverlap: int = 64) -> np.ndarray:
    """
    STFT 段均值（去 DC bin），固定 127 维（不足补零）
    """
    if signal is None:
        raise RuntimeError("需要 scipy.signal，请安装 scipy")
    f, t, Zxx = signal.stft(x_dc, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, boundary=None)
    mag = np.abs(Zxx[1:, :]) if Zxx.shape[0] > 1 else np.abs(Zxx)  # 去掉 DC bin
    seg_means = np.mean(mag, axis=0)  # (T,)
    # 固定为 127 维（不足补零，超出截断）
    out = np.zeros(127, dtype=float)
    L = min(seg_means.size, 127)
    if L > 0:
        out[:L] = seg_means[:L]
    return out

# ---------------------------------------------------------------------
# 3) Welch PSD（1–2000 Hz，1 Hz 栅格；基于去直流信号）
# ---------------------------------------------------------------------
def compute_psd(x_dc: np.ndarray, fs: float, fmin: int = 1, fmax: int = 2000, df: int = 1,
                nperseg: Optional[int] = None) -> np.ndarray:
    """
    Welch PSD（1–2000 Hz，1 Hz 栅格；基于去直流信号）
    """
    if signal is None:
        raise RuntimeError("需要 scipy.signal，请安装 scipy")
    x_dc = np.asarray(x_dc, dtype=float).ravel()
    if nperseg is None:
        nperseg = len(x_dc) // 2 if len(x_dc) >= 2 else 1
    f_raw, Pxx_raw = signal.welch(
        x_dc, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2,
        nfft=len(x_dc), detrend=False, scaling="density"
    )
    mask = (f_raw >= fmin) & (f_raw <= fmax)
    f_valid, P_valid = f_raw[mask], Pxx_raw[mask]
    freqs_target = np.arange(fmin, fmax + 1, df)
    if f_valid.size < 2:
        psd_interp = np.full_like(freqs_target, P_valid.mean() if P_valid.size else 0.0, dtype=float)
    else:
        psd_interp = np.interp(freqs_target, f_valid, P_valid)
    return psd_interp.astype(float)

# ---------------------------------------------------------------------
# 4) 高频指标（幅值比 + 功率比）
# ---------------------------------------------------------------------
def compute_high_frequency_ratios(psd: np.ndarray, fs: float, threshold_hz: float = 1000.0,
                                  fmin: float = 1.0, fmax: float = 2000.0) -> Tuple[float, float]:
    """高频幅值比 + 高频功率占比"""
    psd = np.asarray(psd, dtype=float).ravel()
    if psd.size == 0:
        return 0.0, 0.0
    freqs = np.linspace(fmin, fmax, psd.size)
    low_mask  = freqs < threshold_hz
    high_mask = (freqs >= threshold_hz) & (freqs <= fmax)
    if not (np.any(low_mask) and np.any(high_mask)):
        return 0.0, 0.0
    low_vals  = psd[low_mask]
    high_vals = psd[high_mask]
    amp_ratio = float(np.mean(high_vals) / (np.mean(low_vals) + 1e-12))
    total_power = float(np.trapz(psd, freqs))
    hf_power    = float(np.trapz(high_vals, freqs[high_mask]))
    power_ratio = float(hf_power / (total_power + 1e-12))
    return amp_ratio, power_ratio

# ---------------------------------------------------------------------
# 5) 经验指标评分体系
# ---------------------------------------------------------------------
def score_power_entropy(h: float) -> float:
    return 0.5671 * np.exp(-((h - 3.1) / 0.5895) ** 2) + 0.9038 * np.exp(-((h - 2.147) / 0.945) ** 2)

def score_certainty(c: float) -> float:
    return 0.3972 * np.exp(-((c - (-0.08728)) / 0.09103) ** 2) + 0.9079 * np.exp(-((c - 0.08263) / 0.2975) ** 2)

def score_volatility(v: float) -> float:
    return 0.9412 * np.exp(0.3344 * v) + (-0.9361) * np.exp(-57.31 * v)

def score_highfreq(hf: float) -> float:
    return 0.9552 * np.exp(0.1019 * hf) + (-0.489) * np.exp(-43.76 * hf)

def compute_health_index(H: float, C: float, V_pct: float, HF_pct: float,
                         weights: Dict[str, float] | None = None) -> float:
    if weights is None:
        weights = {"H": 0.5, "C": 0.1, "V": 0.1, "HF": 0.3}
    v_ratio  = float(V_pct) / 100.0
    hf_ratio = float(HF_pct) / 100.0
    sH = score_power_entropy(float(H))
    sC = score_certainty(float(C))
    sV = score_volatility(v_ratio)
    sHF= score_highfreq(hf_ratio)
    tot = (weights.get("H", 0.0) * sH +
           weights.get("C", 0.0) * sC +
           weights.get("V", 0.0) * sV +
           weights.get("HF", 0.0) * sHF)
    return float(tot * 100.0)

def compute_violation_flags(H: float, C: float, V_pct: float, HF_pct: float,
                            thr_H: float, thr_C: float, thr_V: float, thr_HF: float) -> Tuple[int, int, int, int]:
    flag_H  = 1 if H  >= thr_H else 0
    flag_C  = 1 if C  <= thr_C else 0
    flag_V  = 1 if V_pct >= thr_V else 0
    flag_HF = 1 if HF_pct >= thr_HF else 0
    return flag_H, flag_C, flag_V, flag_HF

def classify_severity(flags: Tuple[int, int, int, int]) -> str:
    h, c, v, hf = [int(f) for f in flags]
    total = h + c + v + hf
    if h == 1 and (c + v + hf) >= 2:
        return "严重"
    elif total == 0:
        return "正常"
    elif total == 1:
        return "注意"
    else:
        return "异常"

def classify_health_level(hi: float) -> str:
    if hi <= 60: return "正常"
    if hi <= 80: return "注意"
    if hi <= 90: return "异常"
    return "严重"

# ---------------------------------------------------------------------
# 6) RQA (DET) —— 安全实现（避免卡死）
# ---------------------------------------------------------------------
# ==== RQA 加速/容错配置 ====
# ==== 极简 & 快速 RQA(DET) 近似：只看有限带宽 K（默认 64）====
RQA_ENABLE = True
RQA_MAX_LEN = 2048
RQA_DECIM   = 4
RQA_M       = 2
RQA_TAU     = 1
RQA_EPS     = 0.1
RQA_LMIN    = 2
RQA_BAND    = 64   # 只看 |i-j|<=64 的对角带

def _runlen_sum_ge(arr01: np.ndarray, lmin: int):
    if arr01.size == 0: return 0, 0.0
    a = np.concatenate(([0], arr01.astype(np.uint8), [0]))
    d = np.diff(a.astype(np.int16))
    st = np.where(d == 1)[0]; ed = np.where(d == -1)[0]
    lens = ed - st
    if lens.size == 0: return 0, 0.0
    sel = lens >= int(lmin)
    return int(lens[sel].sum()), float(lens.mean())

def rqa_det(x: np.ndarray,
            m: int = RQA_M, tau: int = RQA_TAU, eps: float = RQA_EPS,
            lmin: int = RQA_LMIN, max_len: int = RQA_MAX_LEN,
            decim: int = RQA_DECIM, band: int = RQA_BAND):
    # 预处理：降采样 + 截断
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if decim > 1: x = x[::int(decim)]
    if max_len and x.size > max_len: x = x[:int(max_len)]
    if x.size < (m - 1) * tau + 2: return 0.0, 0.0, 0.0, 0.0

    # 延迟嵌入
    N = x.size - (m - 1) * tau
    emb = np.column_stack([x[i:i+N] for i in range(0, m * tau, tau)]).astype(np.float32)
    if N <= 1: return 0.0, 0.0, 0.0, 0.0

    eps2 = float(eps) * float(eps)
    band = int(max(1, min(band, N - 1)))

    rec_points = 0   # 带内命中总数（用于 RR、DET 分母）
    det_points = 0   # 连续段(≥lmin)的点数总和（DET 分子）
    Lmeans: List[float] = []

    for k in range(1, band + 1):
        T = N - k
        if T <= 0: break
        diffs = emb[:T] - emb[k:k+T]             # (T, m)
        d2 = np.einsum('ij,ij->i', diffs, diffs) # (T,)
        hits = (d2 <= eps2).astype(np.uint8)     # 一维 0/1
        hsum = int(hits.sum())
        rec_points += hsum * 2                   # 上下带对称
        dp, Lm = _runlen_sum_ge(hits, int(lmin))
        det_points += dp * 2
        if Lm > 0: Lmeans.append(Lm)

    denom_pairs = sum(max(0, (N - k)) for k in range(1, band + 1)) * 2
    RR   = (rec_points / float(denom_pairs)) if denom_pairs > 0 else 0.0
    DET  = (det_points / float(rec_points))  if rec_points   > 0 else 0.0
    Lavg = float(np.mean(Lmeans)) if Lmeans else 0.0
    C = DET  # 占位：复杂度=DET
    return float(DET), float(C), float(RR), float(Lavg)

# ---------------------------------------------------------------------
# 7) 计算 H / C / V% / HF%
# ---------------------------------------------------------------------
def _psd_distribution(psdx: np.ndarray, f_stop: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    psdx -> 累积分布(0..f_stop,1Hz)：
      返回：
        PSD_alpha_total: (f_stop+1, C) 累计能量占比 [0,1]
        power_cum:       (f_stop+1, C) 累计能量
    """
    psdx = np.asarray(psdx)
    F, C = psdx.shape
    if F < f_stop + 1:
        raise ValueError(f"psdx长度 {F} < f_stop+1={f_stop+1}")
    ps = psdx[:f_stop+1, :]
    power_cum = np.cumsum(ps, axis=0)
    power_total = power_cum[-1:, :] + 1e-12
    PSD_alpha_total = power_cum / power_total
    return PSD_alpha_total, power_cum

def _fca_psd_based(PSD_alpha_total: np.ndarray, f_initial: int, f_step: int, f_stop: int) -> np.ndarray:
    """FCA：带频率权重的 PSD 熵（与 MATLAB 等价），返回 (C,)"""
    A = np.asarray(PSD_alpha_total)
    frq_interval = 10
    centers = np.arange(f_initial, f_stop + 1, f_step, dtype=int)
    lefts  = np.clip(centers - frq_interval, 0, f_stop)
    rights = np.clip(centers + frq_interval, 0, f_stop)
    rights[-1] = f_stop
    band_energy = A[rights, :] - A[lefts, :]
    wf = centers.astype(float) / float(f_stop)
    Ef_total = (wf @ band_energy) + 1e-12
    pf = (band_energy * wf[:, None]) / Ef_total
    with np.errstate(divide='ignore', invalid='ignore'):
        plogp = np.where(pf > 0, pf * np.log(pf), 0.0)
    return -np.sum(plogp, axis=0)

def energy_weighted_signal(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据每个通道的时间域能量（均方值）计算权重，得到加权合成的一维信号。

    参数
    ------
    X : np.ndarray
        形状 (T, U) 的多通道信号，T 为时间点数，U 为通道数。

    返回
    ------
    x_weighted : np.ndarray
        一维合成信号，长度 T。
    w : np.ndarray
        通道权重向量，长度 U，非负且和为 1。

    说明
    ------
    - 通道能量 E_u = mean(x_u^2)，只依赖时间域，不依赖采样率 fs；
      与全频段 PSD 能量成正比（Parseval 定理），适合 fs 未来变化的场景。
    """
    X = np.asarray(X)
    if X.ndim == 1:
        x = X.astype(np.float64).ravel()
        return x, np.ones(1, dtype=np.float64)
    T, U = X.shape
    if U == 1:
        return X[:, 0].astype(np.float64), np.ones(1, dtype=np.float64)

    # 每个通道的均方能量
    E = np.mean(X.astype(np.float64) ** 2, axis=0)  # (U,)
    E = E + 1e-12  # 防止全零
    w = E / np.sum(E)
    x_weighted = X @ w  # (T,)
    return x_weighted.astype(np.float64), w.astype(np.float64)

def _fluct_rate(signal_in: np.ndarray, Fs: int, f: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """每电源周期峰值的波动率（var/mean, std/mean）"""
    X = np.asarray(signal_in)
    if X.ndim == 1: X = X[:, None]
    N, C = X.shape; win = int(np.floor(Fs / f)); nwin = int(np.floor(N / win))
    if nwin < 1: raise ValueError("数据长度不足以形成一个周期窗口")
    fr_var = np.zeros(C); fr_std = np.zeros(C)
    for c in range(C):
        x = X[:nwin*win, c]
        mv = [np.max(x[i*win:(i+1)*win]) for i in range(nwin)]
        mv = np.asarray(mv, float); mu = np.mean(mv) + 1e-12
        fr_var[c] = np.var(mv) / mu; fr_std[c] = np.std(mv) / mu
    return fr_var, fr_std

def _magratio_above1000_below1000(Amp: np.ndarray) -> np.ndarray | float:
    """1000Hz 上/下最大幅比；Amp 为 1Hz 栅格的幅值/PSD"""
    A = np.asarray(Amp)
    if A.ndim == 1: A = A[:, None]
    F, C = A.shape
    hi_l = min(999, F-1); hi_r = min(2001, F)
    lo_r = min(951, F)
    above = np.max(A[hi_l:hi_r, :], axis=0)
    below = np.max(A[:lo_r, :], axis=0)
    ratio = above / (below + 1e-12)
    return ratio if C > 1 else float(ratio[0])

def compute_metrics(X: np.ndarray, psd_1to2000: np.ndarray, fmin: int, fmax: int,
                    hf_threshold_hz: float = 2000.0) -> Tuple[float, float, float, float]:
    """
    计算 H/C/V%/HF%（轻量近似版）
    """
    psd = np.asarray(psd_1to2000, dtype=float).ravel()
    # ---- H: 振动功率熵 ----
    psd = np.asarray(psd).astype(float).ravel()
    F_full = int(fmax) + 1
    psd_full = np.zeros((F_full, 1), dtype=float)  # (f=0..fmax, 1)
    i0 = int(max(0, fmin)); i1 = int(min(fmax, fmin + len(psd) - 1))
    psd_full[i0:i1+1, 0] = psd[:(i1 - i0 + 1)]
    PSD_alpha_total, _ = _psd_distribution(psd_full, f_stop=int(fmax))
    H_vec = _fca_psd_based(PSD_alpha_total, f_initial=50, f_step=50, f_stop=int(fmax))
    H = float(np.mean(H_vec))

    # ---- C: 振动确定度（在能量加权通道上做 RQA）----
    if X.ndim == 2 and X.shape[1] > 1:
        x_rqa, _ = energy_weighted_signal(X)
    else:
        x_rqa = np.asarray(X).ravel()
    if RQA_ENABLE:
        try:
            _, C_val, _, _ = rqa_det(x_rqa.reshape(-1,))
        except Exception:
            C_val = float('nan')
    else:
        C_val = float('nan')

    # ---- V_pct: 振动波动率（加权通道的周期峰值波动） ----
    FS_DEFAULT = 8192  # 这里默认 8192，如采样率变化可在上层统一修改
    fr_var, fr_std = _fluct_rate(x_rqa, Fs=FS_DEFAULT, f=50)
    # x_rqa 作为一维信号，返回长度 1 的数组
    V_pct = float(fr_std[0] * 100.0)

    # ---- HF_pct: 高频能量占比 ----
    HF_ratio = _magratio_above1000_below1000(psd_full)
    HF_pct = float((np.mean(HF_ratio) if isinstance(HF_ratio, np.ndarray) else HF_ratio) * 100.0)
    return float(H), float(C_val), float(V_pct), float(HF_pct)


# ---------------------------------------------------------------------
# 8) 单通道 S_ij（2144 维）与多通道聚合
# ---------------------------------------------------------------------
def extract_Sij(x_raw: np.ndarray, fs: float, include_empirical: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    按照当前 FEAT_SCHEMA 提取单通道特征向量。

    返回：
        Sij : ndarray, shape=(TOTAL_FEAT_DIM,)
            按顺序拼接的时域、STFT、PSD、HF 向量；若对应开关关闭，则用零填充。
        empirical : Optional[ndarray], shape=(6,)
            [H, C, V%, HF%, HI, SeverityBin] 经验指标；仅当 include_empirical=True 时返回。
    """
    x_raw = np.asarray(x_raw, dtype=float).ravel()
    if x_raw.size == 0:
        raise ValueError("x_raw 为空")

    # === 时域特征 ===
    if USE_TIME_FEATURES:
        time_feat = compute_time_features(x_raw)
    else:
        time_feat = np.zeros(15, dtype=float)

    # === 去直流信号 ===
    x_dc = x_raw - float(np.mean(x_raw))

    # === STFT 特征 ===
    if USE_STFT_FEATURES:
        try:
            stft_feat = stft_segment_means(x_dc, fs=fs, nperseg=128, noverlap=64)
        except Exception:
            # 无 scipy 或计算失败时返回零向量
            stft_feat = np.zeros(127, dtype=float)
    else:
        stft_feat = np.zeros(127, dtype=float)

    # === PSD 特征与高频指标 ===
    if USE_PSD_FEATURES:
        # 计算完整 PSD (1..PSD_FMAX Hz, df=1)；使用预设窗口长度
        try:
            psd_full = compute_psd(x_dc, fs=fs, fmin=1, fmax=int(PSD_FMAX), df=1, nperseg=int(PSD_NPERSEG))
        except Exception:
            # 兜底：若计算失败，则用零填充
            psd_full = np.zeros(int(PSD_FMAX), dtype=float)
        # 低频 1–1000 Hz（1Hz 栅格）
        low_limit = 1000
        psd_low = psd_full[:low_limit].copy() if psd_full.size >= low_limit else np.pad(psd_full.copy(), (0, max(0, low_limit - psd_full.size)), 'constant')[:low_limit]
        # 中频 1001–2000 Hz 聚合为 50 段（每 20Hz 一段）
        start_idx = low_limit  # 对应 1001 Hz (索引 1000)
        seg_len = 20
        num_segments = 50
        psd_mid = np.zeros(num_segments, dtype=float)
        for i in range(num_segments):
            l = start_idx + i * seg_len
            r = start_idx + (i + 1) * seg_len
            if l >= psd_full.size:
                seg = np.empty(0, dtype=float)
            else:
                seg = psd_full[l:min(r, psd_full.size)]
            psd_mid[i] = float(np.mean(seg)) if seg.size > 0 else 0.0
        psd_feat = np.concatenate([psd_low, psd_mid], axis=0)
        # 高频 8 维指标：阈值 1000/2000/3000/4000 Hz 上的幅值比 & 功率占比
        hf_vals: List[float] = []
        for thr in [1000.0, 2000.0, 3000.0, 4000.0]:
            ar, pr = compute_high_frequency_ratios(psd_full, fs=fs, threshold_hz=float(thr), fmin=1.0, fmax=float(PSD_FMAX))
            hf_vals.extend([ar, pr])
        hf_feat = np.array(hf_vals, dtype=float)
    else:
        psd_feat = np.zeros(1050, dtype=float)
        hf_feat = np.zeros(8, dtype=float)

    # === 拼接完整特征 ===
    Sij = np.concatenate([time_feat, stft_feat, psd_feat, hf_feat], axis=0)
    if Sij.size != TOTAL_FEAT_DIM:
        raise ValueError(f"S_ij 维度应为 {TOTAL_FEAT_DIM}，当前为 {Sij.size}")

    # === 经验指标计算（可选） ===
    empirical: Optional[np.ndarray] = None
    if include_empirical:
        # 为经验指标，始终使用 1–2000 Hz 的 PSD；不依赖上述 PSD 开关
        try:
            psd_1to2000_emp = compute_psd(x_dc, fs=fs, fmin=1, fmax=2000, df=1, nperseg=int(PSD_NPERSEG))
        except Exception:
            psd_1to2000_emp = np.zeros(2000, dtype=float)
        H_val, C_val, V_pct, HF_pct = compute_metrics(x_dc.reshape(-1, 1), psd_1to2000_emp, fmin=1, fmax=2000, hf_threshold_hz=2000.0)
        hi = compute_health_index(H_val, C_val, V_pct, HF_pct)
        flags = compute_violation_flags(H_val, C_val, V_pct, HF_pct, thr_H=3.0, thr_C=0.0, thr_V=30.0, thr_HF=30.0)
        sev_txt = classify_severity(flags)
        severity_bin = 1.0 if sev_txt in ("异常", "严重") else 0.0
        empirical = np.array([H_val, C_val, V_pct, HF_pct, hi, severity_bin], dtype=float)

    return Sij.astype(float), (empirical.astype(float) if empirical is not None else None)

def aggregate_features(S_list: List[np.ndarray],
                       x_raw_list: List[np.ndarray],
                       method: str = "mean",
                       lse_beta: float = 5.0) -> np.ndarray:
    """
    多通道聚合：mean / energy / variance_weight / pca / lse
    """
    if len(S_list) == 0:
        # 返回与特征长度一致的零向量
        return np.zeros(TOTAL_FEAT_DIM, dtype=float)

    # 对输入特征列表做堆叠，形状 (U, D)
    S = np.stack(S_list, axis=0)  # (U, TOTAL_FEAT_DIM)
    U = S.shape[0]

    # ---- 普通几种聚合 ----
    if method == "mean" or U == 1:
        return S.mean(axis=0)

    if method == "energy":
        energies = np.array([float(np.mean(x**2)) for x in x_raw_list], dtype=float)
        w = energies / (energies.sum() + 1e-12)
        return (S.T @ w).astype(float)

    if method == "variance_weight":
        var = np.var(S, axis=1) + 1e-12
        w = var / var.sum()
        return (S.T @ w).astype(float)

    if method == "pca":
        S0 = S - S.mean(axis=0, keepdims=True)
        try:
            _, _, Vt = np.linalg.svd(S0, full_matrices=False)
            v1 = Vt[0]
            scores = (S0 @ v1)
            w = np.abs(scores) / (np.abs(scores).sum() + 1e-12)
            return (S.T @ w).astype(float)
        except np.linalg.LinAlgError:
            return S.mean(axis=0)

    # ---- MIL 风格：Log-Sum-Exp pooling ----
    if method == "lse":
        """
        LSE pooling：
            对于每一维 d：
                m_d = max_u S[u,d]
                z[u,d] = exp( beta * (S[u,d] - m_d) )
                w[u,d] = z[u,d] / sum_u z[u,d]
                f[d]  = sum_u w[u,d] * S[u,d]
        beta 越大越接近 max pooling，越小越接近 mean pooling。
        """
        beta = float(lse_beta)
        # (U, D)
        S_shift = S - S.max(axis=0, keepdims=True)
        Z = np.exp(beta * S_shift)                    # (U, D)
        Z_sum = Z.sum(axis=0, keepdims=True) + 1e-12  # (1, D)
        W = Z / Z_sum                                 # (U, D)
        pooled = (W * S).sum(axis=0)                  # (D,)
        return pooled.astype(float)

    # 兜底
    return S.mean(axis=0)

# =============================================================================
# 9) 向量 -> 单通道灰度条带（不插值版，列复制 + 归一化留到上游做）
# =============================================================================
def vector_to_strip_gray(v: np.ndarray, height: int) -> np.ndarray:
    """
    将 1D 向量竖直铺成 (H, W=1) 的灰度条带（后续再做列复制扩展），不做插值。
    """
    x = np.asarray(v, dtype=float).ravel()
    if x.size == 0:
        return np.zeros((height, 1), dtype=float)
    # 高度与元素数不一致时，采用“折行”到 H：按行优先映射，保证顺序不变
    H = int(height)
    tiles = int(np.ceil(x.size / H))
    pad = tiles * H - x.size
    if pad > 0:
        x = np.concatenate([x, np.zeros(pad, dtype=float)], axis=0)
    M = x.reshape(tiles, H).T  # (H, tiles)
    return M  # 宽度=tiles


def build_named_strips_from_2144(vec_2144: np.ndarray, height: int = IMG_TARGET_SIDE) -> Dict[str, np.ndarray]:
    """
    根据固定 schema，把 2144 向量拆为 4 段，并各自构造成不插值的灰度条带。
    返回：
      {
        "time_features": (H, W_t),
        "stft":          (H, W_s),
        "psd":           (H, W_p),
        "indices":       (H, W_hf)
      }
    """
    segs = split_feature_vector(vec_2144)
    strips = {
        "time_features": vector_to_strip_gray(segs.get("time", np.array([])), height),
        "stft":          vector_to_strip_gray(segs.get("stft", np.array([])), height),
        "psd":           vector_to_strip_gray(segs.get("psd",  np.array([])), height),
        "indices":       vector_to_strip_gray(segs.get("hf",   np.array([])), height),
    }
    return strips


def assemble_image_from_strips(vec_2144: np.ndarray,
                               target_side: int = IMG_TARGET_SIDE,
                               weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    """
    生成灰度图（H=target_side），横向由各段条带按视觉权重列复制后拼接；
    不做缩放；若最终宽/高 < target_side，则仅做 SquarePad 填充至正方形。
    返回：np.ndarray, shape=(target_side, target_side)  （若原内容更大，则返回原尺寸）
    """
    if weights is None:
        weights = STRIP_VISUAL_WEIGHTS

    H = int(target_side)
    named_strips = build_named_strips_from_2144(vec_2144, height=H)

    # 估一个总列数：各段原始列数之和
    total_cols_raw = sum(s.shape[1] for s in named_strips.values())
    total_cols_target = max(total_cols_raw, H)  # 给足一点横向空间
    canvas = hstack_by_visual_weights(named_strips, weights, total_cols_target)
    if canvas is None:
        canvas = np.zeros((H, H), dtype=float)

    # 若 canvas 比 target 小，则仅做 pad；若更大，保持不缩放直接返回
    if max(canvas.shape[:2]) <= target_side:
        return square_pad_np(canvas, target=target_side, fill=PAD_FILL)
    else:
        return canvas  # 上游存 PNG 时直接按原尺寸保存


# ---------------------------------------------------------------------
# 10) 归一化与向量->图像
# ---------------------------------------------------------------------
def normalize_features(feature_matrix: np.ndarray,
                       eps: float = 1e-8,
                       minv: Optional[np.ndarray] = None,
                       maxv: Optional[np.ndarray] = None) -> np.ndarray:
    """
    按列归一化：
      - 如果不给 minv/maxv：在当前矩阵上计算列最小/最大值；
      - 如果给了 minv/maxv：使用外部提供的列最小/最大值做归一化。
    """
    A = np.asarray(feature_matrix, dtype=float)
    if A.ndim == 1:
        A = A.reshape(1, -1)

    if minv is None or maxv is None:
        minv = A.min(axis=0, keepdims=True)
        maxv = A.max(axis=0, keepdims=True)
    else:
        minv = np.asarray(minv, dtype=float)
        maxv = np.asarray(maxv, dtype=float)
        if minv.ndim == 1:
            minv = minv[None, :]
        if maxv.ndim == 1:
            maxv = maxv[None, :]

    rng = np.maximum(maxv - minv, eps)
    A_norm = (A - minv) / rng
    return np.clip(A_norm, 0.0, 1.0)

def _to_uint8(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)

def vector_to_image(vec01: np.ndarray,
                    img_size: Tuple[int, int] = (150, 150),
                    replicate_channels: bool = True,
                    colormap: Optional[str] = None) -> np.ndarray:
    """
    legacy 条带图：根据 ``FEAT_SCHEMA`` 切分一维特征向量并生成灰度条带，按视觉权重拼接到目标宽度，再
    pad 到目标高宽。返回 (C,H,W) ∈ [0,1] float32；C=3 时会应用 colormap 或灰度三通道复制。
    """
    H, W = int(img_size[0]), int(img_size[1])
    strips = build_named_strips_from_2144(np.asarray(vec01, dtype=np.float32), height=H)
    merged = hstack_by_visual_weights(strips, STRIP_VISUAL_WEIGHTS, target_cols=W)
    if merged is None:
        merged = np.zeros((H, W), dtype=np.float32)
    # 保证尺寸恰好 H×W：必要时裁切或 pad（不缩放）
    if merged.shape[0] != H or merged.shape[1] != W:
        merged = merged[:H, :W]
        if merged.shape[0] < H or merged.shape[1] < W:
            merged = square_pad_np(merged, target=max(H, W), fill=PAD_FILL)[:H, :W]
    gray01 = np.clip(merged, 0.0, 1.0)

    if colormap:
        try:
            import matplotlib
            cmap = matplotlib.colormaps.get_cmap(colormap)
            rgb = cmap(gray01)[:, :, :3]
            img = np.transpose(rgb, (2, 0, 1)).astype(np.float32)
        except Exception:
            img = np.repeat(gray01[None, ...], 3, axis=0).astype(np.float32)
    else:
        if replicate_channels:
            img = np.repeat(gray01[None, ...], 3, axis=0).astype(np.float32)
        else:
            img = gray01[None, ...].astype(np.float32)
    return img


# =============================================================================
# 11) 多通道样本构建
# =============================================================================
def _ensure_len(x: np.ndarray, T: int = 8192) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).ravel()
    if x.size >= T: return x[:T]
    out = np.zeros(T, dtype=np.float32); out[:x.size] = x; return out

def build_sample_from_multichannel(
    *,
    signals: List[np.ndarray],
    fs: float,
    agg_method: str = "energy",
    include_empirical: bool = True,
    return_empirical: bool = True,
    lse_beta: float = 5.0
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    多通道 → 聚合特征（长度 = TOTAL_FEAT_DIM）; 可选返回经验 6 维。

    参数
    ----
    signals : List[np.ndarray]    多通道原始信号（长度可不等，将按 8192 对齐/截断）
    fs      : float               采样率
    agg_method: str               'mean'/'energy'/'variance_weight'/'pca'/'lse'
    include_empirical: bool       是否计算经验指标（用于作为附加信息或拼接）
    return_empirical : bool       是否把经验 6 维作为第二返回值
    lse_beta: float               LSE pooling 强度

    返回
    ----
    (T_kj_2144, empirical6 或 None)
    """
    if (signals is None) or (len(signals) == 0):
        return None, None

    x_list = [_ensure_len(x, 8192) for x in signals]
    S_list: List[np.ndarray] = []
    for x in x_list:
        Sij, _ = extract_Sij(x, fs=fs, include_empirical=False)
        S_list.append(Sij)

    # 聚合特征向量
    T_kj = aggregate_features(S_list, x_list, method=agg_method, lse_beta=lse_beta)

    # 经验 6
    empirical6 = None
    if include_empirical:
        # 使用多通道信号计算经验指标：
        # 1) 合并多通道信号至 (T,U)
        X2D = np.stack(x_list, axis=1)  # (8192, U)
        # 2) 计算能量加权的一维信号
        x_weighted, _ = energy_weighted_signal(X2D)
        # 3) 计算 1–2000 Hz 的 PSD（1 Hz 栅格）用于经验指标
        try:
            psd_emp = compute_psd(x_weighted, fs=fs, fmin=1, fmax=2000, df=1, nperseg=int(PSD_NPERSEG))
        except Exception:
            psd_emp = np.zeros(2000, dtype=np.float32)
        # 4) 计算 H/C/V%/HF% 指标
        H_val, C_val, V_pct, HF_pct = compute_metrics(X2D, psd_emp, fmin=1, fmax=2000, hf_threshold_hz=2000.0)
        # 5) 健康指数与等级
        hi = compute_health_index(H_val, C_val, V_pct, HF_pct)
        flags = compute_violation_flags(H_val, C_val, V_pct, HF_pct, thr_H=3.0, thr_C=0.0, thr_V=30.0, thr_HF=30.0)
        sev_txt = classify_severity(flags)
        severity_bin = 1.0 if sev_txt in ("异常", "严重") else 0.0
        empirical6 = np.array([H_val, C_val, V_pct, HF_pct, hi, severity_bin], dtype=np.float32)

    if return_empirical:
        return T_kj.astype(np.float32), (empirical6.astype(np.float32) if empirical6 is not None else None)
    else:
        return T_kj.astype(np.float32), None

# 兼容旧名
build_multi = build_sample_from_multichannel
