# -*- coding: utf-8 -*-
"""
hetero_data.py
数据加载核心（Zerone风格）：统一 JSON/JSONL 读取 → 按 data_time 聚合为多通道 → 能量加权成单通道 →
对齐长度与归一化 → 转 3 通道图像 (CWT / STFT 占位 / Context)。
严格依赖 hetero_config.py，不额外引入常量。
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import pywt
from scipy.signal import stft

import hetero_config as cfg

# ========= 目录标签关键词（仅用于“按目录过滤正常/故障”） =========
NORMAL_KEYS = ("正常", "normal")
FAULT_KEYS  = ("故障","异常", "fault", "abnormal", "error")

def _is_normal_dir(p: Path) -> bool:
    s = str(p).lower()
    return any(k in s for k in NORMAL_KEYS)

def _is_fault_dir(p: Path) -> bool:
    s = str(p).lower()
    return any(k in s for k in FAULT_KEYS)

# ========= 读取与解析 =========
def _read_all_records(fp: Path) -> List[dict]:
    """
    读一个 .jsonl / .json 文件的“原始记录列表”（每行/每条为一个 dict）。
    兼容 .json: list[dict] 或 {"data":[...]} / {"records":[...]}。
    """
    text = fp.read_text(encoding="utf-8", errors="ignore")
    if fp.suffix.lower() == ".jsonl":
        recs: List[dict] = []
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                recs.append(json.loads(s))
            except Exception:
                # 跳过坏行
                continue
        return recs

    # .json
    try:
        data = json.loads(text)
    except Exception:
        return []
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict):
        for k in ("data", "records", "list", "items"):
            v = data.get(k)
            if isinstance(v, list):
                return [d for d in v if isinstance(d, dict)]
    return []

def _pick_timekey(rec: dict) -> Optional[str]:
    """尽量稳健地取时间键。优先 data_time，其次 dataTime/timestamp/acq_time 等。"""
    for k in ("data_time", "dataTime", "timestamp", "acq_time", "time"):
        if k in rec and rec[k] is not None:
            return str(rec[k])
    return None

def _parse_signal_any(v: Any) -> Optional[np.ndarray]:
    """
    将 rec["signal_value"] 解析为 1D np.ndarray(float)：
    - 字符串：去括号/空白/分隔符，逗号分割
    - list/tuple：逐个转 float
    - 其他：返回 None
    """
    T = int(cfg.SIGNAL_LEN)
    try:
        if isinstance(v, str):
            s = v.replace("[", "").replace("]", "").replace("\n", "").replace("\r", " ")
            parts = [p.strip() for p in s.split(",") if p.strip()]
            arr = np.asarray([float(p) for p in parts], dtype=np.float64)
        elif isinstance(v, (list, tuple)):
            arr = np.asarray([float(x) for x in v], dtype=np.float64)
        else:
            return None
    except Exception:
        return None
    # 长度对齐到 T
    if arr.size >= T:
        return arr[:T]
    out = np.zeros((T,), dtype=np.float64)
    out[:arr.size] = arr
    return out

def _group_by_time(records: List[dict]) -> Dict[str, List[np.ndarray]]:
    """
    按时间戳聚合：time_key -> [sig_u]（多通道列表）
    仅保留 signal_value 可解析成功的记录。
    """
    groups: Dict[str, List[np.ndarray]] = {}
    for r in records:
        tk = _pick_timekey(r)
        if not tk:
            continue
        sig = _parse_signal_any(r.get("signal_value"))
        if sig is None:
            continue
        groups.setdefault(tk, []).append(sig)
    return groups

def _energy_weighted_1d(signals: List[np.ndarray]) -> np.ndarray:
    """
    多通道 → 能量加权一维：
      w_u = mean(sig_u^2) / sum(mean(sig_j^2))
      x = sum_u w_u * sig_u
    """
    X = np.stack(signals, axis=1)  # (T, U)
    E = (X ** 2).mean(axis=0) + 1e-12
    w = E / E.sum()
    x = X @ w
    return x.astype(np.float32)

def _zscore(x: np.ndarray) -> np.ndarray:
    m, s = float(x.mean()), float(x.std())
    return (x - m) / (s + 1e-8)

def _to_rgb3_from_1d(x: np.ndarray) -> np.ndarray:
    """
    1D → 3 通道 224x224：
      ch0: CWT(morl) → resize → 0-1
      ch1: STFT |Zxx| → resize → 0-1 （占位“Zerone通道”）
      ch2: 折叠 Context → resize → 0-1
    """
    H = W = int(cfg.INPUT_SIZE)

    # --- ch0: CWT ---
    scales = np.arange(1, 1 + min(128, cfg.SIGNAL_LEN // 16))
    cwtmatr, _ = pywt.cwt(x, scales, "morl", sampling_period=1.0 / float(cfg.FS))
    cwt_abs = np.abs(cwtmatr).astype(np.float32)
    c0 = cv2.resize(cwt_abs, (W, H))
    c0 = (c0 - c0.min()) / (c0.max() - c0.min() + 1e-8)

    # --- ch1: STFT ---
    f, t, Zxx = stft(x, fs=float(cfg.FS), nperseg=256, noverlap=128, boundary=None)
    mag = np.abs(Zxx).astype(np.float32)
    c1 = cv2.resize(mag, (W, H))
    c1 = (c1 - c1.min()) / (c1.max() - c1.min() + 1e-8)

    # --- ch2: Context（把 1D 折叠成 2D 再缩放）---
    h_fold = int(max(1, cfg.SIGNAL_LEN // 256))
    mat = x.reshape(h_fold, -1) if (h_fold * (cfg.SIGNAL_LEN // h_fold) == cfg.SIGNAL_LEN) else x.reshape(-1, 1)
    c2 = cv2.resize(mat.astype(np.float32), (W, H))
    c2 = (c2 - c2.min()) / (c2.max() - c2.min() + 1e-8)

    img = np.stack([c0, c1, c2], axis=0)  # [3, H, W]
    return img.astype(np.float32)

# ========= 数据集 =========
class TransformerVibrationDataset(Dataset):
    """
    Zerone式样本定义：一个“样本”=某文件中**同一 data_time 下的多传感器记录的聚合**。
    索引项是 (file_path, time_key)。训练/验证若 only_normal=True，则仅在“正常”目录下取样本。
    """
    def __init__(self, root_dir: Path | str, *, only_normal: bool, mode: str):
        self.root_dir = Path(root_dir)
        self.only_normal = bool(only_normal)
        self.mode = str(mode)
        self.index: List[Tuple[Path, str]] = []  # (file, time_key)
        self._build_index()

    def _build_index(self) -> None:
        if not self.root_dir.exists():
            print(f"[{self.mode}] 路径不存在：{self.root_dir}")
            return
        
        files = [p for p in self.root_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".jsonl", ".json")]

        # 过滤“正常/故障”目录 —— 只看直属父目录名，不看整条大路径
        keep_files: List[Path] = []
        normal_count = 0
        fault_count = 0
        for fp in files:
            parent_name = fp.parent.name.lower()
            is_normal_parent = any(k in parent_name for k in NORMAL_KEYS)
            is_fault_parent  = any(k in parent_name for k in FAULT_KEYS)

            if self.only_normal:
                # 训练/验证阶段：只保留“父目录名里带‘正常/normal’”的文件
                if not is_normal_parent:
                    continue

            keep_files.append(fp)
            if is_normal_parent:
                normal_count += 1
            if is_fault_parent:
                fault_count += 1

        # 按文件构建“时间戳组索引”
        total_groups = 0
        for fp in keep_files:
            recs = _read_all_records(fp)
            groups = _group_by_time(recs)  # {time_key: [sig_u]}
            g_keys = list(groups.keys())
            self.index.extend((fp, tk) for tk in g_keys)
            total_groups += len(g_keys)

        # 友好打印
        if self.mode.lower().startswith("train"):
            print(f"[train] 正常文件={normal_count} | 故障文件={fault_count} | 实际载入(正常过滤后)={len(keep_files)}")
        elif self.mode.lower().startswith("val"):
            print(f"[val]   正常文件={normal_count} | 故障文件={fault_count} | 实际载入(正常过滤后)={len(keep_files)}")
        else:
            print(f"[{self.mode}] 文件数={len(keep_files)}")

        print(f"[{self.mode}] 基于 data_time 的样本组数: {total_groups}\r\n")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> torch.Tensor:
        fp, tk = self.index[i]
        # 为安全起见，每次按 time_key 重新抽取该组通道
        recs = _read_all_records(fp)
        groups = _group_by_time(recs)
        sig_list = groups.get(tk, None)
        if not sig_list:
            # 兜底：返回全 0 图
            x = np.zeros((cfg.SIGNAL_LEN,), dtype=np.float32)
            img = _to_rgb3_from_1d(x)
            return torch.from_numpy(img)

        # 多通道能量加权 → zscore
        x = _energy_weighted_1d(sig_list)
        x = _zscore(x)
        img = _to_rgb3_from_1d(x)  # [3, H, W]
        return torch.from_numpy(img)
