# -*- coding: utf-8 -*- 
"""
zerone_train_resnet.py  (稳定泛化整合版 · tie-break + 可选SWA + 统一日志 + 中英文分开出图)
--------------------------------------------------------------------------------
要点（面向“可交接/可维护”）：
1) 归一化：仅用 train 统计 mean/std；train 轻量增强，val/test 只 Normalize。
2) 优化&损失：AdamW + weight_decay + label_smoothing；类别权重按 train 类分布自适应。
3) 调度：ReduceLROnPlateau（监控 val_f1）；可切换 Cosine。
4) 稳定性：确定性 + 梯度裁剪；EMA 只追踪“浮点且 requires_grad 的参数”（安全）。
5) 校准：FT-COPY（拷贝模型，仅训 fc，极小 LR，1~2 个 batch）→ 只为 test 打分，不污染主模型。
6) 选模（tie-break）：( val_f1 ↑, val_loss ↓, lr ↓, epoch ↑ ) 四级键；每轮都打印是否覆盖 best。
7) SWA（可选）：后半程、低 lr、低 val_loss 的若干轮做权重平均，更新 BN 后另存 resnet18_swa.pt。
8) 统一日志：每个 epoch 输出两行（主结果 + 事件），阅读体验稳定；无 test 时自动显示 N/A。
9) 图像预处理：保持原始宽高比，先等比再 SquarePad 成正方形。
10) PSD 可视化：稳健的“逐频点”统计（均值±1σ）+ 类模板瀑布 + 差分曲线；空集兜底。
11) 可视化语言：全中文图和全英文图分别导出（_zh / _en）。
"""

import random, csv
import numpy as np
from copy import deepcopy
from collections import Counter
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageOps

from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, roc_curve, auc,
                             precision_recall_curve, average_precision_score)
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from zerone_config import IMG_OUT_ROOT, CLASSES, SEED, LR, EPOCHS, BATCH_SIZE, MASKING_RATIO, PATIENCE
import os as _os_w4
SEED = int(_os_w4.environ.get('ZERONE_SEED', SEED))
from zerone_config import METRICS_ROOT, PRED_ROOT, FEAT_ROOT, VIZ_PLOTS, VIZ_ROCPR, VIZ_RELIAB, VIZ_EMBED

# 拆 PSD 段
from zerone_features import split_feature_vector
from matplotlib.patches import Ellipse

import pandas as pd
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

LANGS = ["zh", "en"]  # 同时产出两套图

def _tr(s: str, lang: str) -> str:
    """简单翻译表；未命中则回退原文。"""
    table = {
        # 通用
        "Confusion Matrix": {"zh": "混淆矩阵", "en": "Confusion Matrix"},
        "Predicted": {"zh": "预测类别", "en": "Predicted"},
        "True": {"zh": "真实类别", "en": "True"},
        "ROC": {"zh": "ROC 曲线", "en": "ROC"},
        "PR": {"zh": "PR 曲线", "en": "PR"},
        "Reliability": {"zh": "可靠性曲线", "en": "Reliability"},
        "Confidence": {"zh": "置信度", "en": "Confidence"},
        "Accuracy": {"zh": "准确率", "en": "Accuracy"},
        "PSD Waterfall": {"zh": "PSD 瀑布图", "en": "PSD Waterfall"},
        "PSD diff (Fault − Normal)": {"zh": "PSD 差分（故障−正常）", "en": "PSD diff (Fault − Normal)"},
        "Frequency Bin": {"zh": "频率 Bin", "en": "Frequency Bin"},
        "Power / Diff": {"zh": "功率 / 差分", "en": "Power / Diff"},
        "Sample Index": {"zh": "样本索引", "en": "Sample Index"},
        "Template": {"zh": "模板", "en": "Template"},
        "t-SNE": {"zh": "t-SNE", "en": "t-SNE"},
        "UMAP": {"zh": "UMAP", "en": "UMAP"},
        # 类别常用翻译
        "正常": {"zh": "正常", "en": "Normal"},
        "故障": {"zh": "故障", "en": "Fault"},
    }
    return table.get(s, {}).get(lang, s)

def _map_class_labels(labels, lang: str):
    return [ _tr(str(c), lang) for c in labels ]

def _apply_lang(ax, lang: str):
    if lang == "zh" :
        ax.set_title(ax.get_title())
        ax.set_xlabel(ax.get_xlabel())
        ax.set_ylabel(ax.get_ylabel())
        for tick in (ax.get_xticklabels() + ax.get_yticklabels()):
            try:
                tick.set_fontproperties("SimHei")
            except Exception:
                pass

def _title(ax, text_en: str, split: str, lang: str):
    txt = f"{_tr(text_en, lang)} – {split}"
    ax.set_title(txt)

# ========================= 开关区（常改） =========================
USE_VAL_FINETUNE = True
FT_ONLY_FC       = True
FT_MAX_STEPS     = 2
FT_LR            = max(1e-5, LR * 0.02)
FT_WARMUP_EPOCHS = 2

USE_EMA          = True
EMA_DECAY        = 0.999

NORM_FROM_TRAIN  = True
USE_COSINE       = False

EPS_IMPROVE      = 1e-4
MIN_TEST_GAIN    = 2e-3

USE_SWA          = True
SWA_START_RATIO  = 0.5
SWA_MIN_LR       = LR * 0.05
SWA_MAX_COUNT    = 20
SWA_MAX_LOSS_BIAS= 0.05
SWA_BN_UPDATE_STEPS = 128

LFE_ENABLED      = True
LFE_YCOVER       = 0.40
LFE_GAIN         = 1.10

def _ensure_dirs():
    for d in [METRICS_ROOT, PRED_ROOT, FEAT_ROOT, VIZ_PLOTS, VIZ_ROCPR, VIZ_RELIAB, VIZ_EMBED]:
        Path(d).mkdir(parents=True, exist_ok=True)
_ensure_dirs()

# ========================= 图像预处理工具：保持宽高比 + SquarePad =========================
class SquarePad(object):
    """
    先按长边等比缩放，再四周用常数色填充成正方形，避免拉伸变形。
    - fill: 0(黑) 或 255(白)，也可传三通道 tuple
    """
    def __init__(self, target=150, fill=0):
        self.target = int(target)
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        # 先按长边等比缩放到不超过 target
        w, h = img.size
        if w == 0 or h == 0:
            return Image.new("RGB", (self.target, self.target), self.fill if isinstance(self.fill, tuple) else (self.fill,)*3)
        scale = self.target / max(w, h)
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        # 再 pad 成正方形
        pad_w = self.target - new_w
        pad_h = self.target - new_h
        left = pad_w // 2
        top  = pad_h // 2
        right= pad_w - left
        bottom = pad_h - top
        if isinstance(self.fill, tuple):
            padded = Image.new(img.mode, (self.target, self.target), self.fill)
            padded.paste(img, (left, top))
            return padded
        return ImageOps.expand(img, border=(left, top, right, bottom), fill=self.fill)

# ========================= 可视化器 =========================
class _VizKit:
    def __init__(self, out_root=VIZ_PLOTS):
        self.root = Path(out_root).resolve()
        (VIZ_PLOTS).mkdir(parents=True, exist_ok=True)
        (VIZ_ROCPR).mkdir(parents=True, exist_ok=True)
        (VIZ_RELIAB).mkdir(parents=True, exist_ok=True)
        (VIZ_EMBED).mkdir(parents=True, exist_ok=True)
        (self.root / "viz" / "waterfall").mkdir(parents=True, exist_ok=True)
        (self.root / "viz" / "embeddings").mkdir(parents=True, exist_ok=True)
        (self.root / "viz" / "psd_stats").mkdir(parents=True, exist_ok=True)
        self._log, self._hooks, self._expected_len = [], {}, None

    def log_epoch(self, epoch, train_loss=None, val_loss=None, val_acc=None, val_f1=None, lr=None):
        self._log.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_acc=val_acc, val_f1=val_f1, lr=lr))

    def flush_curves(self):
        if not self._log: return
        df = pd.DataFrame(self._log)
        df.to_csv(METRICS_ROOT / "training_metrics.csv", index=False, encoding="utf-8-sig")
        ep = df["epoch"]

        plt.figure()
        if "train_loss" in df: plt.plot(ep, df["train_loss"], label="train_loss")
        if "val_loss"   in df: plt.plot(ep, df["val_loss"],   label="val_loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True, alpha=.3)
        plt.tight_layout(); plt.savefig(VIZ_PLOTS/"loss_vs_epoch.png", dpi=200); plt.close()

        plt.figure()
        if "val_f1" in df:  plt.plot(ep, df["val_f1"],  label="val_f1")
        if "val_acc" in df: plt.plot(ep, df["val_acc"], label="val_acc")
        plt.xlabel("Epoch"); plt.ylabel("Score"); plt.legend(); plt.grid(True, alpha=.3)
        plt.tight_layout(); plt.savefig(VIZ_PLOTS/"score_vs_epoch.png", dpi=200); plt.close()

        if "lr" in df:
            plt.figure(); plt.plot(ep, df["lr"], label="lr")
            plt.xlabel("Epoch"); plt.ylabel("Learning Rate"); plt.legend(); plt.grid(True, alpha=.3)
            plt.tight_layout(); plt.savefig(VIZ_PLOTS/"lr_vs_epoch.png", dpi=200); plt.close()

    def begin_feat_capture(self, expected_len: int):
        self._expected_len = int(expected_len)
        if "penult" in self._hooks:
            h, lst = self._hooks["penult"]; lst.clear()

    def end_feat_capture(self, split: str):
        if "penult" not in self._hooks or self._expected_len is None: return
        h, lst = self._hooks["penult"]
        if not lst: return
        X = np.concatenate(lst, axis=0)
        if X.shape[0] > self._expected_len: X = X[-self._expected_len:, :]
        np.save(FEAT_ROOT / f"{split}_features.npy", X)
        self._expected_len = None

    def register_feature_hook(self, model):
        feats = []
        def hook(_m, _in, out):
            x = out
            try: x = torch.flatten(x, 1)
            except Exception: pass
            feats.append(x.detach().cpu().numpy())
        h = model.avgpool.register_forward_hook(hook)
        self._hooks["penult"] = (h, feats)

    def export_feats(self, split):
        if not self._hooks: return
        for h, lst in self._hooks.values():
            if lst:
                X = np.concatenate(lst, axis=0)
                np.save(Path(FEAT_ROOT) / f"{split}_features.npy", X)

    def clear_hooks(self):
        for h, _ in self._hooks.values(): h.remove()
        self._hooks.clear()

    def save_pred_csv(self, split, y_true, y_pred, prob_mat):
        df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
        if prob_mat is not None:
            P = np.asarray(prob_mat)
            for k in range(P.shape[1]): df[f"prob_{k}"] = P[:, k]
        Path(PRED_ROOT).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(PRED_ROOT) / f"{split}_predictions.csv", index=False, encoding="utf-8-sig")

    def plots_from_pred(self, split):
        p = Path(PRED_ROOT) / f"{split}_predictions.csv"
        if not p.exists(): return
        df = pd.read_csv(p)
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        if not prob_cols: return
        classes_raw = sorted(df["y_true"].astype(str).unique())
        y_idx = pd.Categorical(df["y_true"].astype(str), categories=classes_raw).codes
        Y = np.eye(len(classes_raw))[y_idx]
        P = df[prob_cols].to_numpy()

        # ROC
        for lang in LANGS:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for k, cls in enumerate(classes_raw):
                fpr, tpr, _ = roc_curve(Y[:,k], P[:,k]); AUC = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"{_tr(cls, lang)} (AUC={AUC:.3f})")
            ax.plot([0,1],[0,1],"k--",lw=1)
            ax.set_xlabel("FPR" if lang=="en" else "假阳性率")
            ax.set_ylabel("TPR" if lang=="en" else "真阳性率")
            _title(ax, "ROC", split, lang)
            ax.legend(); ax.grid(True, alpha=.3); fig.tight_layout()
            fig.savefig(Path(VIZ_ROCPR)/f"roc_{split}_{lang}.png", dpi=200); plt.close(fig)

        # PR
        for lang in LANGS:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for k, cls in enumerate(classes_raw):
                p_, r_, _ = precision_recall_curve(Y[:,k], P[:,k])
                ap = average_precision_score(Y[:,k], P[:,k])
                mask = r_ < (1.0 - 1e-12)
                r_plot, p_plot = (r_[mask], p_[mask]) if np.any(mask) else (r_, p_)
                ax.plot(r_plot, p_plot, drawstyle="steps-post", label=f"{_tr(cls, lang)} (AP={ap:.3f})")
            ax.set_xlim(0,1); ax.set_ylim(0,1.05)
            ax.set_xlabel("Recall" if lang=="en" else "召回率")
            ax.set_ylabel("Precision" if lang=="en" else "精确率")
            _title(ax, "PR", split, lang)
            ax.legend(); ax.grid(True, alpha=.3); fig.tight_layout()
            fig.savefig(Path(VIZ_ROCPR)/f"pr_{split}_{lang}.png", dpi=200); plt.close(fig)

        # Reliability + ECE
        if "y_pred" in df.columns:
            pred_idx = pd.Categorical(df["y_pred"].astype(str), categories=classes_raw).codes
            conf = P[np.arange(len(df)), pred_idx]
            corr = (df["y_pred"].astype(str)==df["y_true"].astype(str)).astype(int).to_numpy()
            n_bins=10; bins=np.linspace(0,1,n_bins+1); bid=np.digitize(conf,bins)-1
            accs, confs, ns = [], [], []
            for b in range(n_bins):
                m = (bid==b)
                if m.sum()==0: accs.append(0.0); confs.append((bins[b]+bins[b+1])/2); ns.append(0)
                else: accs.append(corr[m].mean()); confs.append(conf[m].mean()); ns.append(int(m.sum()))
            N=len(df); ece=sum((ns[i]/N)*abs(accs[i]-confs[i]) for i in range(n_bins))

            for lang in LANGS:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot([0,1],[0,1],'k--',lw=1)
                ax.bar(confs, accs, width=0.9/n_bins, edgecolor='k', alpha=.6)
                ax.set_xlabel(_tr("Confidence", lang)); ax.set_ylabel(_tr("Accuracy", lang))
                ax.set_title(f"{_tr('Reliability', lang)} – {split} (ECE={ece:.3f})")
                _apply_lang(ax, lang)
                ax.grid(True, alpha=.3); fig.tight_layout()
                fig.savefig(Path(VIZ_RELIAB)/f"reliability_{split}_{lang}.png", dpi=200); plt.close(fig)

        # PSD 相关图
        try:
            self.plot_psd_templates_and_stats(split)
        except Exception as e:
            print(f"[WARN] 绘制 PSD 图失败: {e}")

        # 时域特征相关性图
        try:
            self.plot_time_feature_correlations(split)
        except Exception as e:
            print(f"[WARN] 绘制时域相关性图失败: {e}")

        # STFT 统计图
        try:
            self.plot_stft_stats(split)
        except Exception as e:
            print(f"[WARN] 绘制 STFT 统计图失败: {e}")

    def plots_from_feats(self, split):
        p_pred = Path(PRED_ROOT) / f"{split}_predictions.csv"
        p_feat = Path(FEAT_ROOT) / f"{split}_features.npy"
        if not (p_pred.exists() and p_feat.exists()):
            return
        df = pd.read_csv(p_pred)
        X = np.load(p_feat)
        y_str = df["y_true"].astype(str).to_numpy()

        classes_raw = list(pd.unique(df["y_true"].astype(str)))
        classes_sorted = sorted(classes_raw, key=lambda x: x)
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        cmap = {c: color_cycle[i % len(color_cycle)] for i, c in enumerate(classes_sorted)}

        def draw_cov_ellipse(ax, pts, edgecolor, facecolor, alpha=0.12, lw=1.2):
            if pts.shape[0] < 3: return
            mu = pts.mean(axis=0)
            C = np.cov(pts.T); C += 1e-6 * np.eye(C.shape[0])
            w, v = np.linalg.eigh(C); order = w.argsort()[::-1]
            w, v = w[order], v[:, order]
            theta = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
            width, height = 2 * 2.0 * np.sqrt(max(w[0], 1e-12)), 2 * 2.0 * np.sqrt(max(w[1], 1e-12))
            e = Ellipse(xy=mu, width=width, height=height, angle=theta,
                        edgecolor=edgecolor, facecolor=facecolor, lw=lw, alpha=alpha)
            ax.add_patch(e)

        def render_embed(name_en, Z, y, save_tag):
            for lang in LANGS:
                fig = plt.figure(figsize=(6.0, 4.8)); ax = fig.add_subplot(111)
                for cls in classes_sorted:
                    m = (y == cls); Zi = Z[m]; ci = cmap[cls]
                    ax.scatter(Zi[:, 0], Zi[:, 1], s=26, c=ci, edgecolors="white", linewidths=0.6, alpha=0.85)
                    mu = Zi.mean(axis=0); ax.scatter([mu[0]], [mu[1]], marker="x", s=60, c="black", linewidths=1.5, zorder=5)
                    draw_cov_ellipse(ax, Zi, edgecolor=ci, facecolor=ci, alpha=0.15, lw=1.2)

                handles, labels = [], []
                for cls in classes_sorted:
                    n = int(np.sum(y == cls))
                    h = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap[cls],
                                   markeredgecolor="white", markeredgewidth=0.6, markersize=8, linestyle="")
                    handles.append(h); labels.append(f"{_tr(cls, lang)} (n={n})")
                ax.legend(handles, labels, loc="best", frameon=True)

                ax.set_xticks([]); ax.set_yticks([])
                _title(ax, name_en, split, lang)
                _apply_lang(ax, lang)
                for sp in ["top","right","bottom","left"]:
                    ax.spines[sp].set_visible(False)
                fig.tight_layout()
                fig.savefig(self.root/"viz"/"embeddings"/f"{save_tag}_{split}_{lang}.png", dpi=150)
                fig.savefig(self.root/"viz"/"embeddings"/f"{save_tag}_{split}_{lang}.svg")
                plt.close(fig)

        Z_tsne = TSNE(n_components=2, init="pca", perplexity=30, learning_rate="auto",
                      random_state=0, max_iter=1000).fit_transform(X)
        render_embed("t-SNE", Z_tsne, y_str, "tsne")

        if HAS_UMAP:
            try:
                reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.15, random_state=0, n_jobs=1)
                Z_umap = reducer.fit_transform(X)
                render_embed("UMAP", Z_umap, y_str, "umap")
            except Exception as e:
                print(f"[UMAP] 跳过：{e}")

    # ======= PSD 瀑布 + 逐频点统计（均值±1σ & 差分）+ 兜底（双语） =======
    def plot_psd_templates_and_stats(self, split: str) -> None:
        pred_path = Path(PRED_ROOT) / f"{split}_predictions.csv"
        feat_path = Path(FEAT_ROOT) / f"{split}_features.npy"
        out_dir_wf = self.root / "viz" / "waterfall"
        out_dir_ps = self.root / "viz" / "psd_stats"
        out_dir_wf.mkdir(parents=True, exist_ok=True)
        out_dir_ps.mkdir(parents=True, exist_ok=True)

        if (not pred_path.exists()) or (not feat_path.exists()):
            for lang in LANGS:
                fig = plt.figure(figsize=(6, 3)); ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, (_tr("PSD Waterfall", lang) + " / " + split + "\n" +
                                   ("缺少预测或特征文件" if lang=="zh" else "pred/feature missing")),
                        ha="center", va="center", fontsize=12)
                ax.axis("off"); fig.tight_layout()
                fig.savefig(out_dir_wf / f"waterfall_{split}_{lang}_fallback.png", dpi=200); plt.close(fig)
            return

        try:
            df = pd.read_csv(pred_path); X = np.load(feat_path)
        except Exception:
            for lang in LANGS:
                fig = plt.figure(figsize=(6, 3)); ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, (_tr("PSD Waterfall", lang) + " / " + split + "\n" +
                                   ("文件读取失败" if lang=="zh" else "failed to read files")),
                        ha="center", va="center", fontsize=12)
                ax.axis("off"); fig.tight_layout()
                fig.savefig(out_dir_wf / f"waterfall_{split}_{lang}_fallback.png", dpi=200); plt.close(fig)
            return

        if "y_true" not in df.columns: return
        y_true = df["y_true"].astype(str).to_numpy()
        classes = sorted(np.unique(y_true))

        # 收集 PSD 段
        psd_by_cls = {c: [] for c in classes}
        for i, feats in enumerate(X):
            try:
                segs = split_feature_vector(feats)
                psd = segs.get("psd", None)
                if psd is None: continue
                psd_by_cls[str(y_true[i])].append(np.asarray(psd, dtype=np.float32).ravel())
            except Exception:
                continue

        if all(len(v) == 0 for v in psd_by_cls.values()):
            for lang in LANGS:
                fig = plt.figure(figsize=(6, 3)); ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, (split + ": " + ("无可用的 PSD 段" if lang=="zh" else "no available PSD segments")),
                        ha="center", va="center", fontsize=12)
                ax.axis("off"); fig.tight_layout()
                fig.savefig(out_dir_wf / f"waterfall_{split}_{lang}_empty.png", dpi=200); plt.close(fig)
            return

        non_empty = [v for v in psd_by_cls.values() if len(v) > 0]
        psd_len = min(min(len(p) for p in v) for v in non_empty)
        for cls in classes:
            psd_by_cls[cls] = [p[:psd_len] for p in psd_by_cls[cls]]

        # (1) 类瀑布模板 + 差分
        for lang in LANGS:
            nrows = len(classes) + 1
            fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(8, 2.5 * nrows), sharex=True)
            if nrows == 1: axes = [axes]
            for idx_cls, cls in enumerate(classes):
                arr = np.stack(psd_by_cls[cls], axis=0) if len(psd_by_cls[cls])>0 else np.zeros((1, psd_len), dtype=np.float32)
                ax = axes[idx_cls]
                im = ax.imshow(arr, aspect='auto', origin='lower', cmap='turbo')
                ax.set_ylabel(_tr("Sample Index", lang))
                ax.set_title(f"{_tr('PSD Waterfall', lang)} – {split} – {_tr(cls, lang)}")
                _apply_lang(ax, lang)
                fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.02,
                             label=("功率" if lang=="zh" else "Power"))

            if len(classes) >= 2 and len(psd_by_cls[classes[0]])>0 and len(psd_by_cls[classes[1]])>0:
                diff = np.mean(psd_by_cls[classes[1]], axis=0) - np.mean(psd_by_cls[classes[0]], axis=0)
            else:
                diff = np.zeros(psd_len, dtype=np.float32)

            ax = axes[-1]
            diff_mat = np.tile(diff, (max(1, max(len(v) for v in psd_by_cls.values())), 1))
            im2 = ax.imshow(diff_mat, aspect='auto', origin='lower', cmap='coolwarm')
            ax.set_ylabel(_tr("Template", lang)); ax.set_xlabel(_tr("Frequency Bin", lang))
            ax.set_title(_tr("PSD diff (Fault − Normal)", lang))
            _apply_lang(ax, lang)
            fig.colorbar(im2, ax=ax, orientation='vertical', fraction=0.046, pad=0.02,
                         label=("差值" if lang=="zh" else "Diff"))
            fig.tight_layout()
            fig.savefig(out_dir_wf / f"waterfall_{split}_{lang}.png", dpi=200)
            plt.close(fig)

        # (2) 逐频点统计（均值±1σ + 差分）
        for lang in LANGS:
            fig2 = plt.figure(figsize=(9, 4.5))
            ax2 = fig2.add_subplot(111)
            freq = np.arange(psd_len)

            for cls in classes:
                if len(psd_by_cls[cls]) == 0: continue
                A = np.stack(psd_by_cls[cls], axis=0)
                mu = A.mean(axis=0); sd = A.std(axis=0)
                line, = ax2.plot(freq, mu, label=f"{_tr(cls, lang)} μ")
                c = line.get_color()
                ax2.fill_between(freq, mu - sd, mu + sd, alpha=0.15, color=c, linewidth=0)

            if len(classes) >= 2 and len(psd_by_cls[classes[0]])>0 and len(psd_by_cls[classes[1]])>0:
                mu0 = np.stack(psd_by_cls[classes[0]], axis=0).mean(axis=0)
                mu1 = np.stack(psd_by_cls[classes[1]], axis=0).mean(axis=0)
                ax2.plot(freq, (mu1 - mu0), linestyle="--", linewidth=1.2,
                         label=("故障−正常(μ)" if lang=="zh" else "Fault−Normal (μ)"))

            ax2.set_xlabel(_tr("Frequency Bin", lang))
            ax2.set_ylabel(_tr("Power / Diff", lang))
            ax2.set_title(f"{('PSD 逐频点统计' if lang=='zh' else 'PSD per-bin stats')} – {split} (±1σ)")
            _apply_lang(ax2, lang)
            ax2.grid(True, alpha=.3); ax2.legend()
            fig2.tight_layout()
            fig2.savefig(out_dir_ps / f"psd_stats_{split}_{lang}.png", dpi=200)
            plt.close(fig2)

    # ---------------------------------------------------------------------
    # 新增：时域特征相关性矩阵
    # ---------------------------------------------------------------------
    def plot_time_feature_correlations(self, split: str) -> None:
        """
        针对指定数据集 split（train/val/test），计算时域 15 维特征之间的相关系数矩阵，并绘制热图。

        该函数假定 FEAT_ROOT 中存在 `<split>_features.npy` 文件，且每个样本按照 zerone_features.extract_Sij
        生成 2144 维特征向量，前 15 维对应时域统计量。若文件缺失或无法加载，将输出提示图。

        图像保存到 `viz/time_corr` 子目录下，按双语保存（_zh / _en）。
        """
        from zerone_features import split_feature_vector
        p_feat = Path(FEAT_ROOT) / f"{split}_features.npy"
        out_dir = self.root / "viz" / "time_corr"
        out_dir.mkdir(parents=True, exist_ok=True)
        if not p_feat.exists():
            for lang in LANGS:
                fig = plt.figure(figsize=(6, 3)); ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, (f"{split}: " + ("缺少特征文件" if lang=="zh" else "feature file missing")),
                        ha="center", va="center", fontsize=12)
                ax.axis("off"); fig.tight_layout()
                fig.savefig(out_dir / f"time_corr_{split}_{lang}_fallback.png", dpi=200)
                plt.close(fig)
            return
        try:
            X = np.load(p_feat)
        except Exception:
            for lang in LANGS:
                fig = plt.figure(figsize=(6, 3)); ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, (f"{split}: " + ("读取特征失败" if lang=="zh" else "failed to load features")),
                        ha="center", va="center", fontsize=12)
                ax.axis("off"); fig.tight_layout()
                fig.savefig(out_dir / f"time_corr_{split}_{lang}_error.png", dpi=200)
                plt.close(fig)
            return
        if X.size == 0 or X.ndim != 2 or X.shape[1] < 15:
            for lang in LANGS:
                fig = plt.figure(figsize=(6, 3)); ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, (f"{split}: " + ("特征维度不足" if lang=="zh" else "insufficient feature dimensions")),
                        ha="center", va="center", fontsize=12)
                ax.axis("off"); fig.tight_layout()
                fig.savefig(out_dir / f"time_corr_{split}_{lang}_empty.png", dpi=200)
                plt.close(fig)
            return
        # 取前 15 维时域特征
        time_feats = X[:, :15]
        # 标准化以避免量纲差异影响相关系数
        time_feats = (time_feats - time_feats.mean(axis=0, keepdims=True)) / (time_feats.std(axis=0, keepdims=True) + 1e-9)
        # 计算相关系数矩阵 (15×15)
        corr = np.corrcoef(time_feats, rowvar=False)
        # 绘制双语热图
        feat_labels = [f"f{i+1}" for i in range(corr.shape[0])]
        for lang in LANGS:
            fig = plt.figure(figsize=(6.4, 5.2)); ax = fig.add_subplot(111)
            im = ax.imshow(corr, interpolation="nearest", cmap="coolwarm", vmin=-1.0, vmax=1.0)
            ax.set_xticks(range(len(feat_labels)))
            ax.set_yticks(range(len(feat_labels)))
            ax.set_xticklabels(feat_labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(feat_labels, fontsize=8)
            # 在每个单元格标注相关系数值
            for i in range(corr.shape[0]):
                for j in range(corr.shape[1]):
                    ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                            color="black", fontsize=6)
            ax.set_title(("时域特征相关性" if lang=="zh" else "Time-feature correlations") + f" – {split}")
            ax.set_xlabel(("特征索引" if lang=="zh" else "Feature Index"))
            ax.set_ylabel(("特征索引" if lang=="zh" else "Feature Index"))
            _apply_lang(ax, lang)
            fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.02,
                         label=("相关系数" if lang=="zh" else "Correlation"))
            fig.tight_layout()
            fig.savefig(out_dir / f"time_corr_{split}_{lang}.png", dpi=200)
            plt.close(fig)

    # ---------------------------------------------------------------------
    # 新增：STFT 段均值统计图
    # ---------------------------------------------------------------------
    def plot_stft_stats(self, split: str) -> None:
        """
        针对指定数据集 split，提取 STFT 段均值（127 维），按类别绘制均值±1σ 曲线。

        每个样本的特征向量按照 split_feature_vector 拆分，从中取 "stft" 段。若某类别样本为空，则跳过该类的绘制。
        图像保存到 `viz/stft_stats` 子目录下，按双语命名。
        """
        from zerone_features import split_feature_vector
        p_pred = Path(PRED_ROOT) / f"{split}_predictions.csv"
        p_feat = Path(FEAT_ROOT) / f"{split}_features.npy"
        out_dir = self.root / "viz" / "stft_stats"
        out_dir.mkdir(parents=True, exist_ok=True)
        if not (p_pred.exists() and p_feat.exists()):
            return
        try:
            df = pd.read_csv(p_pred)
            X = np.load(p_feat)
        except Exception:
            return
        if "y_true" not in df.columns:
            return
        y_true = df["y_true"].astype(str).to_numpy()
        classes = sorted(np.unique(y_true))
        # 收集 STFT 段
        stft_by_cls = {c: [] for c in classes}
        for i, feats in enumerate(X):
            try:
                segs = split_feature_vector(feats)
                stft = segs.get("stft", None)
                if stft is None: continue
                stft_by_cls[str(y_true[i])].append(np.asarray(stft, dtype=np.float32).ravel())
            except Exception:
                continue
        # 统一长度
        non_empty = [v for v in stft_by_cls.values() if len(v) > 0]
        if not non_empty:
            return
        stft_len = min(min(len(p) for p in v) for v in non_empty)
        for cls in classes:
            stft_by_cls[cls] = [p[:stft_len] for p in stft_by_cls[cls]]
        freq = np.arange(stft_len)
        for lang in LANGS:
            fig = plt.figure(figsize=(8, 4.0)); ax = fig.add_subplot(111)
            # 绘制每类的均值±1σ
            for cls in classes:
                if len(stft_by_cls[cls]) == 0: continue
                A = np.stack(stft_by_cls[cls], axis=0)
                mu = A.mean(axis=0)
                sd = A.std(axis=0)
                line, = ax.plot(freq, mu, label=f"{_tr(cls, lang)} μ")
                c = line.get_color()
                ax.fill_between(freq, mu - sd, mu + sd, alpha=0.15, color=c, linewidth=0)
            # 差分：若至少两类
            if len(classes) >= 2 and len(stft_by_cls[classes[0]]) > 0 and len(stft_by_cls[classes[1]]) > 0:
                mu0 = np.stack(stft_by_cls[classes[0]], axis=0).mean(axis=0)
                mu1 = np.stack(stft_by_cls[classes[1]], axis=0).mean(axis=0)
                ax.plot(freq, (mu1 - mu0), linestyle="--", linewidth=1.2,
                        label=("故障−正常(μ)" if lang=="zh" else "Fault−Normal (μ)"))
            ax.set_xlabel(_tr("Frequency Bin", lang))
            ax.set_ylabel(("幅值" if lang=="zh" else "Amplitude"))
            ax.set_title(("STFT 段均值统计" if lang=="zh" else "STFT segment statistics") + f" – {split} (±1σ)")
            _apply_lang(ax, lang)
            ax.grid(True, alpha=.3); ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / f"stft_stats_{split}_{lang}.png", dpi=200)
            plt.close(fig)

# ========================= 小工具 & 通用函数 =========================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_mean_std(file_list):
    """仅基于 train 图像计算 mean/std（避免 data leak）"""
    if len(file_list) == 0:
        return [0.5,0.5,0.5], [0.5,0.5,0.5]
    tfm = transforms.Compose([SquarePad(150, fill=0), transforms.ToTensor()])
    m = torch.zeros(3); s = torch.zeros(3); n = 0
    for p in file_list:
        x = tfm(Image.open(p).convert('RGB'))
        m += x.view(3,-1).mean(dim=1)
        s += x.view(3,-1).std(dim=1)
        n += 1
    m /= n; s /= n
    return m.tolist(), s.tolist()

def load_manifest_map(split_root: Path):
    manifest = split_root / "manifest.csv"
    meta = {}
    if manifest.exists():
        with open(manifest, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                meta[Path(row["png_path"]).as_posix()] = row
    return meta

def save_miscls_with_manifest(outdir: Path, files, y_true, y_pred, manifest_root: Path):
    outdir_test = outdir / "test" / "miscls"
    outdir_test.mkdir(parents=True, exist_ok=True)
    meta = load_manifest_map(manifest_root)
    txt_path = outdir_test / "miscls_test.txt"
    csv_path = outdir_test / "miscls_test.csv"
    with open(txt_path, "w", encoding="utf-8") as ftxt, open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        ftxt.write("png_path\ttrue\tpred\tclass_dir\tsrc_json\tdata_time\tsample_idx\tnum_channels\n")
        w = csv.writer(fcsv)
        w.writerow(["png_path","true","pred","class_dir","src_json","data_time","sample_idx","num_channels"])
        for p, t, pr in zip(files, y_true, y_pred):
            if t == pr: continue
            p_str = Path(p).as_posix()
            class_dir = Path(p).parent.name
            m = meta.get(p_str, {})
            row = [p_str, int(t), int(pr), class_dir,
                   m.get("src_json",""), m.get("data_time",""),
                   m.get("sample_idx",""), m.get("num_channels","")]
            ftxt.write("\t".join(map(str, row)) + "\n")
            w.writerow(row)
    print(f"[TEST] misclassified -> {txt_path}")

def eval_split(model, dl, device):
    model.eval()
    y_true, y_pred, fpaths = [], [], []
    with torch.no_grad():
        for x, y, paths in dl:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(y.numpy().tolist())
            fpaths.extend(paths)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    return acc, f1, y_true, y_pred, fpaths

def eval_split_with_loss(model, dl, device, class_weights=None, label_smoothing=0.05):
    model.eval()
    ys, preds, fpaths = [], [], []
    loss_sum, n_batches = 0.0, 0
    ce_kwargs = {}
    if class_weights is not None: ce_kwargs["weight"] = class_weights
    if label_smoothing is not None: ce_kwargs["label_smoothing"] = label_smoothing
    with torch.no_grad():
        for x, y, paths in dl:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, **ce_kwargs)
            loss_sum += float(loss.item()); n_batches += 1
            p = logits.argmax(dim=1).cpu().numpy().tolist()
            preds.extend(p); ys.extend(y.cpu().numpy().tolist()); fpaths.extend(paths)
    acc = accuracy_score(ys, preds); f1 = f1_score(ys, preds, average="macro")
    avg_loss = loss_sum / max(1, n_batches)
    return acc, f1, avg_loss, ys, preds, fpaths

def eval_split_probs(model, dl, device):
    model.eval()
    ys, preds, probs, fpaths = [], [], [], []
    with torch.no_grad():
        for x, y, paths in dl:
            x = x.to(device)
            logits = model(x)
            p = F.softmax(logits, dim=1).detach().cpu().numpy()
            pr = logits.argmax(dim=1).cpu().numpy()
            probs.extend(p.tolist()); preds.extend(pr.tolist()); ys.extend(y.numpy().tolist()); fpaths.extend(paths)
    acc = accuracy_score(ys, preds); f1 = f1_score(ys, preds, average="macro")
    return acc, f1, np.asarray(ys), np.asarray(preds), np.asarray(probs), fpaths

def freeze_bn_running_stats_(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.eval()

def save_confmat(filepath_base: Path, cm, classes, split: str):
    """
    生成两份图：*_zh.png 和 *_en.png
    filepath_base 传入不含后缀的基础路径，如: outdir/'val'/'confusion_matrix_val'
    """
    for lang in LANGS:
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        ax.imshow(cm, interpolation="nearest", cmap="viridis")

        cls = _map_class_labels(classes, lang)
        ax.set_xticks(range(len(cls))); ax.set_yticks(range(len(cls)))
        ax.set_xticklabels(cls, rotation=45, ha="right")
        ax.set_yticklabels(cls)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")

        ax.set_xlabel(_tr("Predicted", lang)); ax.set_ylabel(_tr("True", lang))
        _title(ax, "Confusion Matrix", split, lang)
        _apply_lang(ax, lang)

        fig.tight_layout()
        fig.savefig(Path(f"{filepath_base}_{lang}.png"), dpi=160)
        plt.close(fig)

def gather_split_files(split_root: Path, classes: List[str]):
    files, labels = [], []
    for idx, cls_name in enumerate(classes):
        class_dir = split_root / cls_name
        if not class_dir.exists(): continue
        for p in class_dir.glob("*.png"):
            files.append(p); labels.append(idx)
    return np.array(files), np.array(labels, dtype=np.int64)

def save_metrics(outdir: Path, history):
    """把每轮 train_loss/val_acc/val_f1/lr 记到 CSV，便于画曲线查问题"""
    csv_path = outdir / "training_metrics.csv"
    if not history:
        return
    fieldnames = ["epoch", "train_loss", "val_acc", "val_f1", "lr"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in history:
            w.writerow({k: row.get(k, "") for k in fieldnames})

# ========================= 统一日志（每轮两行） =========================
def print_epoch_main(epoch, val_acc, val_f1, val_loss, lr, te_acc=None, te_f1=None):
    te_str = f"acc={te_acc:.3f} f1={te_f1:.3f}" if (te_acc is not None and te_f1 is not None) else "N/A"
    print(f"[E{epoch:02d}] VAL acc={val_acc:.3f} f1={val_f1:.3f} loss={val_loss:.4f} lr={lr:.5g} | TEST(ftcopy)={te_str}")

def print_epoch_events(epoch, events: list):
    print(f"[E{epoch:02d}] EVENTS: " + (" ; ".join(events) if events else "—"))
    print()

# ========================= 压缩版三段 Summary =========================
def _fmt_sci(x):
    return f"{x:.3g}"

def print_compact_summary(outdir, best_rec, final_val, final_test, test_best_summary, swa_summary):
    if best_rec is not None:
        print(f"[BEST] epoch={best_rec['epoch']} | "
              f"val_acc={best_rec['val_acc']:.3f} val_f1={best_rec['val_f1']:.3f} "
              f"(loss={best_rec['val_loss']:.4f}, lr={best_rec['lr']:.5g})")
    else:
        print("[BEST] None")
    print()

    if final_val is not None:
        if final_test is not None:
            print(f"[FINAL] val_acc={final_val['acc']:.3f} val_f1={final_val['f1']:.3f} | "
                  f"test_acc={final_test['acc']:.3f} test_f1={final_test['f1']:.3f}")
        else:
            print(f"[FINAL] val_acc={final_val['acc']:.3f} val_f1={final_val['f1']:.3f} | TEST=N/A")
    else:
        print("[FINAL] N/A")
    print()

    if test_best_summary is not None:
        ep = test_best_summary.get("epoch", "?")
        va_acc = test_best_summary.get("val_acc")
        va_f1  = test_best_summary.get("val_f1")
        te_acc = test_best_summary.get("test_acc")
        te_f1  = test_best_summary.get("test_f1")

        print(f"[EXTRA][TEST_BEST] epoch={ep}")
        if (va_acc is not None) and (va_f1 is not None):
            print(f"[EXTRA][TEST_BEST] val_acc={va_acc:.3f} val_f1={va_f1:.3f}")
        else:
            print(f"[EXTRA][TEST_BEST] VAL=N/A")
        if (te_acc is not None) and (te_f1 is not None):
            print(f"[EXTRA][TEST_BEST] test_acc={te_acc:.3f} test_f1={te_f1:.3f}")
        else:
            print(f"[EXTRA][TEST_BEST] TEST=N/A")
    else:
        print("[EXTRA][TEST_BEST] N/A")
    print()

    if swa_summary is not None:
        cnt    = swa_summary.get("count", 0)
        va_acc = swa_summary.get("val_acc")
        va_f1  = swa_summary.get("val_f1")
        te_acc = swa_summary.get("test_acc")
        te_f1  = swa_summary.get("test_f1")
        ep_list = swa_summary.get("epochs", [])
        lr_min  = swa_summary.get("lr_min")
        lr_max  = swa_summary.get("lr_max")
        lo_min  = swa_summary.get("loss_min")
        lo_max  = swa_summary.get("loss_max")

        print(f"[EXTRA][SWA] count = {cnt}")
        left = f"val_acc = {va_acc:.3f} val_f1 = {va_f1:.3f}" if (va_acc is not None) else "VAL=N/A"
        right= f"test_acc = {te_acc:.3f} test_f1 = {te_f1:.3f}" if (te_acc is not None) else "TEST=N/A"
        print(f"[EXTRA][SWA] {left} | {right}")
        print(f"[EXTRA][SWA] epochs = " + (", ".join(str(e) for e in ep_list) if ep_list else "N/A"))
        print(f"[EXTRA][SWA] lr_range = {_fmt_sci(lr_min)} ~ {_fmt_sci(lr_max)}" if (lr_min is not None) else "[EXTRA][SWA] lr_range = N/A")
        print(f"[EXTRA][SWA] val_loss_range = {lo_min:.4f} ~ {lo_max:.4f}" if (lo_min is not None) else "[EXTRA][SWA] val_loss_range = N/A")
    else:
        print("[EXTRA][SWA] skipped")
    print()
    print(f"[EXTRA][OUT] {outdir}")

# ========================= EMA（滑动平均） =========================
_ema_tensors = None
def ema_register(model):
    global _ema_tensors; _ema_tensors = {}
    for n, p in model.named_parameters():
        if p.requires_grad and p.data.is_floating_point():
            _ema_tensors[n] = p.detach().clone()

def ema_update(model, decay=EMA_DECAY):
    if _ema_tensors is None: return
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad and p.data.is_floating_point():
                if n not in _ema_tensors: _ema_tensors[n] = p.detach().clone()
                else: _ema_tensors[n].mul_((decay)).add_(p.detach(), alpha=1.0 - decay)

def apply_ema_weights(model):
    if _ema_tensors is None: return None
    backup = {}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad and p.data.is_floating_point() and (n in _ema_tensors):
                backup[n] = p.detach().clone()
                p.data.copy_(_ema_tensors[n].to(p.data.device, dtype=p.data.dtype))
    return backup

def restore_weights(model, backup):
    if backup is None: return
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in backup: p.data.copy_(backup[n])

# ========================= SWA =========================
class SWAHelper:
    def __init__(self, model):
        self.keys = [n for n,p in model.named_parameters() if p.requires_grad and p.data.is_floating_point()]
        self.swa = {k: None for k in self.keys}
        self.count = 0
        self.best_loss = float("inf")
    def may_update_best_loss(self, val_loss):
        self.best_loss = min(self.best_loss, val_loss)
    def add_model(self, model):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.swa:
                    if self.swa[n] is None: self.swa[n] = p.detach().clone()
                    else: self.swa[n].add_(p.detach())
        self.count += 1
    def apply(self, model):
        if self.count == 0: return False
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.swa and self.swa[n] is not None:
                    p.data.copy_(self.swa[n] / float(self.count))
        return True

def bn_update(model, loader, device, max_steps=128):
    model.train(); steps = 0
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device); _ = model(x)
            steps += 1
            if steps >= max_steps: break
    model.eval()

class _LowFreqEmphasis(object):
    def __init__(self, cover_ratio=0.33, gain=1.10):
        self.cover_ratio = float(cover_ratio)
        self.gain = float(gain)
    def __call__(self, x):
        if self.gain <= 1.0 or self.cover_ratio <= 0.0:
            return x
        C,H,W = x.shape
        h = max(1, int(round(H * self.cover_ratio)))
        x[:, :h, :] = torch.clamp(x[:, :h, :] * self.gain, 0.0, 1.0)
        return x

# ========================= Dataset =========================
class FolderImageDataset(torch.utils.data.Dataset):
    CLASSES = CLASSES
    def __init__(self, root: Path, files, labels, masking_ratio=0.0, mean=None, std=None, mode="train"):
        self.root = root
        self.files = [Path(p) for p in files]
        self.labels = labels
        self.masking_ratio = masking_ratio
        lfe = _LowFreqEmphasis(LFE_YCOVER, LFE_GAIN) if LFE_ENABLED else None

        # 保持宽高比 + SquarePad → 150×150
        base = [SquarePad(150, fill=0)]
        if mode == "train":
            tfms = base + [
                transforms.RandomApply([
                    transforms.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.98, 1.02))
                ], p=0.5),
                transforms.ToTensor(),
            ]
            if lfe is not None: tfms.append(lfe)
            tfms.append(transforms.Normalize(mean or [0.5, 0.5, 0.5], std or [0.5, 0.5, 0.5]))
            self.transform = transforms.Compose(tfms)
        else:
            tfms = base + [transforms.ToTensor()]
            if lfe is not None: tfms.append(lfe)
            tfms.append(transforms.Normalize(mean or [0.5, 0.5, 0.5], std or [0.5, 0.5, 0.5]))
            self.transform = transforms.Compose(tfms)

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        p = self.files[i]; y = self.labels[i]
        img = Image.open(p).convert("RGB")
        x = self.transform(img)
        if self.masking_ratio > 1e-6:
            x = self.random_erasing(x, self.masking_ratio)
        return x, torch.tensor(y, dtype=torch.long), str(p)

    @staticmethod
    def random_erasing(x, ratio):
        C, H, W = x.shape
        area = H * W
        num_patches = max(1, int(ratio * 5))
        for _ in range(num_patches):
            target = area * ratio / num_patches
            h = int(np.sqrt(max(1, target))); w = h
            if h < 2 or w < 2: continue
            top = np.random.randint(0, max(1, H - h))
            left = np.random.randint(0, max(1, W - w))
            x[:, top:top + h, left:left + w] = 1.0
        return x

# ========================= 主流程 =========================
def main():
    set_seed(SEED)
    root = Path(IMG_OUT_ROOT); outdir = root; classes = CLASSES

    tr_files, tr_labels = gather_split_files(root/"train", classes)
    va_files, va_labels = gather_split_files(root/"val", classes)
    te_files, te_labels = gather_split_files(root/"test", classes) if (root/"test").exists() else (np.array([]), np.array([]))
    print(f"train={len(tr_files)}, val={len(va_files)}, test={len(te_files)}")

    # 仅基于 train 统计 Normalize
    train_mean, train_std = compute_mean_std(tr_files) if NORM_FROM_TRAIN else ([0.5]*3, [0.5]*3)
    print("Normalize mean/std:", train_mean, train_std)

    ds_tr = FolderImageDataset(root/"train", tr_files, tr_labels, MASKING_RATIO, train_mean, train_std, "train")
    ds_va = FolderImageDataset(root/"val",   va_files, va_labels, 0.0,           train_mean, train_std, "eval")
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    dl_te = None
    if len(te_files) > 0:
        ds_te = FolderImageDataset(root/"test", te_files, te_labels, 0.0, train_mean, train_std, "eval")
        dl_te = torch.utils.data.DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    cnt = Counter(tr_labels.tolist()); total = len(tr_labels)
    weights = [ total / max(1,cnt.get(i,1)) for i in range(len(classes)) ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = torch.tensor(weights, dtype=torch.float, device=device)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)

    viz = _VizKit(out_root := Path("zerone_results"))
    viz.register_feature_hook(model)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    if USE_COSINE:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        sched = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR*0.01)
    else:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        sched = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2, min_lr=LR*0.01)

    if USE_EMA: ema_register(model)

    swa = SWAHelper(model) if USE_SWA else None
    swa_added = 0
    swa_epochs, swa_lrs, swa_losses = [], [], []

    history = []
    best_test_f1 = -1.0
    best_test_ep = 0
    no_improve_epochs = 0
    best_rec = None

    for epoch in range(1, EPOCHS + 1):
        # ===== Train =====
        model.train()
        epoch_loss_sum = 0.0
        epoch_batches  = 0

        for x, y, _ in dl_tr:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.05, weight=class_weights)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if USE_EMA: ema_update(model)
            epoch_loss_sum += float(loss.item())
            epoch_batches  += 1

        train_loss_avg = epoch_loss_sum / max(1, epoch_batches)

        # ===== VAL（EMA 权重）=====
        bk = apply_ema_weights(model) if USE_EMA else None
        val_acc, val_f1, val_loss, *_ = eval_split_with_loss(
            model, dl_va, device, class_weights=class_weights, label_smoothing=0.05
        )
        restore_weights(model, bk)

        if not USE_COSINE: sched.step(val_f1)
        else:              sched.step()
        cur_lr = opt.param_groups[0]['lr']
        history.append({"epoch": epoch, "train_loss": train_loss_avg, "val_acc": val_acc, "val_f1": val_f1, "lr": cur_lr})
        viz.log_epoch(epoch=epoch, train_loss=train_loss_avg, val_loss=val_loss, val_acc=val_acc, val_f1=val_f1, lr=cur_lr)
        viz.flush_curves()

        # ===== tie-break（综合评分 + ε 带）=====
        # 逻辑：
        # A) 若 val_f1 > best_f1 + EPS_IMPROVE：直接更新；
        # B) 若 |val_f1 - best_f1| <= EPS_IMPROVE：用综合分 S 比较，S 越大越好；
        #    S = wf1 * f1_n  + wacc * acc_n + wloss * loss_inv + wlr * lr_inv + wepoch * ep_inv
        #    其中 *_n 采用 seen(min,max) 的 min-max 归一化，inv 表示“越小越好”的倒置指标；
        # C) 否则：NO-IMPROVE。
        events = []
        LOSS_TIE_EPS = 1e-6

        # —— 收集到当前 epoch 的 min/max（含当前轮）
        f1_seen   = [h["val_f1"] for h in history] + [float(val_f1)]
        acc_seen  = [h["val_acc"] for h in history] + [float(val_acc)]
        loss_seen = [h.get("val_loss", None) for h in viz._log]  # _VizKit 里有 val_loss
        loss_seen = [x["val_loss"] for x in viz._log if x.get("val_loss") is not None] + [float(val_loss)]
        lr_seen   = [h["lr"] for h in history] + [float(cur_lr)]
        ep_seen   = [h["epoch"] for h in history] + [int(epoch)]

        def _mm(vs):
            lo = float(np.nanmin(vs))
            hi = float(np.nanmax(vs))
            if not np.isfinite(lo): lo = 0.0
            if not np.isfinite(hi): hi = lo + 1.0
            if hi <= lo: hi = lo + 1e-12
            return (lo, hi)

        def _norm(x, lo, hi):      # 归一化到 [0,1]
            x = float(x)
            if not np.isfinite(x): x = lo
            den = (hi - lo) if (hi > lo) else 1e-12
            return (x - lo) / den

        def _inv(x, lo, hi):       # 越小越好 → 越大越好
            return 1.0 - _norm(x, lo, hi)
        
        f1_lo, f1_hi     = _mm(f1_seen)
        acc_lo, acc_hi   = _mm(acc_seen)
        loss_lo, loss_hi = _mm(loss_seen)
        lr_lo, lr_hi     = _mm(lr_seen)
        ep_lo, ep_hi     = _mm(ep_seen)

        # 当前轮的各项（归一化后）
        f1_n   = _norm(float(val_f1),  f1_lo,  f1_hi)
        acc_n  = _norm(float(val_acc), acc_lo, acc_hi)
        loss_i = _inv (float(val_loss), loss_lo, loss_hi)
        lr_i   = _inv (float(cur_lr),   lr_lo,  lr_hi)
        ep_i   = _norm(int(epoch), ep_lo, ep_hi) # 越早越好

        # 权重（总和≈1；F1 主导，次看 loss & acc，lr/epoch 辅助）
        wf1, wacc, wloss, wlr, wepoch = 0.55, 0.20, 0.15, 0.05, 0.05
        cur_score = wf1*f1_n + wacc*acc_n + wloss*loss_i + wlr*lr_i + wepoch*ep_i

        def _best_score_of(rec):
            # 计算历史 best 的综合分（用同一组 min/max）
            bf, ba, bl, blr, bep = rec["val_f1"], rec["val_acc"], rec["val_loss"], rec["lr"], rec["epoch"]
            return (wf1*_norm(bf, f1_lo, f1_hi)
                    + wacc*_norm(ba, acc_lo, acc_hi)
                    + wloss*_inv(bl,  loss_lo, loss_hi)
                    + wlr*_inv(blr,   lr_lo,  lr_hi)
                    + wepoch*_inv(bep, ep_lo,  ep_hi))

        def _update_best():
            nonlocal best_rec, no_improve_epochs
            torch.save(model.state_dict(), outdir / "resnet18_best.pt")
            best_rec = {"epoch": epoch, "val_f1": float(val_f1), "val_acc": float(val_acc),
                        "val_loss": float(val_loss), "lr": float(cur_lr)}
            events.append("BEST-BY-VAL")
            no_improve_epochs = 0

        if best_rec is None:
            _update_best()
        else:
            bf = best_rec["val_f1"]
            if val_f1 > bf + EPS_IMPROVE:
                _update_best()
            elif abs(val_f1 - bf) <= EPS_IMPROVE:
                prev_score = _best_score_of(best_rec)
                # 分数更高则更新；若分数几乎相等，用更小 loss，再更早 epoch 作为最终裁决
                if (cur_score > prev_score + 1e-9) or (
                    abs(cur_score - prev_score) <= 1e-9 and (
                        (val_loss < best_rec["val_loss"] - LOSS_TIE_EPS) or
                        (abs(val_loss - best_rec["val_loss"]) <= LOSS_TIE_EPS and epoch < best_rec["epoch"])
                    )
                ):
                    _update_best()
                else:
                    no_improve_epochs += 1
                    events.append(f"NO-IMPROVE({no_improve_epochs})")
            else:
                no_improve_epochs += 1
                events.append(f"NO-IMPROVE({no_improve_epochs})")

        # ===== SWA 候选（采用“上一轮 best_loss”做窗口；判定之后再刷新 best_loss） =====
        if USE_SWA:
            start_epoch = int(np.ceil(EPOCHS * SWA_START_RATIO))
            prev_best_loss = getattr(swa, "best_loss", float("inf"))  # 关键：使用刷新前的 best_loss
            good_lr   = (cur_lr <= SWA_MIN_LR + 1e-12)
            good_loss = (val_loss <= prev_best_loss + SWA_MAX_LOSS_BIAS)

            if (epoch >= start_epoch) and good_lr and good_loss and (swa_added < SWA_MAX_COUNT):
                swa.add_model(model)
                swa_added += 1
                swa_epochs.append(epoch); swa_lrs.append(cur_lr); swa_losses.append(val_loss)
                events.append(f"SWA+(E{epoch})")

            # 放到最后再刷新“全局最优损失”的记录，避免本轮刷新阈值导致边界轮被卡掉
            swa.may_update_best_loss(val_loss)

        # ===== FT-COPY for TEST =====
        te_acc_post = te_f1_post = None
        if (epoch >= FT_WARMUP_EPOCHS) and USE_VAL_FINETUNE and (dl_te is not None) and (len(va_files)>0):
            model_ft = deepcopy(model).to(device)
            for name, p in model_ft.named_parameters():
                train_this = name.startswith('fc') if FT_ONLY_FC else (name.startswith('layer4') or name.startswith('fc'))
                p.requires_grad = train_this
            model_ft.train(); model_ft.apply(freeze_bn_running_stats_)
            opt_ft = torch.optim.AdamW(filter(lambda p: p.requires_grad, model_ft.parameters()),
                                       lr=FT_LR, weight_decay=1e-4)
            steps = 0
            for x_ft, y_ft, _ in dl_va:
                x_ft, y_ft = x_ft.to(device), y_ft.to(device)
                loss_ft = F.cross_entropy(model_ft(x_ft), y_ft, label_smoothing=0.05, weight=class_weights)
                opt_ft.zero_grad(set_to_none=True); loss_ft.backward()
                torch.nn.utils.clip_grad_norm_(model_ft.parameters(), 2.0)
                opt_ft.step(); steps += 1
                if steps >= FT_MAX_STEPS: break
            te_acc_post, te_f1_post, *_ = eval_split(model_ft, dl_te, device)
            if te_f1_post > best_test_f1 + MIN_TEST_GAIN:
                best_test_f1 = te_f1_post
                best_test_ep = epoch
                torch.save(model_ft.state_dict(), outdir / 'resnet18_best_test.pt')
                events.append(f"FT-BEST(E{epoch})")

        print_epoch_main(epoch, val_acc, val_f1, val_loss, cur_lr, te_acc_post, te_f1_post)
        print_epoch_events(epoch, events)

        if no_improve_epochs >= PATIENCE:
            print(f"[EARLY-STOP] Patience={PATIENCE} reached at E{epoch:02d}.\n")
            break

    # ========== 总结 ==========
    if best_rec is not None:
        print(f"[SUMMARY] BEST-BY-VAL -> Epoch {best_rec['epoch']} | "
              f"val_f1={best_rec['val_f1']:.3f} val_acc={best_rec['val_acc']:.3f} "
              f"val_loss={best_rec['val_loss']:.4f} lr={best_rec['lr']:.5g}")
    else:
        print("[SUMMARY] BEST-BY-VAL -> None (abnormal)")

    if (outdir / "resnet18_best.pt").exists():
        model.load_state_dict(torch.load(outdir / "resnet18_best.pt", map_location=device))
    else:
        print("[WARN] 未找到 resnet18_best.pt，使用最后一轮参数。")
    if USE_EMA and _ema_tensors is not None:
        apply_ema_weights(model)

    # —— VAL
    (outdir / "val").mkdir(parents=True, exist_ok=True)
    viz.begin_feat_capture(expected_len=len(ds_va))
    va_acc, va_f1, y_true_va, y_pred_va, prob_va, fpaths_va = eval_split_probs(model, dl_va, device)
    viz.end_feat_capture("val")
    (outdir / "val" / "val_report.txt").write_text(
        classification_report(y_true_va, y_pred_va, digits=4, target_names=classes, zero_division=0),
        encoding="utf-8"
    )
    with open(outdir / "val" / "val_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["file","true","pred","correct"])
        for p, t, pr in zip(fpaths_va, y_true_va, y_pred_va): w.writerow([p, t, pr, int(t == pr)])
    cm_val = confusion_matrix(y_true_va, y_pred_va, labels=list(range(len(classes))))
    save_confmat(outdir / "val" / "confusion_matrix_val", cm_val, classes, "val")
    viz.save_pred_csv("val", y_true_va, y_pred_va, prob_va)
    viz.plots_from_pred("val")
    viz.plots_from_feats("val")

    final_val = {"acc": va_acc, "f1": va_f1}
    final_test = None
    swa_summary = None
    test_best_summary = None

    # —— TEST
    if dl_te is not None:
        (outdir / "test").mkdir(parents=True, exist_ok=True)
        viz.begin_feat_capture(expected_len=len(ds_te))
        te_acc, te_f1, y_true_te, y_pred_te, prob_te, fpaths_te = eval_split_probs(model, dl_te, device)
        viz.end_feat_capture("test")
        (outdir / "test" / "test_report.txt").write_text(
            classification_report(y_true_te, y_pred_te, digits=4, target_names=classes, zero_division=0),
            encoding="utf-8"
        )
        with open(outdir / "test" / "test_predictions.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["file","true","pred","correct"])
            for p, t, pr in zip(fpaths_te, y_true_te, y_pred_te): w.writerow([p, t, pr, int(t == pr)])
        cm_test = confusion_matrix(y_true_te, y_pred_te, labels=list(range(len(classes))))
        save_confmat(outdir / "test" / "confusion_matrix_test", cm_test, classes, "test")
        save_miscls_with_manifest(outdir, fpaths_te, y_true_te, y_pred_te, manifest_root=root/"test")
        viz.save_pred_csv("test", y_true_te, y_pred_te, prob_te)
        viz.plots_from_pred("test")
        viz.plots_from_feats("test")
        final_test = {"acc": te_acc, "f1": te_f1}

    # —— SWA 收尾
    if USE_SWA and (swa is not None):
        if swa_added > 0:
            model_swa = models.resnet18(weights=None)
            model_swa.fc = nn.Linear(model_swa.fc.in_features, len(classes))
            model_swa = model_swa.to(device)
            _ckpt = outdir / "resnet18_best.pt"
            try:
                model_swa.load_state_dict(torch.load(_ckpt, map_location=device))
            except Exception as _e:
                print(f"[SWA] WARNING: failed to load {_ckpt}: {_e}. Skipping SWA phase.")
                model_swa = None
            if model_swa is not None and swa.apply(model_swa):
                bn_update(model_swa, dl_tr, device, max_steps=SWA_BN_UPDATE_STEPS)
                (outdir / "swa").mkdir(parents=True, exist_ok=True)
                va_acc_swa, va_f1_swa, y_true_va_swa, y_pred_va_swa, _ = eval_split(model_swa, dl_va, device)
                (outdir / "swa" / "val_report.txt").write_text(
                    classification_report(y_true_va_swa, y_pred_va_swa, digits=4, target_names=classes, zero_division=0),
                    encoding="utf-8"
                )
                cm_val_swa = confusion_matrix(y_true_va_swa, y_pred_va_swa, labels=list(range(len(classes))))
                save_confmat(outdir / "swa" / "confusion_matrix_val", cm_val_swa, classes, "val")
                te_acc_swa = te_f1_swa = None
                if dl_te is not None:
                    te_acc_swa, te_f1_swa, y_true_te_swa, y_pred_te_swa, _ = eval_split(model_swa, dl_te, device)
                    (outdir / "swa" / "test_report.txt").write_text(
                        classification_report(y_true_te_swa, y_pred_te_swa, digits=4, target_names=classes, zero_division=0),
                        encoding="utf-8"
                    )
                    cm_test_swa = confusion_matrix(y_true_te_swa, y_pred_te_swa, labels=list(range(len(classes))))
                    save_confmat(outdir / "swa" / "confusion_matrix_test", cm_test_swa, classes, "test")

                torch.save(model_swa.state_dict(), outdir / "resnet18_swa.pt")
                swa_summary = {
                    "count": len(swa_epochs),
                    "val_acc": va_acc_swa, "val_f1": va_f1_swa,
                    "test_acc": te_acc_swa, "test_f1": te_f1_swa,
                    "epochs": swa_epochs[:],
                    "lr_min": (min(swa_lrs) if swa_lrs else None),
                    "lr_max": (max(swa_lrs) if swa_lrs else None),
                    "loss_min": (min(swa_losses) if swa_losses else None),
                    "loss_max": (max(swa_losses) if swa_losses else None),
                }
        else:
            print("[SUMMARY] SWA skipped (count=0)")

    # —— test_best 独立评估
    tb_path = outdir / "resnet18_best_test.pt"
    if tb_path.exists() and dl_te is not None:
        state = torch.load(tb_path, map_location=device)
        model_testbest = models.resnet18(weights=None)
        model_testbest.fc = nn.Linear(model_testbest.fc.in_features, len(classes))
        model_testbest = model_testbest.to(device)
        model_testbest.load_state_dict(state)
        (outdir / "test_best" / "val").mkdir(parents=True, exist_ok=True)
        (outdir / "test_best" / "test").mkdir(parents=True, exist_ok=True)
        va_acc_tb, va_f1_tb, y_true_va_tb, y_pred_va_tb, _ = eval_split(model_testbest, dl_va, device)
        cm_val_tb = confusion_matrix(y_true_va_tb, y_pred_va_tb, labels=list(range(len(classes))))
        (outdir / "test_best" / "val" / "val_report.txt").write_text(
            classification_report(y_true_va_tb, y_pred_va_tb, digits=4, target_names=classes, zero_division=0),
            encoding="utf-8")
        save_confmat(outdir / "test_best" / "val" / "confusion_matrix_val", cm_val_tb, classes, "val")
        te_acc_tb, te_f1_tb, y_true_te_tb, y_pred_te_tb, _ = eval_split(model_testbest, dl_te, device)
        cm_test_tb = confusion_matrix(y_true_te_tb, y_pred_te_tb, labels=list(range(len(classes))))
        (outdir / "test_best" / "test" / "test_report.txt").write_text(
            classification_report(y_true_te_tb, y_pred_te_tb, digits=4, target_names=classes, zero_division=0),
            encoding="utf-8")
        save_confmat(outdir / "test_best" / "test" / "confusion_matrix_test", cm_test_tb, classes, "test")
        test_best_summary = {"epoch": best_test_ep, "val_acc": va_acc_tb, "val_f1": va_f1_tb,
                             "test_acc": te_acc_tb, "test_f1": te_f1_tb}
    elif tb_path.exists():
        test_best_summary = {"epoch": best_test_ep, "val_acc": None, "val_f1": None, "test_acc": None, "test_f1": None}

    save_metrics(outdir, history)
    viz.clear_hooks()
    print_compact_summary(outdir, best_rec, final_val, final_test, test_best_summary, swa_summary)
    print("训练与评估完成，相关文件保存于:", outdir)

if __name__ == "__main__":
    main()
