# -*- coding: utf-8 -*-
"""
baseline_flatvae_posteval.py  —  放在 hetero（VAE）目录下运行
================================================================================
★ 不需要重新训练！★
加载已保存的 flatvae_best.pth，重新计算更全面的评估指标：
  - 原有: Acc / Pre(N) / Rec(N) / Rec(F) / Macro-F1（固定阈值）
  - 新增: ROC-AUC / AUPRC / Score Margin（阈值无关指标）
  - 新增: ROC 曲线图 / Score distribution overlay 对比图

为什么需要这个：
  FlatVAE 在 97.5% 分位数阈值下恰好 acc=99.25% > HetSpatVAE 98.25%，
  但这是单一阈值的偶然结果。空间潜码的真正优势体现在跨阈值的整体分离能力
  （ROC-AUC）和正常/故障分数分布的间距（score margin）上。

运行：
    cd <hetero 工作目录>
    python baseline_flatvae_posteval.py
================================================================================
"""

from __future__ import annotations
import csv, random, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    roc_auc_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 200; plt.rcParams["savefig.dpi"] = 200

import hetero_config as cfg
from hetero_data import TransformerVibrationDataset
from torchvision import models

warnings.filterwarnings("ignore")

SEED = 42
DEVICE = torch.device(cfg.DEVICE if hasattr(cfg, "DEVICE") else "cuda")
OUT_ROOT = Path("./baseline_flatvae_results").resolve()
ALPHA = 0.6
THRESH_QUANTILE = 0.975
C_NORM = "#3B76AF"; C_ANOM = "#D62728"

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# ── FlatResNetVAE 模型定义（与 baseline_flatvae.py 完全一致） ──
class FlatResNetVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.encoder_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        self.latent_dim = latent_dim
        self.decoder_fc = nn.Linear(latent_dim, 512 * 7 * 7)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.ConvTranspose2d(64,  64,  4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.ConvTranspose2d(64,  3,   4, stride=2, padding=1), nn.Sigmoid(),
        )
    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.randn_like(torch.exp(0.5*logvar)) * torch.exp(0.5*logvar)
        return mu
    def forward(self, x):
        h = self.encoder_stem(x)
        h_flat = self.gap(h).view(h.size(0), -1)
        mu = self.fc_mu(h_flat); logvar = self.fc_logvar(h_flat)
        z = self.reparameterize(mu, logvar)
        dec_in = self.decoder_fc(z).view(-1, 512, 7, 7)
        recon = self.decoder_conv(dec_in)
        if recon.shape != x.shape: recon = F.interpolate(recon, size=x.shape[2:])
        return recon, mu, logvar

# ── 评分工具 ──
def _channel_weighted_l1(recon, inp, w=(0.4, 0.5, 0.1)):
    e0 = torch.mean(torch.abs(recon[:,0]-inp[:,0]), dim=[1,2])
    e1 = torch.mean(torch.abs(recon[:,1]-inp[:,1]), dim=[1,2])
    e2 = torch.mean(torch.abs(recon[:,2]-inp[:,2]), dim=[1,2])
    return (w[0]*e0 + w[1]*e1 + w[2]*e2).detach().cpu().numpy()

def _collect_scores(model, loader, device):
    model.eval(); rec_all, lat_all = [], []
    with torch.no_grad():
        for imgs in tqdm(loader, desc="  Scoring", leave=False):
            imgs = imgs.to(device)
            recon, mu, _ = model(imgs)
            rec_all.append(_channel_weighted_l1(recon, imgs))
            lat_all.append(mu.cpu().numpy())
    return np.concatenate(rec_all), np.vstack(lat_all)

def _mahalanobis(lat, mean, inv_cov):
    diff = lat - mean[None,:]
    return np.sqrt(np.maximum(np.einsum("bi,ij,bj->b", diff, inv_cov, diff), 0.0))


def main():
    print("=" * 64)
    print("  FlatVAE 后处理评估（加载已有权重，不重新训练）")
    print("=" * 64)

    # 加载模型
    model_path = OUT_ROOT / "flatvae_best.pth"
    if not model_path.exists():
        print(f"[ERROR] 未找到 {model_path}，请先运行 baseline_flatvae.py")
        return
    model = FlatResNetVAE(latent_dim=cfg.LATENT_CHANNELS).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"  模型加载自: {model_path}")

    # 训练正常基准
    train_path = Path(cfg.TRAIN_DIR)
    test_path = Path(cfg.TEST_DIR)
    train_normal_set = TransformerVibrationDataset(train_path, only_normal=True, mode="train_baseline")
    train_normal_loader = DataLoader(train_normal_set, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  正常基准样本: {len(train_normal_set)}")

    tr_rec, tr_lat = _collect_scores(model, train_normal_loader, DEVICE)
    mean_lat = tr_lat.mean(axis=0)
    cov_lat = np.cov(tr_lat.T) + 1e-6 * np.eye(tr_lat.shape[1])
    inv_cov = np.linalg.pinv(cov_lat)
    tr_md = _mahalanobis(tr_lat, mean_lat, inv_cov)

    mu_rec, sd_rec = tr_rec.mean(), max(tr_rec.std(), 1e-9)
    mu_md, sd_md = tr_md.mean(), max(tr_md.std(), 1e-9)
    tr_scores = ALPHA * (tr_rec - mu_rec)/sd_rec + (1-ALPHA) * (tr_md - mu_md)/sd_md
    threshold = np.quantile(tr_scores, THRESH_QUANTILE)

    # 测试集
    subdirs = [d for d in test_path.iterdir() if d.is_dir()]
    if not subdirs: subdirs = [test_path]

    all_scores, all_labels = [], []

    for sub in subdirs:
        name = sub.name.lower()
        is_fault = any(k in name for k in ["故障","fault","异常"])
        label = 1 if is_fault else 0

        ds = TransformerVibrationDataset(sub, only_normal=False, mode="test")
        if len(ds) == 0: continue
        loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

        te_rec, te_lat = _collect_scores(model, loader, DEVICE)
        te_md = _mahalanobis(te_lat, mean_lat, inv_cov)
        te_scores = ALPHA * (te_rec - mu_rec)/sd_rec + (1-ALPHA) * (te_md - mu_md)/sd_md

        all_scores.extend(te_scores.tolist())
        all_labels.extend([label] * len(te_scores))

    scores = np.array(all_scores)
    labels = np.array(all_labels)

    # ════ 指标计算 ════
    preds = (scores < threshold).astype(int)
    acc = accuracy_score(labels, preds)
    pn  = precision_score(labels, preds, pos_label=0, zero_division=0)
    rn  = recall_score(labels, preds, pos_label=0, zero_division=0)
    rf  = recall_score(labels, preds, pos_label=1, zero_division=0)
    mf1 = f1_score(labels, preds, average="macro", zero_division=0)

    # 阈值无关指标
    # 注意：分数越低越异常，所以 y_score = -scores（sklearn 要求 score 越高越正）
    neg_scores = -scores
    roc_auc = roc_auc_score(labels, neg_scores)
    auprc = average_precision_score(labels, neg_scores)

    # Score margin：正常分数中位数 - 故障分数中位数
    normal_scores = scores[labels == 0]
    fault_scores = scores[labels == 1]
    score_margin = float(np.median(normal_scores) - np.median(fault_scores))
    # 正常/故障分数的重叠度（overlap coefficient 近似）
    normal_below_thr = (normal_scores < threshold).sum()
    fault_above_thr = (fault_scores >= threshold).sum()

    print(f"\n  ┌── FlatVAE 完整评估 ──")
    print(f"  │ [固定阈值 τ={threshold:.3f}]")
    print(f"  │ Acc     = {acc*100:.2f}%")
    print(f"  │ Pre(N)  = {pn:.3f}   Rec(N) = {rn:.3f}   Rec(F) = {rf:.3f}")
    print(f"  │ Macro-F1= {mf1:.3f}")
    print(f"  │")
    print(f"  │ [阈值无关指标]")
    print(f"  │ ROC-AUC = {roc_auc:.4f}")
    print(f"  │ AUPRC   = {auprc:.4f}")
    print(f"  │ Score Margin (median) = {score_margin:.3f}")
    print(f"  │ Normal scores: mean={normal_scores.mean():.3f} std={normal_scores.std():.3f}")
    print(f"  │ Fault  scores: mean={fault_scores.mean():.3f} std={fault_scores.std():.3f}")
    print(f"  │ FP={normal_below_thr}  FN={fault_above_thr}")
    print(f"  └{'─'*50}\n")

    # ════ 可视化 ════

    # 1. ROC 曲线
    fpr, tpr, _ = roc_curve(labels, neg_scores)
    for lang, tl, xl, yl in [
        ("en", f"ROC — FlatVAE (AUC={roc_auc:.3f})", "FPR", "TPR"),
        ("zh", f"ROC 曲线 — FlatVAE (AUC={roc_auc:.3f})", "假阳性率", "真阳性率"),
    ]:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.plot(fpr, tpr, color=C_NORM, lw=2)
        ax.plot([0,1],[0,1],"k--",lw=0.8)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(tl, fontweight="bold")
        ax.grid(True, alpha=0.3); fig.tight_layout()
        fig.savefig(OUT_ROOT / f"roc_FlatVAE_{lang}.png"); plt.close(fig)

    # 2. PR 曲线
    prec, rec, _ = precision_recall_curve(labels, neg_scores)
    for lang, tl in [
        ("en", f"PR — FlatVAE (AUPRC={auprc:.3f})"),
        ("zh", f"PR 曲线 — FlatVAE (AUPRC={auprc:.3f})"),
    ]:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.plot(rec, prec, color=C_ANOM, lw=2, drawstyle="steps-post")
        ax.set_xlim(0,1); ax.set_ylim(0,1.05)
        ax.set_xlabel("Recall" if "en" in lang else "召回率")
        ax.set_ylabel("Precision" if "en" in lang else "精确率")
        ax.set_title(tl, fontweight="bold"); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT_ROOT / f"pr_FlatVAE_{lang}.png"); plt.close(fig)

    # 3. Score distribution overlay（正常 vs 故障）
    for lang, tl, xl, l_n, l_f, l_t in [
        ("en", "Score Distribution — FlatVAE", "Anomaly Score",
         "Normal", "Fault", "Threshold"),
        ("zh", "评分分布 — FlatVAE", "异常评分",
         "正常", "故障", "阈值"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.kdeplot(normal_scores, ax=ax, color=C_NORM, fill=True, alpha=0.4, label=f"{l_n} (n={len(normal_scores)})")
        sns.kdeplot(fault_scores, ax=ax, color=C_ANOM, fill=True, alpha=0.4, label=f"{l_f} (n={len(fault_scores)})")
        ax.axvline(threshold, color="black", linestyle="--", lw=1.5, label=f"{l_t}={threshold:.2f}")
        ax.set_xlabel(xl); ax.set_ylabel("Density"); ax.set_title(tl, fontweight="bold")
        ax.legend(); sns.despine(); ax.grid(axis="y", linestyle=":", alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT_ROOT / f"score_overlay_FlatVAE_{lang}.png"); plt.close(fig)

    # ════ 更新 CSV ════
    csv_path = OUT_ROOT / "summary_flatvae.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=[
            "method","paradigm","acc","pre_n","rec_n","rec_f","macro_f1","roc_auc","auprc","score_margin"
        ])
        w.writeheader()
        w.writerow(dict(
            method="FlatVAE (GAP, no spatial)", paradigm="One-class",
            acc=f"{acc:.4f}", pre_n=f"{pn:.4f}", rec_n=f"{rn:.4f}",
            rec_f=f"{rf:.4f}", macro_f1=f"{mf1:.4f}",
            roc_auc=f"{roc_auc:.4f}", auprc=f"{auprc:.4f}",
            score_margin=f"{score_margin:.4f}",
        ))
    print(f"  更新 CSV → {csv_path}")
    print("\n  完成！请同时对 HetSpatVAE 计算同样的 ROC-AUC / AUPRC / Score Margin")
    print("  以便在论文中进行公平对比。")


if __name__ == "__main__":
    main()
