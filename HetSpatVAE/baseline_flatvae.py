# -*- coding: utf-8 -*-
"""
baseline_flatvae.py  —  放在 hetero（VAE）目录下运行
================================================================================
基线 (b): FlatVAE — 标准平坦向量 VAE（Global Average Pooling → 64-dim 平坦潜码）
          作为 HetSpatVAE 的消融对照。

★ 仅依赖 hetero 侧代码：hetero_config.py / hetero_data.py / hetero_model.py
★ 不导入任何 zerone_* 模块
★ 复用 hetero_data.py 的 TransformerVibrationDataset（3 通道时频图像）
★ 复用 hetero_diagnose.py 的复合评分流程（重建误差 + Mahalanobis）
★ 输出：混淆矩阵、t-SNE、评分分布直方图、分类报告、汇总 CSV

运行：
    cd <你的 hetero 工作目录>
    python baseline_flatvae.py
================================================================================
"""

from __future__ import annotations
import os, csv, random, warnings
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, precision_score, recall_score,
)
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.dpi"] = 200

# ── hetero 侧依赖 ──
import hetero_config as cfg
from hetero_data import TransformerVibrationDataset
from torchvision import models

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════════
# 0. 全局设置
# ════════════════════════════════════════════════════════════════════════════
SEED = 42
DEVICE = torch.device(cfg.DEVICE if hasattr(cfg, "DEVICE") else "cuda")
OUT_ROOT = Path("./baseline_flatvae_results").resolve()
OUT_ROOT.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

# 异常评分超参数（与 hetero_diagnose.py 对齐）
ALPHA = 0.6       # 重构误差权重
THRESH_QUANTILE = 0.975   # 阈值分位数

# 配色
C_NORM = "#3B76AF"; C_ANOM = "#D62728"


# ════════════════════════════════════════════════════════════════════════════
# 1. FlatResNetVAE 模型
# ════════════════════════════════════════════════════════════════════════════

class FlatResNetVAE(nn.Module):
    """
    与 SpatialResNetVAE 使用完全相同的 ResNet18 编码器骨干，
    ★ 唯一区别：layer4 输出 (512×7×7) → Global Average Pooling → 512-dim
      → FC 投影到 64-dim 平坦潜码（而非 7×7×64 空间潜码）
    解码器从 64-dim 向量 → 512×7×7 → 转置卷积重建 3×224×224 图像
    """
    def __init__(self, latent_dim=64):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.encoder_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        # ★ GAP → 平坦向量
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        self.latent_dim = latent_dim

        # 解码器
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
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, x):
        h = self.encoder_stem(x)                    # (B, 512, 7, 7)
        h_flat = self.gap(h).view(h.size(0), -1)    # (B, 512) ← ★ GAP
        mu = self.fc_mu(h_flat)                      # (B, 64) ← 平坦潜码
        logvar = self.fc_logvar(h_flat)              # (B, 64)
        z = self.reparameterize(mu, logvar)
        dec_in = self.decoder_fc(z).view(-1, 512, 7, 7)
        recon = self.decoder_conv(dec_in)
        if recon.shape != x.shape:
            recon = F.interpolate(recon, size=x.shape[2:])
        return recon, mu, logvar


def flat_vae_loss(recon_x, x, mu, logvar, beta=1.0):
    B = x.size(0)
    rec = F.l1_loss(recon_x, x, reduction="sum") / B
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
    return rec + beta * kld, rec, kld


# ════════════════════════════════════════════════════════════════════════════
# 2. 异常评分工具（与 hetero_diagnose.py 对齐）
# ════════════════════════════════════════════════════════════════════════════

def _channel_weighted_l1(recon, inp, w=(0.4, 0.5, 0.1)):
    e0 = torch.mean(torch.abs(recon[:, 0] - inp[:, 0]), dim=[1, 2])
    e1 = torch.mean(torch.abs(recon[:, 1] - inp[:, 1]), dim=[1, 2])
    e2 = torch.mean(torch.abs(recon[:, 2] - inp[:, 2]), dim=[1, 2])
    return (w[0] * e0 + w[1] * e1 + w[2] * e2).detach().cpu().numpy()


def _collect_scores(model, loader, device):
    """推理并收集 (重建误差, 潜码)。"""
    model.eval()
    rec_all, lat_all = [], []
    with torch.no_grad():
        for imgs in tqdm(loader, desc="  Scoring", leave=False):
            imgs = imgs.to(device)
            recon, mu, _ = model(imgs)
            rec_all.append(_channel_weighted_l1(recon, imgs))
            # FlatVAE: mu 已经是 (B, latent_dim)
            lat_all.append(mu.cpu().numpy())
    return np.concatenate(rec_all), np.vstack(lat_all)


def _mahalanobis(lat, mean, inv_cov):
    diff = lat - mean[None, :]
    d2 = np.einsum("bi,ij,bj->b", diff, inv_cov, diff)
    return np.sqrt(np.maximum(d2, 0.0))


# ════════════════════════════════════════════════════════════════════════════
# 3. 可视化
# ════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, y_pred, save_dir, tag="FlatVAE"):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    for lang, labels, xl, yl, title in [
        ("en", ["Normal", "Fault"], "Predicted", "True", f"Confusion Matrix — {tag}"),
        ("zh", ["正常", "故障"], "预测类别", "真实类别", f"混淆矩阵 — {tag}"),
    ]:
        fig, ax = plt.subplots(figsize=(5, 4.2))
        ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title, fontweight="bold")
        for i in range(2):
            for j in range(2):
                c = "white" if cm_pct[i, j] > 0.5 else "black"
                ax.text(j, i - 0.05, f"{cm[i,j]}", ha="center", va="center", color=c, fontsize=16, fontweight="bold")
                ax.text(j, i + 0.18, f"({cm_pct[i,j]:.1%})", ha="center", va="center", color=c, fontsize=9, alpha=0.7)
        fig.tight_layout(); fig.savefig(save_dir / f"cm_{tag}_{lang}.png"); plt.close(fig)


def plot_score_histogram(scores, threshold, name, save_dir, tag="FlatVAE"):
    for lang, xl, yl, tl, leg in [
        ("en", "Anomaly Score", "Density", f"Score Distribution ({name}) — {tag}",
         ["Score Dist.", "Threshold"]),
        ("zh", "异常评分", "概率密度", f"评分分布 ({name}) — {tag}",
         ["评分分布", "判决阈值"]),
    ]:
        fig = plt.figure(figsize=(6, 4))
        sns.histplot(scores, kde=True, stat="density", color=C_NORM,
                     edgecolor="white", linewidth=0.5, alpha=0.7, label=leg[0])
        plt.axvline(threshold, color=C_ANOM, linestyle="--", linewidth=1.5,
                    label=f"{leg[1]}: {threshold:.2f}")
        plt.title(tl, fontsize=12, fontweight="bold")
        plt.xlabel(xl); plt.ylabel(yl); plt.legend(loc="upper right", fontsize=9)
        sns.despine(); plt.grid(axis="y", linestyle=":", alpha=0.3)
        fig.tight_layout(); fig.savefig(save_dir / f"hist_{name}_{tag}_{lang}.png"); plt.close(fig)


def plot_tsne(lat_train, lat_test_normal, lat_test_fault, save_dir, tag="FlatVAE"):
    if lat_train.size == 0: return
    n_tr = min(len(lat_train), 1000)
    idx_tr = np.random.choice(len(lat_train), n_tr, replace=False)
    X_list = [lat_train[idx_tr]]
    labels_list = [np.zeros(n_tr, dtype=int)]
    if lat_test_normal.size > 0:
        X_list.append(lat_test_normal); labels_list.append(np.ones(len(lat_test_normal), dtype=int))
    if lat_test_fault.size > 0:
        X_list.append(lat_test_fault); labels_list.append(np.full(len(lat_test_fault), 2, dtype=int))
    X = np.vstack(X_list); labels = np.concatenate(labels_list)
    Z = TSNE(n_components=2, perplexity=30, random_state=SEED, init="pca").fit_transform(X)

    for lang, l1, l2, l3, tl in [
        ("en", "Train (Baseline)", "Test (Normal)", "Test (Fault)", f"t-SNE Latent Space — {tag}"),
        ("zh", "训练集 (基准)", "测试集 (正常)", "测试集 (故障)", f"隐空间 t-SNE — {tag}"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(Z[labels == 0, 0], Z[labels == 0, 1], c="#DDDDDD", s=30, alpha=0.5, label=l1)
        if np.any(labels == 1):
            ax.scatter(Z[labels == 1, 0], Z[labels == 1, 1], c=C_NORM, s=40, alpha=0.8,
                       marker="o", edgecolors="white", linewidth=0.5, label=l2)
        if np.any(labels == 2):
            ax.scatter(Z[labels == 2, 0], Z[labels == 2, 1], c=C_ANOM, s=50, alpha=0.9,
                       marker="X", edgecolors="white", linewidth=0.5, label=l3)
        ax.legend(fontsize=9); ax.set_title(tl, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([]); sns.despine()
        fig.tight_layout(); fig.savefig(save_dir / f"tsne_{tag}_{lang}.png"); plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# 4. 主流程
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 64)
    print("  W1 基线实验 (b) — FlatVAE (GAP, no spatial latent)")
    print(f"  输出目录: {OUT_ROOT}")
    print(f"  设备: {DEVICE}")
    print("=" * 64)

    # ──── 数据加载（与 hetero_train.py 对齐） ────
    train_path = Path(cfg.TRAIN_DIR)
    val_path = Path(cfg.VAL_DIR)
    test_path = Path(cfg.TEST_DIR)

    print("\n[1/4] 加载数据 ...")
    train_set = TransformerVibrationDataset(train_path, only_normal=False, mode="train")
    val_set   = TransformerVibrationDataset(val_path,   only_normal=False, mode="val")
    if len(train_set) == 0:
        print("[ERROR] 训练集为空，请检查 hetero_config.py 的路径。"); return

    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

    # ──── 训练 ────
    print("\n[2/4] 训练 FlatVAE ...")
    model = FlatResNetVAE(latent_dim=cfg.LATENT_CHANNELS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    best_val_loss = float("inf"); best_state = None

    for epoch in range(1, cfg.EPOCHS + 1):
        beta = min(cfg.BETA_MAX, cfg.BETA_MAX * epoch / max(cfg.BETA_WARMUP_EPOCHS, 1))

        model.train(); tr_loss = 0.0
        for x in tqdm(train_loader, desc=f"  E{epoch:02d} [Train]", leave=False):
            x = x.to(DEVICE)
            recon, mu, logvar = model(x)
            loss, _, _ = flat_vae_loss(recon, x, mu, logvar, beta=beta)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            tr_loss += loss.item()
        avg_tr = tr_loss / max(1, len(train_loader))

        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(DEVICE)
                recon, mu, logvar = model(x)
                loss, _, _ = flat_vae_loss(recon, x, mu, logvar, beta=beta)
                va_loss += loss.item()
        avg_va = va_loss / max(1, len(val_loader))

        if avg_va < best_val_loss:
            best_val_loss = avg_va; best_state = deepcopy(model.state_dict())
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [E{epoch:02d}] train={avg_tr:.4f}  val={avg_va:.4f}  beta={beta:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    # 保存模型
    torch.save(model.state_dict(), OUT_ROOT / "flatvae_best.pth")

    # ──── 异常检测评估 ────
    # ★ 用 only_normal=True 的 loader 建立正常基准（与 hetero_diagnose.py 对齐）
    print("\n[3/4] 异常检测评估 ...")
    train_normal_set = TransformerVibrationDataset(train_path, only_normal=True, mode="train_baseline")
    train_normal_loader = DataLoader(train_normal_set, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  正常基准样本数: {len(train_normal_set)}")
    tr_rec, tr_lat = _collect_scores(model, train_normal_loader, DEVICE)
    mean_lat = tr_lat.mean(axis=0)
    cov_lat = np.cov(tr_lat.T) + 1e-6 * np.eye(tr_lat.shape[1])
    inv_cov = np.linalg.pinv(cov_lat)
    tr_md = _mahalanobis(tr_lat, mean_lat, inv_cov)

    mu_rec, sd_rec = tr_rec.mean(), max(tr_rec.std(), 1e-9)
    mu_md, sd_md = tr_md.mean(), max(tr_md.std(), 1e-9)
    tr_scores = ALPHA * (tr_rec - mu_rec) / sd_rec + (1 - ALPHA) * (tr_md - mu_md) / sd_md
    threshold = np.quantile(tr_scores, THRESH_QUANTILE)
    print(f"  训练基准样本数: {len(tr_scores)}")
    print(f"  阈值 (α={ALPHA}, q={THRESH_QUANTILE}): {threshold:.4f}")

    # 测试子目录
    subdirs = [d for d in test_path.iterdir() if d.is_dir()]
    if not subdirs: subdirs = [test_path]

    all_yt, all_yp = [], []
    lat_tn_list, lat_tf_list = [], []

    for sub in subdirs:
        name = sub.name.lower()
        is_fault = any(k in name for k in ["故障", "fault", "异常"])
        label = 1 if is_fault else 0
        display_name = "test_fault" if is_fault else "test_normal"

        ds = TransformerVibrationDataset(sub, only_normal=False, mode="test")
        if len(ds) == 0: continue
        loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

        te_rec, te_lat = _collect_scores(model, loader, DEVICE)
        te_md = _mahalanobis(te_lat, mean_lat, inv_cov)
        te_scores = ALPHA * (te_rec - mu_rec) / sd_rec + (1 - ALPHA) * (te_md - mu_md) / sd_md
        preds = (te_scores < threshold).astype(int)  # 低于阈值 → 异常

        n_abn = int(preds.sum())
        print(f"  [{display_name}] 总数={len(preds)} | 检出异常={n_abn} ({n_abn/len(preds)*100:.1f}%)")

        all_yt.extend([label] * len(preds))
        all_yp.extend(preds.tolist())

        # 收集潜码用于 t-SNE
        if is_fault: lat_tf_list.append(te_lat)
        else:        lat_tn_list.append(te_lat)

        # 直方图
        plot_score_histogram(te_scores, threshold, display_name, OUT_ROOT)

    if not all_yt:
        print("[ERROR] 测试集为空"); return

    y_true = np.array(all_yt); y_pred = np.array(all_yp)

    acc = accuracy_score(y_true, y_pred)
    pn  = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    rn  = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    rf  = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    mf1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n  ┌── FlatVAE (GAP, no spatial) ──")
    print(f"  │ Acc    = {acc*100:.2f}%")
    print(f"  │ Pre(N) = {pn:.3f}   Rec(N) = {rn:.3f}   Rec(F) = {rf:.3f}")
    print(f"  │ Macro-F1 = {mf1:.3f}")
    print(f"  └{'─'*44}\n")

    # 分类报告
    report = classification_report(y_true, y_pred, target_names=["正常", "故障"], digits=4, zero_division=0)
    (OUT_ROOT / "classification_report.txt").write_text(report, encoding="utf-8")
    print(report)

    # ──── 可视化 ────
    print("\n[4/4] 生成可视化 ...")
    plot_confusion_matrix(y_true, y_pred, OUT_ROOT)

    lat_tn = np.vstack(lat_tn_list) if lat_tn_list else np.empty((0, tr_lat.shape[1]))
    lat_tf = np.vstack(lat_tf_list) if lat_tf_list else np.empty((0, tr_lat.shape[1]))
    plot_tsne(tr_lat, lat_tn, lat_tf, OUT_ROOT)

    # 汇总 CSV
    csv_path = OUT_ROOT / "summary_flatvae.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["method", "paradigm", "acc", "pre_n", "rec_n", "rec_f", "macro_f1"])
        w.writeheader()
        w.writerow(dict(method="FlatVAE (GAP, no spatial)", paradigm="One-class",
                        acc=f"{acc:.4f}", pre_n=f"{pn:.4f}", rec_n=f"{rn:.4f}",
                        rec_f=f"{rf:.4f}", macro_f1=f"{mf1:.4f}"))
    print(f"  CSV → {csv_path}")

    # 预测详情
    with open(OUT_ROOT / "predictions.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["idx", "y_true", "y_pred", "correct"])
        for i in range(len(y_true)):
            w.writerow([i, y_true[i], y_pred[i], int(y_true[i] == y_pred[i])])

    print(f"\n  所有结果输出到: {OUT_ROOT}")
    print("=" * 64)
    print("  FlatVAE 基线实验完成！")
    print("=" * 64)


if __name__ == "__main__":
    main()
