# -*- coding: utf-8 -*-
"""
baseline_flatvae.py  —  放在 hetero（VAE）目录下运行
================================================================================
基线 (b): FlatVAE — 标准平坦向量 VAE（Global Average Pooling → 平坦潜码）
          作为 HetSpatVAE 的消融对照。

★ 仅依赖 hetero 侧代码：hetero_config.py / hetero_data.py
★ 不导入任何 zerone_* 模块

与 HetSpatVAE 的关键差异（消融点）：
  1. GAP 压缩空间信息 → 8-dim 平坦潜码（vs HetSpatVAE 64×7×7 空间潜码）
  2. 解码器从平坦向量重建 → 丢失局部时频定位能力
  3. 更高的 β 值 → 更强的 KL 正则化，潜码被压缩得更"模糊"

运行：
    cd <hetero 工作目录>
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

# ── FlatVAE 专用超参数（与 HetSpatVAE 的差异在这里体现）──
FLAT_LATENT_DIM    = 8       # 平坦潜码维度（HetSpatVAE: 64×7×7=3136 个潜变量）
FLAT_BETA_MAX      = 0.5     # 更高的 β → KL 正则更强 → 重建更模糊
FLAT_BETA_WARMUP   = 10      # β 预热期
FLAT_EPOCHS        = 20      # 训练轮数
FLAT_LR            = 1e-4    # 学习率

# 异常评分：只用重建误差，不用 Mahalanobis（平坦潜码太低维，协方差估计不稳定）
THRESH_QUANTILE    = 0.975

# 配色
C_NORM = "#3B76AF"; C_ANOM = "#D62728"


# ════════════════════════════════════════════════════════════════════════════
# 1. FlatResNetVAE 模型
# ════════════════════════════════════════════════════════════════════════════

class FlatResNetVAE(nn.Module):
    """
    标准平坦向量 VAE。与 SpatialResNetVAE 使用相同的 ResNet18 编码器，
    但 layer4 输出 (512×7×7) 经 GAP → 512 → FC → latent_dim 维平坦潜码。
    解码器从 latent_dim → 512×7×7 → 转置卷积重建 3×224×224。
    """
    def __init__(self, latent_dim=8):
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
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, x):
        h = self.encoder_stem(x)                    # (B, 512, 7, 7)
        h_flat = self.gap(h).view(h.size(0), -1)    # (B, 512)
        mu = self.fc_mu(h_flat)                      # (B, latent_dim)
        logvar = self.fc_logvar(h_flat)
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
# 2. 异常评分（仅重建误差，不用 Mahalanobis）
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
            lat_all.append(mu.cpu().numpy())
    return np.concatenate(rec_all), np.vstack(lat_all)


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
    Z = TSNE(n_components=2, perplexity=min(30, len(X) - 1), random_state=SEED, init="pca").fit_transform(X)

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
    print(f"  latent_dim={FLAT_LATENT_DIM}  β_max={FLAT_BETA_MAX}  epochs={FLAT_EPOCHS}")
    print("=" * 64)

    # ──── 数据加载 ────
    train_path = Path(cfg.TRAIN_DIR)
    val_path = Path(cfg.VAL_DIR)
    test_path = Path(cfg.TEST_DIR)

    print("\n[1/4] 加载数据 ...")
    train_set = TransformerVibrationDataset(train_path, only_normal=False, mode="train")
    val_set   = TransformerVibrationDataset(val_path,   only_normal=False, mode="val")
    if len(train_set) == 0:
        print("[ERROR] 训练集为空"); return

    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

    # ──── 训练 ────
    print("\n[2/4] 训练 FlatVAE ...")
    model = FlatResNetVAE(latent_dim=FLAT_LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAT_LR)
    best_val_loss = float("inf"); best_state = None

    for epoch in range(1, FLAT_EPOCHS + 1):
        beta = min(FLAT_BETA_MAX, FLAT_BETA_MAX * epoch / max(FLAT_BETA_WARMUP, 1))

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
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [E{epoch:02d}] train={avg_tr:.4f}  val={avg_va:.4f}  beta={beta:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    torch.save(model.state_dict(), OUT_ROOT / "flatvae_best.pth")

    # ──── 异常检测评估（仅重建误差，不用 Mahalanobis）────
    print("\n[3/4] 异常检测评估 ...")
    train_normal_set = TransformerVibrationDataset(train_path, only_normal=True, mode="train_baseline")
    train_normal_loader = DataLoader(train_normal_set, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"  正常基准样本数: {len(train_normal_set)}")

    tr_rec, tr_lat = _collect_scores(model, train_normal_loader, DEVICE)

    # 仅用重建误差做评分（平坦 VAE 的 8-dim 潜码太低维，Mahalanobis 不稳定）
    mu_rec, sd_rec = tr_rec.mean(), max(tr_rec.std(), 1e-9)
    tr_scores = (tr_rec - mu_rec) / sd_rec  # z-score 化的重建误差
    # 注意方向：重建误差越高越异常 → 分数越高越异常
    # 阈值取训练正常样本的 97.5% 分位数
    threshold = np.quantile(tr_scores, THRESH_QUANTILE)
    print(f"  训练基准样本数: {len(tr_scores)}")
    print(f"  阈值 (q={THRESH_QUANTILE}): {threshold:.4f}")

    # 测试
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
        te_scores = (te_rec - mu_rec) / sd_rec
        # 高于阈值 → 异常(pred=1)
        preds = (te_scores > threshold).astype(int)

        n_abn = int(preds.sum())
        print(f"  [{display_name}] 总数={len(preds)} | 检出异常={n_abn} ({n_abn/len(preds)*100:.1f}%)")

        all_yt.extend([label] * len(preds))
        all_yp.extend(preds.tolist())

        if is_fault: lat_tf_list.append(te_lat)
        else:        lat_tn_list.append(te_lat)

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

    report = classification_report(y_true, y_pred, target_names=["正常", "故障"], digits=4, zero_division=0)
    (OUT_ROOT / "classification_report.txt").write_text(report, encoding="utf-8")
    print(report)

    # ──── 可视化 ────
    print("\n[4/4] 生成可视化 ...")
    plot_confusion_matrix(y_true, y_pred, OUT_ROOT)

    lat_tn = np.vstack(lat_tn_list) if lat_tn_list else np.empty((0, tr_lat.shape[1]))
    lat_tf = np.vstack(lat_tf_list) if lat_tf_list else np.empty((0, tr_lat.shape[1]))
    plot_tsne(tr_lat, lat_tn, lat_tf, OUT_ROOT)

    # 汇总 CSV（字段名与 merge_table3.py 对齐）
    csv_path = OUT_ROOT / "summary_flatvae.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["method", "paradigm", "acc", "pre_n", "rec_n", "rec_f", "macro_f1"])
        w.writeheader()
        w.writerow(dict(method="FlatVAE (GAP, no spatial)", paradigm="One-class",
                        acc=f"{acc:.4f}", pre_n=f"{pn:.4f}", rec_n=f"{rn:.4f}",
                        rec_f=f"{rf:.4f}", macro_f1=f"{mf1:.4f}"))
    print(f"  CSV → {csv_path}")

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
