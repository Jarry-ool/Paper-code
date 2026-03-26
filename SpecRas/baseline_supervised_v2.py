# -*- coding: utf-8 -*-
"""
baseline_supervised.py  —  放在 zerone 目录下运行 (v2 修正版)
================================================================================
基线 (a): RawVec-ResNet18 — 不做光栅化，1,200 维向量纯 reshape+tile → ResNet18
基线 (c): 1D-CNN (Wen 2018) — 输入原始时域波形 8,192 点（忠实于原文定义）

修正要点（相对 v1）：
  1. (a) 去掉可学习适配层（Conv+Upsample），改用不可学习的纯 reshape+tile；
     这样消融对照更纯粹——唯一区别就是"有无光栅化编码"。
  2. (c) 1D-CNN 输入改为能量加权后的原始时域波形（8,192 维），而非预提取的
     1,200 维频谱特征。Wen 2018 原文是在原始信号上做 1D 卷积。

★ 仅依赖 zerone 侧：config.py / zerone_config.py / zerone_features.py
★ 不导入 hetero_*

运行：
    cd <zerone 工作目录>
    python baseline_supervised.py
================================================================================
"""

from __future__ import annotations
import os, sys, json, csv, random, warnings
from pathlib import Path
from copy import deepcopy
from collections import Counter
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, precision_score, recall_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 200; plt.rcParams["savefig.dpi"] = 200

# ── zerone 侧依赖 ──
from config import TRAIN_DIRS, VAL_DIRS, TEST_DIRS, FS, CLASS_MAP
from zerone_config import SEED, CLASSES
from zerone_features import (
    TOTAL_FEAT_DIM, build_sample_from_multichannel, normalize_features,
    energy_weighted_signal,
)

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════════
# 0. 全局设置
# ════════════════════════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_ROOT = Path("./baseline_supervised_results").resolve()
OUT_ROOT.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

CLASS_NAMES = list(CLASS_MAP.keys())
NUM_CLASSES = len(CLASS_NAMES)

# ════════════════════════════════════════════════════════════════════════════
# 1. JSON 解析工具
# ════════════════════════════════════════════════════════════════════════════
def _read_all_records(fp: Path) -> List[dict]:
    text = fp.read_text(encoding="utf-8", errors="ignore")
    if fp.suffix.lower() == ".jsonl":
        recs = []
        for line in text.splitlines():
            s = line.strip()
            if not s: continue
            try: recs.append(json.loads(s))
            except: continue
        return recs
    try: data = json.loads(text)
    except: return []
    if isinstance(data, list): return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict):
        for k in ("data", "records", "list", "items"):
            v = data.get(k)
            if isinstance(v, list): return [d for d in v if isinstance(d, dict)]
    return []

def _parse_signal(v, T=8192):
    try:
        if isinstance(v, str):
            s = v.replace("[","").replace("]","").replace("\n","")
            arr = np.array([float(p) for p in s.split(",") if p.strip()], dtype=np.float64)
        elif isinstance(v, (list, tuple)):
            arr = np.array([float(x) for x in v], dtype=np.float64)
        else: return None
    except: return None
    if arr.size >= T: return arr[:T]
    out = np.zeros(T, dtype=np.float64); out[:arr.size] = arr; return out

def _pick_timekey(rec):
    for k in ("data_time","dataTime","timestamp","acq_time","time"):
        if k in rec and rec[k] is not None: return str(rec[k])
    return None

def _iter_samples(split_dirs):
    """迭代 yield (signals_list, label)。"""
    for cls_name, file_list in split_dirs.items():
        label = CLASS_MAP[cls_name]
        for fp_str in file_list:
            fp = Path(fp_str)
            if not fp.exists(): continue
            recs = _read_all_records(fp)
            groups: Dict[str, List[np.ndarray]] = {}
            for r in recs:
                tk = _pick_timekey(r)
                if not tk: continue
                sig = _parse_signal(r.get("signal_value"))
                if sig is not None: groups.setdefault(tk, []).append(sig)
            for tk, sigs in groups.items():
                yield sigs, label


# ════════════════════════════════════════════════════════════════════════════
# 2. 数据加载
# ════════════════════════════════════════════════════════════════════════════

def load_feature_vectors(split_dirs):
    """加载 1,200-dim 频谱特征向量（用于 RawVec-ResNet18）。"""
    all_X, all_y = [], []
    for sigs, label in _iter_samples(split_dirs):
        feat, _ = build_sample_from_multichannel(
            signals=sigs, fs=FS, agg_method="energy",
            include_empirical=False, return_empirical=False)
        if feat is None: continue
        v = np.zeros(TOTAL_FEAT_DIM, dtype=np.float32)
        v[:min(feat.size, TOTAL_FEAT_DIM)] = feat[:TOTAL_FEAT_DIM]
        all_X.append(v); all_y.append(label)
    if not all_X:
        return np.empty((0, TOTAL_FEAT_DIM), np.float32), np.empty(0, np.int64)
    return np.stack(all_X), np.array(all_y, np.int64)


def load_raw_waveforms(split_dirs, T=8192):
    """
    加载能量加权后的原始时域波形（用于 1D-CNN）。
    每个样本做 z-score 归一化。
    """
    all_X, all_y = [], []
    for sigs, label in _iter_samples(split_dirs):
        x_list = [np.asarray(s, dtype=np.float64)[:T] for s in sigs]
        if len(x_list) == 0: continue
        if len(x_list) == 1:
            x_fused = x_list[0]
        else:
            X2d = np.stack(x_list, axis=1)
            x_fused, _ = energy_weighted_signal(X2d)
        mu, sd = x_fused.mean(), x_fused.std()
        if sd > 1e-8: x_fused = (x_fused - mu) / sd
        v = np.zeros(T, dtype=np.float32)
        v[:min(x_fused.size, T)] = x_fused[:T].astype(np.float32)
        all_X.append(v); all_y.append(label)
    if not all_X:
        return np.empty((0, T), np.float32), np.empty(0, np.int64)
    return np.stack(all_X), np.array(all_y, np.int64)


class VecDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ════════════════════════════════════════════════════════════════════════════
# 3. 模型定义
# ════════════════════════════════════════════════════════════════════════════

class RawVecResNet18(nn.Module):
    """
    (a) 消融对照：不做光栅化，纯 reshape + tile → ResNet18。
    1,200-dim → reshape (1,30,40) → repeat tile (1,150,160) → 裁切 (1,150,150)
    → 灰度扩展 3 通道 → ResNet18。
    ★ 无可学习适配层。
    """
    def __init__(self, feat_dim=TOTAL_FEAT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.h0, self.w0 = 30, 40
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self._feats = []
        self._hook = self.backbone.avgpool.register_forward_hook(self._hook_fn)

    def _hook_fn(self, m, inp, out):
        self._feats.append(torch.flatten(out, 1).detach().cpu().numpy())

    def forward(self, x):
        B = x.size(0)
        img = x.view(B, 1, self.h0, self.w0)
        img = img.repeat(1, 1, 5, 4)[:, :, :150, :150]
        img = img.expand(-1, 3, -1, -1)
        return self.backbone(img)


class SpectralCNN1D(nn.Module):
    """
    (c) Wen et al. (2018) 风格 1D-CNN，输入 1,200-dim 频谱特征向量。

    与 SpecRas 使用完全相同的 1,200 维输入特征，区别仅在于信号表示：
      - SpecRas: 1,200-dim → 2D 光栅图像 → 2D CNN (ResNet18)
      - 本基线: 1,200-dim → 1D 序列 → 1D CNN

    这种设定确保了公平对比：输入信息量相同，只比信号表示方式。
    Wen 2018 原文也是在频域特征（振动频谱）上做 1D 卷积分类。

    架构（按 Wen 2018 的多层 Conv-Pool 风格）：
      Conv1d(1→16,  k=7, s=1, p=3) → BN → ReLU → MaxPool(4)   → 300
      Conv1d(16→32, k=5, s=1, p=2) → BN → ReLU → MaxPool(4)   → 75
      AdaptiveAvgPool(1) → FC(32→16) → ReLU → Dropout(0.6) → FC(16→2)
    """
    def __init__(self, in_dim=TOTAL_FEAT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(16), nn.ReLU(True), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(True), nn.MaxPool1d(4),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 16), nn.ReLU(True), nn.Dropout(0.6),
            nn.Linear(16, num_classes),
        )
        self._feats = []
        self._hook = self.features[-1].register_forward_hook(self._hook_fn)

    def _hook_fn(self, m, inp, out):
        self._feats.append(torch.flatten(out, 1).detach().cpu().numpy())

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)  # (B,1,1200)
        h = self.features(x)
        return self.classifier(h)


# ════════════════════════════════════════════════════════════════════════════
# 4. 可视化工具
# ════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, y_pred, save_dir, tag):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    for lang, labels, xl, yl, title in [
        ("en", ["Normal","Fault"], "Predicted", "True", f"Confusion Matrix — {tag}"),
        ("zh", ["正常","故障"], "预测类别", "真实类别", f"混淆矩阵 — {tag}"),
    ]:
        fig, ax = plt.subplots(figsize=(5,4.2))
        ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title, fontweight="bold")
        for i in range(2):
            for j in range(2):
                c = "white" if cm_pct[i,j]>0.5 else "black"
                ax.text(j, i-0.05, f"{cm[i,j]}", ha="center", va="center", color=c, fontsize=16, fontweight="bold")
                ax.text(j, i+0.18, f"({cm_pct[i,j]:.1%})", ha="center", va="center", color=c, fontsize=9, alpha=0.7)
        fig.tight_layout(); fig.savefig(save_dir/f"cm_{tag}_{lang}.png"); plt.close(fig)

def plot_roc_pr(y_true, prob, save_dir, tag):
    Y = np.eye(NUM_CLASSES)[y_true]
    for lang, roc_tl, pr_tl, fpr_l, tpr_l, prec_l, rec_l in [
        ("en", f"ROC — {tag}", f"PR — {tag}", "FPR","TPR","Precision","Recall"),
        ("zh", f"ROC 曲线 — {tag}", f"PR 曲线 — {tag}", "假阳性率","真阳性率","精确率","召回率"),
    ]:
        cls_labels = ["Normal","Fault"] if lang=="en" else ["正常","故障"]
        fig,ax=plt.subplots(figsize=(5,4.5))
        for k in range(NUM_CLASSES):
            fpr,tpr,_=roc_curve(Y[:,k],prob[:,k]); a=auc(fpr,tpr)
            ax.plot(fpr,tpr,label=f"{cls_labels[k]} (AUC={a:.3f})")
        ax.plot([0,1],[0,1],"k--",lw=0.8)
        ax.set_xlabel(fpr_l);ax.set_ylabel(tpr_l);ax.set_title(roc_tl,fontweight="bold")
        ax.legend();ax.grid(True,alpha=0.3);fig.tight_layout()
        fig.savefig(save_dir/f"roc_{tag}_{lang}.png");plt.close(fig)
        fig,ax=plt.subplots(figsize=(5,4.5))
        for k in range(NUM_CLASSES):
            p_,r_,_=precision_recall_curve(Y[:,k],prob[:,k])
            ap=average_precision_score(Y[:,k],prob[:,k])
            ax.plot(r_,p_,drawstyle="steps-post",label=f"{cls_labels[k]} (AP={ap:.3f})")
        ax.set_xlim(0,1);ax.set_ylim(0,1.05)
        ax.set_xlabel(rec_l);ax.set_ylabel(prec_l);ax.set_title(pr_tl,fontweight="bold")
        ax.legend();ax.grid(True,alpha=0.3);fig.tight_layout()
        fig.savefig(save_dir/f"pr_{tag}_{lang}.png");plt.close(fig)

def plot_tsne(feats, y_true, save_dir, tag):
    if feats.shape[0]<5: return
    Z=TSNE(n_components=2,perplexity=min(30,feats.shape[0]-1),
           init="pca",random_state=SEED,max_iter=1000).fit_transform(feats)
    colors={0:"#3B76AF",1:"#D62728"}
    for lang,cls_labels,tl in [
        ("en",{0:"Normal",1:"Fault"},f"t-SNE — {tag}"),
        ("zh",{0:"正常",1:"故障"},f"t-SNE — {tag}"),
    ]:
        fig,ax=plt.subplots(figsize=(6,5))
        for c in [0,1]:
            m=y_true==c
            ax.scatter(Z[m,0],Z[m,1],s=30,c=colors[c],alpha=0.75,
                       edgecolors="white",linewidths=0.4,label=f"{cls_labels[c]} (n={m.sum()})")
        ax.legend(loc="best");ax.set_title(tl,fontweight="bold")
        ax.set_xticks([]);ax.set_yticks([])
        for sp in ax.spines.values():sp.set_visible(False)
        fig.tight_layout();fig.savefig(save_dir/f"tsne_{tag}_{lang}.png");plt.close(fig)

def plot_reliability(y_true,y_pred,prob,save_dir,tag):
    pred_idx=np.array(y_pred)
    conf=prob[np.arange(len(y_pred)),pred_idx]
    correct=(np.array(y_pred)==np.array(y_true)).astype(int)
    n_bins=10;bins=np.linspace(0,1,n_bins+1);bid=np.digitize(conf,bins)-1
    accs,confs,ns=[],[],[]
    for b in range(n_bins):
        m=bid==b
        if m.sum()==0: accs.append(0.0);confs.append((bins[b]+bins[b+1])/2);ns.append(0)
        else: accs.append(correct[m].mean());confs.append(conf[m].mean());ns.append(int(m.sum()))
    N=len(y_pred);ece=sum((ns[i]/N)*abs(accs[i]-confs[i]) for i in range(n_bins))
    for lang,xl,yl,tl in [
        ("en","Confidence","Accuracy",f"Reliability — {tag} (ECE={ece:.3f})"),
        ("zh","置信度","准确率",f"可靠性曲线 — {tag} (ECE={ece:.3f})"),
    ]:
        fig,ax=plt.subplots(figsize=(5,4.5))
        ax.plot([0,1],[0,1],"k--",lw=0.8)
        ax.bar(confs,accs,width=0.9/n_bins,edgecolor="k",alpha=0.6)
        ax.set_xlabel(xl);ax.set_ylabel(yl);ax.set_title(tl,fontweight="bold")
        ax.grid(True,alpha=0.3);fig.tight_layout()
        fig.savefig(save_dir/f"reliability_{tag}_{lang}.png");plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════
# 5. 训练 + 评估
# ════════════════════════════════════════════════════════════════════════════

def _train_and_eval(model, tag, dl_tr, dl_va, dl_te, lr=8e-5, epochs=30, patience=10):
    out_dir = OUT_ROOT / tag; out_dir.mkdir(parents=True, exist_ok=True)
    all_y = []
    for _,yb in dl_tr: all_y.extend(yb.numpy().tolist())
    counts=Counter(all_y);total=sum(counts.values())
    weights=torch.tensor([total/(NUM_CLASSES*counts.get(i,1)) for i in range(NUM_CLASSES)],dtype=torch.float32).to(DEVICE)

    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",patience=5,factor=0.5)
    best_val_f1=-1.0;best_state=None;wait=0

    print(f"\n{'─'*60}\n  Training: {tag}   (device={DEVICE})\n{'─'*60}")

    for epoch in range(1,epochs+1):
        model.train(); model._feats.clear(); running_loss=0.0
        for xb,yb in dl_tr:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE)
            logits=model(xb)
            loss=F.cross_entropy(logits,yb,weight=weights,label_smoothing=0.05)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.0);optimizer.step()
            running_loss+=loss.item()
        model._feats.clear()

        if dl_va:
            model.eval();model._feats.clear();yt,yp=[],[]
            with torch.no_grad():
                for xb,yb in dl_va:
                    pred=model(xb.to(DEVICE)).argmax(1).cpu().numpy()
                    yt.extend(yb.numpy().tolist());yp.extend(pred.tolist())
            model._feats.clear()
            vf1=f1_score(yt,yp,average="macro",zero_division=0)
            vacc=accuracy_score(yt,yp)
            scheduler.step(vf1)
            if vf1>best_val_f1+1e-4: best_val_f1=vf1;best_state=deepcopy(model.state_dict());wait=0
            else: wait+=1
            if epoch%5==0 or epoch==1:
                print(f"  [E{epoch:02d}]  loss={running_loss/len(dl_tr):.4f}  val_acc={vacc:.3f}  val_f1={vf1:.3f}")
            if wait>=patience: print(f"  [EARLY-STOP] epoch {epoch}");break

    if best_state: model.load_state_dict(best_state)

    if dl_te is None: print("[WARN] 无测试集"); return None
    model.eval();model._feats.clear()
    y_true_all,y_pred_all,prob_all=[],[],[]
    with torch.no_grad():
        for xb,yb in dl_te:
            logits=model(xb.to(DEVICE))
            prob=F.softmax(logits,dim=1).cpu().numpy()
            pred=logits.argmax(1).cpu().numpy()
            y_true_all.extend(yb.numpy().tolist());y_pred_all.extend(pred.tolist());prob_all.append(prob)
    feats_te=np.concatenate(model._feats,axis=0) if model._feats else np.empty((0,1))
    model._feats.clear()

    y_true=np.array(y_true_all);y_pred=np.array(y_pred_all);prob=np.vstack(prob_all)
    acc=accuracy_score(y_true,y_pred)
    pn=precision_score(y_true,y_pred,pos_label=0,zero_division=0)
    rn=recall_score(y_true,y_pred,pos_label=0,zero_division=0)
    rf=recall_score(y_true,y_pred,pos_label=1,zero_division=0)
    mf1=f1_score(y_true,y_pred,average="macro",zero_division=0)

    metrics=dict(method=tag,paradigm="Supervised",acc=acc,pre_n=pn,rec_n=rn,rec_f=rf,macro_f1=mf1)
    print(f"\n  ┌── {tag} ──")
    print(f"  │ Acc    = {acc*100:.2f}%")
    print(f"  │ Pre(N) = {pn:.3f}   Rec(N) = {rn:.3f}   Rec(F) = {rf:.3f}")
    print(f"  │ Macro-F1 = {mf1:.3f}")
    print(f"  └{'─'*40}\n")
    report=classification_report(y_true,y_pred,target_names=CLASS_NAMES,digits=4,zero_division=0)
    (out_dir/"classification_report.txt").write_text(report,encoding="utf-8")
    print(report)

    print(f"  [VIZ] 生成可视化 → {out_dir}")
    plot_confusion_matrix(y_true,y_pred,out_dir,tag)
    plot_roc_pr(y_true,prob,out_dir,tag)
    plot_reliability(y_true,y_pred,prob,out_dir,tag)
    if feats_te.shape[0]>5 and feats_te.shape[1]>1:
        plot_tsne(feats_te,y_true,out_dir,tag)

    with open(out_dir/"predictions.csv","w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f);w.writerow(["idx","y_true","y_pred","prob_normal","prob_fault","correct"])
        for i in range(len(y_true)):
            w.writerow([i,y_true[i],y_pred[i],f"{prob[i,0]:.4f}",f"{prob[i,1]:.4f}",int(y_true[i]==y_pred[i])])

    return metrics


# ════════════════════════════════════════════════════════════════════════════
# 6. 主入口
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("="*64)
    print("  W1 基线实验 (a)(c) — 有监督基线 v2")
    print(f"  输出目录: {OUT_ROOT}")
    print(f"  设备: {DEVICE}")
    print("="*64)

    results = []

    # ════════════ (a) RawVec-ResNet18 ════════════
    print("\n[1/4] 加载 1,200-dim 特征向量 ...")
    X_tr, y_tr = load_feature_vectors(TRAIN_DIRS)
    X_va, y_va = load_feature_vectors(VAL_DIRS)
    X_te, y_te = load_feature_vectors(TEST_DIRS)
    print(f"  Train={X_tr.shape[0]}  Val={X_va.shape[0]}  Test={X_te.shape[0]}")
    if X_tr.shape[0] == 0:
        print("[ERROR] 训练集为空"); return

    tr_min, tr_max = X_tr.min(axis=0), X_tr.max(axis=0)
    X_tr_n = normalize_features(X_tr, minv=tr_min, maxv=tr_max)
    X_va_n = normalize_features(X_va, minv=tr_min, maxv=tr_max) if X_va.shape[0]>0 else X_va
    X_te_n = normalize_features(X_te, minv=tr_min, maxv=tr_max) if X_te.shape[0]>0 else X_te

    dl_tr_a = DataLoader(VecDataset(X_tr_n,y_tr), batch_size=16, shuffle=True,  num_workers=0)
    dl_va_a = DataLoader(VecDataset(X_va_n,y_va), batch_size=32, shuffle=False, num_workers=0) if X_va.shape[0]>0 else None
    dl_te_a = DataLoader(VecDataset(X_te_n,y_te), batch_size=32, shuffle=False, num_workers=0) if X_te.shape[0]>0 else None

    print("\n[2/4] 训练基线 (a): RawVec-ResNet18 (纯 reshape+tile, 无可学习适配) ...")
    model_a = RawVecResNet18().to(DEVICE)
    m_a = _train_and_eval(model_a, "RawVec_ResNet18", dl_tr_a, dl_va_a, dl_te_a, lr=8e-5, epochs=30)
    if m_a: results.append(m_a)

    # ════════════ (c) 1D-CNN on same 1200-dim features ════════════
    # 复用已加载的特征向量（与 RawVec-ResNet18 完全相同的输入）
    # 对比的是"1D 卷积 vs 2D 光栅化 + ResNet18"的信号表示方式差异
    dl_tr_c = DataLoader(VecDataset(X_tr_n,y_tr), batch_size=32, shuffle=True,  num_workers=0)
    dl_va_c = DataLoader(VecDataset(X_va_n,y_va), batch_size=32, shuffle=False, num_workers=0) if X_va.shape[0]>0 else None
    dl_te_c = DataLoader(VecDataset(X_te_n,y_te), batch_size=32, shuffle=False, num_workers=0) if X_te.shape[0]>0 else None

    print("\n[4/4] 训练基线 (c): 1D-CNN on 1200-dim spectral features (Wen 2018 style) ...")
    model_c = SpectralCNN1D(in_dim=TOTAL_FEAT_DIM).to(DEVICE)
    m_c = _train_and_eval(model_c, "1DCNN_Wen2018", dl_tr_c, dl_va_c, dl_te_c, lr=5e-4, epochs=20)
    if m_c: results.append(m_c)

    # 汇总
    csv_path = OUT_ROOT / "summary_supervised_baselines.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["method","paradigm","acc","pre_n","rec_n","rec_f","macro_f1"])
        w.writeheader()
        for r in results:
            w.writerow({k:(f"{r[k]:.4f}" if isinstance(r[k],float) else r[k]) for k in w.fieldnames})
    print(f"\n  汇总 CSV → {csv_path}")
    print("\n" + "="*64 + "\n  有监督基线实验完成！\n" + "="*64)


if __name__ == "__main__":
    main()
