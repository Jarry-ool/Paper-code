# -*- coding: utf-8 -*-
"""
hetero_diagnose.py (Final Polish: IEEE/Nature Style Visualization)
无监督异常诊断 + 出版级双语图表
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

import hetero_config as cfg
from hetero_model import SpatialResNetVAE
from hetero_data import TransformerVibrationDataset

# ==========================================
# 1. 全局绘图风格设置 (IEEE/Nature Standard)
# ==========================================
# 字体优先级：微软雅黑(Win) > 黑体(Win/Linux) > 宋体 > Arial
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# 画布与线条
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.linewidth'] = 1.0     # 坐标轴线宽 (Nature要求0.5~1.0)
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['font.size'] = 10           # 基础字号
plt.rcParams['font.family'] = 'sans-serif' # 科技论文首选无衬线字体

# 配色盘 (Nature 风格)
# 蓝色(正常/分布): #3B76AF, 红色(故障/阈值): #D62728, 灰色(背景): #E0E0E0
C_NORM = "#3B76AF"
C_ANOM = "#D62728"
C_BG   = "#E0E0E0"

ALPHA = 0.6      # 重构分数权重
USE_PERC = 0.975 # 阈值分位数 (97.5%)
USE_SIGMA = False

# ==========================================
# 2. 核心计算逻辑 (保持不变)
# ==========================================
def _channel_weighted_l1(recon, inp, w=(0.4, 0.5, 0.1)):
    """计算通道加权 L1 误差"""
    e0 = torch.mean(torch.abs(recon[:, 0] - inp[:, 0]), dim=[1, 2])
    e1 = torch.mean(torch.abs(recon[:, 1] - inp[:, 1]), dim=[1, 2])
    e2 = torch.mean(torch.abs(recon[:, 2] - inp[:, 2]), dim=[1, 2])
    w0, w1, w2 = w
    return (w0 * e0 + w1 * e1 + w2 * e2).detach().cpu().numpy()

def _collect_scores(model, loader, device):
    """推理并收集分数与隐向量"""
    model.eval()
    rec_scores = []
    latents = []
    with torch.no_grad():
        for imgs in tqdm(loader, desc="Scoring", leave=False):
            imgs = imgs.to(device)
            recon, mu, _ = model(imgs)
            rec = _channel_weighted_l1(recon, imgs)
            rec_scores.append(rec)
            z = torch.mean(mu, dim=(2, 3)).detach().cpu().numpy()
            latents.append(z)
    return np.concatenate(rec_scores), np.vstack(latents)

def _fit_mahalanobis(latents):
    m = latents.mean(axis=0)
    cov = np.cov(latents.T) + 1e-6 * np.eye(latents.shape[1])
    inv = np.linalg.pinv(cov)
    return m, inv

def _mahalanobis(latents, mean, inv_cov):
    diff = latents - mean[None, :]
    dist2 = np.einsum("bi,ij,bj->b", diff, inv_cov, diff)
    return np.sqrt(np.maximum(dist2, 0.0))

def _zscore(arr):
    mu, sd = arr.mean(), arr.std()
    sd = sd if sd > 1e-9 else 1.0
    return (arr - mu) / sd, mu, sd

def _decide_threshold(train_scores):
    if USE_SIGMA:
        mu, sd = train_scores.mean(), train_scores.std()
        return mu + 3 * sd
    else:
        return np.quantile(train_scores, USE_PERC)

def _maybe_build_loader(root: Path, only_normal: bool, mode: str):
    if not root.exists(): return None
    ds = TransformerVibrationDataset(root, mode=mode, only_normal=only_normal)
    if len(ds) == 0: return None
    return DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

# ==========================================
# 3. 高级可视化函数 (IEEE/Nature 风格)
# ==========================================

def plot_score_histogram_dual(scores, threshold, name, save_dir: Path):
    """
    生成 IEEE/Nature 风格的直方图 (中英双语版)
    """
    # 基础配置
    configs = [
        {
            'lang': 'EN',
            'xlabel': 'Anomaly Score',
            'ylabel': 'Density',
            'title': f'Score Distribution ({name})',
            'legend': ['Score Dist.', 'Threshold'],
            'fname': f"diagnosis_hist_{name}_EN.png"
        },
        {
            'lang': 'CN',
            'xlabel': '异常评分 (Anomaly Score)',
            'ylabel': '概率密度 (Density)',
            'title': f'异常评分分布直方图 ({name})',
            'legend': ['评分分布', '判决阈值'],
            'fname': f"diagnosis_hist_{name}_CN.png"
        }
    ]

    for cfg_plot in configs:
        plt.figure(figsize=(6, 4))
        
        # 1. 绘制直方图 + KDE 曲线
        # 使用 seaborn histplot，颜色为 Nature Blue，半透明填充
        sns.histplot(scores, kde=True, stat="density", color=C_NORM, 
                     edgecolor='white', linewidth=0.5, alpha=0.7, label=cfg_plot['legend'][0])
        
        # 2. 绘制阈值线
        plt.axvline(threshold, color=C_ANOM, linestyle='--', linewidth=1.5, 
                    label=f"{cfg_plot['legend'][1]}: {threshold:.2f}")
        
        # 3. 美化细节 (IEEE Style)
        plt.title(cfg_plot['title'], fontsize=12, fontweight='bold', pad=12)
        plt.xlabel(cfg_plot['xlabel'], fontsize=11)
        plt.ylabel(cfg_plot['ylabel'], fontsize=11)
        plt.legend(loc='upper right', frameon=True, fontsize=9)
        
        # 去除上方和右侧的边框 (Despine)
        sns.despine()
        
        # 添加微弱的网格线
        plt.grid(axis='y', linestyle=':', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / cfg_plot['fname'])
        plt.close()

def plot_confusion_matrix_dual(y_true, y_pred, save_dir: Path):
    """
    生成 IEEE/Nature 风格的混淆矩阵 (中英双语版)
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    configs = [
        {
            'lang': 'EN',
            'labels': ['Normal', 'Fault'],
            'xlabel': 'Predicted Class',
            'ylabel': 'True Class',
            'title': 'Confusion Matrix',
            'fname': 'diagnosis_confusion_EN.png'
        },
        {
            'lang': 'CN',
            'labels': ['正常', '故障'],
            'xlabel': '预测类别',
            'ylabel': '真实类别',
            'title': '混淆矩阵',
            'fname': 'diagnosis_confusion_CN.png'
        }
    ]

    for cfg_plot in configs:
        plt.figure(figsize=(5, 4.2))
        
        # 使用 seaborn heatmap, 颜色使用 Blues
        ax = sns.heatmap(
            cm, annot=False, fmt='d', cmap='Blues', cbar=False,
            xticklabels=cfg_plot['labels'], yticklabels=cfg_plot['labels'],
            linewidths=1, linecolor='black', clip_on=False
        )
        
        # 手动添加文字，确保格式美观 (数量 + 百分比)
        for i in range(2):
            for j in range(2):
                count = cm[i, j]
                percent = cm_norm[i, j]
                color = 'white' if percent > 0.5 else 'black'
                
                # 第一行：数量
                ax.text(j + 0.5, i + 0.45, f"{count}", 
                        ha='center', va='center', color=color, fontsize=16, fontweight='bold')
                # 第二行：百分比
                ax.text(j + 0.5, i + 0.65, f"({percent:.1%})", 
                        ha='center', va='center', color=color, fontsize=10, alpha=0.7)

        plt.xlabel(cfg_plot['xlabel'], fontsize=11, fontweight='bold', labelpad=10)
        plt.ylabel(cfg_plot['ylabel'], fontsize=11, fontweight='bold', labelpad=10)
        plt.title(cfg_plot['title'], fontsize=13, fontweight='bold', pad=15)
        
        # 调整刻度样式
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10, rotation=0)
        
        plt.tight_layout()
        plt.savefig(save_dir / cfg_plot['fname'])
        plt.close()

def plot_tsne_dual(latents_train, latents_test_norm, latents_test_fault, save_dir: Path):
    """
    生成 IEEE/Nature 风格的 t-SNE 散点图 (中英双语版)
    """
    if latents_train.size == 0 or (latents_test_norm.size == 0 and latents_test_fault.size == 0):
        return

    print(">> 计算 t-SNE (Rendering t-SNE)...")
    # 下采样
    n_train = min(len(latents_train), 1000)
    idx_train = np.random.choice(len(latents_train), n_train, replace=False)

    X_list = [latents_train[idx_train]]
    labels_list = [np.zeros(n_train, dtype=int)]

    if latents_test_norm.size > 0:
        X_list.append(latents_test_norm)
        labels_list.append(np.ones(len(latents_test_norm), dtype=int))
    if latents_test_fault.size > 0:
        X_list.append(latents_test_fault)
        labels_list.append(np.full(len(latents_test_fault), 2, dtype=int))

    X = np.vstack(X_list)
    labels = np.concatenate(labels_list)

    # 计算 t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_emb = tsne.fit_transform(X)

    configs = [
        {
            'lang': 'EN',
            'l1': 'Train (Baseline)', 'l2': 'Test (Normal)', 'l3': 'Test (Fault)',
            'title': 'Latent Space Manifold (t-SNE)', 'fname': 'diagnosis_tsne_EN.png'
        },
        {
            'lang': 'CN',
            'l1': '训练集 (基准)', 'l2': '测试集 (正常)', 'l3': '测试集 (故障)',
            'title': '隐空间特征流形分布', 'fname': 'diagnosis_tsne_CN.png'
        }
    ]

    for cfg_plot in configs:
        plt.figure(figsize=(7, 6))
        
        # 绘制散点，使用 alpha 混合效果
        # 1. 灰色背景 (Train)
        plt.scatter(X_emb[labels == 0, 0], X_emb[labels == 0, 1],
                    c='#DDDDDD', label=cfg_plot['l1'], alpha=0.5, s=30, edgecolors='none') # 灰色背景
        
        # 2. 蓝色 (Test Normal)
        if np.any(labels == 1):
            plt.scatter(X_emb[labels == 1, 0], X_emb[labels == 1, 1],
                        c=C_NORM, label=cfg_plot['l2'], alpha=0.8, s=40, marker='o', edgecolors='white', linewidth=0.5)
        
        # 3. 红色 (Test Fault)
        if np.any(labels == 2):
            plt.scatter(X_emb[labels == 2, 0], X_emb[labels == 2, 1],
                        c=C_ANOM, label=cfg_plot['l3'], alpha=0.9, s=50, marker='X', edgecolors='white', linewidth=0.5)

        # 美化
        plt.legend(fontsize=9, frameon=True, loc='best')
        plt.title(cfg_plot['title'], fontsize=13, fontweight='bold')
        plt.xlabel("t-SNE Dimension 1", fontsize=10)
        plt.ylabel("t-SNE Dimension 2", fontsize=10)
        
        # 去除刻度值，只保留边框
        plt.xticks([])
        plt.yticks([])
        sns.despine()
        
        plt.tight_layout()
        plt.savefig(save_dir / cfg_plot['fname'])
        plt.close()

def analyze_failure_cases(model, loader, thr, mu_rec, sd_rec, mu_md, sd_md,
                          mean_mu, inv_cov, device, save_dir: Path,
                          case_type: str, split_name: str):
    """
    溯源分析 (保持原有的 3x3 绘图逻辑，仅微调风格)
    """
    model.eval()
    with torch.no_grad():
        for imgs in loader:
            imgs = imgs.to(device)
            recon, mu_tensor, _ = model(imgs)

            rec_err = _channel_weighted_l1(recon, imgs)
            z_latent = torch.mean(mu_tensor, dim=(2, 3)).detach().cpu().numpy()
            md = _mahalanobis(z_latent, mean_mu, inv_cov)

            zrec = (rec_err - mu_rec) / (sd_rec if sd_rec > 1e-9 else 1.0)
            zmd  = (md      - mu_md)  / (sd_md  if sd_md  > 1e-9 else 1.0)
            scores = ALPHA * zrec + (1 - ALPHA) * zmd

            if case_type == "False_Positive":
                idxs = np.where(scores < thr)[0]
            elif case_type == "False_Negative":
                idxs = np.where(scores >= thr)[0]
            else:
                return

            if len(idxs) == 0: continue

            # 取第一个错误样本
            idx = int(idxs[0])
            orig_np = imgs[idx].cpu().numpy()
            recon_np = recon[idx].cpu().numpy()
            diff_np = np.abs(orig_np - recon_np)
            score = float(scores[idx])

            configs = [
                {'lang': 'EN', 't': 'Failure Analysis', 'fname': f"failure_{case_type}_EN_{split_name}.png"},
                {'lang': 'CN', 't': '错误溯源分析', 'fname': f"failure_{case_type}_CN_{split_name}.png"}
            ]
            
            # 使用简单的绘图逻辑，无需过度美化，保证信息清晰即可
            for cfg_plot in configs:
                fig, axes = plt.subplots(3, 3, figsize=(10, 10))
                titles = ['CWT', 'Zerone', 'Context']
                cols = ['Input', 'Recon', 'Residual']
                
                for c in range(3):
                    axes[c, 0].imshow(orig_np[c], cmap='jet', aspect='auto')
                    axes[c, 0].set_ylabel(titles[c], fontweight='bold')
                    if c==0: axes[c, 0].set_title(cols[0])
                    
                    axes[c, 1].imshow(recon_np[c], cmap='jet', aspect='auto')
                    if c==0: axes[c, 1].set_title(cols[1])
                    
                    axes[c, 2].imshow(diff_np[c], cmap='inferno', aspect='auto')
                    if c==0: axes[c, 2].set_title(cols[2])
                
                plt.suptitle(f"{cfg_plot['t']}: {split_name}\nScore: {score:.2f} (Thr: {thr:.2f})")
                plt.tight_layout()
                plt.savefig(save_dir / cfg_plot['fname'])
                plt.close()
            return

# ==========================================
# 4. 主程序
# ==========================================
def diagnose():
    device = torch.device(cfg.DEVICE)
    out_dir = cfg.CHECKPOINT_DIR / "diagnosis_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    # 1. 加载模型
    model_path = cfg.CHECKPOINT_DIR / "model" / "epoch_30.pth" # 优先
    if not model_path.exists(): model_path = cfg.CHECKPOINT_DIR / "model" / "best_model.pth"
    if not model_path.exists(): model_path = cfg.CHECKPOINT_DIR / "model" / "final_model.pth"
    
    print(f"Loading Model: {model_path}")
    model = SpatialResNetVAE(latent_channels=cfg.LATENT_CHANNELS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. 训练集基准
    train_loader = _maybe_build_loader(Path(cfg.TRAIN_DIR), True, "train")
    if not train_loader: return
    
    print(">> Calculating Baseline (Train)...")
    tr_rec, tr_mu = _collect_scores(model, train_loader, device)
    mean_mu, inv_cov = _fit_mahalanobis(tr_mu)
    tr_md = _mahalanobis(tr_mu, mean_mu, inv_cov)
    
    # 统计量
    mu_rec, sd_rec = tr_rec.mean(), tr_rec.std()
    mu_md, sd_md = tr_md.mean(), tr_md.std()
    
    tr_scores = ALPHA * (tr_rec - mu_rec)/sd_rec + (1-ALPHA) * (tr_md - mu_md)/sd_md
    threshold = _decide_threshold(tr_scores)
    
    print(f"Baseline Samples: {len(tr_scores)}")
    print(f"Threshold (Alpha={ALPHA}): {threshold:.4f}\n")

    # 3. 测试集评估
    test_root = Path(cfg.TEST_DIR)
    results = []
    
    # 自动识别子目录
    subdirs = [d for d in test_root.iterdir() if d.is_dir()]
    if not subdirs:
        subdirs = [test_root] 

    all_y_true, all_y_pred = [], []
    l_tn_list, l_tf_list = [], []

    for sub in subdirs:
        # === 修改开始：逻辑重命名与标签定义 ===
        raw_name = sub.name
        
        # 判定逻辑：根据文件夹名包含的关键词，强制改名为标准格式
        # 并据此确定 is_fault (Ground Truth)
        if any(k in raw_name.lower() for k in ["正常", "normal"]):
            name = "test_normal"   # 强制显示为 test_normal
            is_fault = False       # 标签设为 0 (正常)
        elif any(k in raw_name.lower() for k in ["故障", "fault", "异常"]):
            name = "test_fault"    # 强制显示为 test_fault
            is_fault = True        # 标签设为 1 (故障)
        else:
            # 如果既没写正常也没写故障，保留原名或默认为 test_unknown
            name = raw_name        
            is_fault = False
        loader = _maybe_build_loader(sub, False, "test")
        if not loader: continue
        
        # 推理
        rec, mu = _collect_scores(model, loader, device)
        md = _mahalanobis(mu, mean_mu, inv_cov)
        zrec = (rec - mu_rec) / sd_rec
        zmd  = (md - mu_md) / sd_md
        scores = ALPHA * zrec + (1-ALPHA) * zmd
        
        preds = (scores < threshold).astype(int)
        detected = int(preds.sum())
        total = len(scores)
        
        # === 核心输出恢复 ===
        print(f"[{name}] Total={total} | Abnormal={detected} ({detected/total*100:.2f}%)")
        
        # 画直方图 (双语)
        plot_score_histogram_dual(scores, threshold, name, out_dir)
        
        # 收集数据
        is_fault = any(k in name.lower() for k in ["fault", "故障", "异常"])
        label = 1 if is_fault else 0
        
        all_y_true.extend([label]*total)
        all_y_pred.extend(preds)
        
        if is_fault:
            l_tf_list.append(mu)
            # 漏报溯源
            if np.any(preds == 0):
                analyze_failure_cases(model, loader, threshold, mu_rec, sd_rec, mu_md, sd_md, mean_mu, inv_cov, device, out_dir, "False_Negative", name)
        else:
            l_tn_list.append(mu)
            # 误报溯源
            if np.any(preds == 1):
                analyze_failure_cases(model, loader, threshold, mu_rec, sd_rec, mu_md, sd_md, mean_mu, inv_cov, device, out_dir, "False_Positive", name)

    # 4. 全局图表
    print("\n>> Generating Global Plots...")
    if all_y_true:
        plot_confusion_matrix_dual(all_y_true, all_y_pred, out_dir)
        
    l_tr = tr_mu
    l_tn = np.vstack(l_tn_list) if l_tn_list else np.empty((0, tr_mu.shape[1]))
    l_tf = np.vstack(l_tf_list) if l_tf_list else np.empty((0, tr_mu.shape[1]))
    
    if l_tn.shape[0] > 0 or l_tf.shape[0] > 0:
        plot_tsne_dual(l_tr, l_tn, l_tf, out_dir)
        
    print(f"Done. Reports at: {out_dir}")

if __name__ == "__main__":
    diagnose()