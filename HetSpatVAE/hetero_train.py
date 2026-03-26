# -*- coding: utf-8 -*-
"""
hetero_train.py (Updated)
包含训练过程的可视化（Loss曲线 & 学习率曲线）
"""
import os
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import hetero_config as cfg
from hetero_data import TransformerVibrationDataset
from hetero_model import SpatialResNetVAE, loss_function

# --- 绘图设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def plot_training_curves(train_losses, val_losses, lrs, save_dir):
    """绘制训练过程曲线"""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Loss Curve
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss (训练误差)', linewidth=1.5)
    ax1.plot(epochs, val_losses, 'r--', label='Val Loss (验证误差)', linewidth=1.5)
    ax1.set_xlabel('Epochs (轮次)')
    ax1.set_ylabel('Loss (MAE + KL)')
    ax1.set_title('Training & Validation Loss / 训练与验证误差曲线')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()
    
    # 2. LR Curve
    ax2.plot(epochs, lrs, 'g-', label='Learning Rate (学习率)', linewidth=1.5)
    ax2.set_xlabel('Epochs (轮次)')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule / 学习率变化曲线')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=300)
    print(f"Training curves saved to {save_dir / 'training_curves.png'}")
    plt.close()

def train():
    device = torch.device(cfg.DEVICE)
    print(f"正在使用设备: {device.type}")

    train_path = Path(cfg.TRAIN_DIR)
    val_path = Path(cfg.VAL_DIR)
    
    #   判断 VAL_DIR 是否有效 (存在且不为空)
    has_explicit_val = val_path.exists() and any(val_path.iterdir())
    
    if has_explicit_val:
        print(f"✅ 检测到独立的验证集目录: {val_path}")
        print(">> 模式 A: 加载独立的 Train 和 Val 文件夹")
        
        # 模式 A: 物理文件夹分离
        # only_normal=False: 无监督模式下，允许混入少量故障数据，不进行过滤
        train_set = TransformerVibrationDataset(train_path, only_normal=False, mode="train")
        val_set   = TransformerVibrationDataset(val_path,   only_normal=False, mode="val")
        
    else:
        print(f"⚠️ 未检测到有效验证集目录 (或路径不存在): {val_path}")
        print(">> 模式 B: 从训练集中自动切分 10% 作为验证集")
        
        # 模式 B: 自动切分 (On-the-fly Split)
        full_dataset = TransformerVibrationDataset(train_path, only_normal=False, mode="mixed_pool")
        
        total_size = len(full_dataset)
        val_size = int(total_size * 0.1)  # 10% 用于验证
        train_size = total_size - val_size
        
        print(f"   自动划分: 总数={total_size} -> 训练={train_size} | 验证={val_size}")
        
        # 固定随机种子，保证每次跑的结果一致
        train_set, val_set = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

    model = SpatialResNetVAE(latent_channels=cfg.LATENT_CHANNELS).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.LR)

    # 记录器
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }

    best_val = float("inf")
    
    for epoch in range(cfg.EPOCHS):
        # Beta 预热
        if cfg.BETA_WARMUP_EPOCHS > 0:
            beta = min(cfg.BETA_MAX, cfg.BETA_MAX * (epoch + 1) / cfg.BETA_WARMUP_EPOCHS)
        else:
            beta = cfg.BETA_MAX

        # === 训练 ===
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Train]")
        
        for x in pbar:
            x = x.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(x)
            loss, rec, kld = loss_function(recon, x, mu, logvar, beta=beta)
            loss.backward()
            opt.step()
            train_loss += float(loss.item())
            pbar.set_postfix(beta=f"{beta:.4f}", loss=f"{loss.item():.4f}")

        avg_train = train_loss / max(1, len(train_loader))
        
        # === 验证 ===
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                recon, mu, logvar = model(x)
                loss, rec, kld = loss_function(recon, x, mu, logvar, beta=beta)
                val_loss += float(loss.item())
        avg_val = val_loss / max(1, len(val_loader))
        
        # === 记录 ===
        current_lr = opt.param_groups[0]['lr']
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        history['lr'].append(current_lr)
        
        print(f"Epoch {epoch+1} | Train={avg_train:.4f} | Val={avg_val:.4f} | LR={current_lr:.2e}")

        # 保存快照 (每10轮)
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), cfg.CHECKPOINT_DIR / "model" / f"epoch_{epoch+1}.pth")

        # 保存最佳
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), cfg.CHECKPOINT_DIR / "model" / "best_model.pth")

    # === 结束 ===
    torch.save(model.state_dict(), cfg.CHECKPOINT_DIR / "model" / "final_model.pth")
    
    # 绘制曲线
    plot_training_curves(history['train_loss'], history['val_loss'], history['lr'], cfg.CHECKPOINT_DIR / "model")
    print("训练完成。")

if __name__ == "__main__":
    train()