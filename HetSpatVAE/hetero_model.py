# -*- coding: utf-8 -*-
"""
hetero_model.py
Hetero-CWT-Zerone-VAE 模型定义
包含 Encoder (ResNet Backbone) 和 Decoder (De-ResNet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SpatialResNetVAE(nn.Module):
    def __init__(self, latent_channels=64):
        super(SpatialResNetVAE, self).__init__()
        
        # ================= Encoder =================
        # 使用 ResNet18 作为骨干网络
        resnet = models.resnet18(weights=None) # 不使用预训练权重，因为输入模式不同
        
        # 修改第一层卷积：虽然 ResNet 默认也是 3 通道，但我们的通道含义完全不同 (CWT, Zerone, Context)
        # 我们保留结构，重新初始化权重
        self.encoder_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # -> 64 x 56 x 56
            resnet.layer2,  # -> 128 x 28 x 28
            resnet.layer3,  # -> 256 x 14 x 14
            resnet.layer4   # -> 512 x 7 x 7
        )
        
        # 空间隐变量投影 (Spatial Latent Projection)
        # 保持 7x7 的空间结构，不压扁成全连接向量，保留局部故障特征
        self.mu_conv = nn.Conv2d(512, latent_channels, kernel_size=1)
        self.logvar_conv = nn.Conv2d(512, latent_channels, kernel_size=1)
        
        # ================= Decoder =================
        # 解码器输入映射
        self.decoder_input = nn.Conv2d(latent_channels, 512, kernel_size=1)
        
        # De-ResNet 结构 (使用转置卷积进行上采样)
        self.decoder = nn.Sequential(
            # Block 1 (对应 layer4): 7x7 -> 14x14
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Block 2 (对应 layer3): 14x14 -> 28x28
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Block 3 (对应 layer2): 28x28 -> 56x56
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Block 4 (对应 layer1): 56x56 -> 112x112
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Final Up (对应 stem): 112x112 -> 224x224
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # 输出归一化到 [0, 1] 以匹配输入图像
        )

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        # 1. 编码
        h = self.encoder_stem(x)
        
        # 2. 投影到隐空间
        mu = self.mu_conv(h)
        logvar = self.logvar_conv(h)
        
        # 3. 重参数化采样
        z = self.reparameterize(mu, logvar)
        
        # 4. 解码
        z_dec = self.decoder_input(z)
        recon = self.decoder(z_dec)
        
        # 确保输出尺寸完全匹配 (处理 padding 可能导致的细微差异)
        if recon.shape != x.shape:
            recon = F.interpolate(recon, size=x.shape[2:])
            
        return recon, mu, logvar

def loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE 损失函数 = 重构损失 (L1) + Beta * KL散度
    使用 L1 Loss 而非 MSE，因为文档提到 L1 对边缘更清晰
    """
    # 重构损失 (Sum over all pixels)
    B = x.size(0)
    recon_loss = F.l1_loss(recon_x, x, reduction='sum') / B
    
    # KL 散度
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
    
    return recon_loss + beta * kld_loss, recon_loss, kld_loss