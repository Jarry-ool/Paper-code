# -*- coding: utf-8 -*-
"""
hetero_config.py
项目配置文件（统一 train / val / test 路径）
"""
from pathlib import Path

# ================= 路径配置 =================
# 三个文件夹的共同父目录，例如：
#   E:\我2\...\20251016
PROJECT_ROOT = Path(
    r"E:\我2\专业实践-工程专项\3-生技中心\1-项目：变压器深度学习诊断故障\3-code\diagnosis\test\20251016"
)

# 明确三个数据子目录（请确保真实存在）
TRAIN_DIR = PROJECT_ROOT / "train"   # 训练用数据（建议尽量是正常）
VAL_DIR   = PROJECT_ROOT / "val"     # 验证用数据（一般也是正常，少量异常也无妨）
TEST_DIR  = PROJECT_ROOT / "test"    # 诊断/测试用数据（可以是混合）

# 如果旧代码还在用 DATA_ROOT / RAW_DATA_DIRS，就保留兼容：
DATA_ROOT = TRAIN_DIR
RAW_DATA_DIRS = [TRAIN_DIR, VAL_DIR, TEST_DIR]  # 或按需写成 [TRAIN_DIR, VAL_DIR, TEST_DIR]

# 模型与结果输出目录
CHECKPOINT_DIR = Path("./outputs")
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

# ================= 物理/信号参数 =================
FS = 8192              # 采样率 (Hz)
SIGNAL_LEN = 8192      # 每条振动信号长度
INPUT_SIZE = 224       # 输入图像尺寸 (224 x 224)

# ================= VAE 模型与训练超参数 =================
LATENT_CHANNELS = 64   # 空间隐变量通道数

BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 50

BETA_INIT = 0.0
BETA_MAX = 0.01        # 先按你原始值来（后面想强化“异常敏感度”可以调大到 0.03~0.05）
BETA_WARMUP_EPOCHS = 20

DEVICE = "cuda"        # 或 "cpu"

# ================= Zerone 特征维度（暂时只在 Zerone 相关脚本里用） =================
FEAT_DIM_TIME = 15
FEAT_DIM_STFT = 127
FEAT_DIM_PSD = 1050
FEAT_DIM_HF = 8

TOTAL_ZERONE_DIM = FEAT_DIM_TIME + FEAT_DIM_STFT + FEAT_DIM_PSD + FEAT_DIM_HF
