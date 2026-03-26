# -*- coding: utf-8 -*-
"""
ZERONE 的配置文件。

本文件集中定义数据路径、特征提取参数以及训练超参数。
与原始示例不同，当前工程从 `config.py` 中读取训练、验证、测试数据的分割，
并统一按类组织。生成的图像按 split（train/val/test）和类名（正常/故障）
保存在 `IMG_OUT_ROOT` 之下，评估指标亦会输出到 `SCORES_OUT_ROOT`。

主要内容包括：
    * 数据集路径：从 config.TRAIN_DIRS/VAL_DIRS/TEST_DIRS 中读取
    * 图像尺寸与颜色映射开关
    * Welch PSD 参数与时域特征开关
    * 四个判别指标的阈值及健康指数权重
    * 训练超参数
"""
# zerone_config.py  —— 只看“输出目录”这一段，其余保持你当前参数
from pathlib import Path

# ====== 从外部 config.py 读取数据划分与基础阈值 ======
# 引入原始 config 中的划分与采样率/阈值设置
try:
    from config import (
        TRAIN_DIRS, VAL_DIRS, TEST_DIRS,
        FS as CONFIG_FS,
        THR_PWR_ENTROPY,
        THR_CERTAINTY,
        THR_VOLATILITY_PCT,
        THR_HIGHFREQ_PCT,
    )
except ImportError:
    # 当 config.py 不存在或导入失败时给出占位符，避免运行时报错
    TRAIN_DIRS = {}
    VAL_DIRS   = {}
    TEST_DIRS  = {}
    CONFIG_FS  = 8192
    THR_PWR_ENTROPY    = 2.0
    THR_CERTAINTY      = 0.38
    THR_VOLATILITY_PCT = 2.0
    THR_HIGHFREQ_PCT   = 5.0

# ---- 数据划分 ----
# CLASS 列表由 TRAIN_DIRS 的键决定，目前仅支持“正常”和“故障”两类
CLASSES = list(TRAIN_DIRS.keys()) if TRAIN_DIRS else ["正常", "故障"]

# 为每个数据集（train/val/test）保存对应类别的文件列表
SPLIT_DIRS = {
    "train": TRAIN_DIRS,
    "val":   VAL_DIRS,
    "test":  TEST_DIRS,
}

# ---- 输出目录 ----

RESULTS_ROOT   = Path("./zerone_results").resolve() # === 统一结果根目录 ===

# 图像（按 split/类组织）、经验分数（如需）
IMG_OUT_ROOT     = RESULTS_ROOT / "images"         # 生成的 150×150 PNG 图像根目录。各 split/类会形成二级子目录
SCORES_OUT_ROOT  = RESULTS_ROOT / "scores"         # 评估得分与判别结果的输出目录

# 训练/评估可视化
VIZ_ROOT   = RESULTS_ROOT / "viz"
VIZ_PLOTS  = VIZ_ROOT / "plots"
VIZ_ROCPR  = VIZ_ROOT / "roc_pr"
VIZ_RELIAB = VIZ_ROOT / "reliability"
VIZ_EMBED  = VIZ_ROOT / "embeddings"

# 预测与特征缓存、训练度量
PRED_ROOT    = RESULTS_ROOT / "predictions"        # <split>_predictions.csv
FEAT_ROOT    = RESULTS_ROOT / "features"           # <split>_features.npy
METRICS_ROOT = RESULTS_ROOT / "metrics"            # training_metrics.csv 等

for d in [IMG_OUT_ROOT, SCORES_OUT_ROOT, VIZ_PLOTS, VIZ_ROCPR, VIZ_RELIAB, VIZ_EMBED,
          PRED_ROOT, FEAT_ROOT, METRICS_ROOT]:
    d.mkdir(parents=True, exist_ok=True)

# ====== 图像/特征参数（保留 legacy + 新 raster 开关） ======
IMAGE_SIZE = (150, 150)  # 高×宽，单位像素
CHANNELS   = 3          # 输出图像通道数，3 表示彩色

# 每个特征竖条的宽度（像素）。默认 2 像素。若启用 STFT 分组特征可按需要调小。
STRIPE_WIDTH = 2

# 是否使用颜色归一化，以及是否二值化
USE_COLOR_NORMALIZATION = True
USE_BINARIZATION        = False

# ---- PSD 频谱提取参数 ----
# 使用 Welch 方法计算功率谱密度，采样率来自原 config
PSD_FS    = CONFIG_FS      # 采样率 Hz，例如 8192
PSD_FMIN  = 1           # 最低频率，1 Hz
PSD_FMAX  = 4000           # 最高频率，4 kHz
PSD_DF    = 1              # 频率步长，1 Hz
PSD_NPERSEG = 4096         # Welch 的窗长度
PSD_NOVER   = 2048         # Welch 的窗重叠长度
PSD_WINDOW  = "hann"       # 窗函数

# ---- STFT 设置 ----
STFT_NPERSEG  = 128
STFT_NOVERLAP = 64

# ---- 特征开关 ----
# 是否提取时域统计量（15 个：均值/标准差/方差/RMS/绝对最大值等）
USE_TIME_FEATURES = True

# 是否提取密集 PSD（主干）；如果关闭，则不计算频域分量
USE_PSD_FEATURES  = True

# 是否提取 STFT 均值特征并按组平均后作为额外条带特征。
# 如果打开，将从每个通道的 STFT 频谱（约 65 维）中分组计算平均值，用于图像生成。
USE_STFT_FEATURES = True

# 移除各通道直流偏移（减去均值）后进行全局归一化
REMOVE_DC = True

# ---- 四个判别指标阈值 ----
# 这些阈值来自文档“注意值”，用于判定指标是否超过注意值
THR_H  = THR_PWR_ENTROPY     # 振动功率熵 H 的注意阈值（H ≥ THR_H 记为 1）
THR_C  = THR_CERTAINTY       # 振动确定度 C 的注意阈值（C > THR_C 记为 1）
THR_V  = THR_VOLATILITY_PCT  # 振动波动率 V% 的注意阈值（V% ≥ THR_V 记为 1）
THR_HF = THR_HIGHFREQ_PCT    # 高频振动占比 HF% 的注意阈值（HF% ≥ THR_HF 记为 1）

# ---- 健康指数权重 ----
# 按照文档拟合权重：功率熵 0.5，确定度 0.1，波动率 0.1，高频占比 0.3
HEALTH_WEIGHTS = {
    "H":  0.5,
    "C":  0.1,
    "V":  0.1,
    "HF": 0.3,
}

# ---- 训练超参数 ----
# Note: 训练仅使用 train/val 划分，不再在训练脚本中做随机拆分
SEED        = 42
LR          = 8e-6
EPOCHS      = 30 #20
BATCH_SIZE  = 16
MASKING_RATIO = 0.05

# ---- 其他参数 ----
# 当没有显式 val/test 划分时可使用 VAL_RATIO 做随机拆分；当前保持未用
VAL_RATIO   = 0.2
PATIENCE = 20   # 连续 5 个 epoch val_f1 没提升就早停
NUM_WORKERS = 4  # DataLoader 的 num_workers 参数

# ==== 可视化布局比例（与 vector_to_image / ColumnMask 一致）====
# 把图像宽度按列分成四段；总和应为 1.0
LAYOUT_RATIOS = {
    "time": 0.10,   # 15 维时域
    "stft": 0.15,   # STFT 条带
    "psd" : 0.65,   # PSD 1–4000 Hz 条带
    "hf"  : 0.10,   # HF 等其他小段
}

# ==== 低频关注：仅在“张量阶段”生效；磁盘 PNG 不变 ====
LOWFREQ_ATTEN = {
    "enabled": True,
    "cutoff_hz": 1000.0,
    "alpha": 2.0,
}

# ============= 无插值列复制 + SquarePad 的统一开关 =============

# 最终正方形边长；与训练侧 Dataset 的 SquarePad(target) 对齐
IMG_TARGET_SIDE  = 150
PAD_FILL         = 0       # 0=黑
KEEP_RATIO_FIRST = True    # 先按原始条带高宽比拼接，最后再 SquarePad

# PSD 变换策略：raw / log / log1p / octave（倍频带汇聚）
PSD_MODE          = "log1p"
OCTAVE_NUM_BANDS  = 30     # 倍频带数量（仅当 PSD_MODE="octave" 时生效）

# ====== 条带布局（列复制的视觉权重）======
# 键为条带名称（需与生成脚本中使用的名称一致），值为该条带相对宽度权重。
# 生成阶段将根据这些权重，把每条带的列“复制”到相应的目标视觉宽度。
STRIP_VISUAL_WEIGHTS = {
    "time_features": 1.0,  # 时域统计（例如 15 维 → 15 列）
    "psd":           3.0,  # Welch PSD 条带更宽
    "stft":          2.0,  # STFT 分组/完整频谱
    "indices":       1.0,  # H/C/V%/HF% 等综合指标
}

# ====== 其他可视化参数（不强制使用）======
CMAP_PSD   = "turbo"
CMAP_STFT  = "magma"
CMAP_OTHER = "viridis"

# ====== Raster-Stripe 无插值布局 ======
# layout_mode: "raster" 使用逐点栅格；"legacy" 使用原 150×150 条带图。
LAYOUT_MODE = "raster"

RASTER_STRIPE = {
    # 单元像素尺寸（每个特征点映射的色块大小）
    # 单元像素尺寸：调大 W_UNIT 可以使横向宽度更合理；保持 H_UNIT 为 2
    "W_UNIT": 3,
    "H_UNIT": 2,

    # 各面板是否折行显示
    "wrap_panels": {"time": True, "stft": True, "psd": True, "hf": True},
    
    # 折行方向：horizontal / vertical
    "stack_axis": "vertical",

    # 图像与排布
    "IMG_SIZE" : 224,           # 训练图片边长（替代原 150）

    # 训练时低频增强的轴向（配合上面的取向）
    # 当 STRIPE_ORIENT="vertical" 时，低频在“上方行”，LFE_AXIS="rows"
    # 当 STRIPE_ORIENT="horizontal" 时，低频在“左侧列”，LFE_AXIS="cols"
    "LFE_AXIS" : "rows",        # 可选 "rows" 或 "cols"
    "LFE_YCOVER" : 0.40,        # 覆盖比例
    "LFE_GAIN" : 1.25,          # 增益倍数

    # 面板折行宽度（单位：特征点个数/列）
    "tile_widths": {
        # 调整折行宽度使整体宽高比接近 1:1。
        # time: 15 维，16 列刚好 1 行。
        "time": 16,
        # stft: 127 维，16 列 → 8 行。
        "stft": 16,
        # psd: 1050 维，32 列 → 33 行。
        "psd": 32,
        # hf: 8 维，8 列 → 1 行。
        "hf": 8,
    },

    # 面板顺序（自上而下），不建议改动
    "panel_order": ["time", "stft", "psd", "hf"],

    # 面板间是否插入 1 列空白
    "insert_gap_after": {"time": True, "stft": True, "psd": True, "hf": False},

    # 空白列宽（单位：特征点列数；会乘以 W_UNIT 变为像素）
    "gap_tile_width": 1,

    # 颜色映射方案
    "colormap": "jet",

    # 低频掩膜（在训练张量阶段应用；PNG 不变）
    "lowfreq_emphasis": {"enabled": True, "alpha": 2.0},
    
    "canvas": {
        "width": 150,      # 基准宽
        "height": 150,     # 基准高
        "allow_grow_w": False,  # 横向是否允许变宽
        "allow_grow_h": True,   # 纵向是否允许变高
        "max_w": 150,      # 允许增长的上限（像素）
        "max_h": 150,      # 允许增长的上限（像素）
        "gap_rows": 1      # 面板间空白行（像素）
    }
}

# ---- 工具函数 ----
def ensure_dirs():
    """创建输出目录。"""
    IMG_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    SCORES_OUT_ROOT.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # 打印分割检查
    print("== SPLIT CHECK ==")
    for name in ["train", "val", "test"]:
        dirs = SPLIT_DIRS.get(name, {})
        total = sum(len(v) for v in dirs.values()) if dirs else 0
        print(f"{name.upper()}: total_files={total}")
        for cls in CLASSES:
            paths = dirs.get(cls, []) if dirs else []
            print(f"  {cls}: {len(paths)}")
            for s in paths[:3]:
                print("     ", s)
    print("\\n== PATHS ==")
    print("RESULTS_ROOT   :", RESULTS_ROOT)
    print("IMG_OUT_ROOT   :", IMG_OUT_ROOT)
    print("VIZ_PLOTS      :", VIZ_PLOTS)
    print("PRED_ROOT      :", PRED_ROOT)
    print("FEAT_ROOT      :", FEAT_ROOT)
    print("METRICS_ROOT   :", METRICS_ROOT)
    print("\\n== TRAIN HYPERPARAMS ==")
    print("SEED, LR, EPOCHS, BATCH_SIZE, MASKING_RATIO, PATIENCE =",
          SEED, LR, EPOCHS, BATCH_SIZE, MASKING_RATIO, PATIENCE)
