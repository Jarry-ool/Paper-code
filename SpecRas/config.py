
"""
Config for the Set-VAE vibration classifier.
Fill in your folder paths below, then run train_setvae.py.

How to read your data:
- Each folder contains one or more .json files.
- Each .json file is a dict with ONE long key. Its value is a list of records.
- Each record is like:
    {"data_time": "...", "sensor_id": "1", "signal_value": "-0.00287,0.00038, ... (8192 numbers) ..."}
- A "sample" is built by grouping all records that share the same `data_time`:
    one sample -> X ∈ R^{8192 × U}, where U varies (number of sensors at that timestamp).

Why Set-VAE:
- We encode each channel with a *shared* encoder, then aggregate with attention pooling.
- This makes the model robust to channel order and even to varying channel counts.
- We add a small classification head on top of the latent variable z.
"""

from pathlib import Path

# ==== REQUIRED: point these to your local folders ====
ROOT = Path(r"E:\我2\专业实践-工程专项\3-生技中心\1-项目：变压器深度学习诊断故障\3-code\diagnosis\test")

# 定义各类别文件夹（只需给到文件夹，不用具体文件）
# CLASS_DIRS = {
#     "正常":  [ROOT / "113--正常--换流变压器"],
#     "异常":  [ROOT / "137--异常--交流变压器"],
#     "老化":  [ROOT / "139--老化--交流变压器"],
#     "故障":  [ROOT / "159--故障--交流变压器"],
# }

 
# CLASS_DIRS = {
#     "正常":  [ROOT / "output_folder/134--正常--交流变压器"],
#     "故障":  [ROOT / "output_folder/135--故障--交流变压器"],
# }

# === 数据集目录 ===
TRAIN_ROOT = ROOT / "20251016/train"
VAL_ROOT   = ROOT / "20251016/val"
TEST_ROOT  = ROOT / "20251016/test"

# Classes to numeric labels
# CLASS_MAP = {"正常":0, "异常":1, "老化":2, "故障":3}
CLASS_MAP = {"正常":0, "故障":1}

# 用于从目录名里“判定类别”的关键词（可按需增减）
CLASS_KEYWORDS = {
    "正常": ("正常",),
    "故障": ("故障",),
}

def collect_split_dirs(split_root: Path, class_keywords=CLASS_KEYWORDS):
    """
    从 split_root (train/val/test) 下面的所有子目录里，递归搜集 .json / .jsonl 文件，
    并根据“子目录名是否包含关键词”来决定该文件属于哪个类别。
    返回字典 {类名: [文件绝对路径, ...]}
    """
    out = {cls: [] for cls in class_keywords.keys()}
    if not split_root.exists():
        return out

    # 遍历 split_root 下的所有子目录（不限层级；直接用 rglob 匹配文件更稳）
    for fp in split_root.rglob("*"):
        if fp.is_file() and fp.suffix.lower() in {".json", ".jsonl"}:
            # 向上找一个“带类名关键词”的目录
            cls_assigned = None
            for parent in fp.parents:
                name = parent.name
                for cls, kws in class_keywords.items():
                    if any(kw in name for kw in kws):
                        cls_assigned = cls
                        break
                if cls_assigned:
                    break

            if cls_assigned is None:
                # 若找不到，最后尝试文件名本身
                for cls, kws in class_keywords.items():
                    if any(kw in fp.name for kw in kws):
                        cls_assigned = cls
                        break

            if cls_assigned:
                out[cls_assigned].append(str(fp.resolve()))
            else:
                # 找不到类别就跳过（也可以 print 提示）
                print(f"[WARN] 无法判定类别，已跳过: {fp}")
                pass

    return out

# 训练参数
VAL_RATIO  = 0.2      # fraction of data reserved for validation (unused when explicit split folders exist)
SEED       = 42       # random seed for reproducibility
LR         = 1e-3     # base learning rate for Adam
EPOCHS     = 30       # maximum number of training epochs
BATCH_SIZE = 8        # batch size for DataLoader
LAMBDA_REC = 0.3      # weight of reconstruction loss (reduced for frequency backbone)
LAMBDA_CLS = 1.0      # weight of classification loss

# === Variational parameters ===
# We anneal the KL weight over the first `BETA_KL_WARMUP` epochs from
# `BETA_KL_START` to `BETA_KL_END`.  If you wish to disable annealing
# simply set START and END to the same value and WARMUP to 0.
BETA_KL_START  = 0.0
BETA_KL_END    = 0.2
BETA_KL_WARMUP = 10

# === Classification loss parameters ===
# Use FocalLoss to counter class imbalance.  Set FOCAL_GAMMA=0.0 to
# recover standard cross-entropy.  Optionally specify label smoothing
# between 0 and 1 to soften hard targets.
FOCAL_GAMMA    = 2.0
LABEL_SMOOTHING = 0.0

# === Miscellaneous training knobs ===
GRAD_CLIP = 1.0         # maximum gradient norm (0 to disable clipping)
EVAL_TEST_EVERY_EPOCH = True  # evaluate on the test set at the end of each epoch

# === Preprocessing options ===
# Remove per-channel DC offset (subtract mean) before global normalisation.
REMOVE_DC = True

# === Frequency-domain backbone options ===
# We perform Welch PSD on each sample and extract dense frequency
# vectors between FMIN_DENSE and FMAX_DENSE (inclusive) at DF_DENSE step.
# Set USE_FREQ_BACKBONE=True to enable the frequency-domain main branch.
# FS controls the sampling rate of your vibration signals.  With 8192 points
# per second you typically have FS=8192.  Adjust FMIN_DENSE/FMAX_DENSE/DF_DENSE
# as needed to reflect the important frequency range for your system.
FS = 8192
USE_FREQ_BACKBONE = True
FMIN_DENSE = 1
FMAX_DENSE = 4000
DF_DENSE   = 1
# Welch parameters: nperseg should be chosen such that FS/nperseg gives
# sufficient frequency resolution.  With FS=8192 and nperseg=4096 the
# bin spacing is 2 Hz; we later interpolate to 1 Hz resolution.
N_PERSEG = 4096
N_OVER   = 2048
WINDOW   = "hann"

# === Empirical rule thresholds for vibration quality assessment ===
# These thresholds correspond to the "注意值" in your DOCX for the four
# summary features: vibration power entropy (H), certainty (C),
# volatility percentage (V%), and high-frequency energy ratio (HF%).
THR_PWR_ENTROPY     = 2.0   # H ≥ 2.0 triggers
THR_CERTAINTY       = 0.38  # C ≤ 0.38 triggers
THR_VOLATILITY_PCT  = 2.0   # V% ≥ 2.0 triggers
THR_HIGHFREQ_PCT    = 5.0   # HF% ≥ 5.0 triggers

# === Model dimensions for the new VibNet ===
# d: channel embedding dimension; zdim: latent dimension; spec_hd: dense spectrogram encoder dimension.
D_ENC   = 64
Z_DIM   = 64
SPEC_HD = 128


# 生成三套清单
TRAIN_DIRS = collect_split_dirs(TRAIN_ROOT)
VAL_DIRS   = collect_split_dirs(VAL_ROOT)
TEST_DIRS  = collect_split_dirs(TEST_ROOT)

# Where to save checkpoints and logs
# OUTDIR = Path("./VAE/outputs_setvae").resolve()
# OUTDIR.mkdir(parents=True, exist_ok=True)

# 打印检查
if __name__ == "__main__":
    print("== SPLIT CHECK ==")
    for name, d in [("TRAIN", TRAIN_DIRS), ("VAL", VAL_DIRS), ("TEST", TEST_DIRS)]:
        total = sum(len(v) for v in d.values())
        print(f"{name}: total_files={total}")
        for k, v in d.items():
            print(f"  {k}: {len(v)}")
            for s in v[:3]:
                try:
                    print("     ", Path(s).relative_to(ROOT))
                except Exception:
                    print("     ", s)