# Transformer Fault Diagnosis via Deep Learning on Vibration Signals

**Code and dataset for the paper submitted to *Expert Systems with Applications* (ESWA)**

This repository provides the complete implementation of two complementary deep learning pipelines for power transformer fault diagnosis from multi-channel vibration signals, along with the accompanying dataset.

---

## Overview

Power transformer health monitoring is critical for grid reliability. This work proposes and evaluates two parallel diagnostic pipelines on real-world vibration data (8192-sample signals, 8192 Hz):

| Pipeline | Paradigm | Core Method | Input |
|---|---|---|---|
| **SpecRas** | Supervised | ResNet18 on spectral raster images | 1200-dim feature → 150×150 PNG |
| **HetSpatVAE** | Unsupervised | Spatial VAE with heterogeneous channels | CWT / STFT / Context → 224×224 |

Both pipelines target binary classification: **Normal (正常)** vs **Fault (故障 / 异常 / 老化)**.

---

## Repository Structure

```
.
├── SpecRas/                        # Supervised spectral-raster pipeline
│   ├── config.py                   # Dataset paths, class keywords, hyperparams
│   ├── zerone_config.py            # Image/feature/output directory settings
│   ├── zerone_features.py          # Feature extraction (time/STFT/PSD/HF, 1200-dim)
│   ├── zerone_make_images.py       # JSON → feature vector → 150×150 raster PNG
│   ├── zerone_train_resnet.py      # ResNet18 training (AdamW + SWA + EMA)
│   ├── zerone_eval_empirical.py    # Empirical threshold evaluation (H/C/V%/HF%)
│   ├── zerone_gradcam.py           # GradCAM visualization
│   ├── baseline_supervised_v1.py   # Baseline: RawVec-ResNet18, 1D-CNN (Wen 2018)
│   ├── w1_unit_discrimination.py   # Ablation W1: unit-identity confound test
│   ├── w2_baselines.py             # Ablation W2: MLP-1200, PatchCore
│   ├── w4_multiseed_stats.py       # Ablation W4: multi-seed statistical significance
│   ├── w5_gradcam_quantitative.py  # Ablation W5: GradCAM vs physical fault frequencies
│   ├── w5_spectral_independent.py  # Ablation W5: spectral independence validation
│   └── zerone_results/             # Output: metrics/, predictions/, viz/, w*/ reports
│
├── HetSpatVAE/                     # Unsupervised spatial VAE pipeline
│   ├── hetero_config.py            # Paths and model hyperparameters
│   ├── hetero_data.py              # JSON loading, CWT/STFT/Context image construction
│   ├── hetero_model.py             # SpatialResNetVAE (ResNet18 encoder + de-ResNet decoder)
│   ├── hetero_train.py             # VAE training with beta-KL annealing
│   ├── hetero_diagnose.py          # Anomaly detection (rec. error + Mahalanobis distance)
│   ├── hetero_viz.py               # Multi-sample channel visualization
│   ├── hetero_viz_single.py        # Single-sample detailed analysis
│   ├── baseline_flatvae.py         # Baseline: FlatVAE (no spatial structure)
│   ├── w3_weight_sensitivity.py    # Ablation W3: composite score weight sensitivity
│   └── outputs/                    # Output: diagnosis_report/, w3_sensitivity/
│
└── vibration datasets for transformers/   # Real-world vibration dataset
    ├── train/                      # 15 transformer units (normal + fault)
    ├── val/                        # 2 transformer units
    └── test/                       # 2 transformer units (held-out evaluation)
```

---

## Dataset

The dataset contains multi-channel vibration measurements from real AC power transformers collected at a substation. Each `.jsonl` file is a 1-second measurement window:

```json
{
  "<key>": [
    {
      "data_time": "2025-09-09 15:28:XX",
      "sensor_id": "1",
      "signal_value": "-0.00287,0.00038,...  (8192 floats)"
    }
  ]
}
```

**Specifications:**
- Sampling rate: **8192 Hz**, window length: **8192 samples** (1 s per sample)
- Channels: 2–4 per timestamp (multi-sensor, energy-weighted aggregation)
- Labels inferred from folder names: `正常` → Normal (0), `故障/异常/老化` → Fault (1)

**Split (train / val / test):**

| Split | Units | Classes |
|---|---|---|
| train | 114, 120, 127–129, 132, 136, 143, 152–155, 157–159 | Normal + Fault |
| val | 145 (Fault), 146 (Normal) | Normal + Fault |
| test | 134 (Normal), 135 (Fault) | Normal + Fault |

---

## Methods

### SpecRas — Spectral Raster ResNet18

1. **Feature extraction** (`zerone_features.py`): Extract a 1200-dim vector per sample:
   - `time` (15): RMS, kurtosis, skewness, crest/impulse/margin/waveform factors, etc.
   - `stft` (127): STFT segment means (nperseg=128, noverlap=64)
   - `psd` (1050): Welch PSD — 1–1000 Hz @ 1 Hz (1000 bins) + 1001–2000 Hz @ 20 Hz (50 bins)
   - `hf` (8): High-frequency amplitude & power ratios at 1000/2000/3000/4000 Hz

2. **Image encoding** (`zerone_make_images.py`): Feature vector → raster-stripe 150×150 PNG (no interpolation; panels: time | STFT | PSD | HF; global min-max normalization over the training set)

3. **Training** (`zerone_train_resnet.py`): Pretrained ResNet18, AdamW, ReduceLROnPlateau on val F1, optional SWA in later epochs

4. **Empirical baseline** (`zerone_eval_empirical.py`): Four vibration health metrics — **H** (power entropy ≥ 2.0), **C** (RQA determinism ≤ 0.38), **V%** (peak volatility ≥ 2.0%), **HF%** (high-freq ratio ≥ 5.0%)

### HetSpatVAE — Spatial Variational Autoencoder

1. **3-channel input** (`hetero_data.py`): Each signal → 224×224 RGB image with channels:
   - Ch. 0: CWT (Morlet wavelet, scales 1–128)
   - Ch. 1: STFT magnitude
   - Ch. 2: Folded temporal context

2. **Model** (`hetero_model.py`): **SpatialResNetVAE**
   - Encoder: ResNet18 backbone → `mu_conv` / `logvar_conv` (1×1 Conv2d, 64 channels), preserving 7×7 spatial structure
   - Decoder: De-ResNet with `ConvTranspose2d` blocks (7→14→28→56→112→224)
   - Loss: L1 reconstruction + β·KL (beta annealing from 0 to 0.01 over 20 epochs)

3. **Anomaly detection** (`hetero_diagnose.py`): Composite score = 0.6 × (channel-weighted L1 error) + 0.4 × (Mahalanobis distance on spatial latent); threshold at 97.5th percentile of training scores

---

## Installation

```bash
# Python 3.11 recommended (tested on Windows, conda)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchvision`, `scipy`, `numpy`, `scikit-learn`, `matplotlib`, `Pillow`, `pandas`, `tqdm`

---

## Usage

### SpecRas Pipeline

```bash
cd SpecRas

# 1. Generate raster images from JSON signals
python zerone_make_images.py

# 2. Train ResNet18 classifier
python zerone_train_resnet.py

# 3. Evaluate empirical health metrics
python zerone_eval_empirical.py

# 4. GradCAM visualization
python zerone_gradcam.py

# 5. Run ablation experiments
python w1_unit_discrimination.py
python w2_baselines.py
python w4_multiseed_stats.py
python w5_gradcam_quantitative.py
```

### HetSpatVAE Pipeline

```bash
cd HetSpatVAE

# 1. Train the spatial VAE (normal samples only)
python hetero_train.py

# 2. Anomaly detection on test set
python hetero_diagnose.py

# 3. Visualize reconstructions
python hetero_viz.py

# 4. Weight sensitivity ablation
python w3_weight_sensitivity.py
```

> **Before running:** update the data paths in `SpecRas/config.py` and `HetSpatVAE/hetero_config.py` to match your local clone. See [Configuring Data Paths](#configuring-data-paths) below.

---

## What Is Not Included (and How to Reproduce It)

To keep the repository at a manageable size, the following generated artifacts are **excluded** from version control. All of them can be fully reproduced by running the scripts in order.

| Excluded path | Size | How to regenerate |
|---|---|---|
| `SpecRas/zerone_results/images/` | ~186 MB | `python zerone_make_images.py` |
| `HetSpatVAE/outputs/model/` | ~1.4 GB | `python hetero_train.py` |
| All `*.pth` / `*.pt` files | varies | run the respective training script |

Everything else — source code, pre-computed metrics/CSVs, result figures, and the full vibration dataset — **is included**.

---

## Configuring Data Paths

Both pipelines use hard-coded absolute paths that **must be updated** before running on a new machine.

### SpecRas — `SpecRas/config.py`

Locate the `ROOT` and split variables near the top of the file and replace with your local path:

```python
# SpecRas/config.py  (lines ~23–42)

ROOT = Path(r"/path/to/repo")          # ← change to your repo root

TRAIN_ROOT = ROOT / "vibration datasets for transformers/train"
VAL_ROOT   = ROOT / "vibration datasets for transformers/val"
TEST_ROOT  = ROOT / "vibration datasets for transformers/test"
```

> The dataset folder is included in this repo at `vibration datasets for transformers/`, so set `ROOT` to wherever you cloned the repository.

### HetSpatVAE — `HetSpatVAE/hetero_config.py`

Locate `PROJECT_ROOT` at the top and update it:

```python
# HetSpatVAE/hetero_config.py  (lines ~11–13)

PROJECT_ROOT = Path(
    r"/path/to/repo/vibration datasets for transformers"
)
# TRAIN_DIR, VAL_DIR, TEST_DIR are derived automatically as subfolders
```

---

## Results

Model checkpoints and generated raster images are excluded (see section above). Re-run the training scripts to reproduce them. Pre-computed metrics, CSVs, and result figures are already included in `SpecRas/zerone_results/` and `HetSpatVAE/outputs/diagnosis_report/`.

---

## Citation

If you use this code or dataset, please cite:

```
[To be updated upon acceptance]
```

---

## License

This repository is released for research purposes accompanying the EAAI submission. The vibration dataset was collected from real-world power transformer units and is provided solely for academic reproducibility.
