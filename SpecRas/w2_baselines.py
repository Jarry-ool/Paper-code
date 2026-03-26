# -*- coding: utf-8 -*-
"""
W2: Additional Baselines — MLP (fair supervised) + PatchCore (fair unsupervised)
=================================================================================
Goal: Provide two missing baselines that address the unfair-comparison concern:
  (B1) MLP-1200: 2-layer MLP on the same 1200-dim spectral feature vector.
       This isolates 2D-raster advantage from model-architecture effects.
  (B2) PatchCore: memory-bank nearest-neighbour anomaly detection on the
       same 224×224 3-channel images as HetSpatVAE, using ResNet18 features.
       This is a competitive unsupervised baseline cited in related work.

Outputs (saved to zerone_results/w2_baselines/):
  w2_baselines_results.csv    - accuracy / F1 / FP / FN for both baselines
  w2_baselines_report.txt     - formatted results table

Usage:
  cd code/ZERONE
  python w2_baselines.py

  # For PatchCore (requires VAE data pipeline):
  # Run from project root so both ZERONE and VAE imports are accessible.
"""

import sys
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "VAE"))

# ── shared helper: on-the-fly ResNet18 avgpool feature extraction ─────────────
def _load_split_features(split: str):
    """
    Extract 512-dim avgpool features directly from raster PNG images.
    Returns (X: np.ndarray shape (N,512), y: np.ndarray shape (N,))
    No dependency on pre-saved .npy files.
    """
    import torch
    from torchvision import models, transforms
    from PIL import Image
    from zerone_config import IMG_OUT_ROOT, CLASSES

    NORMAL_KEYS = ("正常", "normal")
    FAULT_KEYS  = ("故障", "fault", "abnormal", "anomaly")
    IMG_ROOT    = Path(IMG_OUT_ROOT)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    for cp in [IMG_ROOT / "resnet18_best_test.pt", IMG_ROOT / "resnet18_best.pt"]:
        if cp.exists():
            model.load_state_dict(torch.load(cp, map_location=device, weights_only=True))
            break
    buf = []
    model.avgpool.register_forward_hook(
        lambda _m, _in, out: buf.append(
            out.squeeze(-1).squeeze(-1).detach().cpu().numpy()))
    model.to(device).eval()

    tfm = transforms.Compose([transforms.Resize((150, 150)), transforms.ToTensor()])
    split_dir = IMG_ROOT / split
    all_X, all_y = [], []

    for cls_dir in sorted(split_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        name_lc = cls_dir.name.lower()
        if any(k in name_lc for k in NORMAL_KEYS):
            label = 0
        elif any(k in name_lc for k in FAULT_KEYS):
            label = 1
        else:
            continue
        paths = sorted(cls_dir.glob("*.png"))
        batch_t = []
        def flush():
            if not batch_t: return
            with torch.no_grad():
                model(torch.stack(batch_t).to(device))
            all_X.extend(buf); buf.clear(); batch_t.clear()
        for p in paths:
            batch_t.append(tfm(Image.open(p).convert("RGB")))
            if len(batch_t) >= 64: flush()
        flush()
        all_y.extend([label] * len(paths))

    return np.vstack(all_X), np.array(all_y)


# ── ImageNet-pretrained (no fine-tuning) feature extractor ────────────────────
def _load_split_features_imagenet(split: str):
    """
    Extract 512-dim avgpool features using ImageNet-pretrained ResNet18
    with NO domain-specific fine-tuning.
    Returns (X: np.ndarray shape (N,512), y: np.ndarray shape (N,))
    """
    import torch
    from torchvision import models, transforms
    from PIL import Image
    from zerone_config import IMG_OUT_ROOT, CLASSES

    NORMAL_KEYS = ("正常", "normal")
    FAULT_KEYS  = ("故障", "fault", "abnormal", "anomaly")
    IMG_ROOT    = Path(IMG_OUT_ROOT)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    buf = []
    model.avgpool.register_forward_hook(
        lambda _m, _in, out: buf.append(
            out.squeeze(-1).squeeze(-1).detach().cpu().numpy()))
    model.to(device).eval()

    tfm = transforms.Compose([transforms.Resize((150, 150)), transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406],
                                                    [0.229, 0.224, 0.225])])
    split_dir = IMG_ROOT / split
    all_X, all_y = [], []

    for cls_dir in sorted(split_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        name_lc = cls_dir.name.lower()
        if any(k in name_lc for k in NORMAL_KEYS):
            label = 0
        elif any(k in name_lc for k in FAULT_KEYS):
            label = 1
        else:
            continue
        paths = sorted(cls_dir.glob("*.png"))
        batch_t = []
        def flush():
            if not batch_t: return
            with torch.no_grad():
                model(torch.stack(batch_t).to(device))
            all_X.extend(buf); buf.clear(); batch_t.clear()
        for p in paths:
            batch_t.append(tfm(Image.open(p).convert("RGB")))
            if len(batch_t) >= 64: flush()
        flush()
        all_y.extend([label] * len(paths))

    return np.vstack(all_X), np.array(all_y)


# ═══════════════════════════════════════════════════════════
# B0: MLP on ImageNet ResNet18 avgpool features (no fine-tuning)
# ═══════════════════════════════════════════════════════════
def run_mlp_imagenet_baseline():
    """
    Shallow MLP (512->256->2) on IMAGENET-pretrained ResNet18 avgpool features,
    with NO domain-specific fine-tuning. This is the true fair comparison:
    generic features + MLP vs. end-to-end SpecRas SWA.
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (accuracy_score, f1_score,
                                  confusion_matrix, roc_auc_score)

    print("[W2-B0] Extracting ImageNet features (no fine-tuning) from train images ...")
    tr_X, tr_y = _load_split_features_imagenet("train")
    print("[W2-B0] Extracting ImageNet features (no fine-tuning) from test  images ...")
    te_X, te_y = _load_split_features_imagenet("test")
    print(f"[W2-B0] Train: {len(tr_y)} samples | Test: {len(te_y)} samples")

    n0, n1 = (tr_y == 0).sum(), (tr_y == 1).sum()
    cw = {0: len(tr_y)/(2*n0), 1: len(tr_y)/(2*n1)}

    clf = Pipeline([
        ("sc", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(256,),
            activation="relu",
            max_iter=500,
            learning_rate_init=1e-3,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            verbose=False,
        ))
    ])
    clf.fit(tr_X, tr_y)

    te_pred = clf.predict(te_X)
    te_prob = clf.predict_proba(te_X)[:, 1]

    cm = confusion_matrix(te_y, te_pred)
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(te_y, te_pred)
    f1  = f1_score(te_y, te_pred, average="macro")
    auc = roc_auc_score(te_y, te_prob)
    prec_n = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_n  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    rec_f  = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print(f"[W2-B0] MLP-ImageNet: Acc={acc*100:.2f}% Macro-F1={f1:.3f} AUC={auc:.3f} "
          f"FP={fp} FN={fn}")
    return {
        "method": "MLP-ImageNet (ImageNet ResNet18, no fine-tune)",
        "paradigm": "Supervised",
        "accuracy": round(acc*100, 2),
        "precision_N": round(prec_n, 3),
        "recall_N": round(rec_n, 3),
        "recall_F": round(rec_f, 3),
        "macro_F1": round(f1, 3),
        "AUC": round(auc, 3),
        "FP": int(fp), "FN": int(fn),
    }


# ═══════════════════════════════════════════════════════════
# B1: MLP on ResNet18 avgpool features (fair architecture comparison)
# ═══════════════════════════════════════════════════════════
def run_mlp_baseline():
    """
    Shallow MLP (512 -> 256 -> 2) on ResNet18 avgpool features extracted
    from the same raster images as SpecRas. This tests whether the full
    end-to-end fine-tuning (SpecRas) outperforms a frozen-encoder + MLP
    head, isolating the value of task-specific representation learning.
    Features are extracted on-the-fly from zerone_results/images/.
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (accuracy_score, f1_score,
                                  confusion_matrix, roc_auc_score)

    print("[W2-B1] Extracting features from train images ...")
    tr_X, tr_y = _load_split_features("train")
    print("[W2-B1] Extracting features from test  images ...")
    te_X, te_y = _load_split_features("test")

    # class weights
    n0, n1 = (tr_y == 0).sum(), (tr_y == 1).sum()
    cw = {0: len(tr_y)/(2*n0), 1: len(tr_y)/(2*n1)}

    print(f"[W2-B1] Train: {len(tr_y)} samples | Test: {len(te_y)} samples")
    print("[W2-B1] Training MLP (1200→256→2) ...")

    clf = Pipeline([
        ("sc", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(256,),
            activation="relu",
            max_iter=500,
            learning_rate_init=1e-3,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            verbose=False,
        ))
    ])
    clf.fit(tr_X, tr_y)

    te_pred = clf.predict(te_X)
    te_prob = clf.predict_proba(te_X)[:, 1]

    cm = confusion_matrix(te_y, te_pred)
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(te_y, te_pred)
    f1  = f1_score(te_y, te_pred, average="macro")
    auc = roc_auc_score(te_y, te_prob)
    prec_n = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_n  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    rec_f  = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print(f"[W2-B1] MLP-1200: Acc={acc*100:.2f}% Macro-F1={f1:.3f} AUC={auc:.3f} "
          f"FP={fp} FN={fn}")
    return {
        "method": "MLP-1200 (1200→256→2)",
        "paradigm": "Supervised",
        "accuracy": round(acc*100, 2),
        "precision_N": round(prec_n, 3),
        "recall_N": round(rec_n, 3),
        "recall_F": round(rec_f, 3),
        "macro_F1": round(f1, 3),
        "AUC": round(auc, 3),
        "FP": int(fp), "FN": int(fn),
    }


# ═══════════════════════════════════════════════════════════
# B2: PatchCore on 3-channel 224×224 images (HetSpatVAE input)
# ═══════════════════════════════════════════════════════════
def run_patchcore_baseline():
    """
    PatchCore-style nearest-neighbour anomaly detection:
      - Encoder: ResNet18 layer3 features (no fine-tuning, pretrained on ImageNet)
      - Memory bank: all training-normal patch features (subsampled to 10%)
      - Anomaly score: max k-NN distance across spatial positions
    Evaluated on the same test split as HetSpatVAE.
    """
    import torch
    import torch.nn.functional as F
    from torchvision import models, transforms
    from torch.utils.data import DataLoader

    try:
        import hetero_config as vae_cfg
        from hetero_data import TransformerVibrationDataset
    except ImportError as e:
        print(f"[W2-B2] VAE imports failed: {e}. Skipping PatchCore.")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── encoder: ResNet18 layer3, pretrained ImageNet ────────────────────────
    encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    feature_extractor = torch.nn.Sequential(
        encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool,
        encoder.layer1, encoder.layer2, encoder.layer3,   # (N, 256, 14, 14)
    ).to(device).eval()

    def _extract_patches(loader):
        """Extract (N_samples, 256, 14, 14) → flatten to (N*196, 256)."""
        all_feats = []
        with torch.no_grad():
            for imgs in loader:
                imgs = imgs.to(device)
                feats = feature_extractor(imgs)    # (B, 256, 14, 14)
                feats = F.normalize(feats, dim=1)  # L2-normalise channel-wise
                all_feats.append(feats.cpu().numpy())
        feats_all = np.concatenate(all_feats, axis=0)  # (N, 256, 14, 14)
        N, C, H, W = feats_all.shape
        # spatial positions as separate "patches": (N*H*W, C)
        return feats_all.transpose(0, 2, 3, 1).reshape(-1, C)

    def _build_loader(root, only_normal, batch_size=16):
        ds = TransformerVibrationDataset(Path(root), mode="test",
                                          only_normal=only_normal)
        return DataLoader(ds, batch_size=batch_size, shuffle=False,
                          num_workers=0)

    print("[W2-B2] Extracting training-normal features ...")
    tr_loader = _build_loader(vae_cfg.TRAIN_DIR, only_normal=True)
    tr_patches = _extract_patches(tr_loader)       # (M, 256)

    # Subsampling: cap memory bank at 5000 for tractable KNN
    rng = np.random.default_rng(42)
    n_keep = min(5000, len(tr_patches))
    idx = rng.choice(len(tr_patches), n_keep, replace=False)
    memory_bank = tr_patches[idx].astype(np.float32)   # (n_keep, 256)
    bank_sq = (memory_bank ** 2).sum(axis=1)            # precompute for speed
    print(f"[W2-B2] Memory bank: {len(memory_bank)} patch vectors (from {len(tr_patches)})")

    # ── fast batch numpy k-NN (no sklearn, avoids O(N*M) sklearn overhead) ───
    K = 9
    BATCH = 2000

    def _knn_kth_dist(queries):
        """Return k-th NN Euclidean distance for each query; fully vectorised."""
        queries = queries.astype(np.float32)
        n = len(queries)
        out = np.empty(n, dtype=np.float32)
        for i in range(0, n, BATCH):
            q = queries[i:i+BATCH]                          # (B, 256)
            q_sq = (q ** 2).sum(axis=1, keepdims=True)      # (B, 1)
            dists2 = np.maximum(q_sq + bank_sq - 2.0 * (q @ memory_bank.T), 0.0)
            out[i:i+BATCH] = np.sqrt(np.partition(dists2, K-1, axis=1)[:, K-1])
        return out

    def _score_loader(loader):
        """Per-sample anomaly score = max k-NN distance across spatial positions."""
        patches = _extract_patches(loader)          # (N*196, 256)
        knn_score = _knn_kth_dist(patches)          # (N*196,)
        n_samples = len(loader.dataset)
        n_patches_per = len(knn_score) // n_samples
        return knn_score.reshape(n_samples, n_patches_per).max(axis=1)

    print("[W2-B2] Scoring test set ...")
    # training-normal scores for threshold calibration
    tr_scores = _score_loader(_build_loader(vae_cfg.TRAIN_DIR, only_normal=True))
    tau = np.quantile(tr_scores, 0.975)

    # test: load normal and fault separately
    test_root = Path(vae_cfg.TEST_DIR)
    te_scores_all, te_labels_all = [], []
    for sub in test_root.iterdir():
        if not sub.is_dir(): continue
        s = sub.name.lower()
        if any(k in s for k in ("正常", "normal")):
            label = 0
        elif any(k in s for k in ("故障", "fault", "异常")):
            label = 1
        else:
            continue
        loader = _build_loader(sub, only_normal=False)
        sc = _score_loader(loader)
        te_scores_all.append(sc)
        te_labels_all.extend([label] * len(sc))

    te_scores = np.concatenate(te_scores_all)
    te_labels = np.array(te_labels_all)

    preds = (te_scores > tau).astype(int)   # PatchCore: HIGH score = anomalous
    from sklearn.metrics import (accuracy_score, f1_score,
                                  confusion_matrix, roc_auc_score)
    cm = confusion_matrix(te_labels, preds)
    tn, fp, fn, tp_ = cm.ravel()
    acc = accuracy_score(te_labels, preds)
    f1  = f1_score(te_labels, preds, average="macro")
    try:
        auc = roc_auc_score(te_labels, te_scores)
    except Exception:
        auc = float("nan")
    prec_n = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_n  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    rec_f  = tp_ / (tp_ + fn) if (tp_ + fn) > 0 else 0.0

    print(f"[W2-B2] PatchCore: Acc={acc*100:.2f}% Macro-F1={f1:.3f} AUC={auc:.3f} "
          f"FP={fp} FN={fn}")
    return {
        "method": "PatchCore (ResNet18 layer3, k=9)",
        "paradigm": "One-class",
        "accuracy": round(acc*100, 2),
        "precision_N": round(prec_n, 3),
        "recall_N": round(rec_n, 3),
        "recall_F": round(rec_f, 3),
        "macro_F1": round(f1, 3),
        "AUC": round(auc, 3) if not np.isnan(auc) else "N/A",
        "FP": int(fp), "FN": int(fn),
    }


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
def main():
    OUT_DIR = SCRIPT_DIR / "zerone_results" / "w2_baselines"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    print("=" * 60)
    print("W2 Baseline B0: MLP-ImageNet (no fine-tune, fair comparison)")
    print("=" * 60)
    try:
        r0 = run_mlp_imagenet_baseline()
        results.append(r0)
    except Exception as e:
        print(f"[W2-B0] FAILED: {e}")

    print("\n" + "=" * 60)
    print("W2 Baseline B1: MLP-FT (fine-tuned ResNet18 features)")
    print("=" * 60)
    try:
        r1 = run_mlp_baseline()
        results.append(r1)
    except Exception as e:
        print(f"[W2-B1] FAILED: {e}")

    print("\n" + "=" * 60)
    print("W2 Baseline B2: PatchCore")
    print("=" * 60)
    try:
        r2 = run_patchcore_baseline()
        if r2: results.append(r2)
    except Exception as e:
        print(f"[W2-B2] FAILED: {e}")

    if not results:
        print("[W2] No results to report.")
        return

    # ── save CSV ──────────────────────────────────────────────────────────────
    import csv
    csv_path = OUT_DIR / "w2_baselines_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    # ── print report ──────────────────────────────────────────────────────────
    lines = ["=" * 75,
             "W2 Additional Baselines Report",
             "=" * 75,
             f"{'Method':<40} {'Acc%':>6} {'Macro-F1':>9} {'AUC':>7} "
             f"{'FP':>4} {'FN':>4}",
             "-" * 75]
    for r in results:
        lines.append(
            f"{r['method']:<40} {r['accuracy']:>6.2f} {r['macro_F1']:>9.3f} "
            f"{str(r['AUC']):>7} {r['FP']:>4} {r['FN']:>4}")
    lines += [
        "",
        "Context (from Table 2 of the paper):",
        f"  SpecRas SWA            (Supervised, proposed) :  97.00%  F1=0.969  FP=12  FN=0",
        f"  HetSpatVAE             (One-class,  proposed) :  98.25%  F1=0.982  FP=7   FN=0",
        f"  1D-CNN [Wen 2018]      (Supervised, external) :  68.00%  F1=0.659",
        f"  RawVec-ResNet18        (Supervised, ablation) :  79.75%  F1=0.788",
        f"  FlatVAE                (One-class,  ablation) :  38.75%  F1=0.357",
        "",
        "Interpretation:",
        "  MLP-1200 provides a fair 2D-raster vs 1D-flat comparison:",
        "    SpecRas SWA should substantially outperform MLP-1200 on the SAME features,",
        "    attributing the gain entirely to 2D spatial layout (not model capacity).",
        "  PatchCore provides a competitive unsupervised baseline on the SAME images:",
        "    HetSpatVAE should outperform PatchCore, validating the spatial VAE approach.",
    ]

    report_text = "\n".join(lines)
    print(report_text)
    (OUT_DIR / "w2_baselines_report.txt").write_text(report_text, encoding="utf-8")
    print(f"\n[W2] Results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
