# -*- coding: utf-8 -*-
"""
W3: Composite Score Weight Sensitivity Analysis for HetSpatVAE
==============================================================
Goal: Show that the reported 98.25% accuracy is robust to perturbations
      of the two sets of manually-chosen weights:
        (A) channel weights  : (w_cwt, w_stft, w_ctx) summing to 1.0
        (B) composite alpha  : a = alpha * rec_z + (1-alpha) * md_z

Strategy:
  1. Run inference ONCE to collect per-channel raw L1 errors and latents.
  2. Sweep a 2D grid over (alpha, w_stft) without retraining.
  3. For each grid point: recompute composite scores → threshold → accuracy.
  4. Save 2D accuracy heatmap + report min/max/std.

Grid:
  alpha    : 0.0, 0.1, ..., 1.0   (11 values)
  w_stft   : 0.1, 0.2, ..., 0.9   (9 values)
  w_cwt and w_ctx split the remainder evenly: w_cwt = w_ctx = (1 - w_stft) / 2

Outputs (saved to ./outputs/w3_sensitivity/):
  sensitivity_heatmap.png   - 2D heatmap of test accuracy
  sensitivity_results.csv   - all grid results
  sensitivity_report.txt    - summary statistics

Usage:
  cd code/VAE
  python w3_weight_sensitivity.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import hetero_config as cfg
from hetero_model import SpatialResNetVAE
from hetero_data import TransformerVibrationDataset

# ── output directory ─────────────────────────────────────────────────────────
OUT_DIR = cfg.CHECKPOINT_DIR / "w3_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── sweep grid ───────────────────────────────────────────────────────────────
ALPHA_VALS   = np.round(np.arange(0.0, 1.01, 0.1), 2)   # 0.0 .. 1.0 (11 pts)
WSTFT_VALS   = np.round(np.arange(0.1, 0.91, 0.1), 2)   # 0.1 .. 0.9  (9 pts)

# ── helpers ──────────────────────────────────────────────────────────────────
def _collect_raw(model, loader, device):
    """
    Returns per-channel L1 errors and spatial-mean latent vectors.
    e0: CWT channel,  e1: STFT channel,  e2: waveform context channel
    latents: (N, 64) spatial-mean encoder output
    """
    model.eval()
    e0_all, e1_all, e2_all, lat_all = [], [], [], []
    with torch.no_grad():
        for imgs in tqdm(loader, desc="Inference", leave=False):
            imgs = imgs.to(device)
            recon, mu, _ = model(imgs)
            e0 = torch.mean(torch.abs(recon[:, 0] - imgs[:, 0]), dim=[1, 2]).cpu().numpy()
            e1 = torch.mean(torch.abs(recon[:, 1] - imgs[:, 1]), dim=[1, 2]).cpu().numpy()
            e2 = torch.mean(torch.abs(recon[:, 2] - imgs[:, 2]), dim=[1, 2]).cpu().numpy()
            z  = torch.mean(mu, dim=(2, 3)).cpu().numpy()
            e0_all.append(e0); e1_all.append(e1); e2_all.append(e2); lat_all.append(z)
    return (np.concatenate(e0_all),
            np.concatenate(e1_all),
            np.concatenate(e2_all),
            np.vstack(lat_all))


def _fit_mahalanobis(latents):
    m = latents.mean(axis=0)
    cov = np.cov(latents.T) + 1e-6 * np.eye(latents.shape[1])
    inv = np.linalg.pinv(cov)
    return m, inv


def _mahalanobis(latents, mean, inv_cov):
    diff = latents - mean[None, :]
    dist2 = np.einsum("bi,ij,bj->b", diff, inv_cov, diff)
    return np.sqrt(np.maximum(dist2, 0.0))


def _composite_score(e0, e1, e2, md, w_stft, alpha,
                     tr_rec_mu, tr_rec_sd, tr_md_mu, tr_md_sd):
    """
    Given raw per-channel errors and Mahalanobis distances,
    compute composite z-normalised score using (w_stft, alpha).
    w_cwt = w_ctx = (1 - w_stft) / 2
    """
    w_cwt = (1.0 - w_stft) / 2.0
    w_ctx = (1.0 - w_stft) / 2.0
    rec = w_cwt * e0 + w_stft * e1 + w_ctx * e2
    # z-normalise using TRAINING statistics
    rec_z = (rec - tr_rec_mu) / (tr_rec_sd + 1e-9)
    md_z  = (md  - tr_md_mu)  / (tr_md_sd  + 1e-9)
    return alpha * rec_z + (1.0 - alpha) * md_z


def _accuracy(scores, labels, quantile=0.975, tr_scores=None):
    """Threshold at quantile of training scores; return accuracy, fp, fn.
    Score convention: LOW score = anomalous (fault), HIGH score = normal.
    A sample is flagged as fault when score < tau.
    tau = 97.5th-percentile of training-normal scores (upper tail of normal dist).
    """
    tau = np.quantile(tr_scores, quantile)
    preds = (scores < tau).astype(int)    # 1 = anomaly = fault (LOW score)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    acc = (tp + tn) / len(labels) if len(labels) > 0 else 0.0
    return acc, fp, fn


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load model ──────────────────────────────────────────────────────────
    model_candidates = [
        cfg.CHECKPOINT_DIR / "model" / "epoch_30.pth",
        cfg.CHECKPOINT_DIR / "model" / "best_model.pth",
        cfg.CHECKPOINT_DIR / "model" / "final_model.pth",
    ]
    model_path = next((p for p in model_candidates if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError(f"No VAE checkpoint found in {cfg.CHECKPOINT_DIR / 'model'}")

    print(f"[W3] Loading model: {model_path}")
    model = SpatialResNetVAE(latent_channels=cfg.LATENT_CHANNELS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device,
                                      weights_only=True))
    model.eval()

    # ── load data loaders ───────────────────────────────────────────────────
    def _build_loader(root, only_normal):
        ds = TransformerVibrationDataset(Path(root), mode="test",
                                          only_normal=only_normal)
        return DataLoader(ds, batch_size=cfg.BATCH_SIZE,
                          shuffle=False, num_workers=0)

    print("[W3] Collecting training-normal scores ...")
    tr_loader = _build_loader(cfg.TRAIN_DIR, only_normal=True)
    tr_e0, tr_e1, tr_e2, tr_lat = _collect_raw(model, tr_loader, device)
    tr_mean_mu, tr_inv_cov = _fit_mahalanobis(tr_lat)
    tr_md = _mahalanobis(tr_lat, tr_mean_mu, tr_inv_cov)

    # ── test set: normal + fault separately ─────────────────────────────────
    test_root = Path(cfg.TEST_DIR)
    test_subdirs = [d for d in test_root.iterdir() if d.is_dir()]

    te_e0, te_e1, te_e2, te_lat, te_labels = [], [], [], [], []
    for sub in test_subdirs:
        s = sub.name.lower()
        if any(k in s for k in ("正常", "normal")):
            label = 0
        elif any(k in s for k in ("故障", "fault", "异常", "abnormal")):
            label = 1
        else:
            print(f"[W3] Skipping unrecognised test subdir: {sub.name}")
            continue
        loader = _build_loader(sub, only_normal=False)
        e0, e1, e2, lat = _collect_raw(model, loader, device)
        n = len(e0)
        te_e0.append(e0); te_e1.append(e1); te_e2.append(e2); te_lat.append(lat)
        te_labels.extend([label] * n)
        print(f"[W3]   {sub.name}: {n} samples (label={label})")

    te_e0  = np.concatenate(te_e0)
    te_e1  = np.concatenate(te_e1)
    te_e2  = np.concatenate(te_e2)
    te_lat = np.vstack(te_lat)
    te_labels = np.array(te_labels)
    te_md = _mahalanobis(te_lat, tr_mean_mu, tr_inv_cov)

    print(f"[W3] Test samples: {len(te_labels)} "
          f"(normal={int((te_labels==0).sum())}, fault={int((te_labels==1).sum())})")

    # ── sweep grid ───────────────────────────────────────────────────────────
    print(f"[W3] Sweeping {len(ALPHA_VALS)}×{len(WSTFT_VALS)} grid ...")
    acc_grid  = np.zeros((len(ALPHA_VALS), len(WSTFT_VALS)))
    fp_grid   = np.zeros_like(acc_grid, dtype=int)
    fn_grid   = np.zeros_like(acc_grid, dtype=int)

    import csv
    csv_rows = []

    for i, alpha in enumerate(ALPHA_VALS):
        for j, w_stft in enumerate(WSTFT_VALS):
            # compute training composite scores for threshold calibration
            tr_rec_mu_tmp = (((1-w_stft)/2) * tr_e0 + w_stft * tr_e1
                             + ((1-w_stft)/2) * tr_e2).mean()
            tr_rec_sd_tmp = (((1-w_stft)/2) * tr_e0 + w_stft * tr_e1
                             + ((1-w_stft)/2) * tr_e2).std() + 1e-9
            tr_md_mu_tmp  = tr_md.mean()
            tr_md_sd_tmp  = tr_md.std() + 1e-9

            tr_scores = _composite_score(
                tr_e0, tr_e1, tr_e2, tr_md, w_stft, alpha,
                tr_rec_mu_tmp, tr_rec_sd_tmp, tr_md_mu_tmp, tr_md_sd_tmp)
            te_scores = _composite_score(
                te_e0, te_e1, te_e2, te_md, w_stft, alpha,
                tr_rec_mu_tmp, tr_rec_sd_tmp, tr_md_mu_tmp, tr_md_sd_tmp)

            acc, fp, fn = _accuracy(te_scores, te_labels,
                                    quantile=0.975, tr_scores=tr_scores)
            acc_grid[i, j] = acc
            fp_grid[i, j]  = fp
            fn_grid[i, j]  = fn
            csv_rows.append({
                "alpha": alpha, "w_stft": round(w_stft, 2),
                "w_cwt": round((1-w_stft)/2, 3),
                "w_ctx": round((1-w_stft)/2, 3),
                "accuracy": round(acc, 4),
                "FP": fp, "FN": fn,
            })

    # ── save CSV ─────────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "sensitivity_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    # ── summary statistics ───────────────────────────────────────────────────
    all_acc = acc_grid.flatten()
    paper_alpha  = 0.6
    paper_wstft  = 0.5
    i_paper = np.argmin(np.abs(ALPHA_VALS - paper_alpha))
    j_paper = np.argmin(np.abs(WSTFT_VALS - paper_wstft))
    paper_acc = acc_grid[i_paper, j_paper]

    report_lines = [
        "=" * 60,
        "W3 Weight Sensitivity Report",
        "=" * 60,
        f"Grid: alpha in {ALPHA_VALS.tolist()}",
        f"      w_stft in {WSTFT_VALS.tolist()}",
        f"      (w_cwt = w_ctx = (1-w_stft)/2)",
        "",
        f"Paper setting: alpha={paper_alpha}, w_stft={paper_wstft}",
        f"  -> accuracy at paper setting : {paper_acc*100:.2f}%",
        "",
        f"Grid accuracy statistics (all {len(all_acc)} configurations):",
        f"  Mean   : {all_acc.mean()*100:.2f}%",
        f"  Std    : {all_acc.std()*100:.2f}%",
        f"  Min    : {all_acc.min()*100:.2f}%  "
        f"(alpha={ALPHA_VALS[np.unravel_index(all_acc.argmin(), acc_grid.shape)[0]]}, "
        f"w_stft={WSTFT_VALS[np.unravel_index(all_acc.argmin(), acc_grid.shape)[1]]})",
        f"  Max    : {all_acc.max()*100:.2f}%  "
        f"(alpha={ALPHA_VALS[np.unravel_index(all_acc.argmax(), acc_grid.shape)[0]]}, "
        f"w_stft={WSTFT_VALS[np.unravel_index(all_acc.argmax(), acc_grid.shape)[1]]})",
        f"  Range  : {(all_acc.max()-all_acc.min())*100:.2f} pp",
        "",
        "Configs with accuracy >= paper accuracy:",
        f"  Count  : {int((all_acc >= paper_acc - 1e-4).sum())} / {len(all_acc)}",
        "Configs with zero missed faults (FN=0):",
        f"  Count  : {int((fn_grid.flatten() == 0).sum())} / {len(all_acc)}",
    ]
    report_text = "\n".join(report_lines)
    print(report_text)
    (OUT_DIR / "sensitivity_report.txt").write_text(report_text, encoding="utf-8")

    # ── heatmap ──────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # accuracy heatmap
        ax = axes[0]
        im = ax.imshow(acc_grid * 100, cmap="RdYlGn", vmin=85, vmax=100,
                       aspect="auto", origin="lower")
        ax.set_xticks(range(len(WSTFT_VALS)))
        ax.set_xticklabels([f"{v:.1f}" for v in WSTFT_VALS], fontsize=8)
        ax.set_yticks(range(len(ALPHA_VALS)))
        ax.set_yticklabels([f"{v:.1f}" for v in ALPHA_VALS], fontsize=8)
        ax.set_xlabel("STFT channel weight ($w_{\\mathrm{stft}}$)")
        ax.set_ylabel("Composite weight $\\alpha$ (reconstruction)")
        ax.set_title("Test accuracy (%) across weight grid")
        plt.colorbar(im, ax=ax, shrink=0.9)
        # mark paper setting
        ax.plot(j_paper, i_paper, "w*", markersize=14, label="Paper setting")
        ax.legend(fontsize=9, loc="upper right")
        # annotate each cell
        for i in range(len(ALPHA_VALS)):
            for j in range(len(WSTFT_VALS)):
                ax.text(j, i, f"{acc_grid[i,j]*100:.1f}",
                        ha="center", va="center", fontsize=6,
                        color="black" if acc_grid[i, j] > 0.90 else "white")

        # FN heatmap (missed faults)
        ax2 = axes[1]
        im2 = ax2.imshow(fn_grid, cmap="Reds_r", vmin=0,
                          aspect="auto", origin="lower")
        ax2.set_xticks(range(len(WSTFT_VALS)))
        ax2.set_xticklabels([f"{v:.1f}" for v in WSTFT_VALS], fontsize=8)
        ax2.set_yticks(range(len(ALPHA_VALS)))
        ax2.set_yticklabels([f"{v:.1f}" for v in ALPHA_VALS], fontsize=8)
        ax2.set_xlabel("STFT channel weight ($w_{\\mathrm{stft}}$)")
        ax2.set_ylabel("Composite weight $\\alpha$ (reconstruction)")
        ax2.set_title("Missed faults (FN) across weight grid")
        plt.colorbar(im2, ax=ax2, shrink=0.9)
        ax2.plot(j_paper, i_paper, "w*", markersize=14, label="Paper setting")
        ax2.legend(fontsize=9, loc="upper right")
        for i in range(len(ALPHA_VALS)):
            for j in range(len(WSTFT_VALS)):
                ax2.text(j, i, str(int(fn_grid[i, j])),
                         ha="center", va="center", fontsize=7)

        plt.suptitle("W3: HetSpatVAE composite score weight sensitivity analysis",
                     fontsize=11, y=1.01)
        plt.tight_layout()
        fig_path = OUT_DIR / "sensitivity_heatmap.png"
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        print(f"[W3] Heatmap saved: {fig_path}")
        plt.close()
    except Exception as e:
        print(f"[W3] Plotting skipped: {e}")

    print(f"\n[W3] All results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
