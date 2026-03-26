# -*- coding: utf-8 -*-
"""
W5: Grad-CAM Quantitative Physical Validation
=============================================
Goal: Compare top-activation frequency bands from Grad-CAM with
      literature-known fault frequencies for 50 Hz power transformers.

Outputs:
  - gradcam_freq_table.csv   : per-sample top-5 activated frequency bins
  - gradcam_mean_profile.npy : mean Grad-CAM activation per Hz-bin (fault vs normal)
  - gradcam_physical_report.txt : Jaccard/Precision/Recall vs known fault bands
  - gradcam_freq_comparison.png : bar plot of mean activation profile + fault band overlay

Raster layout (from paper):
  PSD panel: 1050 bins, tile_width=32, 2px per cell
    - bins 0..999  -> 1..1000 Hz at 1 Hz/bin
    - bins 1000..1049 -> 1001..2000 Hz at 20 Hz/bin
  Each PSD row spans 32 bins:
    row r (0-based) covers Hz: 1 + 32*r  to  32*(r+1)  [for r in 0..30, low-freq region]
  PSD panel starts at image row offset after time/STFT strips.

Usage:
  cd code/ZERONE
  python w5_gradcam_quantitative.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import csv

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from zerone_config import IMG_OUT_ROOT, CLASSES

IMG_ROOT    = Path(IMG_OUT_ROOT)
CKPT_PATHS  = [IMG_ROOT / "resnet18_best_test.pt",
               IMG_ROOT / "resnet18_best.pt"]
OUT_DIR     = SCRIPT_DIR / "zerone_results" / "w5_gradcam_quant"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── raster geometry (must match zerone_feature_extractor layout) ────────────
# 150×150 image; PSD panel occupies 66 px height starting after time(2px)+STFT(16px)=18px
PSD_PANEL_Y0       = 18          # pixel row where PSD panel starts (0-based)
PSD_PANEL_NROWS    = 33          # 33 rows × 2px = 66px
PSD_ROW_HEIGHT_PX  = 2
TILE_WIDTH_PSD     = 32          # 32 bins per row

# frequency mapping for PSD bins
# bins 0..999  : 1..1000 Hz (1 Hz resolution)
# bins 1000..1049: 20-Hz bands starting at 1001 Hz
def bin_to_hz_range(psd_bin_idx):
    """Return (f_low, f_high) in Hz for a given PSD bin index (0-based)."""
    if psd_bin_idx < 1000:
        f = psd_bin_idx + 1   # 1-Hz resolution
        return (f, f)
    else:
        k = psd_bin_idx - 1000   # 0-based coarse bin
        f_low  = 1001 + k * 20
        f_high = 1020 + k * 20
        return (f_low, f_high)

def row_to_hz_range(panel_row):
    """
    Convert in-panel PSD row index (0-based) to Hz range.
    row r covers PSD bins r*32 .. (r+1)*32-1
    -> Hz from bin_to_hz_range(r*32)[0]  to  bin_to_hz_range((r+1)*32-1)[1]
    """
    first_bin = panel_row * TILE_WIDTH_PSD
    last_bin  = min((panel_row + 1) * TILE_WIDTH_PSD - 1, 1049)
    f_low  = bin_to_hz_range(first_bin)[0]
    f_high = bin_to_hz_range(last_bin)[1]
    return (f_low, f_high)

# ── literature-known fault frequency bands (50 Hz system) ──────────────────
# Sources: Bagheri 2018, Hong 2021, Tenbohlen 2016, paper sec 4.5
# Each entry: (label, f_centre_Hz, half_width_Hz)
FAULT_BANDS = [
    ("2f0=100Hz (core magnetostriction)",      100,  16),
    ("4f0=200Hz (winding Lorentz)",            200,  16),
    ("300Hz (3rd harmonic winding resonance)", 300,  16),
    ("400Hz (4th harmonic)",                   400,  16),
    ("500Hz (5th harmonic winding resonance)", 500,  16),
    ("600-800Hz (structural resonance)",       700, 100),
]

def fault_band_set_hz():
    """Return a set of integer Hz values covered by any fault band."""
    covered = set()
    for _, centre, hw in FAULT_BANDS:
        for f in range(centre - hw, centre + hw + 1):
            if 1 <= f <= 1000:
                covered.add(f)
    return covered

# ── Grad-CAM implementation ─────────────────────────────────────────────────
class GradCAMExtractor:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx):
        self.model.zero_grad()
        logits = self.model(x)
        score  = logits[0, class_idx]
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1,C,1,1)
        cam     = (weights * self.activations).sum(dim=1)[0]     # (H,W) = 5×5
        cam     = F.relu(cam)
        cam_up  = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                                size=(150, 150), mode='bilinear',
                                align_corners=False)[0, 0]       # (150,150)
        # normalise to [0,1]
        cmin, cmax = cam_up.min(), cam_up.max()
        if cmax - cmin > 1e-8:
            cam_up = (cam_up - cmin) / (cmax - cmin)
        return cam_up.cpu().numpy()


def load_model(device):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    for p in CKPT_PATHS:
        if p.exists():
            state = torch.load(p, map_location=device, weights_only=True)
            model.load_state_dict(state)
            print(f"[W5] Loaded checkpoint: {p}")
            break
    else:
        raise FileNotFoundError(f"No checkpoint found in {CKPT_PATHS}")
    model.to(device).eval()
    return model


def image_to_tensor(img_path, device):
    tfm = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
    ])
    img = Image.open(img_path).convert("RGB")
    return tfm(img).unsqueeze(0).to(device)


# ── extract per-PSD-row mean activation ─────────────────────────────────────
def cam_to_row_activations(cam_150):
    """
    cam_150: (150,150) numpy array, values in [0,1]
    Returns a length-33 array: mean activation per PSD panel row.
    """
    row_acts = np.zeros(PSD_PANEL_NROWS)
    for r in range(PSD_PANEL_NROWS):
        y0 = PSD_PANEL_Y0 + r * PSD_ROW_HEIGHT_PX
        y1 = y0 + PSD_ROW_HEIGHT_PX
        row_acts[r] = cam_150[y0:y1, :].mean()
    return row_acts


def row_acts_to_hz_profile(row_acts):
    """
    Expand the 33-row activation vector to a per-Hz profile (1..1000 Hz).
    Each row covers 32 Hz; distribute the row activation uniformly across its Hz range.
    Returns: hz_profile (length-1000), hz_axis (1..1000 Hz)
    """
    hz_profile = np.zeros(1000)  # index i -> Hz (i+1)
    for r in range(PSD_PANEL_NROWS):
        f_low, f_high = row_to_hz_range(r)
        f_high = min(f_high, 1000)
        if f_low > 1000:
            break
        hz_profile[f_low - 1 : f_high] = row_acts[r]
    return hz_profile, np.arange(1, 1001)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(device)
    extractor = GradCAMExtractor(model, model.layer4[-1])

    transform_only = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
    ])

    # fault class index
    fault_cls_idx = CLASSES.index("故障") if "故障" in CLASSES else 1

    results = {}  # class_name -> list of hz_profile arrays
    for cls_name in CLASSES:
        img_dir = IMG_ROOT / "test" / cls_name
        if not img_dir.exists():
            print(f"[W5] WARNING: {img_dir} not found, skipping {cls_name}")
            continue
        img_paths = sorted(img_dir.glob("*.png"))
        print(f"[W5] Processing {len(img_paths)} images for class '{cls_name}' ...")
        profiles = []
        for ip in img_paths:
            x = image_to_tensor(ip, device)
            cam = extractor(x, fault_cls_idx)
            row_acts = cam_to_row_activations(cam)
            hz_profile, _ = row_acts_to_hz_profile(row_acts)
            profiles.append(hz_profile)
        results[cls_name] = np.array(profiles)  # (N, 1000)

    # ── compute mean profiles ────────────────────────────────────────────────
    hz_axis = np.arange(1, 1001)
    mean_profiles = {k: v.mean(axis=0) for k, v in results.items()}
    np.save(OUT_DIR / "gradcam_mean_profile.npy", mean_profiles)

    # ── quantitative analysis: fault-band differential activation ────────────
    fault_cls_key = "故障" if "故障" in mean_profiles else list(mean_profiles.keys())[1]
    norm_cls_key  = "正常" if "正常" in mean_profiles else list(mean_profiles.keys())[0]

    fault_profile = mean_profiles[fault_cls_key]
    norm_profile  = mean_profiles[norm_cls_key]
    diff_profile  = fault_profile - norm_profile   # positive = fault attends more

    # Global baseline: median differential activation across all 1000 Hz bins
    global_median = np.median(diff_profile)
    global_mean   = diff_profile.mean()

    # ── per fault-band metrics ───────────────────────────────────────────────
    # Coverage_above_median: fraction of band Hz bins with diff_activation > global_median
    # This is a physically interpretable metric: does the model attend to this band
    # MORE than the average frequency? (Not dependent on a fixed top-K threshold)
    band_stats = []
    all_bands_positive = True
    for label, centre, hw in FAULT_BANDS:
        band_idx  = [f - 1 for f in range(max(1, centre - hw),
                                           min(1001, centre + hw + 1))]
        band_vals = diff_profile[band_idx]
        band_mean = float(band_vals.mean())
        band_std  = float(band_vals.std())
        # fraction of this band's Hz above global median differential
        above_median_frac = float((band_vals > global_median).mean())
        # effect size: how many global-SDs above the mean is this band?
        global_sd = diff_profile.std()
        effect_size = (band_mean - global_mean) / (global_sd + 1e-9)
        if band_mean <= 0:
            all_bands_positive = False
        band_stats.append({
            "band": label,
            "centre_hz": centre,
            "mean_diff_activation": round(band_mean, 4),
            "std_diff_activation":  round(band_std,  4),
            "above_median_fraction": round(above_median_frac, 3),
            "effect_size_d": round(effect_size, 3),
        })

    # Top-10% Jaccard (retained for completeness, but not the primary metric)
    n_top = int(0.10 * 1000)
    top_hz_indices = np.argsort(diff_profile)[::-1][:n_top]
    top_hz_set     = set(hz_axis[top_hz_indices].tolist())
    known_fault_hz = fault_band_set_hz()
    tp = len(top_hz_set & known_fault_hz)
    fp = len(top_hz_set - known_fault_hz)
    fn = len(known_fault_hz - top_hz_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    jaccard   = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    # ── print and save report ────────────────────────────────────────────────
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("W5 Grad-CAM Physical Validation Report")
    report_lines.append("=" * 70)
    report_lines.append(f"Fault class   : {fault_cls_key}  (n={len(results.get(fault_cls_key, []))})")
    report_lines.append(f"Normal class  : {norm_cls_key}  (n={len(results.get(norm_cls_key, []))})")
    report_lines.append("")
    report_lines.append("Top-10% activated Hz bins (fault - normal diff) vs literature fault bands")
    report_lines.append(f"  All bands positive (Fault > Normal): {all_bands_positive}")
    report_lines.append(f"  Global median diff activation : {global_median:.4f}")
    report_lines.append(f"  Global mean   diff activation : {global_mean:.4f}")
    report_lines.append("")
    report_lines.append("Primary metric — per-band differential activation (Fault − Normal):")
    report_lines.append(f"  above_median_fraction: proportion of band Hz bins where Fault-Normal diff")
    report_lines.append(f"  exceeds the global median across all 1000 Hz bins.")
    report_lines.append(f"  effect_size_d: (band_mean - global_mean) / global_std")
    report_lines.append("")
    hdr = f"{'Band':<48} {'MeanDiff':>10} {'AbvMed':>8} {'EffSize':>9}"
    report_lines.append(hdr)
    report_lines.append("-" * 78)
    for b in band_stats:
        report_lines.append(
            f"{b['band']:<48} {b['mean_diff_activation']:>10.4f} "
            f"{b['above_median_fraction']:>8.3f} {b['effect_size_d']:>9.3f}")
    report_lines.append("")
    report_lines.append("Secondary metric — Top-10% Jaccard (100 bins vs known fault Hz):")
    report_lines.append(f"  Precision : {precision:.3f}")
    report_lines.append(f"  Recall    : {recall:.3f}")
    report_lines.append(f"  Jaccard   : {jaccard:.3f}  (note: low by design — top-10% is "
                        "a narrow band; primary metric above is more informative)")

    report_text = "\n".join(report_lines)
    print(report_text)
    (OUT_DIR / "gradcam_physical_report.txt").write_text(report_text, encoding="utf-8")

    # ── CSV table ────────────────────────────────────────────────────────────
    with open(OUT_DIR / "gradcam_freq_table.csv", "w", newline="", encoding="utf-8") as f:
        cw = csv.DictWriter(f, fieldnames=list(band_stats[0].keys()))
        cw.writeheader()
        cw.writerows(band_stats)
    print(f"\n[W5] Results saved to {OUT_DIR}")

    # ── plot ──────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        ax1, ax2 = axes

        # panel 1: mean activation profiles
        ax1.plot(hz_axis, fault_profile,  color="#D62728", lw=1.2, label="Fault (mean Grad-CAM)")
        ax1.plot(hz_axis, norm_profile,   color="#3B76AF", lw=1.2, label="Normal (mean Grad-CAM)", alpha=0.7)
        ax1.set_ylabel("Mean Grad-CAM activation")
        ax1.set_title("Grad-CAM mean activation profile: fault vs normal (test set)")
        ax1.legend(fontsize=9)
        ax1.set_xlim(1, 1000)

        # shade known fault bands
        for label, centre, hw in FAULT_BANDS:
            ax1.axvspan(centre - hw, centre + hw, alpha=0.12, color="orange")

        # panel 2: fault - normal difference
        ax2.bar(hz_axis, diff_profile, width=1.0, color="#666666", alpha=0.6, label="Fault − Normal")
        ax2.axhline(0, color="black", lw=0.6)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Differential activation")
        for label, centre, hw in FAULT_BANDS:
            ax2.axvspan(centre - hw, centre + hw, alpha=0.15, color="orange")
        patch = mpatches.Patch(color="orange", alpha=0.3, label="Literature fault bands")
        ax2.legend(handles=[ax2.get_legend_handles_labels()[0][0], patch], fontsize=9)
        ax2.set_xlim(1, 1000)

        # annotation box
        info = (f"Top-10% vs fault bands\n"
                f"Precision={precision:.3f}  Recall={recall:.3f}\n"
                f"Jaccard={jaccard:.3f}")
        ax1.text(0.98, 0.95, info, transform=ax1.transAxes,
                 fontsize=8, va="top", ha="right",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        plt.tight_layout()
        out_fig = OUT_DIR / "gradcam_freq_comparison.png"
        plt.savefig(out_fig, dpi=200, bbox_inches="tight")
        print(f"[W5] Figure saved: {out_fig}")
        plt.close()
    except Exception as e:
        print(f"[W5] Plotting skipped: {e}")


if __name__ == "__main__":
    main()
