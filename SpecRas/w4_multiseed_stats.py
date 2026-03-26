# -*- coding: utf-8 -*-
"""
w4_multiseed_stats.py
=====================
W4 multi-seed statistical significance experiment for SpecRas ResNet18+SWA.

Runs zerone_train_resnet.py across seeds [0, 1, 2, 3, 42], collects
test_predictions.csv after each run, computes per-seed and aggregate stats,
and saves:
  - zerone_results/w4_multiseed/predictions/seed_<N>_test_predictions.csv
  - zerone_results/w4_multiseed/w4_multiseed_results.csv
  - zerone_results/w4_multiseed/w4_report.txt
"""

import os
import sys
import shutil
import subprocess
import time
from pathlib import Path

# ---- Locate the correct Python interpreter (needs torch + sklearn) ---------
# The script may be invoked from a bare Python that lacks the ML packages.
# We detect the correct interpreter by checking a known Anaconda env path.
_CANDIDATE_PYTHONS = [
    r"E:\anaconda3\envs\Pytorch\python.exe",
    r"E:\anaconda3\envs\clean_pytorch\python.exe",
    r"D:\ProgramData\Anaconda3\envs\new_pytorch_env\python.exe",
]
_SELF = Path(__file__).resolve()

def _find_ml_python():
    for p in _CANDIDATE_PYTHONS:
        if Path(p).exists():
            return p
    return sys.executable  # fallback

TRAIN_PYTHON = _find_ml_python()

# ---- If this interpreter lacks numpy/sklearn, re-launch with correct one ---
try:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
    _HAS_ML = True
except ImportError:
    _HAS_ML = False

if not _HAS_ML:
    # Re-invoke this script using the ML-capable interpreter
    ml_python = _find_ml_python()
    if ml_python == sys.executable:
        print("ERROR: Cannot find a Python with numpy/sklearn/torch. Please run this script "
              "with the correct conda environment.")
        sys.exit(1)
    print(f"[w4] Current Python lacks ML packages. Re-launching with: {ml_python}")
    result = subprocess.run(
        [ml_python, str(_SELF)] + sys.argv[1:],
        cwd=str(_SELF.parent),
    )
    sys.exit(result.returncode)

# ---- From here on, we have numpy/pandas/sklearn available -----------------

# ---- Paths ----------------------------------------------------------------
ZERONE_DIR   = _SELF.parent
PRED_SRC     = ZERONE_DIR / "zerone_results" / "predictions" / "test_predictions.csv"
OUT_DIR      = ZERONE_DIR / "zerone_results" / "w4_multiseed"
PRED_BACKUP  = OUT_DIR / "predictions"
OUT_CSV      = OUT_DIR / "w4_multiseed_results.csv"
OUT_TXT      = OUT_DIR / "w4_report.txt"

OUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_BACKUP.mkdir(parents=True, exist_ok=True)

# ---- Seeds ----------------------------------------------------------------
SEEDS = [0, 1, 2, 3, 42]

# ---------------------------------------------------------------------------

def compute_metrics(csv_path: Path) -> dict:
    """Read test_predictions.csv and compute key metrics."""
    df = pd.read_csv(csv_path)
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values
    prob_1 = df["prob_1"].values

    acc  = accuracy_score(y_true, y_pred) * 100.0
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        fp = fn = 0

    try:
        auc_val = roc_auc_score(y_true, prob_1)
    except Exception:
        auc_val = float("nan")

    return dict(acc=acc, f1=f1, fp=int(fp), fn=int(fn), auc=auc_val)


def backup_exists(seed: int) -> bool:
    path = PRED_BACKUP / f"seed_{seed}_test_predictions.csv"
    return path.exists()


def backup_predictions(seed: int):
    dst = PRED_BACKUP / f"seed_{seed}_test_predictions.csv"
    shutil.copy2(PRED_SRC, dst)
    print(f"  [backup] Saved predictions for seed={seed} -> {dst}")


def run_seed(seed: int) -> bool:
    """Train the model with the given seed as a subprocess.
    Returns True if the subprocess exited cleanly, False on error.
    Even on error, test_predictions.csv may have been written before the crash
    (predictions are saved before the SWA phase), so callers should still
    attempt to back up the file if it exists.
    """
    env = {**os.environ, "ZERONE_SEED": str(seed)}
    print(f"\n{'='*60}")
    print(f"  Running seed={seed}  ({time.strftime('%H:%M:%S')})")
    print(f"  Using Python: {TRAIN_PYTHON}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        [TRAIN_PYTHON, "zerone_train_resnet.py"],
        env=env,
        cwd=str(ZERONE_DIR),
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  [WARNING] seed={seed} exited with code {result.returncode} "
              f"after {elapsed/60:.1f} min. Will still collect predictions if available.")
        return False
    print(f"  seed={seed} finished in {elapsed/60:.1f} min")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print("W4 Multi-Seed Statistical Significance Experiment")
print(f"Seeds       : {SEEDS}")
print(f"ZERONE_DIR  : {ZERONE_DIR}")
print(f"Train Python: {TRAIN_PYTHON}")
print(f"Output      : {OUT_DIR}")
print()

# Step 1: Handle seed=42 first (may already exist from a prior run)
if backup_exists(42):
    print("seed=42: backup already exists, skipping re-training.")
else:
    if PRED_SRC.exists():
        print("seed=42: using existing test_predictions.csv (no re-training).")
        backup_predictions(42)
    else:
        print("seed=42: no existing predictions found, will train.")
        run_seed(42)
        backup_predictions(42)

# Step 2: Run remaining seeds
for seed in SEEDS:
    if seed == 42:
        continue  # Already handled above
    if backup_exists(seed):
        print(f"seed={seed}: backup already exists, skipping.")
        continue
    ok = run_seed(seed)
    # Even if the subprocess failed (e.g. SWA crash after predictions were saved),
    # try to collect the predictions file if it exists.
    if PRED_SRC.exists():
        backup_predictions(seed)
    elif not ok:
        print(f"  [WARNING] seed={seed}: no predictions file found after failed run.")

# Step 3: Collect metrics
print("\nCollecting metrics from per-seed prediction files...")
records = []
for seed in SEEDS:
    csv_path = PRED_BACKUP / f"seed_{seed}_test_predictions.csv"
    if not csv_path.exists():
        print(f"  WARNING: missing {csv_path}, skipping seed={seed}")
        continue
    m = compute_metrics(csv_path)
    m["seed"] = seed
    records.append(m)
    print(f"  seed={seed:2d}: acc={m['acc']:.2f}%  f1={m['f1']:.4f}  "
          f"FP={m['fp']:3d}  FN={m['fn']:3d}  AUC={m['auc']:.4f}")

if not records:
    print("ERROR: No records collected. Aborting.")
    sys.exit(1)

# Step 4: Save CSV
df_res = pd.DataFrame(records)[["seed", "acc", "f1", "fp", "fn", "auc"]]
df_res.to_csv(OUT_CSV, index=False)
print(f"\nSaved per-seed CSV: {OUT_CSV}")

# Step 5: Compute mean +/- std
accs = np.array([r["acc"] for r in records])
f1s  = np.array([r["f1"]  for r in records])
fps  = np.array([r["fp"]  for r in records], dtype=float)
fns  = np.array([r["fn"]  for r in records], dtype=float)
aucs = np.array([r["auc"] for r in records])

# Step 6: Build report text
lines = []
lines.append("W4 Multi-Seed Statistical Significance Report")
lines.append("=" * 60)
lines.append("Model   : SpecRas ResNet18 + SWA")
lines.append(f"Seeds   : {SEEDS}")
lines.append(f"Date    : {time.strftime('%Y-%m-%d %H:%M:%S')}")
lines.append("")
lines.append(f"{'Seed':<6} {'Acc%':>7} {'F1':>7} {'FP':>5} {'FN':>5} {'AUC':>7}")
lines.append("-" * 45)
for r in sorted(records, key=lambda x: x["seed"]):
    lines.append(
        f"{r['seed']:<6} {r['acc']:>7.2f} {r['f1']:>7.4f} "
        f"{r['fp']:>5d} {r['fn']:>5d} {r['auc']:>7.4f}"
    )
lines.append("-" * 45)
lines.append(
    f"{'Mean':<6} {accs.mean():>7.2f} {f1s.mean():>7.4f} "
    f"{fps.mean():>5.1f} {fns.mean():>5.1f} {aucs.mean():>7.4f}"
)
lines.append(
    f"{'Std':<6} {accs.std():>7.2f} {f1s.std():>7.4f} "
    f"{fps.std():>5.1f} {fns.std():>5.1f} {aucs.std():>7.4f}"
)
lines.append("")
lines.append("Summary (mean +/- std):")
lines.append(f"  Accuracy : {accs.mean():.2f} +/- {accs.std():.2f} %")
lines.append(f"  Macro-F1 : {f1s.mean():.4f} +/- {f1s.std():.4f}")
lines.append(f"  FP       : {fps.mean():.1f} +/- {fps.std():.1f}")
lines.append(f"  FN       : {fns.mean():.1f} +/- {fns.std():.1f}")
lines.append(f"  AUC      : {aucs.mean():.4f} +/- {aucs.std():.4f}")
lines.append("")

report_text = "\n".join(lines)
print("\n" + report_text)

OUT_TXT.write_text(report_text, encoding="utf-8")
print(f"Saved report: {OUT_TXT}")
print("\nDone.")
