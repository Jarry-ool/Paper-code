# -*- coding: utf-8 -*-
"""
merge_table3.py  —  放在任意位置运行
================================================================================
合并 baseline_supervised.py 和 baseline_flatvae.py 的输出 CSV，
加上原论文中三种方法的已知结果，生成最终的 Table 3。

输入文件（自动搜索或手动指定）：
  - baseline_supervised_results/summary_supervised_baselines.csv
  - baseline_flatvae_results/summary_flatvae.csv

输出：
  - merged_table3/table3_full.csv      — 完整 CSV
  - merged_table3/table3_full.tex      — 可直接粘入论文的 LaTeX 表格
  - merged_table3/table3_discussion.tex — 消融讨论段落（带占位符）

运行：
    python merge_table3.py
================================================================================
"""

import csv
from pathlib import Path

OUT = Path("./merged_table3").resolve()
OUT.mkdir(parents=True, exist_ok=True)

# ── 原论文已有的三行结果 ──
PROPOSED = [
    dict(method="SpecRas + ResNet18 (best-val)", paradigm="Supervised",
         acc=0.9425, pre_n=1.000, rec_n=0.907, rec_f=1.000, macro_f1=0.941),
    dict(method="SpecRas + ResNet18 (SWA)", paradigm="Supervised",
         acc=0.9700, pre_n=1.000, rec_n=0.951, rec_f=1.000, macro_f1=0.969),
    dict(method="HetSpatVAE (SpatialResNetVAE)", paradigm="One-class",
         acc=0.9825, pre_n=1.000, rec_n=0.972, rec_f=1.000, macro_f1=0.982),
]

def load_csv(path):
    rows = []
    if not path.exists():
        print(f"  [WARN] 未找到 {path}，跳过")
        return rows
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            for k in ["acc", "pre_n", "rec_n", "rec_f", "macro_f1"]:
                r[k] = float(r[k])
            rows.append(r)
    return rows

def main():
    print("=" * 60)
    print("  合并 Table 3")
    print("=" * 60)

    # 读取基线结果
    sup_csv = Path("./baseline_supervised_results/summary_supervised_baselines.csv")
    vae_csv = Path("./baseline_flatvae_results/summary_flatvae.csv")

    baselines = load_csv(sup_csv) + load_csv(vae_csv)
    all_rows = baselines + PROPOSED

    # ── CSV ──
    csv_path = OUT / "table3_full.csv"
    fields = ["method", "paradigm", "acc", "pre_n", "rec_n", "rec_f", "macro_f1"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]) for k in fields})
    print(f"  CSV → {csv_path}")

    # ── LaTeX ──
    best = {}
    for col in ["acc", "pre_n", "rec_n", "rec_f", "macro_f1"]:
        best[col] = max(r[col] for r in all_rows)

    tex_path = OUT / "table3_full.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(r"""% ---- Auto-generated Table 3 ----
\begin{table}[H]
  \centering
  \caption{%
    Diagnostic performance on the held-out test set
    (246 normal / 154 fault, $N=400$; transformers 134 and 135).
    Best result per metric in \textbf{bold}.
    $^\dagger$: trained without fault labels.
    $^\ddagger$: external baseline.
  }
  \label{tab:main_results}
  \small
  \renewcommand{\arraystretch}{1.25}
  \begin{tabular}{@{}p{4.2cm}p{1.8cm}ccccc@{}}
    \toprule
    Method & Paradigm
      & Acc.$\uparrow$
      & Pre.(N)$\uparrow$
      & Rec.(N)$\uparrow$
      & Rec.(F)$\uparrow$
      & Macro-F1$\uparrow$ \\
    \midrule
""")
        # 先写基线，再写提出方法，中间加 midrule
        def write_row(r):
            name = r["method"]
            suffix = ""
            if "FlatVAE" in name or "HetSpatVAE" in name:
                suffix = "$^\\dagger$"
            if "1D" in name or "Wen" in name:
                suffix = "$^\\ddagger$"
            vals = []
            for col, fmt in [("acc", "pct"), ("pre_n", ".3f"), ("rec_n", ".3f"), ("rec_f", ".3f"), ("macro_f1", ".3f")]:
                v = r[col]
                s = f"{v*100:.2f}\\%" if fmt == "pct" else f"{v:{fmt}}"
                if abs(v - best[col]) < 1e-4:
                    s = f"\\textbf{{{s}}}"
                vals.append(s)
            f.write(f"    {name}{suffix}\n")
            f.write(f"      & {r['paradigm']}   & {' & '.join(vals)} \\\\\n")

        for r in baselines:
            write_row(r)
        f.write("    \\midrule\n")
        for r in PROPOSED:
            write_row(r)

        f.write(r"""    \bottomrule
  \end{tabular}
\end{table}
""")
    print(f"  LaTeX → {tex_path}")

    # ── 讨论段落模板 ──
    disc_path = OUT / "table3_discussion.tex"
    # 尝试自动计算增益
    def find(name_substr):
        for r in all_rows:
            if name_substr.lower() in r["method"].lower():
                return r
        return None

    rawvec = find("rawvec")
    swa    = find("swa")
    cnn1d  = find("1d")
    flat   = find("flatvae") or find("flat")
    spatial= find("hetspatvae") or find("spatial")

    with open(disc_path, "w", encoding="utf-8") as f:
        f.write("% ---- 消融讨论段落（根据实际结果自动填充） ----\n\n")

        if rawvec and swa:
            gain_acc = (swa["acc"] - rawvec["acc"]) * 100
            gain_f1  = swa["macro_f1"] - rawvec["macro_f1"]
            f.write(f"\\paragraph{{Ablation: SpecRas rasterisation}}\n")
            f.write(f"The raster-stripe encoding improves test accuracy by "
                    f"$+${gain_acc:.2f}\\,pp and macro-F1 by $+${gain_f1:.3f} "
                    f"over the RawVec-ResNet18 baseline, confirming that the spatial "
                    f"layout of PSD bins enables convolutional filters to exploit "
                    f"local spectral correlations.\n\n")

        if flat and spatial:
            gain_acc = (spatial["acc"] - flat["acc"]) * 100
            gain_f1  = spatial["macro_f1"] - flat["macro_f1"]
            f.write(f"\\paragraph{{Ablation: spatial latent map}}\n")
            f.write(f"HetSpatVAE achieves $+${gain_acc:.2f}\\,pp accuracy and "
                    f"$+${gain_f1:.3f} macro-F1 over FlatVAE, supporting the claim "
                    f"that the $7\\times7$ spatial structure is critical for "
                    f"localising time-frequency anomalies.\n\n")

        if cnn1d and swa:
            gain_acc = (swa["acc"] - cnn1d["acc"]) * 100
            f.write(f"\\paragraph{{External baseline: 1D-CNN}}\n")
            f.write(f"SpecRas (SWA) outperforms the 1D-CNN by $+${gain_acc:.2f}\\,pp in accuracy, "
                    f"indicating that 2-D raster images preserve multi-scale context "
                    f"that 1-D convolutions cannot efficiently model.\n\n")

    print(f"  讨论段落 → {disc_path}")
    print("\n  完成！")


if __name__ == "__main__":
    main()
