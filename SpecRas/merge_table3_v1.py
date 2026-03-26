# -*- coding: utf-8 -*-
"""
merge_table3.py  —  放在 zerone 目录下运行（需能访问两边的 results 文件夹）
================================================================================
合并 baseline_supervised.py 和 baseline_flatvae.py 的输出 CSV，
加上原论文中三种方法的已知结果，生成：

  1. merged_table3/table3_full.csv          — 完整 CSV
  2. merged_table3/table3_full.tex          — LaTeX 表格
  3. merged_table3/table3_discussion.tex    — 消融讨论段落
  4. merged_table3/table3_barplot_en.png    — 分组柱状图（英文，IEEE 风格）
  5. merged_table3/table3_barplot_zh.png    — 分组柱状图（中文）
  6. merged_table3/table3_radar_en.png      — 雷达图（英文）
  7. merged_table3/table3_radar_zh.png      — 雷达图（中文）

运行：
    python merge_table3.py
================================================================================
"""

import csv
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300

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

# ── 显示用短名（柱状图/雷达图标签） ──
SHORT_NAMES = {
    "RawVec_ResNet18":               "RawVec\nResNet18",
    "1DCNN_Wen2018":                 "1D-CNN\n(Wen '18)",
    "FlatVAE (GAP, no spatial)":     "FlatVAE\n(GAP)",
    "SpecRas + ResNet18 (best-val)": "SpecRas\n(best-val)",
    "SpecRas + ResNet18 (SWA)":      "SpecRas\n(SWA)",
    "HetSpatVAE (SpatialResNetVAE)": "HetSpatVAE",
}
SHORT_NAMES_ZH = {
    "RawVec_ResNet18":               "原始向量\nResNet18",
    "1DCNN_Wen2018":                 "1D-CNN\n(Wen '18)",
    "FlatVAE (GAP, no spatial)":     "平坦VAE\n(GAP)",
    "SpecRas + ResNet18 (best-val)": "SpecRas\n(best-val)",
    "SpecRas + ResNet18 (SWA)":      "SpecRas\n(SWA)",
    "HetSpatVAE (SpatialResNetVAE)": "HetSpatVAE",
}

# 颜色：基线用灰/蓝，提出方法用橙/红
COLORS_BASELINE = ["#9E9E9E", "#78909C", "#607D8B"]
COLORS_PROPOSED = ["#FFB74D", "#FF9800", "#E65100"]


def load_csv(path):
    rows = []
    if not path.exists():
        print(f"  [WARN] 未找到 {path}，跳过")
        return rows
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            for k in ["acc", "pre_n", "rec_n", "rec_f", "macro_f1"]:
                if k in r:
                    r[k] = float(r[k])
            rows.append(r)
    return rows


def get_short(method, lang="en"):
    table = SHORT_NAMES if lang == "en" else SHORT_NAMES_ZH
    return table.get(method, method[:12])


# ════════════════════════════════════════════════════════════════════════════
# 可视化 1：分组柱状图 (Accuracy + Macro-F1)
# ════════════════════════════════════════════════════════════════════════════
def plot_barplot(all_rows, baselines, proposed, lang="en"):
    """IEEE 风格分组柱状图：每个方法两根柱子 (Acc, Macro-F1)"""
    n = len(all_rows)
    names = [get_short(r["method"], lang) for r in all_rows]
    accs = [r["acc"] * 100 for r in all_rows]
    f1s  = [r["macro_f1"] * 100 for r in all_rows]

    n_bl = len(baselines)
    colors = []
    for i in range(n):
        if i < n_bl:
            colors.append(COLORS_BASELINE[i % len(COLORS_BASELINE)])
        else:
            colors.append(COLORS_PROPOSED[(i - n_bl) % len(COLORS_PROPOSED)])

    x = np.arange(n)
    w = 0.32

    fig, ax = plt.subplots(figsize=(max(8, n * 1.4), 5))

    bars1 = ax.bar(x - w/2, accs, w, color=colors, edgecolor="white", linewidth=0.8, label="Accuracy (%)" if lang == "en" else "准确率 (%)")
    bars2 = ax.bar(x + w/2, f1s,  w, color=colors, edgecolor="white", linewidth=0.8, alpha=0.65,
                   hatch="//", label="Macro-F1 (%)" if lang == "en" else "Macro-F1 (%)")

    # 数值标签
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # 分隔线：基线 vs 提出方法
    if n_bl > 0 and n_bl < n:
        ax.axvline(x=n_bl - 0.5, color="#BDBDBD", linestyle="--", linewidth=1)
        y_top = ax.get_ylim()[1]
        if lang == "en":
            ax.text(n_bl/2 - 0.5, y_top - 2, "Baselines", ha="center", fontsize=9, color="#757575", style="italic")
            ax.text((n_bl + n)/2 - 0.5, y_top - 2, "Proposed", ha="center", fontsize=9, color="#BF360C", style="italic")
        else:
            ax.text(n_bl/2 - 0.5, y_top - 2, "基线方法", ha="center", fontsize=9, color="#757575")
            ax.text((n_bl + n)/2 - 0.5, y_top - 2, "本文方法", ha="center", fontsize=9, color="#BF360C")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("%" if lang == "en" else "百分比 (%)", fontsize=11)
    title = "Comparison of Diagnostic Methods on Held-out Test Set" if lang == "en" else "各诊断方法在测试集上的性能对比"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / f"table3_barplot_{lang}.png")
    fig.savefig(OUT / f"table3_barplot_{lang}.svg")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# 可视化 2：雷达图（全部 5 项指标）
# ════════════════════════════════════════════════════════════════════════════
def plot_radar(all_rows, baselines, proposed, lang="en"):
    """IEEE 风格雷达图：每个方法一条闭合折线，5 个轴"""
    metrics = ["acc", "pre_n", "rec_n", "rec_f", "macro_f1"]
    if lang == "en":
        labels = ["Accuracy", "Pre.(N)", "Rec.(N)", "Rec.(F)", "Macro-F1"]
    else:
        labels = ["准确率", "正常精确率", "正常召回率", "故障召回率", "Macro-F1"]

    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    n_bl = len(baselines)
    for i, r in enumerate(all_rows):
        values = [r[m] for m in metrics]
        values += values[:1]
        name = get_short(r["method"], lang).replace("\n", " ")
        if i < n_bl:
            color = COLORS_BASELINE[i % len(COLORS_BASELINE)]
            ls = "--"; lw = 1.5; alpha = 0.5
        else:
            color = COLORS_PROPOSED[(i - n_bl) % len(COLORS_PROPOSED)]
            ls = "-"; lw = 2.5; alpha = 0.15
        ax.plot(angles, values, ls, color=color, linewidth=lw, label=name)
        ax.fill(angles, values, color=color, alpha=alpha)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7.5, color="#999")
    ax.grid(True, linestyle=":", alpha=0.5)

    title = "Multi-metric Radar Comparison" if lang == "en" else "多指标雷达对比图"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(OUT / f"table3_radar_{lang}.png", bbox_inches="tight")
    fig.savefig(OUT / f"table3_radar_{lang}.svg", bbox_inches="tight")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# 主程序
# ════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  合并 Table 3 + 生成可视化")
    print("=" * 60)

    sup_csv = Path("./baseline_supervised_results/summary_supervised_baselines.csv")
    vae_csv = Path("./baseline_flatvae_results/summary_flatvae.csv")

    baselines = load_csv(sup_csv) + load_csv(vae_csv)
    all_rows = baselines + PROPOSED

    if not all_rows:
        print("[ERROR] 未读取到任何数据"); return

    # ── 1. CSV ──
    csv_path = OUT / "table3_full.csv"
    fields = ["method", "paradigm", "acc", "pre_n", "rec_n", "rec_f", "macro_f1"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]) for k in fields})
    print(f"  CSV → {csv_path}")

    # ── 2. LaTeX ──
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

    # ── 3. 讨论段落 ──
    disc_path = OUT / "table3_discussion.tex"
    def find(name_substr):
        for r in all_rows:
            if name_substr.lower() in r["method"].lower(): return r
        return None

    rawvec  = find("rawvec")
    swa     = find("swa")
    cnn1d   = find("1d")
    flat    = find("flatvae") or find("flat")
    spatial = find("hetspatvae") or find("spatial")

    with open(disc_path, "w", encoding="utf-8") as f:
        f.write("% ---- 消融讨论段落（根据实际结果自动填充） ----\n\n")
        if rawvec and swa:
            ga = (swa["acc"] - rawvec["acc"]) * 100
            gf = swa["macro_f1"] - rawvec["macro_f1"]
            f.write(f"\\paragraph{{Ablation: SpecRas rasterisation}}\n")
            f.write(f"The raster-stripe encoding improves test accuracy by "
                    f"$+${ga:.2f}\\,pp and macro-F1 by $+${gf:.3f} "
                    f"over the RawVec-ResNet18 baseline, confirming that the spatial "
                    f"layout of PSD bins enables convolutional filters to exploit "
                    f"local spectral correlations.\n\n")
        if flat and spatial:
            ga = (spatial["acc"] - flat["acc"]) * 100
            gf = spatial["macro_f1"] - flat["macro_f1"]
            f.write(f"\\paragraph{{Ablation: spatial latent map}}\n")
            f.write(f"HetSpatVAE achieves $+${ga:.2f}\\,pp accuracy and "
                    f"$+${gf:.3f} macro-F1 over FlatVAE, supporting the claim "
                    f"that the $7\\times7$ spatial structure is critical for "
                    f"localising time-frequency anomalies.\n\n")
        if cnn1d and swa:
            ga = (swa["acc"] - cnn1d["acc"]) * 100
            f.write(f"\\paragraph{{External baseline: 1D-CNN}}\n")
            f.write(f"SpecRas (SWA) outperforms the 1D-CNN by $+${ga:.2f}\\,pp in accuracy, "
                    f"indicating that 2-D raster images preserve multi-scale context "
                    f"that 1-D convolutions cannot efficiently model.\n\n")
    print(f"  讨论段落 → {disc_path}")

    # ── 4. 可视化 ──
    print("\n  生成可视化 ...")
    for lang in ["en", "zh"]:
        plot_barplot(all_rows, baselines, PROPOSED, lang)
        plot_radar(all_rows, baselines, PROPOSED, lang)
    print(f"  柱状图 → {OUT}/table3_barplot_en/zh.png")
    print(f"  雷达图 → {OUT}/table3_radar_en/zh.png")
    print(f"  (同时生成 SVG 矢量版)")

    print(f"\n  全部输出 → {OUT}")
    print("  完成！")


if __name__ == "__main__":
    main()
