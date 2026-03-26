# zerone_eval_empirical.py
# ========================
# 兼容 zerone_make_images.py 生成的新 scores 文件：
#   SCORES_OUT_ROOT/<split>/<split>_scores.csv
#   列：["split","class","H","C","V","HF","HI","severity"]
#
# 本脚本自动生成两种经验预测列：
#   1) pred_severity: severity >=1 -> 故障, else 正常
#   2) pred_hi_80   : HI > 80     -> 故障, else 正常
#
# 并对 val/test split 计算 acc/F1 和混淆矩阵。

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from zerone_config import SCORES_OUT_ROOT

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def to_binary_normal_fault_from_class(s: pd.Series) -> pd.Series:
    """
    根据 class 列（含“正常”字样视为正常），其余视为故障。
    """
    s = s.astype(str)
    return s.apply(lambda x: "正常" if "正常" in x else "故障")


def save_confmat_png(cm: np.ndarray, classes: List[str], outpath: Path, title: str):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(4.6, 4.0))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest", cmap="viridis")

    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="black", fontsize=11)

    ax.set_xlabel("预测 (Predicted)", fontsize=12)
    ax.set_ylabel("真实 (True)", fontsize=12)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def add_empirical_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    在原 DataFrame 上增加两列预测：
        - pred_severity: severity >=1 -> 故障, else 正常
        - pred_hi_80   : HI > 80     -> 故障, else 正常
    """
    df = df.copy()

    # 真实标签来自 class 列
    df["y_true"] = to_binary_normal_fault_from_class(df["class"])

    # 预测 1：基于 severity（二值：0 正常/注意；1 异常/严重）
    if "severity" in df.columns:
        df["pred_severity"] = df["severity"].apply(
            lambda v: "故障" if float(v) >= 1.0 else "正常"
        )
    else:
        df["pred_severity"] = "正常"

    # 预测 2：基于 HI 阈值（>80 视为异常/故障）
    if "HI" in df.columns:
        df["pred_hi_80"] = df["HI"].apply(
            lambda v: "故障" if float(v) > 80.0 else "正常"
        )
    else:
        df["pred_hi_80"] = "正常"

    return df


def eval_and_save_split(df: pd.DataFrame, split_name: str, scores_root: Path):
    """
    对一个 split 计算两列预测的 acc/f1，保存到该 split 目录下：
      - <split>/metrics_<split>.csv（长表）
      - <split>/accuracy_matrix_<split>.csv（行=split，列=预测列）
      - <split>/f1_matrix_<split>.csv
      - <split>/confmat/confmat__<split>__<列名>.png
    """
    out_dir = scores_root / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    if df is None or df.empty:
        return results

    df = add_empirical_predictions(df)
    y_true = df["y_true"]
    pred_cols = ["pred_severity", "pred_hi_80"]

    row_for_acc = {}
    row_for_f1 = {}

    labels = ["正常", "故障"]

    for col in pred_cols:
        y_pred = df[col]
        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro")

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_path = out_dir / "confmat" / f"confmat__{split_name}__{col}.png"
        save_confmat_png(cm, labels, cm_path, title=f"{split_name} | {col}")

        results.append({
            "split": split_name,
            "pred_col": col,
            "accuracy": acc,
            "f1_macro": f1m,
            "confmat_png": str(cm_path)
        })
        row_for_acc[col] = acc
        row_for_f1[col] = f1m
        print(f"[{split_name}] {col} -> acc={acc:.4f}  f1_macro={f1m:.4f}  ({cm_path.name})")

    # 长表
    df_metrics = pd.DataFrame(results)
    df_metrics.to_csv(out_dir / f"metrics_{split_name}.csv",
                      index=False, encoding="utf-8-sig")

    # 单行矩阵
    acc_matrix = pd.DataFrame([row_for_acc], index=[split_name])
    f1_matrix = pd.DataFrame([row_for_f1], index=[split_name])
    (out_dir / "confmat").mkdir(parents=True, exist_ok=True)
    acc_matrix.to_csv(out_dir / "confmat" / f"accuracy_matrix_{split_name}.csv",
                      encoding="utf-8-sig")
    f1_matrix.to_csv(out_dir / "confmat" / f"f1_matrix_{split_name}.csv",
                     encoding="utf-8-sig")

    return results


def main():
    scores_root = Path(SCORES_OUT_ROOT).resolve()

    val_path = scores_root / "val" / "val_scores.csv"
    test_path = scores_root / "test" / "test_scores.csv"

    all_for_summary = []

    if val_path.exists():
        val_rows = pd.read_csv(val_path)
        print("VAL Columns:", val_rows.columns.tolist())
        all_for_summary.extend(eval_and_save_split(val_rows, "val", scores_root))
    else:
        print("[WARN] 未找到", val_path)

    if test_path.exists():
        test_rows = pd.read_csv(test_path)
        print("TEST Columns:", test_rows.columns.tolist())
        all_for_summary.extend(eval_and_save_split(test_rows, "test", scores_root))
    else:
        print("[WARN] 未找到", test_path, "，跳过 TEST。")

    if all_for_summary:
        summary_df = pd.DataFrame(all_for_summary)
        summary_path = scores_root / "metrics_summary.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"[OK] 总汇总已保存：{summary_path}")


if __name__ == "__main__":
    main()
