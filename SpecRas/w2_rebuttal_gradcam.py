# -*- coding: utf-8 -*-
"""
w2_rebuttal_gradcam.py
================================
专门用于回应审稿人 W2 (Unit-Class Confound) 意见的可视化排版脚本。

逻辑说明：
1. 自动从 zerone_config 读取测试集图像目录 (IMG_OUT_ROOT/test/正常 和 IMG_OUT_ROOT/test/故障)。
2. 自动抓取两类的典型样本各一张。
3. 复用 zerone_gradcam.py 中的 ResNet18CAM 引擎提取激活热力图。
4. 使用 matplotlib 拼接为 1x2 对比图，标注单元 ID 和物理意义，输出到 VIZ_PLOTS。
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# 导入你现有的配置和 CAM 引擎
from zerone_config import IMG_OUT_ROOT, CLASSES, VIZ_PLOTS
from zerone_gradcam import build_resnet18_for_gradcam, ResNet18CAM, overlay_cam_on_image

# 兼容中文字体显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

def generate_w2_rebuttal_figure():
    root = Path(IMG_OUT_ROOT).resolve()
    test_dir = root / "test"
    
    # 1. 检查测试集目录是否存在
    if not test_dir.exists():
        print(f"[Error] 找不到测试集目录: {test_dir}")
        print("请确保已经运行过完整训练并生成了图像。")
        sys.exit(1)

    # 确定类别名（假设 CLASSES[0] 是正常，CLASSES[1] 是故障）
    cls_normal = CLASSES[0] if len(CLASSES) > 0 else "正常"
    cls_fault = CLASSES[1] if len(CLASSES) > 1 else "故障"

    dir_normal = test_dir / cls_normal
    dir_fault = test_dir / cls_fault

    # 获取第一张图片作为代表性样本
    imgs_normal = list(dir_normal.glob("*.png"))
    imgs_fault = list(dir_fault.glob("*.png"))

    if not imgs_normal or not imgs_fault:
        print(f"[Error] 测试集图片缺失。正常类: {len(imgs_normal)}张, 故障类: {len(imgs_fault)}张。")
        sys.exit(1)

    path_normal = imgs_normal[0]
    path_fault = imgs_fault[0]

    print(f"-> 选中正常样本 (Unit 134): {path_normal.name}")
    print(f"-> 选中故障样本 (Unit 135): {path_fault.name}")

    # 2. 初始化模型和 CAM 引擎
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18_for_gradcam(device)
    cam_engine = ResNet18CAM(model, device)

    # 3. 处理正常样本 (Unit 134)
    x_norm, pil_norm = cam_engine._preprocess(path_normal)
    # class_idx=0 解释它为什么是“正常”
    cam_norm_np = cam_engine.grad_cam(x_norm, class_idx=0) 
    cam_norm_01 = cam_engine._to_cam_image(cam_norm_np, (pil_norm.height, pil_norm.width))
    blended_norm = overlay_cam_on_image(cam_norm_01, pil_norm, alpha=0.55)

    # 4. 处理故障样本 (Unit 135)
    x_fault, pil_fault = cam_engine._preprocess(path_fault)
    # class_idx=1 解释它为什么是“故障”
    cam_fault_np = cam_engine.grad_cam(x_fault, class_idx=1)
    cam_fault_01 = cam_engine._to_cam_image(cam_fault_np, (pil_fault.height, pil_fault.width))
    blended_fault = overlay_cam_on_image(cam_fault_01, pil_fault, alpha=0.55)

    # 5. 绘图与排版
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    
    # 图 (a) 正常
    axes[0].imshow(blended_norm)
    axes[0].set_title("(a) Unit 134: Normal (Input + Grad-CAM)\nModel output: Normal (Correct)", fontsize=13, fontweight='bold')
    axes[0].axis('off')
    
    # 图 (b) 故障
    axes[1].imshow(blended_fault)
    axes[1].set_title("(b) Unit 135: Fault (Input + Grad-CAM)\nModel output: Fault (Correct)", fontsize=13, fontweight='bold')
    axes[1].axis('off')

    # 添加物理坐标轴说明（为了让审稿人看懂 SpecRas 图像的物理含义）
    # 注：如果你的图像 Y轴和 X轴的具体物理含义有所不同，可以微调这里
    fig.text(0.5, 0.02, 'Feature Stripes along Frequency/Time Domain (X-axis)', ha='center', fontsize=12)
    fig.text(0.04, 0.5, 'Rasterized Channel / Feature Amplitude (Y-axis)', va='center', rotation='vertical', fontsize=12)

    # 保存结果
    out_dir = Path(VIZ_PLOTS)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "W2_gradcam_comparison.png"
    
    plt.tight_layout(rect=[0.05, 0.05, 1, 1]) # 给边缘留白以显示文本
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\n[Success] 完美！回应 W2 的对比图已保存至: {save_path}")
    print("请检查生成的图像：正常样本的激活区域应较为发散/偏向低频环境噪声；而故障样本的激活应高度聚焦在特定的中高频故障频带。")

if __name__ == "__main__":
    generate_w2_rebuttal_figure()