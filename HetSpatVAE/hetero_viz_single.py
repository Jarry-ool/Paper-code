# -*- coding: utf-8 -*-
import torch
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import pywt
import os
from pathlib import Path
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm

# 引入配置和模型
import hetero_config as cfg
from hetero_model import SpatialResNetVAE

# --- 绘图全局设置 (IEEE/Nature 风格) ---
# 尝试设置英文字体为 Times New Roman 或 Arial
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'stix' # 数学公式字体
plt.rcParams['axes.linewidth'] = 1.0      # 坐标轴线宽
plt.rcParams['xtick.direction'] = 'in'    # 刻度向内
plt.rcParams['ytick.direction'] = 'in'

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def load_snapshot_data(file_path):
    """读取单个 JSONL 文件作为一组快照"""
    lines = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [json.loads(line.strip()) for line in f if line.strip()]
    except Exception as e:
        print(f"读取失败: {e}")
        return None, []
    
    if not lines:
        return None, []

    # 获取元数据
    timestamp = lines[0].get('data_time', 'unknown_time').replace(':', '-').replace('.', '-')
    return timestamp, lines

def process_signal(json_data):
    """处理信号生成 Tensor"""
    sig_str = json_data.get('signal_value', '')
    signal = np.fromstring(sig_str, sep=',')
    
    # 长度对齐
    if len(signal) > cfg.SIGNAL_LEN: signal = signal[:cfg.SIGNAL_LEN]
    else: signal = np.pad(signal, (0, cfg.SIGNAL_LEN - len(signal)))
    
    # CWT 生成
    scales = np.arange(1, 129)
    coef, _ = pywt.cwt(signal, scales, 'morl')
    scalogram = np.log1p(np.abs(coef))
    cwt_img = cv2.resize(scalogram, (cfg.INPUT_SIZE, cfg.INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    cwt_img_norm = (cwt_img - cwt_img.min()) / (cwt_img.max() - cwt_img.min() + 1e-8)
    
    # Zerone (模拟特征图，实际应调用 hetero_data 的逻辑)
    zerone_img = np.random.rand(cfg.INPUT_SIZE, cfg.INPUT_SIZE).astype(np.float32) * 0.1 # 仅演示用占位
    
    # Tensor
    ctx_img = np.zeros((cfg.INPUT_SIZE, cfg.INPUT_SIZE), dtype=np.float32)
    tensor = np.stack([cwt_img_norm, zerone_img, ctx_img], axis=0)
    tensor = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)
    
    return signal, tensor, cwt_img_norm, zerone_img

def plot_summary(save_path, sensor_data_list, timestamp, lang='en'):
    """
    生成总览图：左侧宽幅波形，右侧 CWT 对比
    """
    num_sensors = len(sensor_data_list)
    # 设置画布：高一点，宽一点
    fig = plt.figure(figsize=(24, 4 * num_sensors))
    # 布局：左边 60% 放波形，右边 20% 原图，20% 重构
    gs = gridspec.GridSpec(num_sensors, 4, width_ratios=[3, 1, 1, 0.1], wspace=0.1, hspace=0.3)
    
    titles = {
        'en': {'sig': 'Time-Domain Waveform', 'orig': 'Original Time-Frequency', 'recon': 'Reconstructed T-F', 'axis': 'Amplitude'},
        'cn': {'sig': '时域振动波形', 'orig': '原始时频图 (CWT)', 'recon': '重构时频图 (VAE)', 'axis': '幅值'}
    }
    t = titles[lang]

    for i, item in enumerate(sensor_data_list):
        sensor_id = item['id']
        raw_sig = item['raw']
        orig_cwt = item['orig_cwt']
        recon_cwt = item['recon_cwt']
        
        # 1. 波形图 (Ax1)
        ax_sig = fig.add_subplot(gs[i, 0])
        ax_sig.plot(raw_sig, color='#00509E', linewidth=1.0, alpha=0.9) # IEEE 常用深蓝
        ax_sig.set_xlim(0, len(raw_sig))
        
        # 美化坐标轴
        ax_sig.spines['top'].set_visible(False)
        ax_sig.spines['right'].set_visible(False)
        ax_sig.set_ylabel(f"Sensor {sensor_id}\n({t['axis']})", fontsize=14, fontweight='bold')
        ax_sig.grid(True, which='major', linestyle=':', alpha=0.6)
        
        if i == 0: ax_sig.set_title(t['sig'], fontsize=16, pad=15)
        
        # 2. 原始 CWT (Ax2)
        ax_orig = fig.add_subplot(gs[i, 1])
        im1 = ax_orig.imshow(orig_cwt, cmap='jet', aspect='auto')
        ax_orig.set_xticks([])
        ax_orig.set_yticks([])
        if i == 0: ax_orig.set_title(t['orig'], fontsize=16, pad=15)
        
        # 3. 重构 CWT (Ax3)
        ax_recon = fig.add_subplot(gs[i, 2])
        im2 = ax_recon.imshow(recon_cwt, cmap='jet', aspect='auto')
        ax_recon.set_xticks([])
        ax_recon.set_yticks([])
        if i == 0: ax_recon.set_title(t['recon'], fontsize=16, pad=15)
        
        # Colorbar (Ax4) - 仅在每行最后加一个小条
        ax_cb = fig.add_subplot(gs[i, 3])
        plt.colorbar(im2, cax=ax_cb)
    
    # 总标题
    fig.suptitle(f"Multi-channel Transformer Vibration Snapshot\nTime: {timestamp}", fontsize=20, y=0.92)
    
    filename = f"Summary_Snapshot_{lang.upper()}.png"
    plt.savefig(os.path.join(save_path, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存总图: {filename}")

def plot_individual_sensor(base_folder, sensor_id, raw, orig_cwt, recon_cwt, orig_z, recon_z, lang='en'):
    """
    为单个传感器生成详细的三张图：波形、CWT对比、Zerone对比
    """
    sensor_dir = ensure_dir(os.path.join(base_folder, f"sensor_{sensor_id}"))
    
    # 文本字典
    txt = {
        'en': {
            'wave_title': f'Sensor {sensor_id} - High Resolution Waveform',
            'cwt_title': f'Sensor {sensor_id} - Time-Frequency Analysis',
            'zerone_title': f'Sensor {sensor_id} - Feature Manifold (Zerone)',
            'orig': 'Input', 'recon': 'Reconstruction',
            'xlabel': 'Sample Index', 'ylabel': 'Acceleration (g)'
        },
        'cn': {
            'wave_title': f'传感器 {sensor_id} - 高分辨率波形图',
            'cwt_title': f'传感器 {sensor_id} - 时频特征重构分析',
            'zerone_title': f'传感器 {sensor_id} - 物理流形映射 (Zerone)',
            'orig': '模型输入', 'recon': '模型重构',
            'xlabel': '采样点', 'ylabel': '加速度 (g)'
        }
    }
    t = txt[lang]

    # --- 图1: 纯波形图 (适合放正文) ---
    plt.figure(figsize=(12, 5))
    plt.plot(raw, color='#2C3E50', linewidth=1.2)
    plt.title(t['wave_title'], fontsize=14)
    plt.xlabel(t['xlabel'], fontsize=12)
    plt.ylabel(t['ylabel'], fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.margins(x=0)
    plt.tight_layout()
    plt.savefig(os.path.join(sensor_dir, f"1_Waveform_{lang}.png"), dpi=300)
    plt.close()

    # --- 图2: CWT 对比图 (重点) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1 = axes[0]
    im1 = ax1.imshow(orig_cwt, cmap='jet', aspect='auto')
    ax1.set_title(f"{t['orig']} (CWT)", fontsize=12)
    ax1.axis('off')
    
    ax2 = axes[1]
    im2 = ax2.imshow(recon_cwt, cmap='jet', aspect='auto')
    ax2.set_title(f"{t['recon']} (CWT)", fontsize=12)
    ax2.axis('off')
    
    # 共享 colorbar
    cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
    
    plt.suptitle(t['cwt_title'], fontsize=16)
    plt.savefig(os.path.join(sensor_dir, f"2_TF_Analysis_{lang}.png"), dpi=300)
    plt.close()

    # --- 图3: Zerone 特征图 (解释这是物理基因) ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1 = axes[0]
    # 使用 viridis 或 inferno 这种看起来比较"科学"的配色
    im1 = ax1.imshow(orig_z, cmap='inferno', aspect='auto')
    ax1.set_title(f"{t['orig']} (Feature Grid)", fontsize=12)
    ax1.axis('off')
    
    ax2 = axes[1]
    im2 = ax2.imshow(recon_z, cmap='inferno', aspect='auto')
    ax2.set_title(f"{t['recon']} (Feature Grid)", fontsize=12)
    ax2.axis('off')
    
    plt.suptitle(t['zerone_title'], fontsize=16)
    plt.savefig(os.path.join(sensor_dir, f"3_Manifold_{lang}.png"), dpi=300)
    plt.close()

def run_visualization_pipeline(jsonl_file, model_path):
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型
    print(f"Loading model from {model_path}...")
    model = SpatialResNetVAE(latent_channels=cfg.LATENT_CHANNELS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. 读取数据
    print(f"Reading file: {jsonl_file}")
    timestamp, lines = load_snapshot_data(jsonl_file)
    if not lines: return

    # 3. 创建输出目录
    ensure_dir("outputs/viz")  # 确保模型输出目录存在
    folder_name = f"{timestamp}_{Path(jsonl_file).stem}"
    output_base = ensure_dir(os.path.join("outputs/viz", folder_name))
    print(f"Output directory: {output_base}")

    sensor_results = []

    with torch.no_grad():
        for idx, line_data in enumerate(lines):
            s_id = line_data.get('sensor_id', str(idx+1))
            
            # 处理
            raw_sig, input_tensor, orig_cwt_img, orig_zerone_img = process_signal(line_data)
            
            # 推理
            recon_tensor, _, _ = model(input_tensor.to(device))
            
            # 提取结果 (CPU numpy)
            recon_cwt_img = recon_tensor[0, 0].cpu().numpy()
            recon_zerone_img = recon_tensor[0, 1].cpu().numpy()
            
            # 存入列表供总图使用
            sensor_results.append({
                'id': s_id,
                'raw': raw_sig,
                'orig_cwt': orig_cwt_img,
                'recon_cwt': recon_cwt_img,
                'orig_z': orig_zerone_img,
                'recon_z': recon_zerone_img
            })
            
            # 生成单个传感器的详情图 (中英双语)
            print(f"  Generating detailed plots for Sensor {s_id}...")
            plot_individual_sensor(output_base, s_id, raw_sig, 
                                   orig_cwt_img, recon_cwt_img, 
                                   orig_zerone_img, recon_zerone_img, lang='en')
            plot_individual_sensor(output_base, s_id, raw_sig, 
                                   orig_cwt_img, recon_cwt_img, 
                                   orig_zerone_img, recon_zerone_img, lang='cn')

    # 4. 生成总览图
    print("Generating summary snapshots...")
    plot_summary(output_base, sensor_results, timestamp, lang='en')
    plot_summary(output_base, sensor_results, timestamp, lang='cn')
    
    print("\nAll visualizations complete.")

if __name__ == "__main__":
    # 这里填入你的文件路径
    JSONL_FILE = "E:/我2/专业实践-工程专项/3-生技中心/1-项目：变压器深度学习诊断故障/3-code/diagnosis/test/20251016/train/120--正常--交流变压器/202509100938_2023-12-04T15-09-34-000Z.jsonl" 
    MODEL_FILE = os.path.join("outputs/model", "vae_stage1_epoch_50.pth")
    
    # 检查文件是否存在，避免报错
    if os.path.exists(JSONL_FILE) and os.path.exists(MODEL_FILE):
        run_visualization_pipeline(JSONL_FILE, MODEL_FILE)
    else:
        print("Error: Please check if jsonl file and model file exist.")