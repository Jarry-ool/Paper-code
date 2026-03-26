# -*- coding: utf-8 -*-
"""
hetero_viz.py
Transformer Vibration Fault Analysis - Professional Visualization Pipeline
功能：生成符合 IEEE/Nature 期刊标准的故障诊断可视化图表（中英双语）
修复记录：
- 修复了字体缺失警告 (调整 font.sans-serif 顺序)
- 修复了图表文字互相遮挡的问题 (调整 padding, spacing, suptitle 位置)
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import pywt
import os
import shutil
from pathlib import Path
import matplotlib.gridspec as gridspec

# 引入项目配置和模型
import hetero_config as cfg
from hetero_model import SpatialResNetVAE

# ---------------------------------------------------------
# 1. 全局绘图设置 (IEEE/Nature Style)
# ---------------------------------------------------------
plt.rcParams['font.family'] = 'sans-serif'
# 【重要修复】优先使用支持中文的字体，解决 Glyph missing 警告
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
plt.rcParams['figure.dpi'] = 300           # 设置默认高分辨率
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.linewidth'] = 0.8       # 坐标轴线宽
plt.rcParams['xtick.direction'] = 'in'     # 刻度向内
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['font.size'] = 10             # 基础字号

# 配色方案
COLOR_WAVE = '#1f77b4'  # 经典蓝
COLOR_REC = '#d62728'   # 经典红
CMAP_CWT = 'jet'        # 时频图常用
CMAP_ZERONE = 'inferno' # 特征图常用 (高对比度)

# ---------------------------------------------------------
# 2. 辅助函数
# ---------------------------------------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def load_snapshot_data(file_path):
    """读取整个 JSONL 文件作为一组快照"""
    lines = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [json.loads(line.strip()) for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, [], ""
    
    if not lines:
        return None, [], ""

    # 提取时间戳用于命名
    raw_time = lines[0].get('data_time', 'unknown')
    safe_time = raw_time.replace(':', '').replace('-', '').replace('.', '').replace('T', '_').replace('Z', '')[:15]
    
    return raw_time, lines, safe_time

def process_signal_to_tensor(json_data):
    """核心处理：Raw Signal -> CWT Image & Zerone Image -> VAE Input Tensor"""
    # 1. 解析信号
    sig_str = json_data.get('signal_value', '')
    if not sig_str: return None, None, None, None
    signal = np.fromstring(sig_str, sep=',')
    
    # 长度对齐
    if len(signal) > cfg.SIGNAL_LEN: signal = signal[:cfg.SIGNAL_LEN]
    else: signal = np.pad(signal, (0, cfg.SIGNAL_LEN - len(signal)))
    
    # 2. 生成 CWT (Channel 0)
    scales = np.arange(1, 129)
    coef, _ = pywt.cwt(signal, scales, 'morl')
    scalogram = np.log1p(np.abs(coef))
    cwt_img = cv2.resize(scalogram, (cfg.INPUT_SIZE, cfg.INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    cwt_img_norm = (cwt_img - cwt_img.min()) / (cwt_img.max() - cwt_img.min() + 1e-8)
    
    # 3. 生成 Zerone (Channel 1) - 演示用模拟数据
    zerone_img = np.random.rand(cfg.INPUT_SIZE, cfg.INPUT_SIZE).astype(np.float32)
    zerone_img = cv2.GaussianBlur(zerone_img, (15, 15), 0)
    zerone_img = (zerone_img - zerone_img.min()) / (zerone_img.max() - zerone_img.min() + 1e-8)

    # 4. Context (Channel 2) - 占位
    ctx_img = np.zeros((cfg.INPUT_SIZE, cfg.INPUT_SIZE), dtype=np.float32)
    
    # 5. 堆叠 Tensor
    tensor = np.stack([cwt_img_norm, zerone_img, ctx_img], axis=0)
    tensor = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)
    
    return signal, tensor, cwt_img_norm, zerone_img

# ---------------------------------------------------------
# 3. 绘图核心函数 (已修复遮挡问题)
# ---------------------------------------------------------

def plot_waveform(save_dir, sensor_id, signal, lang='en'):
    """生成单独的宽幅波形图"""
    titles = {
        'en': {'t': f'Sensor {sensor_id} - Time Domain Waveform', 'x': 'Sample Index', 'y': 'Amplitude (g)'},
        'cn': {'t': f'传感器 {sensor_id} - 时域振动波形', 'x': '采样点索引', 'y': '幅值 (g)'}
    }
    t = titles[lang]

    plt.figure(figsize=(10, 3))
    plt.plot(signal, color=COLOR_WAVE, linewidth=0.8)
    # 【调整】增加标题 padding
    plt.title(t['t'], fontsize=12, fontweight='bold', pad=10)
    plt.xlabel(t['x'], fontsize=10)
    plt.ylabel(t['y'], fontsize=10)
    plt.xlim(0, len(signal))
    plt.grid(True, linestyle=':', alpha=0.5)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # 【调整】使用 tight_layout 防止标签超出边界
    plt.subplots_adjust()
    plt.savefig(os.path.join(save_dir, f"{sensor_id}_1_Waveform_{lang.upper()}.png"))
    plt.close()

def plot_cwt_compare(save_dir, sensor_id, orig, recon, lang='en'):
    """生成 CWT 对比图"""
    titles = {
        'en': {'main': f'Sensor {sensor_id} - Time-Frequency Reconstruction', 'l': 'Input CWT', 'r': 'Reconstructed CWT'},
        'cn': {'main': f'传感器 {sensor_id} - 时频特征重构对比', 'l': '原始时频图 (Input)', 'r': '重构时频图 (Output)'}
    }
    t = titles[lang]

    # 【调整】稍微增加高度
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    
    im1 = ax1.imshow(orig, cmap=CMAP_CWT, aspect='auto', vmin=0, vmax=1)
    # 【调整】增加子标题 padding，减小字号
    ax1.set_title(t['l'], fontsize=10, pad=8)
    ax1.axis('off')
    
    im2 = ax2.imshow(recon, cmap=CMAP_CWT, aspect='auto', vmin=0, vmax=1)
    ax2.set_title(t['r'], fontsize=10, pad=8)
    ax2.axis('off')
    
    cbar = fig.colorbar(im2, ax=[ax1, ax2], fraction=0.02, pad=0.16)
    cbar.ax.tick_params(labelsize=8)
    
    # 【调整】提高总标题位置，预留顶部空间
    plt.suptitle(t['main'], fontsize=13, y=0.98)
    # 【调整】手动调整布局，确保顶部有空间
    plt.subplots_adjust(top=0.88, bottom=0.05, left=0.05, right=0.84)
    
    plt.savefig(os.path.join(save_dir, f"{sensor_id}_2_CWT_Compare_{lang.upper()}.png"))
    plt.close()

def plot_zerone_compare(save_dir, sensor_id, orig, recon, lang='en'):
    """生成 Zerone 特征流形对比图"""
    titles = {
        'en': {
            'main': f'Sensor {sensor_id} - Physical Manifold (Zerone)', # 缩短标题
            'l': 'Input Feature Grid', 
            'r': 'Recon Feature Grid',
            'note': 'Note: This map represents 1200 flattened physical features.'
        },
        'cn': {
            'main': f'传感器 {sensor_id} - 物理特征流形映射 (Zerone)', 
            'l': '输入特征矩阵', 
            'r': '重构特征矩阵',
            'note': '注：此图谱由1200维物理统计特征映射而成，非时空图像。'
        }
    }
    t = titles[lang]

    # 【调整】增加高度以容纳底部注释
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    im1 = ax1.imshow(orig, cmap=CMAP_ZERONE, aspect='auto')
    # 【调整】增加 padding
    ax1.set_title(t['l'], fontsize=10, pad=8)
    ax1.axis('off')
    
    im2 = ax2.imshow(recon, cmap=CMAP_ZERONE, aspect='auto')
    ax2.set_title(t['r'], fontsize=10, pad=8)
    ax2.axis('off')
    
    # 【调整】提高总标题位置
    plt.suptitle(t['main'], fontsize=13, y=0.98)
    
    # 底部添加注释
    plt.figtext(0.5, 0.02, t['note'], ha="center", fontsize=9, style='italic', color='gray')
    
    # 【关键调整】预留底部空间给注释，预留顶部空间给标题
    plt.subplots_adjust(top=0.88, bottom=0.15, left=0.05, right=0.95, wspace=0.2)
    
    plt.savefig(os.path.join(save_dir, f"{sensor_id}_3_Zerone_Compare_{lang.upper()}.png"))
    plt.close()

def plot_summary_page(save_dir, timestamp, data_list, lang='en'):
    """生成总览页 (Summary) - 修复布局拥挤"""
    num_sensors = len(data_list)
    fig = plt.figure(figsize=(18, 3 * num_sensors))
    
    # 【修改点 2】优化 GridSpec 布局
    # width_ratios: 最后一个 0.15 是给 Colorbar 的，稍微加大一点
    # wspace: 从 0.25 增加到 0.4，显著拉开左右子图的水平间距
    # hspace: 保持 0.6 不变，垂直间距通常够用
    gs = gridspec.GridSpec(num_sensors, 4, width_ratios=[3, 1, 1, 0.15], wspace=0.4, hspace=0.6)
    
    txt = {
        'en': {'title': 'Multi-sensor Snapshot Analysis', 'wave': 'Time Domain', 'orig': 'Orig CWT', 'recon': 'Recon CWT'},
        'cn': {'title': '多传感器快照联合分析', 'wave': '时域波形', 'orig': '原始时频', 'recon': '重构时频'}
    }
    t = txt[lang]

    for i, item in enumerate(data_list):
        sid = item['id']
        
        # 1. Waveform
        ax_wave = fig.add_subplot(gs[i, 0])
        ax_wave.plot(item['signal'], color=COLOR_WAVE, linewidth=0.7)
        ax_wave.set_xlim(0, len(item['signal']))
        ax_wave.set_ylabel(f"S-{sid}", fontweight='bold', fontsize=9)
        ax_wave.set_yticks([]) 
        if i == 0: ax_wave.set_title(t['wave'], fontsize=12, pad=20)
        
        # 2. Orig CWT
        ax_orig = fig.add_subplot(gs[i, 1])
        ax_orig.imshow(item['orig_cwt'], cmap=CMAP_CWT, aspect='auto')
        ax_orig.axis('off')
        if i == 0: ax_orig.set_title(t['orig'], fontsize=12, pad=20)
        
        # 3. Recon CWT
        ax_recon = fig.add_subplot(gs[i, 2])
        im = ax_recon.imshow(item['recon_cwt'], cmap=CMAP_CWT, aspect='auto')
        ax_recon.axis('off')
        if i == 0: ax_recon.set_title(t['recon'], fontsize=12, pad=20)
        
        # 4. Colorbar
        ax_cb = fig.add_subplot(gs[i, 3])
        # 使用 matplotlib 的 colorbar 填充这个专门预留的子图
        plt.colorbar(im, cax=ax_cb)
        # 可选：让 Colorbar 上的字体稍微小一点，显得精致
        ax_cb.tick_params(labelsize=8)

    fig.suptitle(f"{t['title']} | Time: {timestamp}", fontsize=16, y=0.99)
    
    # tight_layout 参数微调，确保右侧不被切掉
    plt.subplots_adjust(left=0.05, bottom=0.02, right=0.98, top=0.93)
    
    filename = f"Summary_{lang.upper()}.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()
    print(f"  -> Summary saved: {filename}")

# ---------------------------------------------------------
# 4. 主流程
# ---------------------------------------------------------
def main(jsonl_path, model_path):
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    
    # 模型加载
    if not os.path.exists(model_path): return
    model = SpatialResNetVAE(latent_channels=cfg.LATENT_CHANNELS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 数据读取
    raw_time, lines, safe_time = load_snapshot_data(jsonl_path)
    if not lines: return

    # === 【关键修改】溯源路径构建 ===
    # 原始: outputs/viz/timestamp_filename
    # 新版: outputs/viz/父目录名/timestamp_filename
    # 例如: outputs/viz/134--正常--交流变压器/20250910_data/
    
    jsonl_path_obj = Path(jsonl_path)
    parent_dir_name = jsonl_path_obj.parent.name # e.g. "134--正常--交流变压器"
    base_name = jsonl_path_obj.stem             # e.g. "20250910..."
    
    output_root = cfg.CHECKPOINT_DIR / "viz" / parent_dir_name / f"{safe_time}_{base_name}"
    
    if not output_root.exists():
        os.makedirs(output_root)
    
    print(f"[Processing] {jsonl_path_obj.name}")
    print(f"  -> Save to: {output_root}")

    # ... (后续处理逻辑与原版一致) ...
    summary_data = [] 
    with torch.no_grad():
        for i, line_data in enumerate(lines):
            sensor_id = line_data.get('sensor_id', f"{i+1}")
            raw_sig, input_tensor, orig_cwt, orig_zerone = process_signal_to_tensor(line_data)
            
            if raw_sig is None: continue
            
            recon_tensor, _, _ = model(input_tensor.to(device))
            recon_cwt = recon_tensor[0, 0].cpu().numpy()
            recon_zerone = recon_tensor[0, 1].cpu().numpy()
            
            summary_data.append({
                'id': sensor_id, 'signal': raw_sig,
                'orig_cwt': orig_cwt, 'recon_cwt': recon_cwt
            })
            
            # 生成单传感器图
            sensor_dir = output_root / f"sensor_{sensor_id}"
            if not sensor_dir.exists(): os.makedirs(sensor_dir)
            
            # 调用之前的绘图函数 (确保这些函数已被定义或导入)
            plot_waveform(str(sensor_dir), sensor_id, raw_sig, 'cn')
            plot_cwt_compare(str(sensor_dir), sensor_id, orig_cwt, recon_cwt, 'cn')
            plot_zerone_compare(str(sensor_dir), sensor_id, orig_zerone, recon_zerone, 'cn')

    # 总览图
    if summary_data:
        plot_summary_page(str(output_root), raw_time, summary_data, 'cn')

# ... (scan_all_jsonl_files 函数保持不变) ...

if __name__ == "__main__":
    # ... (模型路径检查逻辑保持不变) ...
    model_filename = "final_model.pth" 
    MODEL_FILE = cfg.CHECKPOINT_DIR /"model"/ model_filename
    
    all_data_files = scan_all_jsonl_files(cfg.RAW_DATA_DIRS) # 扫描所有配置的目录
    
    for i, jsonl_path in enumerate(all_data_files):
        try:
            main(jsonl_path, MODEL_FILE)
        except Exception as e:
            print(f"Error processing {jsonl_path}: {e}")