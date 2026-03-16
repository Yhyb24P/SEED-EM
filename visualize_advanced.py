"""
特征与拓扑审计工具 (Advanced QA Probe)
用于直接评估管道产出的最终 .mat 特征包质量，重点验证 GNN 输入的连通性矩阵及全局频域特性。
"""
import os
import numpy as np
import scipy.io as sio
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 确保输出目录存在
OUTPUT_DIR = "Data/QA_Reports/Advanced"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_advanced_metrics(mat_filepath):
    """提取单个被试的综合指标并生成审计看板"""
    subject_id = os.path.basename(mat_filepath).split('.')[0]
    try:
        data = sio.loadmat(mat_filepath)
        data_pure = data['data_pure']     # shape: (3, 15) obj
        adj_matrix = data['adj_matrix']   # shape: (3, 15) obj
        ch_names = [ch.strip() for ch in data['ch_names']]
        
        # 安全剥离 sfreq
        fs_raw = data['sfreq']
        while isinstance(fs_raw, np.ndarray) and fs_raw.size == 1:
            fs_raw = fs_raw.item()
        fs = float(fs_raw)
        
    except Exception as e:
        print(f"[-] 跳过无效文件 {mat_filepath}: {e}")
        return

    # 抽取 Day 1, Trial 1 作为代表性样本并执行深度剥离
    try:
        # [核心修复] 递归解包 MATLAB 的嵌套 Cell Array
        def _unwrap(obj):
            while isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
                obj = obj.item()
            return obj
        
        # 提取并强制映射为纯浮点张量，斩断一切 Object 泄漏
        sample_adj = np.asarray(_unwrap(adj_matrix[0, 0]), dtype=float)
        sample_signal = np.asarray(_unwrap(data_pure[0, 0]), dtype=float)
        
        # 形状矫正防护 (降维打击多余的 1 维度)
        if sample_adj.ndim > 2:
            sample_adj = np.squeeze(sample_adj)
        if sample_signal.ndim > 2:
            sample_signal = np.squeeze(sample_signal)
            
    except Exception as e:
        print(f"[-] {subject_id} 数据结构解析失败: {e}")
        return

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(f'Advanced Feature Topology Audit: {subject_id} (Day 1, Trial 1)', 
                 fontsize=16, fontweight='bold', y=0.98)

    # ==========================================
    # 面板 1：GNN 邻接矩阵拓扑热力图 (Adjacency Matrix)
    # 预期：对角线为1，应呈现出明显的额叶聚集与枕叶聚集区块
    # ==========================================
    ax1 = fig.add_subplot(1, 2, 1)
    
    # 使用自定义发散型色图，0为白色，正相关为红，负相关为蓝
    cmap = mcolors.LinearSegmentedColormap.from_list('rwb', ['#053061', '#FFFFFF', '#67001F'])
    im1 = ax1.imshow(sample_adj, cmap=cmap, vmin=-1.0, vmax=1.0)
    
    # 标记重要的脑区边界
    regions = {'Frontal': (0, 13), 'Central': (23, 31), 'Parietal/Occipital': (41, 61)}
    for region, (start, end) in regions.items():
        rect = plt.Rectangle((start-0.5, start-0.5), end-start+1, end-start+1, 
                             fill=False, edgecolor='lime', linewidth=1.5, alpha=0.8)
        ax1.add_patch(rect)
        ax1.text(int(start), int(start)-1, region, color='lime', fontsize=9, fontweight='bold')

    ax1.set_title("Pearson Connectivity Matrix ($A \in \mathbb{R}^{62 \\times 62}$)", fontsize=12)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Pearson $r$")
    ax1.set_xticks(list(range(0, 62, 5)))
    ax1.set_yticks(list(range(0, 62, 5)))
    
    # ==========================================
    # 面板 2：全局功率谱密度 Welch's PSD
    # 预期：清晰的 1/f 衰减，以及 50Hz 处的深 V 型断崖
    # ==========================================
    ax2 = fig.add_subplot(1, 2, 2)
    
    nperseg = int(fs * 2) # 2秒分辨率
    freqs, psd = signal.welch(sample_signal, fs=fs, nperseg=nperseg, axis=1)
    
    # 计算全脑平均 PSD 并转换为 dB
    mean_psd = np.mean(psd, axis=0)
    mean_psd_db = 10 * np.log10(mean_psd + 1e-12)
    
    # 绘制所有通道的虚影
    for i in range(psd.shape[0]):
        ax2.plot(freqs, 10 * np.log10(psd[i] + 1e-12), color='gray', alpha=0.1, linewidth=0.5)
    
    # 绘制平均能量线
    ax2.plot(freqs, mean_psd_db, color='#D32F2F', linewidth=2, label='Global Average PSD')
    
    # 标定生理频段
    ax2.axvspan(1, 4, color='#BBDEFB', alpha=0.3, label='Delta (1-4Hz)')
    ax2.axvspan(4, 8, color='#C8E6C9', alpha=0.3, label='Theta (4-8Hz)')
    ax2.axvspan(8, 13, color='#FFF9C4', alpha=0.3, label='Alpha (8-13Hz)')
    
    # 检查 50Hz 陷波
    ax2.axvline(50, color='black', linestyle='--', linewidth=1, label='50Hz Notch')
    
    ax2.set_xlim(left=0, right=70)
    ax2.set_ylim(bottom=-30, top=40)
    ax2.set_title("Global Power Spectral Density (Welch's Method)", fontsize=12)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power / Frequency (dB/Hz)")
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"{subject_id}_Advanced_Audit.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[+] 成功生成审计看板: {save_path}")

if __name__ == "__main__":
    mat_dir = "Data/EEG_pure"
    mat_files = [f for f in os.listdir(mat_dir) if f.endswith('.mat')]
    
    print("开始执行特征拓扑审计...")
    for f in mat_files:
        plot_advanced_metrics(os.path.join(mat_dir, f))
    print("审计完成，请查阅 Data/QA_Reports/Advanced 目录。")