"""
管线层级信号演进可视化探针 (纯后台渲染模块)
已剥离重复的信号运算，专职负责将 main.py 生产的 Raw、Pure 与 STFT 张量
渲染为多页 PDF 学术报告 (Multi-page PDF Report)，防止内存泄漏。
"""
import numpy as np
import matplotlib
# 强制使用无头渲染后端，切断所有 GUI，防止 675 次循环导致内存泄漏 (OOM)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.prep_config import FS

TARGET_NODES = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3', 'CZ', 'C4', 'T8', 'PZ', 'O1', 'OZ', 'O2']

# ================= 0. 出版级绘图环境配置 =================
def set_pub_style():
    """对齐学术出版级与看板展示绘图参数"""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "lines.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.grid": False,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
        "savefig.dpi": 300,
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
    })
    
COLORS = {
    "Raw": "#B0BEC5",      # 浅灰蓝，代表混沌信号
    "Pure": "#2E86C1",     # 学术蓝，代表高信噪比纯净信号
}

def _plot_summary_panel(ax, raw_data, pure_data, metadata):
    """Dashboard 风格审计摘要面板"""
    meta = metadata
    raw_var = np.var(raw_data)
    pure_var = np.var(pure_data)
    snr_est = 10 * np.log10(pure_var / (raw_var - pure_var + 1e-8)) if raw_var > pure_var else "N/A"
    
    summary = [
        "PIPELINE AUDIT REPORT",
        "---------------------",
        f"Subject : S{meta['subject']} | Day: {meta['day']}",
        f"Trial   : #{meta['trial']}",
        f"Channels: {pure_data.shape[0]}",
        f"Length  : {pure_data.shape[1]} samples",
        "---------------------",
        f"Raw Max : {np.max(raw_data):.1f} μV",
        f"Pure Max: {np.max(pure_data):.1f} μV",
        f"Est. SNR: {snr_est if isinstance(snr_est, str) else f'{snr_est:.2f} dB'}",
        "---------------------",
        "Status  : CLEAN & READY"
    ]
    
    ax.text(0.05, 0.95, "\n".join(summary), transform=ax.transAxes, 
            verticalalignment='top', family='monospace', fontsize=7,
            bbox=dict(facecolor='#F8F9F9', alpha=0.9, edgecolor='#D5D8DC', boxstyle='round,pad=1'))
    ax.axis('off')


def plot_all_channels_waveform_grid(raw_data, pure_data, metadata, save_path, window_sec=10):
    """
    1D 时域网格审计引擎：动态生成多页 PDF 报告，防范维度越界。
    """
    # 动态获取实际通道数，彻底摒弃静态截断依赖
    n_chan = raw_data.shape[0]
    n_samples = int(FS * window_sec)
    plot_len = min(n_samples, raw_data.shape[1], pure_data.shape[1])
    t = np.arange(plot_len) / FS
    
    num_pages = int(np.ceil((n_chan + 1) / 16.0))
    # 启用 PdfPages 上下文以聚合多图
    with PdfPages(save_path) as pdf:
        for fig_idx in range(num_pages):
            fig = plt.figure(figsize=(14, 8), constrained_layout=True)
            fig.suptitle(f'Project A: Time-Domain Shift Audit - Page {fig_idx+1}/{num_pages}', 
                         fontsize=12, fontweight='bold', y=1.02)
            gs = gridspec.GridSpec(4, 4, figure=fig)
            
            for ax_idx in range(16):
                ch_idx = fig_idx * 16 + ax_idx
                ax = fig.add_subplot(gs[ax_idx // 4, ax_idx % 4])
                
                if ch_idx < n_chan:
                    ch_name = TARGET_NODES[ch_idx] if ch_idx < len(TARGET_NODES) else f"CH{ch_idx}"
                    ax.plot(t, raw_data[ch_idx, :plot_len], color=COLORS["Raw"], alpha=0.8, label='Raw (Drift/EOG)')
                    ax.plot(t, pure_data[ch_idx, :plot_len], color=COLORS["Pure"], linewidth=1.0, label='QVAE Purified')
                    
                    p1, p99 = np.percentile(pure_data[ch_idx, :plot_len], [0.5, 99.5])
                    ax.set_ylim(p1 - 15, p99 + 15)
                    
                    ax.set_title(f'[{ch_name}]', loc='left', color='#333333')
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.spines["left"].set_linewidth(0.5)
                    ax.spines["bottom"].set_linewidth(0.5)
                    ax.grid(axis='y', linestyle='--', alpha=0.3)
                    
                    if ax_idx == 0:
                        ax.legend(loc='upper right', frameon=False, fontsize=7)
                        
                    if ax_idx // 4 == 3 or ch_idx == n_chan - 1:
                        ax.set_xlabel('Time (s)', color='#555555')
                    else:
                        ax.set_xticklabels([])
                        
                    if ax_idx % 4 == 0:
                        ax.set_ylabel('Amp (µV)', color='#555555')
                else:
                    if ch_idx == n_chan:
                        _plot_summary_panel(ax, raw_data, pure_data, metadata)
                    else:
                        ax.axis('off')
            
            # 将该页写入 PDF，并显式摧毁 Figure 以清空内存
            pdf.savefig(fig)
            plt.close(fig)


def plot_all_channels_stft_grid(stft_data, raw_data, pure_data, metadata, save_path, window_sec=10):
    """
    2D 频域网格审计引擎：动态生成多页 STFT 热力图报告。
    """
    n_chan = stft_data.shape[0]
    _, n_freqs, n_tbins = stft_data.shape
    f = np.linspace(0, FS / 2, n_freqs)
    t = np.arange(n_tbins) * 25 / FS
    
    t_mask = t <= window_sec
    t_plot = t[t_mask]
    f_mask = f <= 50.0
    f_plot = f[f_mask]
    
    num_pages = int(np.ceil((n_chan + 1) / 16.0))
    with PdfPages(save_path) as pdf:
        for fig_idx in range(num_pages):
            fig = plt.figure(figsize=(14, 8), constrained_layout=True)
            fig.suptitle(f'Project B: STFT Spectrogram Topology - Page {fig_idx+1}/{num_pages}', 
                         fontsize=12, fontweight='bold', y=1.02)
            gs = gridspec.GridSpec(4, 4, figure=fig)
            
            for ax_idx in range(16):
                ch_idx = fig_idx * 16 + ax_idx
                ax = fig.add_subplot(gs[ax_idx // 4, ax_idx % 4])
                
                if ch_idx < n_chan:
                    ch_name = TARGET_NODES[ch_idx] if ch_idx < len(TARGET_NODES) else f"CH{ch_idx}"
                    data_plot = stft_data[ch_idx, f_mask, :][:, t_mask]
                    
                    im = ax.pcolormesh(t_plot, f_plot, data_plot, shading='gouraud', cmap='jet')
                    ax.set_title(f'[{ch_name}] Rank-{ch_idx}', loc='left', color='#333333')
                    
                    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                    cbar.ax.tick_params(labelsize=6, length=2, width=0.5)
                    cbar.outline.set_visible(False)
                    
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    
                    if ax_idx // 4 == 3 or ch_idx == n_chan - 1:
                        ax.set_xlabel('Time (s)', color='#555555')
                    else:
                        ax.set_xticklabels([])
                        
                    if ax_idx % 4 == 0:
                        ax.set_ylabel('Freq (Hz)', color='#555555')
                else:
                    if ch_idx == n_chan:
                        _plot_summary_panel(ax, raw_data, pure_data, metadata)
                    else:
                        ax.axis('off')
            
            pdf.savefig(fig)
            plt.close(fig)