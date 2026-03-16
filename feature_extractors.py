"""
深度学习特征提取器 (集成 Sub-Epoch Rejection)
[SOTA 升级] 引入亚时间窗动态剔除，保障邻接矩阵的统计鲁棒性。
"""
import numpy as np
from scipy import signal

def _drop_bad_epochs(data, fs=200.0, epoch_sec=1.0, ptp_threshold=150.0):
    """
    亚时间窗动态剔除 (Sub-Epoch Dropout)
    
    参数:
    data: (n_chan, n_samples)
    epoch_sec: 划分小段的时间长度 (默认1秒)
    ptp_threshold: 峰峰值截断阈值，生理脑电PtP很少超过 150μV
    """
    n_chan, n_samples = data.shape
    epoch_len = int(fs * epoch_sec)
    n_epochs = n_samples // epoch_len
    
    if n_epochs == 0:
        return data
        
    # 重塑并转置，形状变为 (n_epochs, n_chan, epoch_len)
    epochs = data[:, :n_epochs * epoch_len].reshape(n_chan, n_epochs, epoch_len).transpose(1, 0, 2)
    
    # 计算每个 epoch 在所有通道上的最大峰峰值 (Peak-to-Peak)
    ptp_max = np.max(np.ptp(epochs, axis=2), axis=1) # shape: (n_epochs,)
    
    # 筛选健康的 epoch
    good_mask = ptp_max < ptp_threshold
    
    # [审计逻辑] 监控剔除率
    n_dropped = n_epochs - np.sum(good_mask)
    if n_dropped > 0:
        # 仅在需要调试时打印，或作为元数据返回
        # print(f"    └── [AutoReject] Dropped {n_dropped}/{n_epochs} sec due to spikes.")
        pass

    if not np.any(good_mask):
        return data
        
    # 拼接健康片段
    clean_data = epochs[good_mask].transpose(1, 0, 2).reshape(n_chan, -1)
    return clean_data

def compute_connectivity_matrix(data, fs=200.0):
    """
    提取皮尔逊图连通性矩阵 (Adjacency Matrix)。
    [SOTA 升级] 在计算协方差前先执行亚秒级伪迹剔除。
    """
    # 1. 剔除包含单通道电极爆音的时间段
    clean_data = _drop_bad_epochs(data, fs=fs, epoch_sec=1.0, ptp_threshold=120.0)
    
    # 2. 标准化处理 (Z-Score)
    std_dev = np.std(clean_data, axis=1, keepdims=True)
    std_dev[std_dev == 0] = 1e-8 
    data_norm = (clean_data - np.mean(clean_data, axis=1, keepdims=True)) / std_dev
    
    # 3. 计算 Pearson 邻接矩阵
    return np.corrcoef(data_norm)

def compute_stft_features(data, fs=200.0):
    """
    提取短时傅里叶变换 (STFT) 时频表征。
    注意：STFT 需要保持时间轴完整，因此严禁使用 _drop_bad_epochs。
    """
    nperseg = 50
    noverlap = nperseg // 2 
    _, _, Zxx = signal.stft(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return np.abs(Zxx)