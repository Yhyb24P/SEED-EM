"""
深度学习特征提取器 (解耦物理时间轴与图拓扑)
[修复 2] 彻底废除残端拼接，dFC/STFT 均基于绝对连续的时间流形运行。遇到坏纪元执行 Drop Graph 而非切碎时序。
"""
import numpy as np
from scipy import signal

def get_valid_epoch_mask(data, fs=200.0, epoch_sec=1.0, ptp_threshold=120.0):
    """
    亚时间窗动态剔除掩码生成
    返回布尔数组，标注哪些 epoch 属于安全神经电生理学范围。绝对不进行物理拼接。
    """
    n_chan, n_samples = data.shape
    epoch_len = int(fs * epoch_sec)
    n_epochs = n_samples // epoch_len
    
    if n_epochs == 0:
        return np.ones(1, dtype=bool)
        
    epochs = data[:, :n_epochs * epoch_len].reshape(n_chan, n_epochs, epoch_len).transpose(1, 0, 2)
    ptp_max = np.max(np.ptp(epochs, axis=2), axis=1) # shape: (n_epochs,)
    
    return ptp_max < ptp_threshold

def compute_connectivity_matrix(data, fs=200.0):
    """
    提取静态皮尔逊图连通性矩阵 (Adjacency Matrix)。
    [逻辑重构] 仅在计算协方差的数学代数空间内使用合法数据段，保护外部主输入流的时间完整性。
    """
    mask = get_valid_epoch_mask(data, fs=fs, epoch_sec=1.0, ptp_threshold=120.0)
    epoch_len = int(fs * 1.0)
    n_epochs = len(mask)
    
    epochs = data[:, :n_epochs * epoch_len].reshape(data.shape[0], n_epochs, epoch_len).transpose(1, 0, 2)
    valid_epochs = epochs[mask]
    
    # 极端情况兜底
    if len(valid_epochs) == 0:
        data_norm = (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-8)
        return np.corrcoef(data_norm)
        
    # 仅在计算相关性的内部空间展平，绝不污染返回主循环的连续时序数据
    clean_data_for_graph = valid_epochs.transpose(1, 0, 2).reshape(data.shape[0], -1)
    
    std_dev = np.std(clean_data_for_graph, axis=1, keepdims=True)
    std_dev[std_dev == 0] = 1e-8 
    data_norm = (clean_data_for_graph - np.mean(clean_data_for_graph, axis=1, keepdims=True)) / std_dev
    
    return np.corrcoef(data_norm)

def compute_dfc_matrix(data, fs=200.0, window_sec=4.0, step_sec=1.0):
    """
    [新增算子] 提取动态功能连通性 (dFC) 张量。
    遵循马尔可夫演化时序：遇到强伪迹坏窗直接执行 Drop Graph，杜绝时序拼接引起的人造空间跳变。
    """
    n_chan, n_samples = data.shape
    win_len = int(fs * window_sec)
    step_len = int(fs * step_sec)
    
    dfc_list = []
    
    for start in range(0, n_samples - win_len + 1, step_len):
        segment = data[:, start:start + win_len]
        
        # Drop Graph：如果当前动态窗包含 >150μV 的高方差爆音，则丢弃整张网络拓扑图
        if np.max(np.ptp(segment, axis=1)) > 150.0:
            continue
            
        if np.any(np.std(segment, axis=1) < 1e-4):
            continue
            
        std_dev = np.std(segment, axis=1, keepdims=True)
        std_dev[std_dev == 0] = 1e-8
        seg_norm = (segment - np.mean(segment, axis=1, keepdims=True)) / std_dev
        
        dfc_list.append(np.corrcoef(seg_norm))
        
    if len(dfc_list) == 0:
        return np.expand_dims(compute_connectivity_matrix(data, fs), axis=0)
        
    return np.array(dfc_list) # 输出时空张量维度: (K, 62, 62)

def compute_stft_features(data, fs=200.0):
    """
    提取短时傅里叶变换 (STFT) 时频表征。
    [防泄露] 直接接收来自主循环的连续物理波形，彻底消除拼凑断崖造成的 Sinc 高频宽带泄露。
    """
    # 调整 STFT 窗长至 2.0 秒，保障 0.5Hz 物理频域分辨率，修复 Delta/Theta 频段塌缩
    nperseg = int(fs * 2)
    noverlap = nperseg // 2 
    _, _, Zxx = signal.stft(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return np.abs(Zxx)