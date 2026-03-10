"""
深度学习特征提取器
负责生成适配图神经网络 (GNN) 与 2D-CNN 的衍生特征矩阵。
"""
import numpy as np
from scipy import signal

def compute_connectivity_matrix(data):
    """
    提取皮尔逊图连通性矩阵 (Adjacency Matrix)，供 GraphConv/GCN 调用。
    """
    std_dev = np.std(data, axis=1, keepdims=True)
    std_dev[std_dev == 0] = 1e-8 # 防止零方差除零异常
    data_norm = (data - np.mean(data, axis=1, keepdims=True)) / std_dev
    return np.corrcoef(data_norm)

def compute_stft_features(data, fs=200.0):
    """
    提取短时傅里叶变换 (STFT) 时频表征，供 EEGNet/TSception 调用。
    分辨率配置：4Hz, 窗长 50 采样点。
    """
    nperseg = 50
    noverlap = nperseg // 2 
    _, _, Zxx = signal.stft(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return np.abs(Zxx)