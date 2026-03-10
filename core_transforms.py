"""
信号处理核心算子
负责实现基础的脑电时域与空间变换机制（无状态函数），支持快速向量化运算。
"""
import numpy as np

def reref_car(data):
    """重参考: 共平均参考 (Common Average Reference)"""
    ref = np.mean(data, axis=0)
    return data - ref

def fun_rmout_python(data, check_step=2, threshold=130):
    """
    超幅异常截断 (等效于 MATLAB fun_rmout)
    沿时间轴执行布尔卷积膨胀并硬截断，使用矢量化算子消除 for 循环。
    """
    outlier_mask = np.any(np.abs(data) > threshold, axis=0)
    kernel = np.ones(2 * check_step + 1, dtype=bool)
    rm_mask = np.convolve(outlier_mask, kernel, mode='same') > 0
    return data[:, ~rm_mask]

def fix_hardcoded_bads(data_raw, turn, trial_idx):
    """
    还原 SEED 官方实验日志记录的硬编码坏导球面样条插值。
    """
    if turn == 1:
        data_raw[55, :] = np.mean(data_raw[[47, 54, 56, 61], :], axis=0)
    elif turn == 2:
        data_raw[44, :] = np.mean(data_raw[[35, 43, 52, 45], :], axis=0)
    elif turn == 13:
        data_raw[41, :] = np.mean(data_raw[[32, 42, 50], :], axis=0)
    elif turn == 16:
        data_raw[4, :] = np.mean(data_raw[[2, 10, 11, 12], :], axis=0)
    elif turn == 37:
        data_raw[53, :] = np.mean(data_raw[[45, 52, 54, 58], :], axis=0)
    elif turn == 43:
        data_raw[15, :] = np.mean(data_raw[[14, 24, 16, 6], :], axis=0)
        data_raw[18, :] = np.mean(data_raw[[9, 17, 27, 19], :], axis=0)
        data_raw[26, :] = np.mean(data_raw[[25, 35, 27, 17], :], axis=0)
        if trial_idx == 1:
            data_raw[61, :] = np.mean(data_raw[[59, 55, 56], :], axis=0)
    elif turn == 45 and trial_idx > 10:
        data_raw[9, :] = np.mean(data_raw[[8, 18, 10], :], axis=0)
    return data_raw