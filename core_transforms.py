"""
信号处理核心算子 (基于微积分与统计学的 SOTA 重构)
[数学重构] 引入基于导数 (Gradient) 的瞬态拦截，取代单纯的绝对幅值“一刀切”。
"""
import numpy as np

def reref_car(data):
    """重参考: 共平均参考 (Common Average Reference)"""
    ref = np.mean(data, axis=0)
    return data - ref

def intercept_gradient_spikes(data, grad_threshold=50.0, check_step=2):
    """
    [新增核心算子] 一阶导数瞬态拦截 (Calculus-based Slew Rate Limiter)
    专杀硬件电涌与接触不良带来的阶跃突变。
    基于导数特性，它对万微伏级别的缓慢基线漂移完全免疫，杜绝大面积误杀。
    """
    clean_data = data.copy()
    n_chan, n_samples = data.shape
    kernel = np.ones(2 * check_step + 1, dtype=bool)
    all_idx = np.arange(n_samples)
    
    for i in range(n_chan):
        # 计算一阶导数 (相邻点电位跳变幅值)，prepend 保持数组长度一致
        grad = np.abs(np.diff(data[i], prepend=data[i, 0]))
        grad_mask = grad > grad_threshold
        
        if not np.any(grad_mask):
            continue
            
        # 膨胀掩码，保护突变点周边的能量溢出
        rm_mask = np.convolve(grad_mask, kernel, mode='same') > 0
        valid_idx = np.where(~rm_mask)[0]
        
        if len(valid_idx) == 0:
            clean_data[i] = 0.0
            continue
            
        # 局部线性插值修复阶跃
        clean_data[i] = np.interp(all_idx, valid_idx, data[i, valid_idx])
        
    return clean_data

def fun_rmout_python(data, check_step=2, threshold=130):
    """
    绝对幅值截断与抖动注入
    [约束] 必须在带通滤波（去除基线）之后执行，此时的 130μV 才是真正的生理学阈值。
    """
    clean_data = data.copy()
    n_chan, n_samples = data.shape
    kernel = np.ones(2 * check_step + 1, dtype=bool)
    all_idx = np.arange(n_samples)
    
    for i in range(n_chan):
        outlier_mask = np.abs(data[i]) > threshold
        if not np.any(outlier_mask):
            continue
            
        rm_mask = np.convolve(outlier_mask, kernel, mode='same') > 0
        valid_idx = np.where(~rm_mask)[0]
        
        if len(valid_idx) == 0:
            clean_data[i] = 0.0
            continue
            
        # 采用线性插值连接断点
        clean_data[i] = np.interp(all_idx, valid_idx, data[i, valid_idx])
        
        # 注入极微弱高斯底噪打破共线性，保护下游 ICA 的非奇异性
        noise = np.random.normal(loc=0.0, scale=1e-2, size=np.sum(rm_mask))
        clean_data[i, rm_mask] += noise
        
    return clean_data

def fix_hardcoded_bads(data_raw, turn, trial_idx):
    """还原 SEED 官方实验日志记录的硬编码坏导球面样条插值"""
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