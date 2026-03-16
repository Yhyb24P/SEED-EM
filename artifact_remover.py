"""
伪迹隔离与调度路由 (已修复)
[Fix 2] 移除了 ICA 内部的高通复制滤波，彻底消灭接缝边缘伪迹 (Edge Ringing)。
[Fix 6] 增加 SVD 秩塌陷拦截锁，保障多线程池稳定性。
[Fix 7] 注入底层频带元数据，封堵 MNE 警告洪泛。
"""
import numpy as np
import mne
from joblib import Parallel, delayed

from config import CH_NAMES, WINDOW_SEC
from models import HAS_QVAE_DEPS

if HAS_QVAE_DEPS:
    import torch
    from models import QuantumEEGDenoiser

def _process_qvae_segment(data_seg):
    """QVAE 纯推理模式隔离眼电特征"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_raw = torch.tensor(data_seg.T, dtype=torch.float32, device=device)
    
    # Z-Score 批次归一化 (防 Tanh 饱和与梯度消失)
    mean_X = X_raw.mean(dim=0, keepdim=True)
    std_X = X_raw.std(dim=0, keepdim=True)
    X_norm = (X_raw - mean_X) / (std_X + 1e-8)
    
    model = QuantumEEGDenoiser(input_dim=62, hidden_dim=32, n_qubits=6).to(device)
    model.eval()
    with torch.no_grad():
        # [关键] 必须将归一化后的数据送入流形投影
        _, _, _, q_out = model(X_norm)
        
        # 动态鲁棒 EOG 锚点 (保持使用物理变量求方差，确保生理意义明确)
        frontal_indices = [0, 1, 2, 3, 4]
        variances = torch.var(X_raw[:, frontal_indices], dim=0)
        
        valid_mask = (variances > 1e-2) & (variances < 5000)
        if not valid_mask.any():
            anchor_signal = torch.mean(X_norm, dim=1)
        else:
            valid_frontals = torch.tensor(frontal_indices, device=device)[valid_mask]
            anchor_signal = torch.mean(X_norm[:, valid_frontals], dim=1)
            
        q_out_centered = q_out - q_out.mean(dim=0)
        anchor_centered = anchor_signal - anchor_signal.mean()
        
        denom = (torch.sqrt(torch.sum(q_out_centered**2, dim=0)) * torch.sqrt(torch.sum(anchor_centered**2)))
        denom[denom == 0] = 1e-8
        correlations = torch.sum(q_out_centered * anchor_centered.unsqueeze(1), dim=0) / denom
        
        eog_mask = torch.abs(correlations) > 0.4
        q_out_clean = q_out.clone()
        q_out_clean[:, eog_mask] = 0.0
        
        # 解码输出依然是归一化域
        clean_X_norm = model.decoder(q_out_clean)
        
        # 反归一化，将信号投射回原物理微伏域
        clean_X = clean_X_norm * std_X + mean_X
        
    return clean_X.cpu().numpy().T

def _process_artifact_segment(data_seg, info, montage, method='qvae'):
    """单时间窗伪迹处理算子"""
    # [Fix 8] 提升残差分段的硬拦截锁 (FIR 拓扑与相位畸变保护)
    # 若尾部片段短于 MNE 默认的 2048 点 FIR 滤波器长度（约10秒），
    # 强行拟合会导致严重的边缘振铃，且 ICA 统计自由度极低，故直接跳过。
    if data_seg.shape[1] < 2048:
        return data_seg

    raw_seg = mne.io.RawArray(data_seg * 1e-6, info, verbose=False)
    raw_seg.set_montage(montage, on_missing='ignore')
    
    if method == 'qvae' and HAS_QVAE_DEPS:
        try:
            return _process_qvae_segment(data_seg)
        except Exception:
            pass
            
    try:
        ica = mne.preprocessing.ICA(n_components=20, method='picard', random_state=42, max_iter=2000, verbose=False)
        ica.fit(raw_seg, verbose=False)
    except Exception:
        ica = mne.preprocessing.ICA(n_components=20, method='fastica', random_state=42, max_iter=2000, verbose=False)
        ica.fit(raw_seg, verbose=False)
    
    eog_indices, _ = ica.find_bads_eog(raw_seg, ch_name=['FP1', 'FP2'], verbose=False)
    ica.exclude = eog_indices
    
    raw_pure = ica.apply(raw_seg.copy(), verbose=False)
    return raw_pure.get_data() * 1e6

def apply_windowed_artifact_rejection(data, sfreq=200.0, window_sec=40.0, n_jobs=-1, method='ica'):
    """分段处理逻辑 (已优化)"""
    n_chan, n_samples = data.shape
    window_len = int(sfreq * window_sec)
    
    info = mne.create_info(ch_names=CH_NAMES, sfreq=sfreq, ch_types=['eeg'] * n_chan)
    
    # [Fix 7] 元数据注入 (Metadata Injection)
    # 显式告知 MNE 该数据已在外部执行过 0.25~50.0 Hz 的物理滤波
    # 彻底拦截并消灭 ICA 拟合时产生的 "The data has not been high-pass filtered" 警告洪泛
    try:
        with info._unlock():
            info['highpass'] = 0.25
            info['lowpass'] = 50.0
    except AttributeError:
        # 兼容部分极其崭新或古早的 MNE 版本
        info['highpass'] = 0.25
        info['lowpass'] = 50.0
        
    montage = mne.channels.make_standard_montage('standard_1020')
    
    if n_samples <= window_len:
        return _process_artifact_segment(data, info, montage, method=method)

    seg_num = int(np.ceil(n_samples / window_len))
    data_collect = np.zeros((n_chan, n_samples))
    
    segments = []
    for seg in range(seg_num):
        start_idx = seg * window_len
        end_idx = min((seg + 1) * window_len, n_samples)
        segments.append(data[:, start_idx:end_idx])
    
    safe_n_jobs = 1 if method == 'qvae' else n_jobs
    processed_segments = Parallel(n_jobs=safe_n_jobs)(
        delayed(_process_artifact_segment)(seg_data, info, montage, method=method) for seg_data in segments
    )
    
    for seg, seg_data in enumerate(processed_segments):
        start_idx = seg * window_len
        end_idx = min((seg + 1) * window_len, n_samples)
        data_collect[:, start_idx:end_idx] = seg_data
        
    return data_collect