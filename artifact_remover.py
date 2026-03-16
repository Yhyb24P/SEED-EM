"""
伪迹隔离与调度路由 (系统级重构版)
[修复 1] 注入全局 Z-Score 统计量下放，实施全局流形映射，消除 QVAE 的 40s 边界断崖伪迹 (Boundary Artifacts)。
"""
import numpy as np
import mne
from joblib import Parallel, delayed

from config import CH_NAMES, WINDOW_SEC
from models import HAS_QVAE_DEPS

if HAS_QVAE_DEPS:
    import torch
    from models import QuantumEEGDenoiser

def _process_qvae_segment(data_seg, global_mean=None, global_std=None):
    """QVAE 纯推理模式隔离眼电特征 (支持全局流形对齐)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_raw = torch.tensor(data_seg.T, dtype=torch.float32, device=device)
    
    # [核心修复 1.1] 摒弃局部批归一化，使用向下传递的全局统计量进行 Z-Score，防缩放塌陷
    if global_mean is not None and global_std is not None:
        g_mean = torch.tensor(global_mean.T, dtype=torch.float32, device=device)
        g_std = torch.tensor(global_std.T, dtype=torch.float32, device=device)
    else:
        g_mean = X_raw.mean(dim=0, keepdim=True)
        g_std = X_raw.std(dim=0, keepdim=True)
        
    X_norm = (X_raw - g_mean) / (g_std + 1e-8)
    
    model = QuantumEEGDenoiser(input_dim=62, hidden_dim=32, n_qubits=6).to(device)
    try:
        model.load_state_dict(torch.load('weights/qvae_pretrained.pt', map_location=device, weights_only=True))
    except FileNotFoundError:
        pass
        
    model.eval()
    with torch.no_grad():
        _, _, _, q_out = model(X_norm)
        
        # 动态鲁棒 EOG 锚点 (基于原始物理变量求方差)
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
        
        clean_X_norm = model.decoder(q_out_clean)
        
        # [核心修复 1.2] 反归一化：将清理后的信号精准投射回基于全局属性的物理微伏域
        clean_X = clean_X_norm * g_std + g_mean
        latent_Z = q_out.cpu().numpy().T
        
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return clean_X.cpu().numpy().T, latent_Z

def _process_artifact_segment(data_seg, sfreq, method='qvae', global_mean=None, global_std=None):
    if np.linalg.matrix_rank(data_seg) < 15:
        return data_seg, None

    if method == 'qvae' and HAS_QVAE_DEPS:
        try:
            return _process_qvae_segment(data_seg, global_mean, global_std)
        except Exception as e:
            print(f"  [QVAE Error] {e}")
            pass
            
    try:
        info = mne.create_info(ch_names=CH_NAMES, sfreq=sfreq, ch_types=['eeg'] * data_seg.shape[0])
        try:
            with info._unlock():
                info['highpass'] = 0.25
                info['lowpass'] = 50.0
        except AttributeError:
            info['highpass'] = 0.25
            info['lowpass'] = 50.0
        montage = mne.channels.make_standard_montage('standard_1020')

        ica = mne.preprocessing.ICA(n_components=20, method='picard', random_state=42, max_iter=2000, verbose=False)
        # 将输入数据转为 V 量级，防止量纲引发奇异矩阵
        raw_seg = mne.io.RawArray(data_seg * 1e-6, info, verbose=False)
        # 忽略标准系统中不存在的小脑导联 (CB1/CB2)，防止触发 ValueError 断崖休眠
        raw_seg.set_montage(montage, on_missing='ignore')
        ica.fit(raw_seg, verbose=False)
        
        eog_indices, _ = ica.find_bads_eog(raw_seg, ch_name=['FP1', 'FP2'], verbose=False)
        ica.exclude = eog_indices
        raw_pure = ica.apply(raw_seg.copy(), verbose=False)
        
        # 投射回物理 μV 域，解除 1e12 级张量爆炸风险
        return raw_pure.get_data() * 1e6, None
    except ValueError:
        return data_seg, None

def apply_windowed_artifact_rejection(data, sfreq=200.0, window_sec=40.0, n_jobs=-1, method='ica', global_mean=None, global_std=None):
    """滑动窗口并行伪迹消除 (支持全局流形锚定)"""
    n_chan, n_samples = data.shape
    window_len = int(sfreq * window_sec)
    
    if n_samples <= window_len:
        seg_data, seg_latent = _process_artifact_segment(data, sfreq, method, global_mean, global_std)
        return seg_data, (seg_latent if method == 'qvae' else None)

    seg_num = int(np.ceil(n_samples / window_len))
    data_collect = np.zeros((n_chan, n_samples))
    latent_collect = np.zeros((6, n_samples)) if method == 'qvae' else None
    
    segments = []
    for seg in range(seg_num):
        start_idx = seg * window_len
        end_idx = min((seg + 1) * window_len, n_samples)
        segments.append(data[:, start_idx:end_idx])
        
    safe_n_jobs = 1 if method == 'qvae' else n_jobs
    # 剥离 MNE info 元数据下传，阻断 _thread.lock 锁引起的子进程 PicklingError
    processed_segments = Parallel(n_jobs=safe_n_jobs)(
        delayed(_process_artifact_segment)(seg_data, sfreq, method, global_mean, global_std) 
        for seg_data in segments
    )
    
    for seg, res in enumerate(processed_segments):
        start_idx = seg * window_len
        end_idx = min((seg + 1) * window_len, n_samples)
        seg_data, seg_latent = res
        
        data_collect[:, start_idx:end_idx] = seg_data
        if method == 'qvae' and seg_latent is not None:
            latent_collect[:, start_idx:end_idx] = seg_latent
        
    return data_collect, latent_collect