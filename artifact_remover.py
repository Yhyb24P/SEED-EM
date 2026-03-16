"""
伪迹隔离与调度路由 (系统级重构版)
[零样本推断] 注入基于试次级仿射不变性 (Trial-level Affine Invariance) 的拓扑手术，实施流形对齐。
"""
import os
import numpy as np
import mne
from joblib import Parallel, delayed

from config import CH_NAMES, WINDOW_SEC
from models import HAS_QVAE_DEPS

if HAS_QVAE_DEPS:
    import torch
    from models import QuantumEEGDenoiser

def _process_qvae_segment(data_seg, global_mean=None, global_std=None):
    """QVAE 纯推断模式：在希尔伯特子空间执行正交掩码手术 (Orthogonal Masking Surgery)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_raw = torch.tensor(data_seg.T, dtype=torch.float32, device=device)
    
    # 1. 试次级域对齐 (Domain Alignment for Affine Invariance)
    if global_mean is not None and global_std is not None:
        g_mean = torch.tensor(global_mean.T, dtype=torch.float32, device=device)
        g_std = torch.tensor(global_std.T, dtype=torch.float32, device=device)
    else:
        g_mean = X_raw.mean(dim=0, keepdim=True)
        g_std = X_raw.std(dim=0, keepdim=True)
        
    X_norm = (X_raw - g_mean) / (g_std + 1e-8)
    
    model = QuantumEEGDenoiser(input_dim=62, hidden_dim=32, n_qubits=6).to(device)
    
    # [核心防线] 零样本推断必须依赖流形泛化权重，绝对禁止静默退化为随机哈希映射
    weight_path = 'weights/qvae_pretrained.pt'
    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"[-] 致命异常: 缺失泛化流形权重 '{weight_path}'。\n"
            f"必须先运行 `python train_qvae.py` 执行无监督预训练以提取跨被试拓扑不变性，\n"
            f"或者在 main.py 中将盲源分离方法显式回退为 method='ica'。"
        )
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True), strict=True)
        
    model.eval()
    with torch.no_grad():
        # 2. 零样本流形嵌入 (Zero-shot Embedding)
        chunk_size = 2000
        Z_infer_list = []
        for i in range(0, X_norm.size(0), chunk_size):
            chunk = X_norm[i:i+chunk_size]
            _, _, _, q_chunk = model(chunk)
            Z_infer_list.append(q_chunk)
        Z_infer = torch.cat(Z_infer_list, dim=0)
        
        # 3. 解剖学眼电锚点构建 (Anatomical Anchor Generation: a_EOG)
        frontal_indices = [0, 1, 2, 3, 4]  # FP1, FPZ, FP2, AF3, AF4
        variances = torch.var(X_raw[:, frontal_indices], dim=0)
        
        valid_mask = (variances > 1e-2) & (variances < 5000)
        if not valid_mask.any():
            a_EOG = torch.mean(X_norm, dim=1)
        else:
            valid_frontals = torch.tensor(frontal_indices, device=device)[valid_mask]
            a_EOG = torch.mean(X_norm[:, valid_frontals], dim=1)
            
        # 4. 拓扑皮尔逊度量与正交掩码剥离 (Orthogonal Masking Surgery)
        Z_centered = Z_infer - Z_infer.mean(dim=0)
        a_EOG_centered = a_EOG - a_EOG.mean()
        
        denom = (torch.sqrt(torch.sum(Z_centered**2, dim=0)) * torch.sqrt(torch.sum(a_EOG_centered**2)))
        denom[denom == 0] = 1e-8
        rho = torch.sum(Z_centered * a_EOG_centered.unsqueeze(1), dim=0) / denom
        
        eog_mask = torch.abs(rho) > 0.4
        Z_clean = Z_infer.clone()
        Z_clean[:, eog_mask] = 0.0
        
        # 5. 逆向解码重构
        clean_X_norm_list = []
        for i in range(0, Z_clean.size(0), chunk_size):
            chunk_clean = model.decoder(Z_clean[i:i+chunk_size])
            clean_X_norm_list.append(chunk_clean)
        clean_X_norm = torch.cat(clean_X_norm_list, dim=0)
        
        # 6. 物理量纲还原 (Inverse Projection Recovery)
        clean_X = clean_X_norm * g_std + g_mean
        
        # 提取执行拓扑手术后、已剔除眼电的纯净潜变量作为多模态独立特征
        latent_Z_clean = Z_clean.cpu().numpy().T
        
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return clean_X.cpu().numpy().T, latent_Z_clean

def _process_artifact_segment(data_seg, sfreq, method='qvae', global_mean=None, global_std=None):
    if np.linalg.matrix_rank(data_seg) < 15:
        return data_seg, None

    if method == 'qvae' and HAS_QVAE_DEPS:
        return _process_qvae_segment(data_seg, global_mean, global_std)
            
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