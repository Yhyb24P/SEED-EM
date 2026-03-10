"""
伪迹隔离与调度路由
封装 MNE ICA 流程与 QVAE 推理隔离器，支持 Joblib 多核分发。
"""
import numpy as np
import mne
from joblib import Parallel, delayed

from config import CH_NAMES
from models import HAS_QVAE_DEPS

if HAS_QVAE_DEPS:
    import torch
    from models import QuantumEEGDenoiser

def _process_qvae_segment(data_seg):
    """QVAE 纯推理模式隔离眼电特征"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.tensor(data_seg.T, dtype=torch.float32, device=device)
    
    model = QuantumEEGDenoiser(input_dim=62, hidden_dim=32, n_qubits=6).to(device)
    model.eval()
    with torch.no_grad():
        _, _, _, q_out = model(X)
        fp_signal = X[:, 0] + X[:, 2] 
        q_out_centered = q_out - q_out.mean(dim=0)
        fp_centered = fp_signal - fp_signal.mean()
        
        correlations = torch.sum(q_out_centered * fp_centered.unsqueeze(1), dim=0) / \
                       (torch.sqrt(torch.sum(q_out_centered**2, dim=0)) * torch.sqrt(torch.sum(fp_centered**2)))
        
        eog_mask = torch.abs(correlations) > 0.4
        q_out_clean = q_out.clone()
        q_out_clean[:, eog_mask] = 0.0
        
        clean_X = model.decoder(q_out_clean)
        
    return clean_X.cpu().numpy().T

def _process_artifact_segment(data_seg, info, montage, method='qvae'):
    """单时间窗伪迹处理算子 (支持 QVAE/ICA 回退机制)"""
    if method == 'qvae' and HAS_QVAE_DEPS:
        try:
            return _process_qvae_segment(data_seg)
        except Exception:
            pass # 隐式回退
            
    # 经典 ICA 兜底流
    raw_seg = mne.io.RawArray(data_seg * 1e-6, info, verbose=False)
    raw_seg.set_montage(montage, on_missing='ignore')
    raw_ica_fit = raw_seg.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
    
    try:
        ica = mne.preprocessing.ICA(n_components=15, method='picard', random_state=42, max_iter=2000, verbose=False)
        ica.fit(raw_ica_fit, verbose=False)
    except Exception:
        ica = mne.preprocessing.ICA(n_components=15, method='fastica', random_state=42, max_iter=2000, verbose=False)
        ica.fit(raw_ica_fit, verbose=False)
    
    eog_indices, _ = ica.find_bads_eog(raw_seg, ch_name=['FP1', 'FP2'], verbose=False)
    ica.exclude = eog_indices
    
    raw_clean = ica.apply(raw_seg.copy(), verbose=False)
    return raw_clean.get_data() * 1e6

def apply_windowed_artifact_rejection(data, sfreq=200.0, window_sec=40.0, n_jobs=-1, method='ica'):
    """
    分段聚合与多进程 CPU 调度主干。
    安全策略：QVAE 模式强制降级为 n_jobs=1，阻断 PyTorch C++ 线程冲突死锁。
    """
    window_len = int(sfreq * window_sec)
    seg_num = data.shape[1] // window_len
    n_chan = data.shape[0]
    data_collect = np.zeros((n_chan, seg_num * window_len))
    
    info = mne.create_info(ch_names=CH_NAMES, sfreq=sfreq, ch_types=['eeg'] * n_chan)
    montage = mne.channels.make_standard_montage('standard_1020')
    
    segments = [data[:, seg * window_len:(seg + 1) * window_len] for seg in range(seg_num)]
    
    safe_n_jobs = 1 if method == 'qvae' else n_jobs
    processed_segments = Parallel(n_jobs=safe_n_jobs)(
        delayed(_process_artifact_segment)(seg_data, info, montage, method=method) for seg_data in segments
    )
    
    for seg, seg_data in enumerate(processed_segments):
        start_idx = seg * window_len
        end_idx = (seg + 1) * window_len
        data_collect[:, start_idx:end_idx] = seg_data
        
    return data_collect