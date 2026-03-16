"""
主运行入口 (终极重构版 - SOTA 工业级拓扑)
[数学逻辑] 大道至简：完全摒弃时域非线性截断与删点，保护 ICA 的空间解析力。
[执行序列] 坏导修复 -> FIR 强效去漂移 -> 安全 CAR -> 并行 ICA ->  [新增] Post-ICA 物理爆音插值
"""
import os
import re
import gc
import numpy as np
import scipy.io as sio
from scipy import signal
import mne

from config import CH_NAMES, FS
from core_transforms import fix_hardcoded_bads, fun_rmout_python  # [重载] 重新引入绝对阈值插值算子
from feature_extractors import compute_connectivity_matrix, compute_stft_features
from artifact_remover import apply_windowed_artifact_rejection
from visualize_pipeline_stages import set_pub_style, plot_all_channels_waveform_grid, plot_all_channels_stft_grid

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def pipeline():
    input_dir = 'Data/Preprocessed_EEG'
    output_dir = 'Data/EEG_pure'
    qa_output_dir = 'Data/QA_Reports'
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(qa_output_dir, exist_ok=True)
    
    set_pub_style()
    print("[System] Visual QA Probe initialized in background mode (Agg).")
    
    all_mat_files = [f for f in os.listdir(input_dir) if f.endswith('.mat')]
    mat_files = [f for f in all_mat_files if 'label' not in f.lower()]
    mat_files.sort(key=natural_sort_key)
    
    data_pure_subject = None
    
    for turn_idx, filename in enumerate(mat_files):
        turn = turn_idx + 1
        subject = int(np.ceil(turn / 3.0))
        day = turn - (subject - 1) * 3
        
        day_qa_dir = os.path.join(qa_output_dir, f'S{subject:02d}', f'Day{day}')
        os.makedirs(day_qa_dir, exist_ok=True)
        
        if day == 1:
            data_pure_subject = np.empty((3, 15), dtype=object)
            adj_matrix_subject = np.empty((3, 15), dtype=object)
            stft_features_subject = np.empty((3, 15), dtype=object)
            
        file_path = os.path.join(input_dir, filename)
        mat_data = sio.loadmat(file_path)
        
        trial_keys = [key for key in mat_data.keys() if not key.startswith('__') and type(mat_data[key]) == np.ndarray]
        trial_keys.sort(key=natural_sort_key)
        
        for trial_idx, trial_key in enumerate(trial_keys):
            print(f"Processing -> S{subject:02d} | Day {day} | Trial {trial_idx + 1:02d}...")
            
            data_raw = mat_data[trial_key].copy()
            
            if data_raw.shape[0] > 62:
                data_raw = data_raw[:62, :]
            
            raw_snapshot = data_raw.copy()
            
            # ==========================================
            #  终极 SOTA 核心清洗流 (非线性后置拓扑)
            # ==========================================
            
            # 1. 修复硬编码物理坏导
            data_raw = fix_hardcoded_bads(data_raw, turn, trial_idx)
            
            # 2. 0均值中心化
            data_raw = data_raw - np.mean(data_raw, axis=1, keepdims=True)
            
            # 3. 工业级 FIR 频域净化 (1.0Hz)
            data_raw = mne.filter.filter_data(
                data_raw, sfreq=FS, l_freq=1.0, h_freq=50.0, 
                method='fir', phase='zero', verbose=False
            )
            data_raw = mne.filter.notch_filter(
                data_raw, Fs=FS, freqs=np.array([50.0]), 
                method='fir', phase='zero', verbose=False
            )
            
            # 4. 自适应安全共平均参考 (Safe CAR)
            valid_mask = np.std(data_raw, axis=1) > 1e-4
            if np.any(valid_mask):
                global_ref = np.mean(data_raw[valid_mask, :], axis=0)
                data_raw[valid_mask, :] -= global_ref
            
            # 5. 盲源分离 (ICA / QVAE) 
            # 提示: 如果想测试 QVAE 的威力，可将 method 改为 'qvae'
            data_collect = apply_windowed_artifact_rejection(data_raw, sfreq=FS, window_sec=40.0, n_jobs=-1, method='qvae')
            
            # 6. [核心增强] Post-ICA 瞬态物理爆音拦截 (Post-Hoc AutoReject)
            # 此时空间线性解耦已完成，安全切除如 Trial_05 中 [P1] 通道残留的 >100μV 肌肉爆音
            data_collect = fun_rmout_python(data_collect, threshold=100.0)
            fun_rmout_python(data_collect, threshold=100.0)
            
            # ==========================================
            
            adj_matrix = compute_connectivity_matrix(data_collect)
            stft_feat = compute_stft_features(data_collect, fs=FS)
            
            metadata = {'subject': subject, 'day': day, 'trial': trial_idx + 1}
            pdf_path_1d = os.path.join(day_qa_dir, f'Trial_{trial_idx + 1:02d}_1D_Waveform.pdf')
            plot_all_channels_waveform_grid(raw_snapshot, data_collect, metadata, pdf_path_1d, window_sec=10)
            pdf_path_2d = os.path.join(day_qa_dir, f'Trial_{trial_idx + 1:02d}_2D_STFT.pdf')
            plot_all_channels_stft_grid(stft_feat, raw_snapshot, data_collect, metadata, pdf_path_2d, window_sec=10)
            
            data_pure_subject[day - 1, trial_idx] = data_collect
            adj_matrix_subject[day - 1, trial_idx] = adj_matrix
            stft_features_subject[day - 1, trial_idx] = stft_feat
            
            del data_raw, raw_snapshot
            gc.collect()
            
        if day == 3:
            save_name = f"S{subject:02d}.mat"
            mdict = {
                'data_pure': data_pure_subject,
                'adj_matrix': adj_matrix_subject,
                'stft_features': stft_features_subject,
                'sfreq': FS,
                'ch_names': CH_NAMES
            }
            sio.savemat(os.path.join(output_dir, save_name), mdict)
            print(f"=== Saved S{subject:02d} with GNN/DL features and QA PDFs ===")
            
            del data_pure_subject, adj_matrix_subject, stft_features_subject, mdict
            gc.collect()

if __name__ == '__main__':
    pipeline()