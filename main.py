"""
主运行入口
负责本地文件 I/O 交互、异常捕捉、过滤器的挂载以及流式内存回收。
执行本脚本将启动完整的脑电清洗管线。
"""
import os
import re
import gc
import numpy as np
import scipy.io as sio
from scipy import signal

from config import CH_NAMES, FS, OUTLIER_THRESHOLD
from core_transforms import fix_hardcoded_bads, reref_car, fun_rmout_python
from feature_extractors import compute_connectivity_matrix, compute_stft_features
from artifact_remover import apply_windowed_artifact_rejection

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def pipeline():
    input_dir = 'Data/Preprocessed_EEG'
    output_dir = 'Data/EEG_pure'
    os.makedirs(output_dir, exist_ok=True)
    
    all_mat_files = [f for f in os.listdir(input_dir) if f.endswith('.mat')]
    mat_files = [f for f in all_mat_files if 'label' not in f.lower()]
    mat_files.sort(key=natural_sort_key)
    
    if len(mat_files) != 45:
        print(f"[Warning] 期望45个文件，实际发现 {len(mat_files)} 个。请检查数据目录。")
        
    data_pure_subject = None
    
    for turn_idx, filename in enumerate(mat_files):
        turn = turn_idx + 1
        subject = int(np.ceil(turn / 3.0))
        day = turn - (subject - 1) * 3
        
        if day == 1:
            data_pure_subject = np.empty((3, 15), dtype=object)
            adj_matrix_subject = np.empty((3, 15), dtype=object)
            stft_features_subject = np.empty((3, 15), dtype=object)
            
        file_path = os.path.join(input_dir, filename)
        mat_data = sio.loadmat(file_path)
        
        trial_keys = [key for key in mat_data.keys() if not key.startswith('__') and type(mat_data[key]) == np.ndarray]
        trial_keys.sort(key=natural_sort_key)
        
        if len(trial_keys) != 15:
            raise ValueError(f"[Error] 被试 {subject} Day {day} 数据异常：包含 {len(trial_keys)} 个 Trials。")
        
        for trial_idx, trial_key in enumerate(trial_keys):
            print(f"Subject: {subject}, Day: {day}, Trial: {trial_idx + 1}")
            
            # 使用深拷贝提取并隔离矩阵，规避字典变异
            data_raw = mat_data[trial_key].copy()
            
            if data_raw.shape[0] != 62:
                raise ValueError(f"[Error] 数据异常：通道数越界。收到 {data_raw.shape[0]} 通道。")
            
            data_raw = fix_hardcoded_bads(data_raw, turn, trial_idx)
            data_raw = reref_car(data_raw)
            
            nyq = FS / 2.0
            b, a = signal.butter(2, [1.0 / nyq, 45.0 / nyq], btype='bandpass')
            data_raw = signal.filtfilt(b, a, data_raw, axis=1)
            
            b_notch, a_notch = signal.iirnotch(50.0, 30.0, FS)
            data_raw = signal.filtfilt(b_notch, a_notch, data_raw, axis=1)
            
            data_raw = signal.detrend(data_raw, axis=1)
            data_raw = fun_rmout_python(data_raw, threshold=OUTLIER_THRESHOLD)
            
            data_collect = apply_windowed_artifact_rejection(data_raw, sfreq=FS, window_sec=40.0, n_jobs=-1, method='ica')
            
            adj_matrix = compute_connectivity_matrix(data_collect)
            stft_feat = compute_stft_features(data_collect, fs=FS)
            
            data_pure_subject[day - 1, trial_idx] = data_collect
            adj_matrix_subject[day - 1, trial_idx] = adj_matrix
            stft_features_subject[day - 1, trial_idx] = stft_feat
            
            del data_raw
            gc.collect()
            
        if day == 3:
            save_name = f"S{subject}.mat"
            mdict = {
                'data_pure': data_pure_subject,
                'adj_matrix': adj_matrix_subject,
                'stft_features': stft_features_subject,
                'sfreq': FS,
                'ch_names': CH_NAMES
            }
            sio.savemat(os.path.join(output_dir, save_name), mdict)
            print(f"=== Saved S{subject} with GNN/DL features ===")
            
            del data_pure_subject, adj_matrix_subject, stft_features_subject, mdict
            gc.collect()

if __name__ == '__main__':
    pipeline()