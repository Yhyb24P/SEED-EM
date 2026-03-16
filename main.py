"""
主运行入口 (终极防御重构版)
[修复 3] 激活 Slew Rate 高阶拦截，实现双端安全 CAR，删减冗余白噪声，提取 dFC 张量流。
"""
import os
import re
import gc
import numpy as np
import scipy.io as sio
from scipy import signal
import mne
from tqdm import tqdm

from config import CH_NAMES, FS
# [重载] 彻底废除暴力的 fun_rmout_python，启用 SOTA 微积分算子
from core_transforms import fix_hardcoded_bads, intercept_gradient_spikes 
from feature_extractors import compute_connectivity_matrix, compute_dfc_matrix, compute_stft_features
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
    
    # 分配进度实例锚定总 trial 数量
    pbar = tqdm(total=len(mat_files) * 15, desc="SEED Pipeline", dynamic_ncols=True)
    
    for turn_idx, filename in enumerate(mat_files):
        turn = turn_idx + 1
        subject = int(np.ceil(turn / 3.0))
        day = turn - (subject - 1) * 3
        
        day_qa_dir = os.path.join(qa_output_dir, f'S{subject:02d}', f'Day{day}')
        os.makedirs(day_qa_dir, exist_ok=True)
        
        if day == 1:
            data_pure_subject = np.empty((3, 15), dtype=object)
            adj_matrix_subject = np.empty((3, 15), dtype=object)
            dfc_matrix_subject = np.empty((3, 15), dtype=object)  # 新增 dFC 容器
            stft_features_subject = np.empty((3, 15), dtype=object)
            qvae_latents_subject = np.empty((3, 15), dtype=object) # 新增 QVAE 潜空间容器
            
        file_path = os.path.join(input_dir, filename)
        mat_data = sio.loadmat(file_path)
        
        trial_keys = [key for key in mat_data.keys() if not key.startswith('__') and type(mat_data[key]) == np.ndarray]
        trial_keys.sort(key=natural_sort_key)
        
        for trial_idx, trial_key in enumerate(trial_keys):
            # 覆写标准输出为动态前缀
            pbar.set_description(f"Processing -> S{subject:02d} | Day {day} | Trial {trial_idx + 1:02d}")
            
            data_raw = mat_data[trial_key].copy()
            if data_raw.shape[0] > 62:
                data_raw = data_raw[:62, :]
            raw_snapshot = data_raw.copy()
            
            # ==========================================
            #  系统级 SOTA 防御流 (微积分拦截与双端掩码)
            # ==========================================
            
            # 1. 修复硬编码物理坏导
            data_raw = fix_hardcoded_bads(data_raw, turn, trial_idx)
            
            # 2. 0均值中心化
            data_raw = data_raw - np.mean(data_raw, axis=1, keepdims=True)
            
            # 3. 工业级 FIR 频域净化
            data_raw = mne.filter.filter_data(
                data_raw, sfreq=FS, l_freq=1.0, h_freq=50.0, method='fir', phase='zero', verbose=False
            )
            data_raw = mne.filter.notch_filter(
                data_raw, Fs=FS, freqs=np.array([50.0]), method='fir', phase='zero', verbose=False
            )
            
            # 4. [防线增强] 双端安全共平均参考 (Dual-bound Safe CAR)
            # 既拦截死导 (<1e-4)，又隔离空间悬空高噪爆音 (>100.0)，防止全脑投毒
            chan_stds = np.std(data_raw, axis=1)
            valid_mask = (chan_stds > 1e-4) & (chan_stds < 100.0)
            if np.any(valid_mask):
                global_ref = np.mean(data_raw[valid_mask, :], axis=0)
                data_raw[valid_mask, :] -= global_ref
            
            # 提取全局统计量，下放给 QVAE 进行流形对齐，防止 40s 边界塌陷
            global_mean = np.mean(data_raw, axis=1, keepdims=True)
            global_std = np.std(data_raw, axis=1, keepdims=True)
            global_std[global_std == 0] = 1e-8
            
            # 5. 盲源分离 (QVAE / ICA) 
            data_collect, q_latents = apply_windowed_artifact_rejection(
                data_raw, sfreq=FS, window_sec=40.0, n_jobs=-1, method='qvae',
                global_mean=global_mean, global_std=global_std
            )
            
            # 6. [防线增强] 微积分压摆率限制器 (Calculus Slew Rate Limiter)
            # 替代双重调用的暴力截断算子，利用导数特征对物理瞬态电涌执行完美拦截，且仅执行一次
            data_collect = intercept_gradient_spikes(data_collect, grad_threshold=50.0, check_step=2)
            
            # ==========================================
            
            # 多模态提取：张量流动已经过解耦，时域拓扑完整，图谱无 Sinc 泄露
            adj_matrix = compute_connectivity_matrix(data_collect, fs=FS)
            dfc_matrix = compute_dfc_matrix(data_collect, fs=FS, window_sec=4.0, step_sec=1.0)
            stft_feat = compute_stft_features(data_collect, fs=FS)
            
            metadata = {'subject': subject, 'day': day, 'trial': trial_idx + 1}
            pdf_path_1d = os.path.join(day_qa_dir, f'Trial_{trial_idx + 1:02d}_1D_Waveform.pdf')
            plot_all_channels_waveform_grid(raw_snapshot, data_collect, metadata, pdf_path_1d, window_sec=10)
            pdf_path_2d = os.path.join(day_qa_dir, f'Trial_{trial_idx + 1:02d}_2D_STFT.pdf')
            plot_all_channels_stft_grid(stft_feat, raw_snapshot, data_collect, metadata, pdf_path_2d, window_sec=10)
            
            # 写入主容器
            data_pure_subject[day - 1, trial_idx] = data_collect
            adj_matrix_subject[day - 1, trial_idx] = adj_matrix
            dfc_matrix_subject[day - 1, trial_idx] = dfc_matrix
            stft_features_subject[day - 1, trial_idx] = stft_feat
            qvae_latents_subject[day - 1, trial_idx] = q_latents
            
            del data_raw, raw_snapshot
            gc.collect()
            
            # 步进更新指针与 ETA
            pbar.update(1)
            
        if day == 3:
            save_name = f"S{subject:02d}.mat"
            mdict = {
                'data_pure': data_pure_subject,
                'adj_matrix': adj_matrix_subject,
                'dfc_matrix': dfc_matrix_subject,
                'stft_features': stft_features_subject,
                'qvae_latents': qvae_latents_subject,
                'sfreq': FS,
                'ch_names': CH_NAMES
            }
            sio.savemat(os.path.join(output_dir, save_name), mdict)
            
            # 隔离缓存区刷新防换行截断
            tqdm.write(f"=== Saved S{subject:02d} fully synchronized hybrid features ===")
            
            del data_pure_subject, adj_matrix_subject, dfc_matrix_subject, stft_features_subject, qvae_latents_subject, mdict
            gc.collect()

    # 释放句柄
    pbar.close()

if __name__ == '__main__':
    pipeline()