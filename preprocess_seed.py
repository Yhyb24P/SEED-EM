import os
import glob
import re
import numpy as np
import scipy.io as sio
from scipy import signal
import mne
from joblib import Parallel, delayed

# 全局常量：确保通道元数据可用于所有函数及输出保存
CH_NAMES = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
            'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1',
            'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
            'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ',
            'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']

# ==========================================
# 辅助函数: 还原 MATLAB 预处理自定义逻辑
# ==========================================

def natural_sort_key(s):
    """用于实现 Trial 与文件的自然排序 (替代 MATLAB `who` 乱序人工还原机制)"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def reref_car(data):
    """重参考: Common Average Reference"""
    ref = np.mean(data, axis=0)
    return data - ref

def fun_rmout_python(data, check_step=2, threshold=130):
    """
    等效于 MATLAB fun_rmout
    剔除包含超过阈值幅度的时间点，并向两侧延伸 check_step 个点。
    注意：此操作沿时间轴 (axis=1) 执行硬截断，会改变时间点总长度。
    
    参数:
    data: np.ndarray, 形状 (n_channels, n_times)
    """
    # 寻找任何通道存在越界(±130)的时间点
    outlier_mask = np.any(np.abs(data) > threshold, axis=0)
    
    # 优化：采用 1D 卷积算子执行布尔膨胀，完全消除 for 循环，提升矢量化执行效率
    kernel = np.ones(2 * check_step + 1, dtype=bool)
    rm_mask = np.convolve(outlier_mask, kernel, mode='same') > 0
    
    # 提取未被标记为删除的列
    clean_data = data[:, ~rm_mask]
    return clean_data

def fix_hardcoded_bads(data_raw, turn, trial_idx):
    """
    严格还原 MATLAB 脚本中硬编码的各被试/批次坏导插值机制。
    注: 参数 turn 从 1 到 45 (1-based), trial_idx 从 0 到 14 (0-based)
        所有矩阵行索引均已转换为 Python 0-based。
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
        if trial_idx == 1:  # 原版为 trial == 2 (MATLAB index)
            data_raw[61, :] = np.mean(data_raw[[59, 55, 56], :], axis=0)
    elif turn == 45 and trial_idx > 10:  # 原版为 trial > 11 (MATLAB index)
        data_raw[9, :] = np.mean(data_raw[[8, 18, 10], :], axis=0)
        
    return data_raw

def _process_ica_segment(data_seg, info, montage):
    """用于 Joblib 并行的单段 ICA 处理核心逻辑"""
    raw_seg = mne.io.RawArray(data_seg * 1e-6, info, verbose=False)
    raw_seg.set_montage(montage, on_missing='ignore')
    
    # [优化 1]：创建用于 ICA 拟合的高通副本 (1.0 Hz)
    # 目的：阻断次低频漂移对盲源分离方差的干扰，消除 RuntimeWarning
    raw_ica_fit = raw_seg.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
    
    # [优化 2]：增加 max_iter
    # 目的：为 FastICA 算法提供更高的迭代深度，消除 ConvergenceWarning
    ica = mne.preprocessing.ICA(n_components=15, random_state=42, max_iter=2000, verbose=False)
    
    # 仅在高通副本上执行空间矩阵解算
    ica.fit(raw_ica_fit, verbose=False)
    
    # 在 0.25Hz 原始数据上通过模式匹配定位眼电成分
    eog_indices, _ = ica.find_bads_eog(raw_seg, ch_name=['FP1', 'FP2'], verbose=False)
    ica.exclude = eog_indices
    
    # 将净化后的 ICA 逆矩阵投射回 0.25Hz 原始数据
    raw_clean = ica.apply(raw_seg.copy(), verbose=False)
    return raw_clean.get_data() * 1e6

def apply_windowed_ica(data, sfreq=200.0, window_sec=40.0, n_jobs=-1):
    """
    等效于 MATLAB 中 seg_num 循环体与 fun_rmeye_test 机制。
    按窗提取、逐段执行 ICA 并剔除伪迹成分，严格舍弃窗外尾部数据。已整合多核并行加速。
    """
    window_len = int(sfreq * window_sec) # 8000
    seg_num = data.shape[1] // window_len
    n_chan = data.shape[0]
    
    data_collect = np.zeros((n_chan, seg_num * window_len))
    
    # 构建信息字典用于 MNE 处理
    info = mne.create_info(ch_names=CH_NAMES, sfreq=sfreq, ch_types=['eeg'] * n_chan)
    montage = mne.channels.make_standard_montage('standard_1020')
    
    # 提取各段时间序列片段以供并行
    segments = [data[:, seg * window_len:(seg + 1) * window_len] for seg in range(seg_num)]
    
    # 使用 joblib 进行 CPU 级并行运算，加速独立成分矩阵的拟合求解
    processed_segments = Parallel(n_jobs=n_jobs)(
        delayed(_process_ica_segment)(seg_data, info, montage) for seg_data in segments
    )
    
    # 拼接处理后的并行块序列
    for seg, seg_data in enumerate(processed_segments):
        start_idx = seg * window_len
        end_idx = (seg + 1) * window_len
        data_collect[:, start_idx:end_idx] = seg_data
        
    return data_collect

# ==========================================
# 主运行管线
# ==========================================

def pipeline():
    input_dir = 'Data/Preprocessed_EEG'
    output_dir = 'Data/EEG_pure'
    os.makedirs(output_dir, exist_ok=True)
    
    fs = 200.0
    check_step = 2
    
    # 获取目录下的 mat 文件
    all_mat_files = [f for f in os.listdir(input_dir) if f.endswith('.mat')]
    
    # 逻辑修复：隔离 label 文件。剔除文件名中包含 'label' 的文件以确保被试文件对齐
    mat_files = [f for f in all_mat_files if 'label' not in f.lower()]
    
    # 假设文件名为 '1_20131027.mat' 形式，使用自然排序
    mat_files.sort(key=natural_sort_key)
    
    if len(mat_files) != 45:
        print(f"[Warning] Expected 45 files, found {len(mat_files)}. Please check input directory.")
        
    data_pure_subject = None
    
    for turn_idx, filename in enumerate(mat_files):
        turn = turn_idx + 1 # MATLAB 的 turn 为 1-based (1 到 45)
        subject = int(np.ceil(turn / 3.0))
        day = turn - (subject - 1) * 3
        
        if day == 1:
            # 初始化 Python Cell Array 替代物 (3天 x 15 trials)
            data_pure_subject = np.empty((3, 15), dtype=object)
            
        file_path = os.path.join(input_dir, filename)
        mat_data = sio.loadmat(file_path)
        
        # 获取所有 EEG 相关主变量，并执行自然排序 (等效于原代码 [1; 8:15; 2:7])
        trial_keys = [key for key in mat_data.keys() if not key.startswith('__') and type(mat_data[key]) == np.ndarray]
        trial_keys.sort(key=natural_sort_key)
        
        # 边界与输入有效性验证 1：确保试验批次结构严格等于 15
        if len(trial_keys) != 15:
            raise ValueError(f"[Error] 被试 {subject} Day {day} 文件 {filename} 数据异常：包含 {len(trial_keys)} 个 Trials（预期值：15）。")
        
        for trial_idx, trial_key in enumerate(trial_keys):
            print(f"Subject: {subject}, Day: {day}, Trial: {trial_idx + 1}")
            
            # (channels, times)
            data_raw = mat_data[trial_key]
            
            # 边界与输入有效性验证 2：确保信号通道数目严格等于 62
            if data_raw.shape[0] != 62:
                raise ValueError(f"[Error] 数据异常：通道数越界。收到 {data_raw.shape[0]} 通道（预期值：62 通道）。")
            
            # 1. 修复硬编码坏导 (对齐物理实验异常)
            data_raw = fix_hardcoded_bads(data_raw, turn, trial_idx)
            
            # 2. 重参考 CAR
            data_raw = reref_car(data_raw)
            
            # 3. 带通滤波 (Butterworth order=2, Wn=[0.25, 50])
            # 对应 MATLAB: Wn = [0.25*2 50*2] / 200
            nyq = fs / 2.0
            b, a = signal.butter(2, [0.25 / nyq, 50.0 / nyq], btype='bandpass')
            # MATLAB filtfilt 会转置执行，Python 直接对最后一维执行 axis=1
            data_raw = signal.filtfilt(b, a, data_raw, axis=1)
            
            # 4. 去基线漂移
            data_raw = signal.detrend(data_raw, axis=1)
            
            # 5. 删除超幅伪迹及其近邻点
            data_raw = fun_rmout_python(data_raw, check_step, threshold=130)
            
            # 6. 分段独立成分去眼电 (舍弃尾段)
            data_collect = apply_windowed_ica(data_raw, sfreq=fs, window_sec=40.0, n_jobs=-1)
            
            # 存储至矩阵体系 (day-1, trial_idx 均为 0-based)
            data_pure_subject[day - 1, trial_idx] = data_collect
            
        # 当 Day=3 完成该被试的整个批次，实施保存
        if day == 3:
            save_name = f"S{subject}.mat"
            # 构建包含元数据的保存字典
            mdict = {
                'data_pure': data_pure_subject,
                'sfreq': fs,
                'ch_names': CH_NAMES
            }
            sio.savemat(os.path.join(output_dir, save_name), mdict)
            print(f"=== Saved S{subject} with metadata ===")

if __name__ == '__main__':
    pipeline()