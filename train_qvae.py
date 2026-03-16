"""
独立非监督预训练脚本 (QVAE 流形映射权重生成器)
[架构升级] 引入 IterableDataset 实现流式懒加载，解决 RAM 瓶颈，并加入 β 退火上限截断以保护高频拓扑。
[Bug修复] 修正 max_subjects 切片逻辑，确保完整覆盖 15 名被试的 45 个物理会话。
"""
import os
import argparse
import re
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
import mne

from config import CH_NAMES, FS
from core_transforms import fix_hardcoded_bads, intercept_gradient_spikes
from models import HAS_QVAE_DEPS, QuantumEEGDenoiser

if not HAS_QVAE_DEPS:
    raise ImportError("QVAE 组件依赖缺失，无法执行预训练。")

def natural_sort_key(s):
    """提取物理数字特征实现自然排序"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

class StreamingEEGDataset(IterableDataset):
    """
    [泛化防线 A] 流式张量生成器 (Iterable Lazy-Loader)
    从全量数据集中动态抽取样本并实时应用数学变换，将 RAM 占用控制在 O(1)。
    """
    def __init__(self, input_dir, max_subjects=15, trials_per_subject=5):
        self.input_dir = input_dir
        self.max_subjects = max_subjects
        self.trials_per_subject = trials_per_subject
        
        all_mat_files = [f for f in os.listdir(input_dir) if f.endswith('.mat') and 'label' not in f.lower()]
        all_mat_files.sort(key=natural_sort_key)
        
        # [核心修复] SEED 共有 15 名被试，每人 3 个文件。
        # 必须乘以 3 以覆盖 max_subjects 指向的真实物理拓扑，避免只截取前 5 人。
        self.mat_files = all_mat_files[:self.max_subjects * 3]

    def process_trial(self, data_raw, turn, trial_idx):
        if data_raw.shape[0] > 62:
            data_raw = data_raw[:62, :]
            
        # 1. 坏导修复 (拓扑补全，传入严谨的真实时空物理索引)
        data_raw = fix_hardcoded_bads(data_raw, turn=turn, trial_idx=trial_idx)
        
        # 2. 去均值与 FIR 滤波 (移除不可学的直流极点)
        data_raw = data_raw - np.mean(data_raw, axis=1, keepdims=True)
        data_raw = mne.filter.filter_data(data_raw, sfreq=FS, l_freq=1.0, h_freq=50.0, method='fir', phase='zero', verbose=False)
        data_raw = mne.filter.notch_filter(data_raw, Fs=FS, freqs=np.array([50.0]), method='fir', phase='zero', verbose=False)
        
        # 3. 双端安全 CAR (防止悬空爆音冲爆协方差)
        chan_stds = np.std(data_raw, axis=1)
        valid_mask = (chan_stds > 1e-4) & (chan_stds < 100.0)
        if np.any(valid_mask):
            global_ref = np.mean(data_raw[valid_mask, :], axis=0)
            data_raw[valid_mask, :] -= global_ref
            
        # 4. 前置压摆率拦截 (防单点异常值毁损 Global STD)
        data_raw = intercept_gradient_spikes(data_raw, grad_threshold=50.0, check_step=2)
        
        # 5. [核心约束] 全局量纲解绑 (Global Z-Score)
        g_mean = np.mean(data_raw, axis=1, keepdims=True)
        g_std = np.std(data_raw, axis=1, keepdims=True)
        g_std[g_std == 0] = 1e-8
        data_norm = (data_raw - g_mean) / g_std
        
        return data_norm.T # (T, C)

    def __iter__(self):
        # 使用 worker_info 处理多进程 DataLoader 的潜在重复采样问题
        worker_info = torch.utils.data.get_worker_info()
        
        for turn_idx, filename in enumerate(self.mat_files):
            turn = turn_idx + 1
            
            # 如果是多进程，分配不同的文件给不同的 worker
            if worker_info is not None:
                if turn_idx % worker_info.num_workers != worker_info.id:
                    continue
                    
            file_path = os.path.join(self.input_dir, filename)
            mat_data = sio.loadmat(file_path)
            trial_keys = [key for key in mat_data.keys() if not key.startswith('__') and type(mat_data[key]) == np.ndarray]
            trial_keys.sort(key=natural_sort_key)
            
            # 均匀降采样：随机抽取 trials_per_subject 个试验
            np.random.seed(turn_idx) # 保证每次 epoch 抽取的 trial 集合一致，但也可用变异种子
            sampled_keys = np.random.choice(trial_keys, min(self.trials_per_subject, len(trial_keys)), replace=False)
            
            for trial_key in sampled_keys:
                # 尝试从 key 提取准确的 trial_idx，如果失败则回退到顺序索引
                match = re.search(r'\d+', trial_key)
                trial_idx = int(match.group()) if match else 1
                
                data_raw = mat_data[trial_key].copy()
                data_norm_t = self.process_trial(data_raw, turn, trial_idx)
                
                # 逐样本生成 (Sample-by-sample Yield)
                for i in range(data_norm_t.shape[0]):
                    yield torch.tensor(data_norm_t[i], dtype=torch.float32)

def train_qvae(epochs=15, batch_size=2000, lr=1e-3, beta_max=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 初始化训练节点，计算设备挂载: {device}")
    
    os.makedirs('weights', exist_ok=True)
    
    # 实例化流式生成器 (抽取 15 人，每人每天抽 5 个 Trial，总计约 1,050,000 样本点)
    dataset = StreamingEEGDataset('Data/Preprocessed_EEG', max_subjects=15, trials_per_subject=5)
    
    # 使用 DataLoader 消费流，避免一次性加载
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=True)
    
    model = QuantumEEGDenoiser(input_dim=62, hidden_dim=32, n_qubits=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    print(f"[*] 开始流式泛化预训练 (Epochs: {epochs}, Batch Size: {batch_size}, β_max: {beta_max})")
    model.train()
    
    for epoch in range(epochs):
        # [泛化防线 B] 变分正则化余弦预热并强制上限截断 (Bounded Cosine Annealing)
        # 保护潜空间不至于过度向高斯坍缩，从而遗失特异性的高频神经拓扑细节
        beta_weight = beta_max * 0.5 * (1 - np.cos(np.pi * min(epoch / (epochs * 0.8), 1.0)))
        
        epoch_loss = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        batch_count = 0
        
        # 进度条无法预知 Iterable 的总长度，改为动态展示
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", dynamic_ncols=True)
        for batch in pbar:
            x_batch = batch.to(device) # IterableDataset 的批次直接是张量
            
            optimizer.zero_grad()
            
            recon, mu, logvar, _ = model(x_batch)
            
            # ELBO 优化目标
            recon_loss = torch.nn.functional.mse_loss(recon, x_batch, reduction='sum') / batch_size
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
            
            loss = recon_loss + beta_weight * kl_loss
            loss.backward()
            
            # 梯度裁剪防梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            batch_count += 1
            
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Recon': f"{recon_loss.item():.4f}", 'KL': f"{kl_loss.item():.4f}"})
            
        print(f"    └── Avg Loss: {epoch_loss/batch_count:.4f} | Recon: {recon_loss_sum/batch_count:.4f} | KL (β={beta_weight:.2f}): {kl_loss_sum/batch_count:.4f}")

    # 强制序列化保存权重
    save_path = 'weights/qvae_pretrained.pt'
    torch.save(model.state_dict(), save_path)
    print(f"[+] 泛化流形权重已落盘: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="QVAE Generalization Pre-training Routine")
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2000, help='Mini-batch size for VRAM safety')
    parser.add_argument('--beta_max', type=float, default=0.5, help='Maximum KL divergence penalty')
    args = parser.parse_args()
    
    train_qvae(epochs=args.epochs, batch_size=args.batch_size, beta_max=args.beta_max)