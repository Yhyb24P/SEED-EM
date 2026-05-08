from __future__ import annotations

"""
Task-aware-ready QVAE pretraining.

// 统一 checkpoint 格式与权重路径，消除推理路径错位。
// 保存训练元数据，为后续 supervised disentanglement 提供 warm start 契约。
"""

import argparse
import os
import random
import re
from typing import Dict, Iterator

import mne
import numpy as np
import scipy.io as sio
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from configs.prep_config import CH_NAMES, FS, TARGET_NODES
from engine_prep.transforms import fix_hardcoded_bads, intercept_gradient_spikes
from engine_quantum.qvae_net import HAS_QVAE_DEPS, QuantumEEGDenoiser

if not HAS_QVAE_DEPS:
    raise ImportError("QVAE 组件依赖缺失，无法执行预训练。")


def natural_sort_key(s: str):
    """Sort names with embedded integers."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class StreamingEEGDataset(IterableDataset):
    """Stream per-sample EEG points from filtered trials."""

    def __init__(self, input_dir: str, max_subjects: int = 15, trials_per_subject: int = 5, in_channels: int = 16):
        self.input_dir = input_dir
        self.max_subjects = max_subjects
        self.trials_per_subject = trials_per_subject
        self.in_channels = in_channels
        self.core_idx = [CH_NAMES.index(name) for name in TARGET_NODES[:in_channels]]
        # // 挂载进程级字典以固化跨 Epoch 的 I/O 与滤波张量
        self._cache = {}
        all_mat_files = [f for f in os.listdir(input_dir) if f.endswith(".mat") and "label" not in f.lower()]
        all_mat_files.sort(key=natural_sort_key)
        self.mat_files = all_mat_files[: self.max_subjects * 3]
        
        # // 装载真实物理情绪标签以支持下游局部验证钩子
        label_path = os.path.join(input_dir, "label.mat")
        if not os.path.exists(label_path):
            label_path = os.path.join(input_dir, "../01_raw_mat/label.mat")
        try:
            self.labels = sio.loadmat(label_path)['label'][0]
        except Exception:
            self.labels = np.zeros(15)

    def process_trial(self, data_raw: np.ndarray, turn: int, trial_idx: int) -> np.ndarray:
        """Apply the same physical cleanup used by run_pipeline."""
        if data_raw.shape[0] > 62:
            data_raw = data_raw[:62, :]
        data_raw = fix_hardcoded_bads(data_raw, turn=turn, trial_idx=trial_idx)
        data_raw = data_raw[self.core_idx, :]
        data_raw = data_raw - np.mean(data_raw, axis=1, keepdims=True)
        data_raw = mne.filter.filter_data(data_raw, sfreq=FS, l_freq=1.0, h_freq=50.0, method="fir", phase="zero", verbose=False)
        data_raw = mne.filter.notch_filter(data_raw, Fs=FS, freqs=np.array([50.0]), method="fir", phase="zero", verbose=False)
        chan_stds = np.std(data_raw, axis=1)
        valid_mask = (chan_stds > 1e-4) & (chan_stds < 100.0)
        if np.any(valid_mask):
            global_ref = np.mean(data_raw[valid_mask, :], axis=0)
            data_raw[valid_mask, :] -= global_ref
        data_raw = intercept_gradient_spikes(data_raw, grad_threshold=50.0, check_step=2)
        g_mean = np.mean(data_raw, axis=1, keepdims=True)
        g_std = np.std(data_raw, axis=1, keepdims=True)
        g_std[g_std == 0] = 1e-8
        data_norm = (data_raw - g_mean) / g_std
        return data_norm.T.astype(np.float32)

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker_info = torch.utils.data.get_worker_info()
        for turn_idx, filename in enumerate(self.mat_files):
            if worker_info is not None and turn_idx % worker_info.num_workers != worker_info.id:
                continue
            turn = turn_idx + 1
            
            # // 命中解析缓存则跳过高负载磁盘读取
            if filename not in self._cache:
                file_path = os.path.join(self.input_dir, filename)
                self._cache[filename] = sio.loadmat(file_path)
            mat_data = self._cache[filename]
            
            trial_keys = [key for key in mat_data.keys() if not key.startswith("__") and isinstance(mat_data[key], np.ndarray)]
            trial_keys.sort(key=natural_sort_key)
            
            # // 释放静态种子绑定以允许跨 Epoch 独立同分布随机采样
            rng = np.random.default_rng()
            sampled_keys = rng.choice(trial_keys, min(self.trials_per_subject, len(trial_keys)), replace=False)
            
            # // 引入局部缓冲区阻断批次内高自相关性
            buffer = []
            label_buffer = []
            
            for trial_key in sampled_keys:
                match = re.search(r"\d+", trial_key)
                trial_idx = int(match.group()) if match else 1
                
                # // 扩展缓存覆盖至滤波层级，将 O(N) 重复物理映射坍缩为 O(1)
                cache_key = f"{filename}_{trial_key}"
                if cache_key not in self._cache:
                    data_raw = mat_data[trial_key].copy()
                    self._cache[cache_key] = self.process_trial(data_raw, turn, trial_idx)
                buffer.append(self._cache[cache_key])
                label_buffer.extend([self.labels[trial_idx - 1]] * self._cache[cache_key].shape[0])
                
            # // 拼接连续 NumPy 内存块后单次强转 Tensor，彻底阻断 Python 对象分配抖动
            buffer_arr = np.concatenate(buffer, axis=0)
            
            label_arr = np.array(label_buffer)
            
            # // 同步打乱波形基底与类别标签张量
            indices = np.arange(buffer_arr.shape[0])
            rng.shuffle(indices)
            buffer_arr = buffer_arr[indices]
            label_arr = label_arr[indices]


            buffer_tensor = torch.from_numpy(buffer_arr)
            label_tensor = torch.from_numpy(label_arr).long()
            for i in range(buffer_tensor.size(0)):
                yield buffer_tensor[i], label_tensor[i]


def build_checkpoint(
    model: QuantumEEGDenoiser,
    epochs: int,
    batch_size: int,
    lr: float,
    beta_max: float,
    input_dir: str,
) -> Dict[str, object]:
    """Build structured checkpoint with self-describing architecture metadata."""
    return {
        "model_state_dict": model.state_dict(),
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "beta_max": beta_max,
            "input_dir": input_dir,
        },
        "input_dim": model.encoder[0].in_features,
        "n_qubits": model.n_qubits,
        "latent_dim": model.n_qubits,
        "channels": CH_NAMES[:model.encoder[0].in_features],
        "fs": FS,
    }


def train_qvae(
    epochs: int = 15,
    batch_size: int = 2000,
    lr: float = 1e-3,
    beta_max: float = 0.5,
    input_dir: str = "data/01_raw_mat",
    output_path: str = "data/04_weights/qvae_pretrained.pt",
    seed: int = 42,
    save_legacy_copy: bool = True,
) -> str:
    """Train QVAE and save a structured checkpoint."""
    seed_everything(seed)
    device = torch.device("cpu")
    print(f"[*] 初始化训练节点，计算设备挂载: {device}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset = StreamingEEGDataset(input_dir, max_subjects=15, trials_per_subject=5, in_channels=16)
    num_workers = max(1, min(4, os.cpu_count() or 1))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )

    model = QuantumEEGDenoiser(input_dim=16, hidden_dim=32, n_qubits=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"[*] 开始流式泛化预训练 (Epochs: {epochs}, Batch Size: {batch_size}, β_max: {beta_max})")
    model.train()
    best_probe_f1 = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        warmup_epochs = max(1, int(np.ceil(epochs * 0.8)))
        beta_progress = min((epoch + 1) / warmup_epochs, 1.0)
        beta_weight = beta_max * 0.5 * (1 - np.cos(np.pi * beta_progress))
        epoch_loss = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        batch_count = 0
        probe_mu, probe_y = [], []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", dynamic_ncols=True)
        for batch in pbar:
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            recon, mu, logvar, _ = model(x_batch)
            recon_loss = torch.nn.functional.mse_loss(recon, x_batch, reduction="sum") / batch_size
            
            # // 细粒度计算独立通道活跃度，用于精准诊断变分后验坍缩
            kl_per_qubit = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
            active_qubits = (kl_per_qubit > 0.1).sum().item()
            kl_loss = kl_per_qubit.sum()
            
            
            loss = recon_loss + beta_weight * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # // 轻量级采样潜变量流形与物理标签，储备探针评估池
            if random.random() < 0.1:
                probe_mu.append(mu.detach().cpu().numpy())
                probe_y.append(y_batch.cpu().numpy())

            epoch_loss += float(loss.item())
            recon_loss_sum += float(recon_loss.item())
            kl_loss_sum += float(kl_loss.item())
            batch_count += 1
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Recon": f"{recon_loss.item():.4f}",
                "KL": f"{kl_loss.item():.4f}",
                "ActQ": f"{active_qubits}/{model.n_qubits}"
            })

        if batch_count == 0:
            raise RuntimeError("预训练未产生任何批次；请检查 input_dir 与 MAT 文件结构。")
        
        
        # // 触发 Epoch 级无感线性探针验证以提取语义增量
        epoch_f1 = 0.0
        if probe_mu:
            import warnings
            from sklearn.exceptions import ConvergenceWarning
            X_val = np.concatenate(probe_mu, axis=0)
            Y_val = np.concatenate(probe_y, axis=0)
            if len(np.unique(Y_val)) > 1:
                clf = LinearSVC(C=1.0, max_iter=1000, dual=False)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    cv_scores = cross_val_score(clf, X_val, Y_val, cv=3, scoring='f1_macro')
                epoch_f1 = cv_scores.mean()
        
        print(
            f"    └── Avg Loss: {epoch_loss / batch_count:.4f} | "
            f"Recon: {recon_loss_sum / batch_count:.4f} | "
            f"KL (β={beta_weight:.2f}): {kl_loss_sum / batch_count:.4f} | "
            f"Active Q: {active_qubits} | Probe F1: {epoch_f1:.4f}"
        )
        
        # // 执行潜流形语义早停熔断，拦截过度拟合高频噪声消耗 CPU 算力
        if epoch_f1 > best_probe_f1 + 0.01:
            best_probe_f1 = epoch_f1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 3 and epoch > warmup_epochs:
                print("  [!] 线性探测器 F1 触及瓶颈，触发提前终止。")
                break   

    checkpoint = build_checkpoint(
        model=model,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        beta_max=beta_max,
        input_dir=input_dir,
    )
    torch.save(checkpoint, output_path)
    print(f"[+] 结构化 checkpoint 已落盘: {output_path}")

    if save_legacy_copy:
        legacy_dir = "weights"
        os.makedirs(legacy_dir, exist_ok=True)
        legacy_path = os.path.join(legacy_dir, "qvae_pretrained.pt")
        torch.save(model.state_dict(), legacy_path)
        print(f"[+] 兼容旧推理路径的 state_dict 已落盘: {legacy_path}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task-aware-ready QVAE warm-start pretraining")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta_max", type=float, default=0.5)
    parser.add_argument("--input_dir", type=str, default="data/01_raw_mat")
    parser.add_argument("--output_path", type=str, default="data/04_weights/qvae_pretrained.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_legacy_copy", action="store_true")
    args = parser.parse_args()
    train_qvae(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta_max=args.beta_max,
        input_dir=args.input_dir,
        output_path=args.output_path,
        seed=args.seed,
        save_legacy_copy=not args.no_legacy_copy,
    )
