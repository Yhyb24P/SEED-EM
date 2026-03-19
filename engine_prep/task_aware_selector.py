from __future__ import annotations

"""
Task-Aware Component Selector & Differentiable Feature Bridge.

// 提供端到端可微的 Selector 网络与 PyTorch 原生 STFT->DE 算子。
// 解决 Proxy 梯度反传时的 NumPy/SciPy 梯度断崖问题。
"""

import torch
import torch.nn as nn

from configs.prep_config import FS


class ComponentSelector(nn.Module):
    """
    Task-Aware Component Selector (Phase A).
    接收 BSS/QVAE 潜变量分量 C_i，输出保留概率 p_i \in (0, 1)。
    """

    def __init__(self, in_channels: int = 1, hidden_dim: int = 32, n_components: int = 6):
        super().__init__()
        self.n_components = n_components
        
        # // 利用 1D-CNN 提取时间流形的高频/低频纹理，捕捉潜在伪迹模式
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=15, stride=5, padding=7),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # // 输出软保留概率，引入 sigmoid 边界
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, z_components: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_components: (B, K, T) 独立分量张量
        Returns:
            p_mask: (B, K, 1) 保留概率掩码
        """
        B, K, T = z_components.shape
        # // 展平为 (B*K, 1, T) 以共享网络权重对各分量独立打分
        x = z_components.view(B * K, 1, T)
        feats = self.feature_extractor(x).squeeze(-1)  # (B*K, hidden_dim*2)
        scores = self.scorer(feats)                    # (B*K, 1)
        
        p_mask = scores.view(B, K, 1)
        return p_mask


class DifferentiableSTFTAndDE(nn.Module):
    """
    可微微分熵 (DE) 提取器。
    严格对齐 scipy.signal.stft 的数学基底，保持前向计算图完整。
    """

    def __init__(self, fs: float = FS, window_sec: float = 2.0):
        super().__init__()
        self.fs = fs
        self.n_fft = int(fs * window_sec)
        self.hop_length = int(fs * 1.0)  # // 默认 1s 步长同步 GNN 动态图
        
        # // 注册汉宁窗为不可导 buffer，防止张量设备错位
        window = torch.hann_window(self.n_fft)
        self.register_buffer("window", window)

        # // 物理频段的索引确界 (Delta, Theta, Alpha, Beta, Gamma)
        freqs = torch.fft.rfftfreq(self.n_fft, d=1.0/fs)
        self.bands = {
            "delta": (freqs >= 1.0) & (freqs < 4.0),
            "theta": (freqs >= 4.0) & (freqs < 8.0),
            "alpha": (freqs >= 8.0) & (freqs < 14.0),
            "beta":  (freqs >= 14.0) & (freqs < 31.0),
            "gamma": (freqs >= 31.0) & (freqs <= 50.0),
        }

    def forward(self, x_pure: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_pure: (B, C, T) 纯净物理波形 (可导)
        Returns:
            node_de: (B, C, 5, T_bins) 微分熵节点特征
        """
        B, C, T = x_pure.shape
        x_flat = x_pure.view(B * C, T)
        
        # // 执行 PyTorch 原生可微 STFT，返回复数张量
        stft_complex = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=False,
            return_complex=True
        ) # (B*C, F, T_bins)
        
        stft_mag_sq = torch.abs(stft_complex) ** 2
        
        de_bands = []
        for band_name, freq_mask in self.bands.items():
            # // 提取对应频带能量并积分
            band_energy = torch.sum(stft_mag_sq[:, freq_mask, :], dim=1)  # (B*C, T_bins)
            # // 计算微分熵近似值，加入 1e-8 防御对数发散极点
            band_de = torch.log(torch.clamp(band_energy, min=1e-8))
            de_bands.append(band_de)
            
        # // (B*C, 5, T_bins) -> (B, C, 5, T_bins)
        node_de = torch.stack(de_bands, dim=1).view(B, C, 5, -1)
        return node_de


def compute_selector_loss(
    p_mask: torch.Tensor,
    artifact_score_prior: torch.Tensor,
    lambda_sparse: float = 0.1,
    lambda_smooth: float = 0.05,
    lambda_anticheat: float = 0.5
) -> torch.Tensor:
    """
    计算 Selector 专属正则化损失 (对应 Step A2 中的稳定项与反作弊项)。
    """
    # 1. 稀疏性惩罚: L1 范数防止全 1 保留 (L_sparse = |p|_1)
    loss_sparse = torch.mean(p_mask)
    
    # 2. 平滑性惩罚: 避免同试次内概率方差过大，维持局部平稳
    mean_p = torch.mean(p_mask, dim=1, keepdim=True)
    loss_smooth = torch.mean(torch.abs(p_mask - mean_p))
    
    # 3. 反作弊惩罚 (L_anti_cheat):
    # 若先验 artifact_score 极高 (疑似 EOG/EMG)，但 Selector 依然给出高保留概率，则施加重罚。
    # 阻断 Selector 将面部肌肉活动作为情绪分类捷径。
    loss_anticheat = torch.mean(p_mask * artifact_score_prior.mean(dim=-1, keepdim=True))
    
    total_reg = lambda_sparse * loss_sparse + \
                lambda_smooth * loss_smooth + \
                lambda_anticheat * loss_anticheat
                
    return total_reg