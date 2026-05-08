from __future__ import annotations

"""
Task-Aware Component Selector (Phase A).

// 仅保留 ComponentSelector 网络定义，供 engine_prep.artifact 加载 Selector 权重。
// DifferentiableSTFTAndDE 与 compute_selector_loss 已统一至 engine_gnn.classifier_taskaware_adapter。
"""

import torch
import torch.nn as nn


class ComponentSelector(nn.Module):
    """
    Task-Aware Component Selector (Phase A).
    接收 BSS/QVAE 潜变量分量 C_i，输出保留概率 p_i ∈ (0, 1)。
    """

    def __init__(self, in_channels: int = 1, hidden_dim: int = 32, n_components: int = 6):
        super().__init__()
        self.n_components = n_components

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=15, stride=5, padding=7),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, z_components: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_components: (B, K, T) 独立分量张量
        Returns:
            p_mask: (B, K, 1) 保留概率掩码
        """
        B, K, T = z_components.shape
        x = z_components.view(B * K, 1, T)
        feats = self.feature_extractor(x).squeeze(-1)
        scores = self.scorer(feats)
        return scores.view(B, K, 1)