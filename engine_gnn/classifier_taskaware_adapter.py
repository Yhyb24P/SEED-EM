from __future__ import annotations

"""
Classifier-side adapter for task-aware preprocessing outputs.

// 适配 run_pipeline 导出的 node_de / artifact meta 至双流 classifier。
// 提供统一 loss 聚合入口，以及 Phase A 所需的 Selector / 可微 STFT->DE / 图批桥接。
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

try:
    from prep_config import FS
except ImportError:
    FS = 200.0


@dataclass
class TaskAwareBatch:
    node_de: torch.Tensor
    adj_matrix: torch.Tensor
    dfc_matrix: torch.Tensor
    emotion_label: torch.Tensor
    subject_label: torch.Tensor
    artifact_scores: Optional[torch.Tensor] = None
    retain_probs: Optional[torch.Tensor] = None


@dataclass
class TaskAwareLossConfig:
    lambda_trait: float = 1.0
    lambda_adv: float = 1.0
    lambda_ortho: float = 0.1
    lambda_supcon: float = 0.5
    lambda_artinv: float = 0.2


@dataclass
class Schedules:
    alpha: float
    gamma: float


@dataclass
class TemporalGraphSpec:
    temporal_window: int = 3
    edge_threshold: float = 0.3
    drop_initial_bins: int = 20


class ComponentSelector(nn.Module):
    """对潜变量分量打分，输出保留概率。"""

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
        if z_components.dim() != 3:
            raise ValueError(f"z_components must be (B,K,T), got {tuple(z_components.shape)}")
        bsz, n_comp, tlen = z_components.shape
        if n_comp != self.n_components:
            raise ValueError(f"expected {self.n_components} components, got {n_comp}")
        x = z_components.reshape(bsz * n_comp, 1, tlen)
        feats = self.feature_extractor(x).squeeze(-1)
        scores = self.scorer(feats)
        return scores.view(bsz, n_comp, 1)


class DifferentiableSTFTAndDE(nn.Module):
    """逐项对齐原始 SciPy STFT 与 dataloader DE 口径的可微桥接器。"""

    def __init__(self, fs: float = FS, window_sec: float = 2.0, hop_sec: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.fs = float(fs)
        self.window_sec = float(window_sec)
        self.hop_sec = float(hop_sec)
        self.n_fft = int(round(self.fs * self.window_sec))
        self.win_length = self.n_fft
        self.hop_length = int(round(self.fs * self.hop_sec))
        self.eps = float(eps)

        # 对齐 scipy.signal.stft 默认: window='hann_periodic'
        window = torch.hann_window(self.win_length, periodic=True)
        self.register_buffer("window", window)
        # 对齐 scipy.signal.stft 默认: scaling='spectrum' -> mode='stft' 时乘以 1 / sum(window)
        self.register_buffer("window_scale", window.sum().clamp_min(self.eps))

        # 对齐 scipy.signal.stft 默认: return_onesided=True, nfft=nperseg
        freqs = torch.fft.rfftfreq(self.n_fft, d=1.0 / self.fs)
        self.register_buffer("freqs", freqs)

        # 对齐 dataloader._stft_to_de 的 bin 切片规则
        self.band_slices = (
            ("delta", 2, 8),
            ("theta", 8, 16),
            ("alpha", 16, 26),
            ("beta", 26, 60),
            ("gamma", 60, 90),
        )

    def _pad_like_scipy(self, x: torch.Tensor) -> torch.Tensor:
        # 对齐 scipy.signal.stft 默认: boundary='zeros'
        pad = self.win_length // 2
        if pad > 0:
            x = F.pad(x, (pad, pad), mode="constant", value=0.0)

        # 对齐 scipy.signal.stft 默认: padded=True
        nstep = self.hop_length
        nadd = (-(x.shape[-1] - self.win_length) % nstep) % self.win_length
        if nadd > 0:
            x = F.pad(x, (0, int(nadd)), mode="constant", value=0.0)
        return x

    def forward_stft(self, x_pure: torch.Tensor) -> torch.Tensor:
        if x_pure.dim() != 3:
            raise ValueError(f"x_pure must be (B,C,T), got {tuple(x_pure.shape)}")
        bsz, n_ch, tlen = x_pure.shape
        if tlen == 0:
            raise ValueError("x_pure time dimension must be > 0")

        # 对齐短序列兜底，避免 torch.stft 在长度不足 win_length 时崩溃
        if tlen < self.win_length:
            x_pure = F.pad(x_pure, (0, self.win_length - tlen), mode="constant", value=0.0)

        x_flat = x_pure.reshape(bsz * n_ch, -1)
        x_flat = self._pad_like_scipy(x_flat)

        stft_complex = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        # 对齐 scipy.signal.stft 默认: scaling='spectrum'
        stft_mag = stft_complex.abs() / self.window_scale
        return stft_mag.view(bsz, n_ch, stft_mag.size(-2), stft_mag.size(-1))

    def forward(self, x_pure: torch.Tensor) -> torch.Tensor:
        stft_mag = self.forward_stft(x_pure)
        de_bands = []
        n_freq = stft_mag.size(2)
        for _, start, end in self.band_slices:
            if start >= n_freq:
                band_energy = torch.zeros_like(stft_mag[:, :, 0, :])
            else:
                end_clamped = min(end, n_freq)
                # 对齐 dataloader._stft_to_de: sum(square(abs(Zxx)), axis=freq)
                band_energy = stft_mag[:, :, start:end_clamped, :].pow(2).sum(dim=2)
            de_bands.append(torch.log(torch.clamp(band_energy, min=self.eps)))
        return torch.stack(de_bands, dim=2)


def compute_schedules(epoch: int, max_epoch: int) -> Schedules:
    """Compute GRL and trait schedules from normalized progress."""
    warmup_epochs = 3
    if epoch <= warmup_epochs:
        return Schedules(alpha=0.0, gamma=0.0)

    progress = float(epoch - warmup_epochs) / float(max(max_epoch - warmup_epochs, 1))
    scalar = 2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * progress))) - 1.0
    alpha = 0.15 * float(scalar.item())
    gamma = (torch.log(torch.tensor(3.0)) / torch.log(torch.tensor(15.0))) * scalar
    return Schedules(alpha=alpha, gamma=float(gamma.item()))


def masked_offdiag_cosine(a_e: torch.Tensor, a_s: torch.Tensor) -> torch.Tensor:
    """Compute off-diagonal cosine similarity for orthogonality regularization."""
    n = a_e.shape[-1]
    eye = torch.eye(n, device=a_e.device, dtype=a_e.dtype)
    mask = 1.0 - eye
    a_e_masked = a_e * mask
    a_s_masked = a_s * mask
    num = torch.sum(a_e_masked * a_s_masked, dim=(-1, -2))
    den = torch.norm(a_e_masked, dim=(-1, -2)) * torch.norm(a_s_masked, dim=(-1, -2))
    den = torch.clamp(den, min=1e-8)
    return torch.mean(num / den)


def artifact_invariance_loss(z_emo: torch.Tensor, artifact_scores: Optional[torch.Tensor]) -> torch.Tensor:
    """Penalize correlation between emotion embedding magnitude and artifact intensity."""
    if artifact_scores is None or artifact_scores.numel() == 0:
        return z_emo.new_tensor(0.0)
    score = torch.nanmean(artifact_scores, dim=-1, keepdim=True)
    score = (score - score.mean(dim=0, keepdim=True)) / (score.std(dim=0, keepdim=True) + 1e-8)
    z_norm = (z_emo - z_emo.mean(dim=0, keepdim=True)) / (z_emo.std(dim=0, keepdim=True) + 1e-8)
    corr = torch.mean(torch.abs(torch.mean(z_norm.unsqueeze(1) * score, dim=0)))
    return corr


# 构造单个时序图样本

def build_temporal_graphs_from_de(
    node_de: torch.Tensor,
    adj_matrix: torch.Tensor,
    label: torch.Tensor,
    subj_id: torch.Tensor,
    spec: TemporalGraphSpec | None = None,
) -> List[Data]:
    spec = spec or TemporalGraphSpec()
    if node_de.dim() != 3:
        raise ValueError(f"node_de must be (C,5,T_bins), got {tuple(node_de.shape)}")
    if adj_matrix.dim() != 2:
        raise ValueError(f"adj_matrix must be (C,C), got {tuple(adj_matrix.shape)}")

    n_ch, n_bands, n_bins = node_de.shape
    start_bin = min(spec.drop_initial_bins, max(n_bins - spec.temporal_window, 0))
    node_de = node_de[:, :, start_bin:]
    n_bins = node_de.shape[-1]
    if n_bins < spec.temporal_window:
        return []

    adj = adj_matrix.clone()
    adj = adj - torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
    edge_indices = torch.where(torch.abs(adj) > spec.edge_threshold)
    if edge_indices[0].numel() == 0:
        full_idx = torch.arange(n_ch, device=adj.device)
        edge_index_base = torch.stack([full_idx, full_idx], dim=0)
        edge_weight_base = torch.ones(n_ch, device=adj.device, dtype=adj.dtype)
    else:
        edge_index_base = torch.stack(edge_indices, dim=0).long()
        edge_weight_base = torch.abs(adj[edge_indices]).float()

    graphs: List[Data] = []
    for sec in range(0, n_bins - spec.temporal_window + 1):
        x_slices = [node_de[:, :, sec + t] for t in range(spec.temporal_window)]
        x_tensor = torch.cat(x_slices, dim=0).float()
        edge_indices_t = [edge_index_base + t * n_ch for t in range(spec.temporal_window)]
        edge_index_t = torch.cat(edge_indices_t, dim=1)
        edge_attr_t = edge_weight_base.repeat(spec.temporal_window)
        graphs.append(
            Data(
                x=x_tensor,
                edge_index=edge_index_t,
                edge_attr=edge_attr_t,
                y=label.view(1).long(),
                subj_id=subj_id.view(1).long(),
                time_len=torch.tensor([spec.temporal_window], dtype=torch.long),
            )
        )
    return graphs


# 合并多个试次图为一个 Batch

def batch_graphs(graphs: Iterable[Data], device: torch.device | None = None) -> Batch | None:
    graphs = list(graphs)
    if not graphs:
        return None
    batch = Batch.from_data_list(graphs)
    return batch.to(device) if device is not None else batch


# 计算 Selector 正则项

def compute_selector_loss(
    p_mask: torch.Tensor,
    artifact_score_prior: torch.Tensor,
    lambda_sparse: float = 0.1,
    lambda_smooth: float = 0.05,
    lambda_anticheat: float = 0.5,
) -> torch.Tensor:
    if artifact_score_prior.dim() == 3:
        prior = artifact_score_prior.mean(dim=-1)
    elif artifact_score_prior.dim() == 2:
        prior = artifact_score_prior
    else:
        raise ValueError(f"artifact_score_prior must be (B,K) or (B,K,T), got {tuple(artifact_score_prior.shape)}")

    loss_sparse = torch.mean(p_mask)
    mean_p = torch.mean(p_mask, dim=1, keepdim=True)
    loss_smooth = torch.mean(torch.abs(p_mask - mean_p))
    loss_anticheat = torch.mean(p_mask.squeeze(-1) * prior)
    return lambda_sparse * loss_sparse + lambda_smooth * loss_smooth + lambda_anticheat * loss_anticheat


def compute_total_loss(
    outputs: Dict[str, torch.Tensor],
    batch: TaskAwareBatch,
    epoch: int,
    max_epoch: int,
    cfg: TaskAwareLossConfig,
) -> Dict[str, torch.Tensor]:
    """Aggregate classifier losses with optional artifact invariance term."""
    schedules = compute_schedules(epoch, max_epoch)
    y_emo = outputs["y_emo"]
    y_subj = outputs["y_subj"]
    y_adv = outputs["y_adv"]
    z_emo = outputs["z_emo"]
    a_e = outputs["a_e"]
    a_s = outputs["a_s"]

    loss_cls = F.cross_entropy(y_emo, batch.emotion_label)
    loss_trait = F.cross_entropy(y_subj, batch.subject_label)
    loss_adv = F.cross_entropy(y_adv, batch.subject_label)
    loss_ortho = masked_offdiag_cosine(a_e, a_s)

    loss_artinv = artifact_invariance_loss(z_emo, batch.artifact_scores)
    total = (
        loss_cls
        + schedules.gamma * cfg.lambda_trait * loss_trait
        + schedules.alpha * cfg.lambda_adv * loss_adv
        + cfg.lambda_ortho * loss_ortho
        + cfg.lambda_artinv * loss_artinv
    )

    return {
        "total": total,
        "loss_cls": loss_cls.detach(),
        "loss_trait": loss_trait.detach(),
        "loss_adv": loss_adv.detach(),
        "loss_ortho": loss_ortho.detach(),
        "loss_artinv": loss_artinv.detach(),
        "alpha": torch.tensor(schedules.alpha),
        "gamma": torch.tensor(schedules.gamma),
    }
