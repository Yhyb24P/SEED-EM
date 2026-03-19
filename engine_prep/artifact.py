from __future__ import annotations

"""
伪迹隔离与调度路由。

// 在保留现有 QVAE/ICA 主路径的同时，导出 Phase A 所需的 z_raw / z_clean / artifact_score / retain_prob。
// 维持 apply_windowed_artifact_rejection 的兼容接口，并为 Task-Aware Selector 暴露可学习先验。
"""

import os
from functools import lru_cache
from typing import Dict, Optional, Tuple

import mne
import numpy as np
from joblib import Parallel, delayed

try:
    from configs.prep_config import CH_NAMES, WINDOW_SEC
except ImportError:
    from prep_config import CH_NAMES, WINDOW_SEC

try:
    from engine_quantum.qvae_net import HAS_QVAE_DEPS, QuantumEEGDenoiser
except ImportError:
    from qvae_net import HAS_QVAE_DEPS, QuantumEEGDenoiser

if HAS_QVAE_DEPS:
    import torch


DEFAULT_QVAE_WEIGHT_CANDIDATES = (
    "data/04_weights/qvae_pretrained.pt",
    "weights/qvae_pretrained.pt",
    "qvae_pretrained.pt",
)


def _resolve_qvae_weight_path(weight_path: Optional[str] = None) -> str:
    candidates = [weight_path] if weight_path else []
    candidates.extend(DEFAULT_QVAE_WEIGHT_CANDIDATES)
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"QVAE pretrained weight not found, searched: {candidates}")


@lru_cache(maxsize=2)
def _load_qvae_model_cached(weight_path: str, device_name: str):
    if not HAS_QVAE_DEPS:
        raise ImportError("QVAE dependencies not available")
    device = torch.device(device_name)
    model = QuantumEEGDenoiser(input_dim=62, hidden_dim=32, n_qubits=6).to(device)
    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# 处理单个 QVAE 窗口并导出 Phase A 所需辅助量

def _process_qvae_segment(
    data_seg: np.ndarray,
    global_mean: Optional[np.ndarray] = None,
    global_std: Optional[np.ndarray] = None,
    weight_path: Optional[str] = None,
):
    if not HAS_QVAE_DEPS:
        raise ImportError("QVAE dependencies not available")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_weight = _resolve_qvae_weight_path(weight_path)
    model = _load_qvae_model_cached(resolved_weight, str(device))

    x_raw = torch.tensor(data_seg.T, dtype=torch.float32, device=device)
    if global_mean is not None and global_std is not None:
        g_mean = torch.tensor(global_mean.T, dtype=torch.float32, device=device)
        g_std = torch.tensor(global_std.T, dtype=torch.float32, device=device)
    else:
        g_mean = x_raw.mean(dim=0, keepdim=True)
        g_std = x_raw.std(dim=0, keepdim=True)
    g_std = torch.clamp(g_std, min=1e-8)
    x_norm = (x_raw - g_mean) / g_std

    with torch.no_grad():
        chunk_size = 2000
        z_raw_list = []
        for i in range(0, x_norm.size(0), chunk_size):
            _, _, _, q_chunk = model(x_norm[i : i + chunk_size])
            z_raw_list.append(q_chunk)
        z_raw = torch.cat(z_raw_list, dim=0)  # (T, K)

        frontal_indices = [0, 1, 2, 3, 4]
        variances = torch.var(x_raw[:, frontal_indices], dim=0)
        valid_mask = (variances > 1e-2) & (variances < 5000)
        if not valid_mask.any():
            a_eog = torch.mean(x_norm, dim=1)
        else:
            valid_frontals = torch.tensor(frontal_indices, device=device)[valid_mask]
            a_eog = torch.mean(x_norm[:, valid_frontals], dim=1)

        z_centered = z_raw - z_raw.mean(dim=0)
        a_centered = a_eog - a_eog.mean()
        denom = torch.sqrt(torch.sum(z_centered**2, dim=0)) * torch.sqrt(torch.sum(a_centered**2))
        denom = torch.clamp(denom, min=1e-8)
        rho = torch.sum(z_centered * a_centered.unsqueeze(1), dim=0) / denom  # (K,)

        artifact_score = torch.abs(rho)
        retain_prob = 1.0 - torch.clamp(artifact_score / 0.4, min=0.0, max=1.0)
        hard_mask = (artifact_score <= 0.4).float()
        z_clean = z_raw * retain_prob.unsqueeze(0)

        clean_norm_list = []
        for i in range(0, z_clean.size(0), chunk_size):
            clean_norm_list.append(model.decoder(z_clean[i : i + chunk_size]))
        clean_x_norm = torch.cat(clean_norm_list, dim=0)
        clean_x = clean_x_norm * g_std + g_mean

    t_len = z_raw.shape[0]
    aux = {
        "latent_raw": z_raw.detach().cpu().numpy().T,
        "latent_clean": z_clean.detach().cpu().numpy().T,
        "artifact_score": artifact_score.detach().cpu().numpy()[:, None].repeat(t_len, axis=1),
        "retain_prob": retain_prob.detach().cpu().numpy()[:, None].repeat(t_len, axis=1),
        "hard_mask": hard_mask.detach().cpu().numpy()[:, None].repeat(t_len, axis=1),
        "weight_path": resolved_weight,
        "artifact_method": "qvae",
    }
    return clean_x.detach().cpu().numpy().T, aux


# 处理单个窗口

def _process_artifact_segment(
    data_seg: np.ndarray,
    sfreq: float,
    method: str = "qvae",
    global_mean: Optional[np.ndarray] = None,
    global_std: Optional[np.ndarray] = None,
    weight_path: Optional[str] = None,
):
    if np.linalg.matrix_rank(data_seg) < 15:
        return data_seg, None

    if method == "qvae" and HAS_QVAE_DEPS:
        return _process_qvae_segment(data_seg, global_mean=global_mean, global_std=global_std, weight_path=weight_path)

    try:
        info = mne.create_info(ch_names=CH_NAMES, sfreq=sfreq, ch_types=["eeg"] * data_seg.shape[0])
        montage = mne.channels.make_standard_montage("standard_1020")
        ica = mne.preprocessing.ICA(n_components=20, method="picard", random_state=42, max_iter=2000, verbose=False)
        raw_seg = mne.io.RawArray(data_seg * 1e-6, info, verbose=False)
        raw_seg.set_montage(montage, on_missing="ignore")
        ica.fit(raw_seg, verbose=False)
        eog_indices, _ = ica.find_bads_eog(raw_seg, ch_name=["FP1", "FP2"], verbose=False)
        ica.exclude = eog_indices
        raw_pure = ica.apply(raw_seg.copy(), verbose=False)
        return raw_pure.get_data() * 1e6, {"artifact_method": "ica"}
    except ValueError:
        return data_seg, {"artifact_method": "ica_fallback"}


# 窗口级伪迹消除

def apply_windowed_artifact_rejection(
    data: np.ndarray,
    sfreq: float = 200.0,
    window_sec: float = 40.0,
    n_jobs: int = -1,
    method: str = "ica",
    global_mean: Optional[np.ndarray] = None,
    global_std: Optional[np.ndarray] = None,
    weight_path: Optional[str] = None,
):
    n_chan, n_samples = data.shape
    window_len = int(sfreq * window_sec)

    if n_samples <= window_len:
        seg_data, seg_aux = _process_artifact_segment(
            data,
            sfreq,
            method=method,
            global_mean=global_mean,
            global_std=global_std,
            weight_path=weight_path,
        )
        return seg_data, seg_aux

    seg_num = int(np.ceil(n_samples / window_len))
    data_collect = np.zeros((n_chan, n_samples), dtype=np.float32)
    aux_collect: Dict[str, np.ndarray] | None = None

    segments = []
    for seg in range(seg_num):
        start_idx = seg * window_len
        end_idx = min((seg + 1) * window_len, n_samples)
        segments.append((start_idx, end_idx, data[:, start_idx:end_idx]))

    safe_n_jobs = 1 if method == "qvae" else n_jobs
    processed_segments = Parallel(n_jobs=safe_n_jobs)(
        delayed(_process_artifact_segment)(
            seg_data,
            sfreq,
            method=method,
            global_mean=global_mean,
            global_std=global_std,
            weight_path=weight_path,
        )
        for _, _, seg_data in segments
    )

    for (start_idx, end_idx, _), (seg_data, seg_aux) in zip(segments, processed_segments):
        data_collect[:, start_idx:end_idx] = seg_data
        if seg_aux is None:
            continue
        if aux_collect is None and method == "qvae":
            aux_collect = {
                "latent_raw": np.zeros((6, n_samples), dtype=np.float32),
                "latent_clean": np.zeros((6, n_samples), dtype=np.float32),
                "artifact_score": np.zeros((6, n_samples), dtype=np.float32),
                "retain_prob": np.zeros((6, n_samples), dtype=np.float32),
                "hard_mask": np.zeros((6, n_samples), dtype=np.float32),
                "artifact_method": np.array([["qvae"]], dtype=object),
            }
        if aux_collect is not None and method == "qvae":
            seg_len = end_idx - start_idx
            for key in ["latent_raw", "latent_clean", "artifact_score", "retain_prob", "hard_mask"]:
                aux_collect[key][:, start_idx:end_idx] = seg_aux[key][:, :seg_len]

    return data_collect, aux_collect
