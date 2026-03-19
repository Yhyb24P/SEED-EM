from __future__ import annotations

"""
Task-aware-ready artifact separation.

// 对齐权重路径并缓存模型以避免重复加载。
// 输出 latent 与 artifact score，为后续 task-aware selector 提供接口。
"""

import os
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
from joblib import Parallel, delayed

from configs.prep_config import CH_NAMES
from engine_quantum.qvae_net import HAS_QVAE_DEPS

if HAS_QVAE_DEPS:
    import torch
    from engine_quantum.qvae_net import QuantumEEGDenoiser


_QVAE_MODEL_CACHE: Dict[Tuple[str, str], "QuantumEEGDenoiser"] = {}
_QVAE_LATENT_DIM = 6
_DEFAULT_QVAE_WEIGHT_CANDIDATES = (
    "data/04_weights/qvae_pretrained.pt",
    "weights/qvae_pretrained.pt",
)


def _resolve_weight_path(weight_path: Optional[str] = None) -> str:
    """Resolve pretrained weight path."""
    candidates = [weight_path] if weight_path else []
    candidates.extend(_DEFAULT_QVAE_WEIGHT_CANDIDATES)
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(
        "缺失 QVAE 预训练权重；已搜索: "
        + ", ".join([p for p in candidates if p])
    )


def _load_qvae_model(device: "torch.device", weight_path: Optional[str] = None) -> Tuple["QuantumEEGDenoiser", str]:
    """Load cached QVAE model."""
    resolved = _resolve_weight_path(weight_path)
    cache_key = (str(device), resolved)
    model = _QVAE_MODEL_CACHE.get(cache_key)
    if model is not None:
        return model, resolved

    model = QuantumEEGDenoiser(input_dim=62, hidden_dim=32, n_qubits=_QVAE_LATENT_DIM).to(device)
    try:
        state_dict = torch.load(resolved, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(resolved, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    _QVAE_MODEL_CACHE[cache_key] = model
    return model, resolved


def _normalize_segment(
    data_seg: np.ndarray,
    device: "torch.device",
    global_mean: Optional[np.ndarray],
    global_std: Optional[np.ndarray],
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Normalize trial segment with optional global stats."""
    x_raw = torch.tensor(data_seg.T, dtype=torch.float32, device=device)
    if global_mean is not None and global_std is not None:
        g_mean = torch.tensor(global_mean.T, dtype=torch.float32, device=device)
        g_std = torch.tensor(global_std.T, dtype=torch.float32, device=device)
    else:
        g_mean = x_raw.mean(dim=0, keepdim=True)
        g_std = x_raw.std(dim=0, keepdim=True)
    g_std = torch.clamp(g_std, min=1e-8)
    x_norm = (x_raw - g_mean) / g_std
    return x_raw, x_norm, g_mean, g_std


def _compute_proxy_artifact_scores(
    z_infer: "torch.Tensor",
    x_raw: "torch.Tensor",
    x_norm: "torch.Tensor",
    rho_threshold: float = 0.4,
    temperature: float = 12.0,
    min_retain: float = 0.15,
) -> Dict[str, np.ndarray]:
    """Compute soft artifact scores from frontal correlation proxy."""
    frontal_indices = [0, 1, 2, 3, 4]
    variances = torch.var(x_raw[:, frontal_indices], dim=0)
    valid_mask = (variances > 1e-2) & (variances < 5000)
    if not valid_mask.any():
        a_eog = torch.mean(x_norm, dim=1)
    else:
        valid_frontals = torch.tensor(frontal_indices, device=x_raw.device)[valid_mask]
        a_eog = torch.mean(x_norm[:, valid_frontals], dim=1)

    z_centered = z_infer - z_infer.mean(dim=0)
    a_centered = a_eog - a_eog.mean()
    denom = torch.sqrt(torch.sum(z_centered ** 2, dim=0)) * torch.sqrt(torch.sum(a_centered ** 2))
    denom = torch.clamp(denom, min=1e-8)
    rho = torch.sum(z_centered * a_centered.unsqueeze(1), dim=0) / denom

    artifact_score = torch.abs(rho)
    hard_mask = (artifact_score > rho_threshold).float()
    retain_prob = 1.0 - torch.sigmoid((artifact_score - rho_threshold) * temperature)
    retain_prob = torch.clamp(retain_prob, min=min_retain, max=1.0)

    return {
        "artifact_score": artifact_score.detach().cpu().numpy(),
        "retain_prob": retain_prob.detach().cpu().numpy(),
        "hard_mask": hard_mask.detach().cpu().numpy(),
    }


def _process_qvae_segment(
    data_seg: np.ndarray,
    global_mean: Optional[np.ndarray] = None,
    global_std: Optional[np.ndarray] = None,
    weight_path: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Run cached QVAE inference and export task-aware-ready aux signals."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_raw, x_norm, g_mean, g_std = _normalize_segment(data_seg, device, global_mean, global_std)
    model, resolved_weight_path = _load_qvae_model(device, weight_path)

    with torch.no_grad():
        chunk_size = 2000
        z_list = []
        for i in range(0, x_norm.size(0), chunk_size):
            chunk = x_norm[i : i + chunk_size]
            _, _, _, q_chunk = model(chunk)
            z_list.append(q_chunk)
        z_infer = torch.cat(z_list, dim=0)

        aux_scores = _compute_proxy_artifact_scores(z_infer=z_infer, x_raw=x_raw, x_norm=x_norm)
        retain_prob = torch.tensor(aux_scores["retain_prob"], dtype=torch.float32, device=device)
        z_clean = z_infer * retain_prob.unsqueeze(0)

        clean_list = []
        for i in range(0, z_clean.size(0), chunk_size):
            clean_chunk = model.decoder(z_clean[i : i + chunk_size])
            clean_list.append(clean_chunk)
        clean_x_norm = torch.cat(clean_list, dim=0)
        clean_x = clean_x_norm * g_std + g_mean

    aux = {
        "artifact_method": np.array("qvae_soft", dtype=object),
        "weight_path": np.array(resolved_weight_path, dtype=object),
        "latent_raw": z_infer.detach().cpu().numpy().T,
        "latent_clean": z_clean.detach().cpu().numpy().T,
        "artifact_score": aux_scores["artifact_score"],
        "retain_prob": aux_scores["retain_prob"],
        "hard_mask": aux_scores["hard_mask"],
    }
    return clean_x.detach().cpu().numpy().T, aux


def _process_ica_segment(data_seg: np.ndarray, sfreq: float) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Run ICA fallback and export empty aux arrays for a stable interface."""
    try:
        info = mne.create_info(ch_names=CH_NAMES, sfreq=sfreq, ch_types=["eeg"] * data_seg.shape[0])
        try:
            with info._unlock():
                info["highpass"] = 0.25
                info["lowpass"] = 50.0
        except AttributeError:
            info["highpass"] = 0.25
            info["lowpass"] = 50.0
        montage = mne.channels.make_standard_montage("standard_1020")

        ica = mne.preprocessing.ICA(n_components=20, method="picard", random_state=42, max_iter=2000, verbose=False)
        raw_seg = mne.io.RawArray(data_seg * 1e-6, info, verbose=False)
        raw_seg.set_montage(montage, on_missing="ignore")
        ica.fit(raw_seg, verbose=False)
        eog_indices, _ = ica.find_bads_eog(raw_seg, ch_name=["FP1", "FP2"], verbose=False)
        ica.exclude = eog_indices
        raw_pure = ica.apply(raw_seg.copy(), verbose=False)
        clean = raw_pure.get_data() * 1e6
    except ValueError:
        clean = data_seg

    aux = {
        "artifact_method": np.array("ica", dtype=object),
        "weight_path": np.array("", dtype=object),
        "latent_raw": np.empty((0, clean.shape[1]), dtype=np.float32),
        "latent_clean": np.empty((0, clean.shape[1]), dtype=np.float32),
        "artifact_score": np.empty((0,), dtype=np.float32),
        "retain_prob": np.empty((0,), dtype=np.float32),
        "hard_mask": np.empty((0,), dtype=np.float32),
    }
    return clean, aux


def _process_artifact_segment(
    data_seg: np.ndarray,
    sfreq: float,
    method: str = "qvae",
    global_mean: Optional[np.ndarray] = None,
    global_std: Optional[np.ndarray] = None,
    weight_path: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Route artifact processing by method."""
    if np.linalg.matrix_rank(data_seg) < 15:
        aux = {
            "artifact_method": np.array("pass_through", dtype=object),
            "weight_path": np.array("", dtype=object),
            "latent_raw": np.empty((0, data_seg.shape[1]), dtype=np.float32),
            "latent_clean": np.empty((0, data_seg.shape[1]), dtype=np.float32),
            "artifact_score": np.empty((0,), dtype=np.float32),
            "retain_prob": np.empty((0,), dtype=np.float32),
            "hard_mask": np.empty((0,), dtype=np.float32),
        }
        return data_seg, aux

    if method == "qvae" and HAS_QVAE_DEPS:
        return _process_qvae_segment(data_seg, global_mean, global_std, weight_path=weight_path)
    return _process_ica_segment(data_seg, sfreq)


def _stack_segment_info(segment_infos: List[Dict[str, np.ndarray]], n_samples: int) -> Dict[str, np.ndarray]:
    """Merge per-segment aux dictionaries into trial-level arrays."""
    latent_raw_blocks = []
    latent_clean_blocks = []
    artifact_scores = []
    retain_probs = []
    hard_masks = []
    methods = []
    weight_paths = []
    segment_bounds = []
    cursor = 0

    for info in segment_infos:
        latent_raw = info.get("latent_raw")
        latent_clean = info.get("latent_clean")
        if latent_raw is not None and latent_raw.size > 0:
            latent_raw_blocks.append(latent_raw)
        if latent_clean is not None and latent_clean.size > 0:
            latent_clean_blocks.append(latent_clean)
        artifact_scores.append(np.asarray(info.get("artifact_score", np.empty((0,), dtype=np.float32)), dtype=np.float32))
        retain_probs.append(np.asarray(info.get("retain_prob", np.empty((0,), dtype=np.float32)), dtype=np.float32))
        hard_masks.append(np.asarray(info.get("hard_mask", np.empty((0,), dtype=np.float32)), dtype=np.float32))
        methods.append(str(np.asarray(info.get("artifact_method", ""), dtype=object).item()))
        weight_paths.append(str(np.asarray(info.get("weight_path", ""), dtype=object).item()))
        seg_len = 0
        if latent_raw is not None and latent_raw.ndim == 2 and latent_raw.size > 0:
            seg_len = latent_raw.shape[1]
        elif latent_clean is not None and latent_clean.ndim == 2 and latent_clean.size > 0:
            seg_len = latent_clean.shape[1]
        segment_bounds.append((cursor, cursor + seg_len))
        cursor += seg_len

    latent_raw_full = np.concatenate(latent_raw_blocks, axis=1) if latent_raw_blocks else np.empty((0, n_samples), dtype=np.float32)
    latent_clean_full = np.concatenate(latent_clean_blocks, axis=1) if latent_clean_blocks else np.empty((0, n_samples), dtype=np.float32)

    max_k = max([arr.shape[0] for arr in artifact_scores if arr.ndim == 1] + [0])
    if max_k == 0:
        score_mat = np.empty((0, len(segment_infos)), dtype=np.float32)
        retain_mat = np.empty((0, len(segment_infos)), dtype=np.float32)
        mask_mat = np.empty((0, len(segment_infos)), dtype=np.float32)
    else:
        score_mat = np.full((max_k, len(segment_infos)), np.nan, dtype=np.float32)
        retain_mat = np.full((max_k, len(segment_infos)), np.nan, dtype=np.float32)
        mask_mat = np.full((max_k, len(segment_infos)), np.nan, dtype=np.float32)
        for idx, arr in enumerate(artifact_scores):
            if arr.size > 0:
                score_mat[: arr.shape[0], idx] = arr
        for idx, arr in enumerate(retain_probs):
            if arr.size > 0:
                retain_mat[: arr.shape[0], idx] = arr
        for idx, arr in enumerate(hard_masks):
            if arr.size > 0:
                mask_mat[: arr.shape[0], idx] = arr

    return {
        "artifact_method": np.array(methods, dtype=object),
        "weight_path": np.array(weight_paths, dtype=object),
        "latent_raw": latent_raw_full,
        "latent_clean": latent_clean_full,
        "artifact_score": score_mat,
        "retain_prob": retain_mat,
        "hard_mask": mask_mat,
        "segment_bounds": np.asarray(segment_bounds, dtype=np.int32),
    }


def apply_windowed_artifact_rejection(
    data: np.ndarray,
    sfreq: float = 200.0,
    window_sec: float = 40.0,
    n_jobs: int = -1,
    method: str = "ica",
    global_mean: Optional[np.ndarray] = None,
    global_std: Optional[np.ndarray] = None,
    weight_path: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Apply artifact rejection and export trial-level aux arrays."""
    n_chan, n_samples = data.shape
    window_len = int(sfreq * window_sec)

    if n_samples <= window_len:
        seg_data, seg_info = _process_artifact_segment(
            data, sfreq, method, global_mean, global_std, weight_path=weight_path
        )
        seg_info["segment_bounds"] = np.asarray([[0, seg_data.shape[1]]], dtype=np.int32)
        return seg_data, seg_info

    seg_num = int(np.ceil(n_samples / window_len))
    data_collect = np.zeros((n_chan, n_samples), dtype=np.float32)
    segments = []
    segment_bounds = []
    for seg in range(seg_num):
        start_idx = seg * window_len
        end_idx = min((seg + 1) * window_len, n_samples)
        segments.append(data[:, start_idx:end_idx])
        segment_bounds.append((start_idx, end_idx))

    safe_n_jobs = 1 if method == "qvae" else n_jobs
    processed_segments = Parallel(n_jobs=safe_n_jobs)(
        delayed(_process_artifact_segment)(
            seg_data,
            sfreq,
            method,
            global_mean,
            global_std,
            weight_path,
        )
        for seg_data in segments
    )

    infos: List[Dict[str, np.ndarray]] = []
    for (start_idx, end_idx), (seg_data, seg_info) in zip(segment_bounds, processed_segments):
        data_collect[:, start_idx:end_idx] = seg_data
        infos.append(seg_info)

    trial_info = _stack_segment_info(infos, n_samples=n_samples)
    trial_info["segment_bounds"] = np.asarray(segment_bounds, dtype=np.int32)
    return data_collect, trial_info
