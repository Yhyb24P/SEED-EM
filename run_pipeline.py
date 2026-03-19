from __future__ import annotations

"""
主运行入口。

// 保留 SciPy/MNE 版高效落盘路径用于常规评测。
// 额外导出 node_de / qvae_latents_raw / qvae_latents_clean / artifact_scores / retain_probs，闭合 Phase A 数据契约。
"""

import argparse
import gc
import os
import re
from typing import Dict, Tuple

import mne
import numpy as np
import scipy.io as sio
from tqdm import tqdm

try:
    from configs.prep_config import CH_NAMES, FS, WINDOW_SEC
except ImportError:
    from prep_config import CH_NAMES, FS, WINDOW_SEC

try:
    from engine_prep.transforms import fix_hardcoded_bads, intercept_gradient_spikes
except ImportError:
    from transforms import fix_hardcoded_bads, intercept_gradient_spikes

try:
    from engine_prep.extractors import compute_connectivity_matrix, compute_dfc_matrix, compute_stft_features
except ImportError:
    # 当前平铺目录未上传原 extractors.py 时，只允许导入失败后提示
    raise ImportError("compute_connectivity_matrix/compute_dfc_matrix/compute_stft_features not found")

try:
    from engine_prep.artifact import apply_windowed_artifact_rejection
except ImportError:
    from artifact import apply_windowed_artifact_rejection

try:
    from tools_audit.probe_1d2d import set_pub_style, plot_all_channels_waveform_grid, plot_all_channels_stft_grid
except ImportError:
    set_pub_style = None
    plot_all_channels_waveform_grid = None
    plot_all_channels_stft_grid = None


BAND_SLICES = {
    "delta": (2, 8),
    "theta": (8, 16),
    "alpha": (16, 26),
    "beta": (26, 60),
    "gamma": (60, 90),
}


def natural_sort_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


# 从 STFT 映射到五频带 DE

def stft_to_node_de(stft_feat: np.ndarray) -> np.ndarray:
    de_features = []
    for _, (start, end) in BAND_SLICES.items():
        band_energy = np.sum(np.square(stft_feat[:, start:end, :]), axis=1)
        band_energy[band_energy <= 1e-8] = 1e-8
        de_features.append(np.log(band_energy))
    return np.stack(de_features, axis=1).astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="SEED preprocessing pipeline")
    parser.add_argument("--input_dir", type=str, default="Data/Preprocessed_EEG")
    parser.add_argument("--output_dir", type=str, default="Data/EEG_pure")
    parser.add_argument("--qa_output_dir", type=str, default="Data/QA_Reports")
    parser.add_argument("--artifact_method", type=str, default="qvae", choices=["qvae", "ica"])
    parser.add_argument("--qvae_weight", type=str, default=None)
    parser.add_argument("--skip_qa", action="store_true")
    return parser.parse_args()


def pipeline():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    qa_output_dir = args.qa_output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(qa_output_dir, exist_ok=True)

    if set_pub_style is not None:
        set_pub_style()

    all_mat_files = [f for f in os.listdir(input_dir) if f.endswith(".mat")]
    mat_files = [f for f in all_mat_files if "label" not in f.lower()]
    mat_files.sort(key=natural_sort_key)

    pbar = tqdm(total=len(mat_files) * 15, desc="SEED Pipeline", dynamic_ncols=True)
    data_pure_subject = None

    for turn_idx, filename in enumerate(mat_files):
        turn = turn_idx + 1
        subject = int(np.ceil(turn / 3.0))
        day = turn - (subject - 1) * 3
        save_name = f"S{subject:02d}.mat"
        save_path = os.path.join(output_dir, save_name)

        if os.path.exists(save_path):
            pbar.update(15)
            continue

        day_qa_dir = os.path.join(qa_output_dir, f"S{subject:02d}", f"Day{day}")
        os.makedirs(day_qa_dir, exist_ok=True)

        if day == 1:
            data_pure_subject = np.empty((3, 15), dtype=object)
            adj_matrix_subject = np.empty((3, 15), dtype=object)
            dfc_matrix_subject = np.empty((3, 15), dtype=object)
            stft_features_subject = np.empty((3, 15), dtype=object)
            node_de_subject = np.empty((3, 15), dtype=object)
            qvae_latents_subject = np.empty((3, 15), dtype=object)
            qvae_latents_raw_subject = np.empty((3, 15), dtype=object)
            artifact_scores_subject = np.empty((3, 15), dtype=object)
            retain_probs_subject = np.empty((3, 15), dtype=object)
            hard_masks_subject = np.empty((3, 15), dtype=object)

        file_path = os.path.join(input_dir, filename)
        mat_data = sio.loadmat(file_path)
        trial_keys = [key for key in mat_data.keys() if not key.startswith("__") and isinstance(mat_data[key], np.ndarray)]
        trial_keys.sort(key=natural_sort_key)

        for trial_idx, trial_key in enumerate(trial_keys):
            pbar.set_description(f"Processing -> S{subject:02d} | Day {day} | Trial {trial_idx + 1:02d}")
            data_raw = mat_data[trial_key].copy()
            if data_raw.shape[0] > 62:
                data_raw = data_raw[:62, :]
            raw_snapshot = data_raw.copy()

            data_raw = fix_hardcoded_bads(data_raw, turn, trial_idx + 1)
            data_raw = data_raw - np.mean(data_raw, axis=1, keepdims=True)
            data_raw = mne.filter.filter_data(data_raw, sfreq=FS, l_freq=1.0, h_freq=50.0, method="fir", phase="zero", verbose=False)
            data_raw = mne.filter.notch_filter(data_raw, Fs=FS, freqs=np.array([50.0]), method="fir", phase="zero", verbose=False)

            chan_stds = np.std(data_raw, axis=1)
            valid_mask = (chan_stds > 1e-4) & (chan_stds < 100.0)
            if np.any(valid_mask):
                global_ref = np.mean(data_raw[valid_mask, :], axis=0)
                data_raw[valid_mask, :] -= global_ref

            data_raw = intercept_gradient_spikes(data_raw, grad_threshold=50.0, check_step=2)
            global_mean = np.mean(data_raw, axis=1, keepdims=True)
            global_std = np.std(data_raw, axis=1, keepdims=True)
            global_std[global_std == 0] = 1e-8

            data_collect, artifact_aux = apply_windowed_artifact_rejection(
                data_raw,
                sfreq=FS,
                window_sec=WINDOW_SEC,
                n_jobs=-1,
                method=args.artifact_method,
                global_mean=global_mean,
                global_std=global_std,
                weight_path=args.qvae_weight,
            )

            adj_matrix = compute_connectivity_matrix(data_collect, fs=FS)
            dfc_matrix = compute_dfc_matrix(data_collect, fs=FS, window_sec=4.0, step_sec=1.0)
            stft_feat = compute_stft_features(data_collect, fs=FS)
            node_de = stft_to_node_de(stft_feat)

            if not args.skip_qa and plot_all_channels_waveform_grid is not None and plot_all_channels_stft_grid is not None:
                metadata = {"subject": subject, "day": day, "trial": trial_idx + 1}
                pdf_path_1d = os.path.join(day_qa_dir, f"Trial_{trial_idx + 1:02d}_1D_Waveform.pdf")
                plot_all_channels_waveform_grid(raw_snapshot, data_collect, metadata, pdf_path_1d, window_sec=10)
                pdf_path_2d = os.path.join(day_qa_dir, f"Trial_{trial_idx + 1:02d}_2D_STFT.pdf")
                plot_all_channels_stft_grid(stft_feat, raw_snapshot, data_collect, metadata, pdf_path_2d, window_sec=10)

            data_pure_subject[day - 1, trial_idx] = data_collect.astype(np.float32)
            adj_matrix_subject[day - 1, trial_idx] = adj_matrix.astype(np.float32)
            dfc_matrix_subject[day - 1, trial_idx] = dfc_matrix.astype(np.float32)
            stft_features_subject[day - 1, trial_idx] = stft_feat.astype(np.float32)
            node_de_subject[day - 1, trial_idx] = node_de.astype(np.float32)

            if artifact_aux is not None and args.artifact_method == "qvae":
                qvae_latents_subject[day - 1, trial_idx] = artifact_aux["latent_clean"].astype(np.float32)
                qvae_latents_raw_subject[day - 1, trial_idx] = artifact_aux["latent_raw"].astype(np.float32)
                artifact_scores_subject[day - 1, trial_idx] = artifact_aux["artifact_score"].astype(np.float32)
                retain_probs_subject[day - 1, trial_idx] = artifact_aux["retain_prob"].astype(np.float32)
                hard_masks_subject[day - 1, trial_idx] = artifact_aux["hard_mask"].astype(np.float32)
            else:
                qvae_latents_subject[day - 1, trial_idx] = np.zeros((6, data_collect.shape[1]), dtype=np.float32)
                qvae_latents_raw_subject[day - 1, trial_idx] = np.zeros((6, data_collect.shape[1]), dtype=np.float32)
                artifact_scores_subject[day - 1, trial_idx] = np.zeros((6, data_collect.shape[1]), dtype=np.float32)
                retain_probs_subject[day - 1, trial_idx] = np.ones((6, data_collect.shape[1]), dtype=np.float32)
                hard_masks_subject[day - 1, trial_idx] = np.ones((6, data_collect.shape[1]), dtype=np.float32)

            del data_raw, raw_snapshot
            gc.collect()
            pbar.update(1)

        if day == 3:
            mdict = {
                "data_pure": data_pure_subject,
                "adj_matrix": adj_matrix_subject,
                "dfc_matrix": dfc_matrix_subject,
                "stft_features": stft_features_subject,
                "node_de": node_de_subject,
                "qvae_latents": qvae_latents_subject,
                "qvae_latents_clean": qvae_latents_subject,
                "qvae_latents_raw": qvae_latents_raw_subject,
                "artifact_scores": artifact_scores_subject,
                "retain_probs": retain_probs_subject,
                "hard_masks": hard_masks_subject,
                "artifact_method": np.array([[args.artifact_method]], dtype=object),
                "band_names": np.array([["delta", "theta", "alpha", "beta", "gamma"]], dtype=object),
                "sfreq": FS,
                "ch_names": CH_NAMES,
            }
            sio.savemat(save_path, mdict)
            tqdm.write(f"=== Saved S{subject:02d} with Phase-A ready tensors ===")
            del data_pure_subject, adj_matrix_subject, dfc_matrix_subject, stft_features_subject, node_de_subject
            del qvae_latents_subject, qvae_latents_raw_subject, artifact_scores_subject, retain_probs_subject, hard_masks_subject, mdict
            gc.collect()

    pbar.close()


if __name__ == "__main__":
    pipeline()
