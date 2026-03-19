from __future__ import annotations

"""
Task-aware-ready preprocessing pipeline.

// 补齐 classifier 侧真正需要的 node_de 与 artifact meta。
// 保持原有波形、邻接与 STFT 产物不变，确保可做 apples-to-apples 对比。
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

from configs.prep_config import CH_NAMES, FS, WINDOW_SEC
from engine_prep.extractors import compute_connectivity_matrix, compute_dfc_matrix, compute_stft_features
from engine_prep.transforms import fix_hardcoded_bads, intercept_gradient_spikes
from engine_prep.artifact import apply_windowed_artifact_rejection
from tools_audit.probe_1d2d import plot_all_channels_stft_grid, plot_all_channels_waveform_grid, set_pub_style


BANDS = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 14.0),
    ("beta", 14.0, 31.0),
    ("gamma", 31.0, 50.0),
)


def natural_sort_key(s: str):
    """Sort names with embedded integers."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def compute_band_de_from_stft(stft_feat: np.ndarray, fs: float = FS) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate STFT magnitude into five-band differential entropy-like node features."""
    if stft_feat.ndim != 3:
        raise ValueError(f"stft_feat 期望形状为 (C, F, T)，实际为 {stft_feat.shape}")
    n_chan, n_freq, n_time = stft_feat.shape
    freqs = np.fft.rfftfreq(int(fs * 2), d=1.0 / fs)
    if freqs.shape[0] != n_freq:
        raise ValueError(f"频率维不匹配: 预期 {freqs.shape[0]}，实际 {n_freq}")

    node_de = np.zeros((n_chan, len(BANDS), n_time), dtype=np.float32)
    for band_idx, (_, low, high) in enumerate(BANDS):
        band_mask = (freqs >= low) & (freqs < high)
        if not np.any(band_mask):
            continue
        band_energy = np.sum(np.square(stft_feat[:, band_mask, :]), axis=1)
        node_de[:, band_idx, :] = np.log(np.maximum(band_energy, 1e-8)).astype(np.float32)
    return node_de, np.array([name for name, _, _ in BANDS], dtype=object)


def summarize_artifact_info(artifact_info: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract classifier-facing artifact stats from trial-level aux dict."""
    latent_clean = np.asarray(artifact_info.get("latent_clean", np.empty((0, 0), dtype=np.float32)), dtype=np.float32)
    artifact_scores = np.asarray(artifact_info.get("artifact_score", np.empty((0, 0), dtype=np.float32)), dtype=np.float32)
    retain_probs = np.asarray(artifact_info.get("retain_prob", np.empty((0, 0), dtype=np.float32)), dtype=np.float32)
    return latent_clean, artifact_scores, retain_probs


def pipeline(
    input_dir: str = "Data/Preprocessed_EEG",
    output_dir: str = "Data/EEG_pure",
    qa_output_dir: str = "Data/QA_Reports",
    artifact_method: str = "qvae",
    weight_path: str | None = None,
    run_qa: bool = True,
) -> None:
    """Run preprocessing and export classifier-ready features."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(qa_output_dir, exist_ok=True)

    if run_qa:
        set_pub_style()
        print("[System] Visual QA Probe initialized in background mode (Agg).")

    all_mat_files = [f for f in os.listdir(input_dir) if f.endswith(".mat")]
    mat_files = [f for f in all_mat_files if "label" not in f.lower()]
    mat_files.sort(key=natural_sort_key)

    data_pure_subject = None
    pbar = tqdm(total=len(mat_files) * 15, desc="SEED Pipeline", dynamic_ncols=True)

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
            artifact_scores_subject = np.empty((3, 15), dtype=object)
            retain_probs_subject = np.empty((3, 15), dtype=object)
            qvae_latents_subject = np.empty((3, 15), dtype=object)
            artifact_method_subject = np.empty((3, 15), dtype=object)

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

            data_collect, artifact_info = apply_windowed_artifact_rejection(
                data_raw,
                sfreq=FS,
                window_sec=WINDOW_SEC,
                n_jobs=-1,
                method=artifact_method,
                global_mean=global_mean,
                global_std=global_std,
                weight_path=weight_path,
            )

            adj_matrix = compute_connectivity_matrix(data_collect, fs=FS)
            dfc_matrix = compute_dfc_matrix(data_collect, fs=FS, window_sec=4.0, step_sec=1.0)
            stft_feat = compute_stft_features(data_collect, fs=FS)
            node_de, band_names = compute_band_de_from_stft(stft_feat, fs=FS)
            latent_clean, artifact_scores, retain_probs = summarize_artifact_info(artifact_info)

            if run_qa:
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
            artifact_scores_subject[day - 1, trial_idx] = artifact_scores.astype(np.float32)
            retain_probs_subject[day - 1, trial_idx] = retain_probs.astype(np.float32)
            qvae_latents_subject[day - 1, trial_idx] = latent_clean.astype(np.float32)
            artifact_method_subject[day - 1, trial_idx] = np.array(str(artifact_method), dtype=object)

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
                "artifact_scores": artifact_scores_subject,
                "retain_probs": retain_probs_subject,
                "qvae_latents": qvae_latents_subject,
                "artifact_method": artifact_method_subject,
                "band_names": band_names,
                "sfreq": FS,
                "ch_names": CH_NAMES,
            }
            sio.savemat(save_path, mdict)
            tqdm.write(f"=== Saved S{subject:02d} classifier-ready hybrid features ===")

            del (
                data_pure_subject,
                adj_matrix_subject,
                dfc_matrix_subject,
                stft_features_subject,
                node_de_subject,
                artifact_scores_subject,
                retain_probs_subject,
                qvae_latents_subject,
                artifact_method_subject,
                mdict,
            )
            gc.collect()

    pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task-aware-ready SEED preprocessing pipeline")
    parser.add_argument("--input_dir", type=str, default="Data/Preprocessed_EEG")
    parser.add_argument("--output_dir", type=str, default="Data/EEG_pure")
    parser.add_argument("--qa_output_dir", type=str, default="Data/QA_Reports")
    parser.add_argument("--artifact_method", type=str, default="qvae", choices=["qvae", "ica"])
    parser.add_argument("--weight_path", type=str, default=None)
    parser.add_argument("--skip_qa", action="store_true")
    args = parser.parse_args()
    pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        qa_output_dir=args.qa_output_dir,
        artifact_method=args.artifact_method,
        weight_path=args.weight_path,
        run_qa=not args.skip_qa,
    )
