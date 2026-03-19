"""
核心数据泵与特征装载引擎 (Dataloader)
负责优先读取 run_pipeline 导出的 node_de，并把 artifact meta 注入图样本。
"""
import os
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.io as sio
import torch
from torch_geometric.data import Data, Dataset


class SEEDGraphDataset(Dataset):
    def __init__(self, root_dir, subject_id, day_idx=0, mode='train', test_trial_idx=None):
        super(SEEDGraphDataset, self).__init__(root_dir)
        self.root_dir = root_dir
        self.subject_id = subject_id
        self.day_idx = day_idx
        self.mode = mode
        self.test_trial_idx = test_trial_idx

        self.band_indices = {
            'delta': (2, 8),
            'theta': (8, 16),
            'alpha': (16, 26),
            'beta': (26, 60),
            'gamma': (60, 90),
        }

        self.data_list = self._process_and_load()

    def _stft_to_de(self, stft_tensor: np.ndarray) -> np.ndarray:
        de_features = []
        for start, end in self.band_indices.values():
            band_energy = np.sum(np.square(stft_tensor[:, start:end, :]), axis=1)
            band_energy[band_energy <= 1e-8] = 1e-8
            de_features.append(np.log(band_energy))
        return np.stack(de_features, axis=1)

    def _resolve_mat_path(self) -> str:
        candidates = [
            os.path.join(self.root_dir, f'S{self.subject_id:02d}.mat'),
            os.path.join(self.root_dir, 'EEG_pure', f'S{self.subject_id:02d}.mat'),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f'未找到 S{self.subject_id:02d}.mat，已搜索: {candidates}')

    def _resolve_label_path(self) -> str:
        candidates = [
            os.path.join(self.root_dir, 'label.mat'),
            os.path.join(self.root_dir, '../01_raw_mat/label.mat'),
            os.path.join(self.root_dir, '../01_raw_mat/Preprocessed_EEG/label.mat'),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f'未找到 label.mat，已搜索: {candidates}')

    def _extract_day_matrix(self, mat: Dict[str, np.ndarray], key: str) -> Optional[np.ndarray]:
        arr = mat.get(key)
        if arr is None:
            return None
        return arr[self.day_idx]

    def _trim_warmup(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim != 3:
            return arr
        if arr.shape[2] > 20:
            return arr[:, :, 20:]
        return arr[:, :, 0:0]

    def _compute_subject_norm(self, day_feature_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        all_trials = []
        for i in range(15):
            if self.test_trial_idx is not None and self.test_trial_idx != -1 and i == self.test_trial_idx:
                continue
            feat = self._trim_warmup(day_feature_matrix[i])
            if feat.shape[2] > 0:
                all_trials.append(feat)
        if not all_trials:
            return 0.0, 1.0
        all_concat = np.concatenate(all_trials, axis=-1)
        subj_mean = np.mean(all_concat, axis=-1, keepdims=True)
        subj_std = np.std(all_concat, axis=-1, keepdims=True) + 1e-8
        return subj_mean, subj_std

    def _coerce_trial_feature(self, node_de_day, stft_day, trial_idx: int) -> np.ndarray:
        if node_de_day is not None:
            feat = self._trim_warmup(node_de_day[trial_idx])
            if feat.shape[2] > 0:
                return feat
        stft = self._trim_warmup(stft_day[trial_idx])
        if stft.shape[2] == 0:
            return stft[:, :5, :]
        return self._stft_to_de(stft)

    def _trial_artifact_meta(self, score_day, retain_day, trial_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        score = np.asarray(score_day[trial_idx], dtype=np.float32) if score_day is not None else np.empty((0,), dtype=np.float32)
        retain = np.asarray(retain_day[trial_idx], dtype=np.float32) if retain_day is not None else np.empty((0,), dtype=np.float32)

        if score.ndim == 2:
            score = np.nanmean(score, axis=1)
        if retain.ndim == 2:
            retain = np.nanmean(retain, axis=1)
        if score.ndim == 0:
            score = score.reshape(1)
        if retain.ndim == 0:
            retain = retain.reshape(1)
        return score.astype(np.float32), retain.astype(np.float32)

    def _process_and_load(self):
        mat_path = self._resolve_mat_path()
        label_path = self._resolve_label_path()

        data = sio.loadmat(mat_path)
        labels = sio.loadmat(label_path)['label'][0]

        node_de_day = self._extract_day_matrix(data, 'node_de')
        stft_day = self._extract_day_matrix(data, 'stft_features')
        adj_day = self._extract_day_matrix(data, 'adj_matrix')
        score_day = self._extract_day_matrix(data, 'artifact_scores')
        retain_day = self._extract_day_matrix(data, 'retain_probs')

        if adj_day is None:
            raise KeyError(f'{mat_path} 缺少 adj_matrix')
        if node_de_day is None and stft_day is None:
            raise KeyError(f'{mat_path} 同时缺少 node_de 与 stft_features')

        day_feature_matrix = node_de_day if node_de_day is not None else np.empty((15,), dtype=object)
        if node_de_day is None:
            for i in range(15):
                day_feature_matrix[i] = self._stft_to_de(self._trim_warmup(stft_day[i]))

        subj_mean, subj_std = self._compute_subject_norm(day_feature_matrix)
        graph_data_list = []

        for trial_idx in range(15):
            if self.test_trial_idx is not None and self.test_trial_idx != -1:
                if self.mode == 'train' and trial_idx == self.test_trial_idx:
                    continue
                if self.mode == 'test' and trial_idx != self.test_trial_idx:
                    continue

            de_feat = self._coerce_trial_feature(node_de_day, stft_day, trial_idx)
            if de_feat.shape[2] == 0:
                continue

            de_feat = (de_feat - subj_mean) / subj_std
            adj = np.asarray(adj_day[trial_idx], dtype=np.float32)
            adj = adj - np.eye(adj.shape[0], dtype=np.float32)
            edge_indices = np.where(np.abs(adj) > 0.3)
            edge_index = torch.tensor(np.array(edge_indices), dtype=torch.long)
            edge_weight = torch.tensor(np.abs(adj[edge_indices]), dtype=torch.float32)

            score_vec, retain_vec = self._trial_artifact_meta(score_day, retain_day, trial_idx)
            label = int(labels[trial_idx]) + 1

            T = 3
            n_seconds = de_feat.shape[2]
            for sec in range(0, n_seconds - T + 1):
                x_slices = [de_feat[:, :, sec + t] for t in range(T)]
                x_tensor = torch.tensor(np.concatenate(x_slices, axis=0), dtype=torch.float32)

                edge_indices_t = [edge_index + t * adj.shape[0] for t in range(T)]
                edge_index_t = torch.cat(edge_indices_t, dim=1)
                edge_weight_t = edge_weight.repeat(T)

                graph = Data(
                    x=x_tensor,
                    edge_index=edge_index_t,
                    edge_attr=edge_weight_t,
                    y=torch.tensor([label], dtype=torch.long),
                    subj_id=torch.tensor([self.subject_id - 1], dtype=torch.long),
                    time_len=torch.tensor([T], dtype=torch.long),
                    artifact_score=torch.from_numpy(score_vec).unsqueeze(0),
                    retain_prob=torch.from_numpy(retain_vec).unsqueeze(0),
                )
                graph_data_list.append(graph)

        return graph_data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
