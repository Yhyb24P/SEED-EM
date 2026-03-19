from __future__ import annotations

"""
Phase A 协同微调引擎。

// 读取落盘的 qvae_latents_raw / artifact_scores / adj_matrix / label，构建可微 Selector -> Decoder -> STFT/DE -> GNN Proxy 闭环。
// 对齐现有 EEG_DGCN / EEG_GCN 的 x / edge_index / edge_attr / batch 输入契约。
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset, DataLoader

from engine_gnn.classifier_taskaware_adapter import (
    ComponentSelector,
    DifferentiableSTFTAndDE,
    TemporalGraphSpec,
    batch_graphs,
    build_temporal_graphs_from_de,
    compute_selector_loss,
)
from engine_gnn.graph_operators import EEG_DGCN, EEG_GCN
from engine_quantum.qvae_net import QuantumEEGDenoiser, HAS_QVAE_DEPS


def seed_everything(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_yaml(path: str | None) -> Dict:
    candidates = [p for p in [path, "train_config.yaml", "configs/train_config.yaml"] if p]
    for candidate in candidates:
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"missing config, searched: {candidates}")


@dataclass
class TrialSample:
    z_raw: torch.Tensor
    art_prior: torch.Tensor
    adj: torch.Tensor
    label: torch.Tensor
    subj_id: torch.Tensor


class PhaseASelectorDataset(Dataset):
    def __init__(self, root_dir: str, subject_ids: Optional[Sequence[int]] = None):
        self.root_dir = root_dir
        self.subject_ids = set(subject_ids) if subject_ids is not None else None
        self.samples = self._load_samples()

    @staticmethod
    def _unwrap_object(arr, i, j):
        return np.asarray(arr[i, j])

    def _load_samples(self) -> List[TrialSample]:
        label_candidates = [
            os.path.join(self.root_dir, "label.mat"),
            os.path.join(self.root_dir, "../01_raw_mat/label.mat"),
            os.path.join(self.root_dir, "../01_raw_mat/Preprocessed_EEG/label.mat"),
        ]
        label_path = next((p for p in label_candidates if os.path.exists(p)), None)
        if label_path is None:
            raise FileNotFoundError(f"label.mat not found in candidates: {label_candidates}")
        labels = sio.loadmat(label_path)["label"][0]

        mat_files = sorted([p for p in Path(self.root_dir).glob("S*.mat")])
        samples: List[TrialSample] = []
        for mat_path in mat_files:
            subject_id = int(mat_path.stem[1:])
            if self.subject_ids is not None and subject_id not in self.subject_ids:
                continue
            data = sio.loadmat(str(mat_path))
            if "qvae_latents_raw" not in data:
                raise KeyError(f"{mat_path.name} missing qvae_latents_raw")
            qvae_raw = data["qvae_latents_raw"]
            art_scores = data.get("artifact_scores")
            adj_matrix = data["adj_matrix"]
            if art_scores is None:
                raise KeyError(f"{mat_path.name} missing artifact_scores")

            for day_idx in range(qvae_raw.shape[0]):
                for trial_idx in range(qvae_raw.shape[1]):
                    z_raw = self._unwrap_object(qvae_raw, day_idx, trial_idx)
                    adj = self._unwrap_object(adj_matrix, day_idx, trial_idx)
                    art_prior = self._unwrap_object(art_scores, day_idx, trial_idx)
                    if z_raw.size == 0 or adj.size == 0 or art_prior.size == 0:
                        continue
                    label = int(labels[trial_idx])
                    samples.append(
                        TrialSample(
                            z_raw=torch.tensor(z_raw, dtype=torch.float32),
                            art_prior=torch.tensor(art_prior, dtype=torch.float32),
                            adj=torch.tensor(adj, dtype=torch.float32),
                            label=torch.tensor(label, dtype=torch.long),
                            subj_id=torch.tensor(subject_id - 1, dtype=torch.long),
                        )
                    )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> TrialSample:
        return self.samples[idx]


def collate_selector_trials(batch: Sequence[TrialSample]) -> Dict[str, torch.Tensor]:
    max_len = max(item.z_raw.shape[-1] for item in batch)
    n_comp = batch[0].z_raw.shape[0]
    z_raw = torch.zeros(len(batch), n_comp, max_len, dtype=torch.float32)
    art_prior = torch.zeros(len(batch), n_comp, max_len, dtype=torch.float32)
    adj = torch.stack([item.adj for item in batch], dim=0)
    y_emo = torch.stack([item.label for item in batch], dim=0)
    subj_id = torch.stack([item.subj_id for item in batch], dim=0)
    lengths = torch.tensor([item.z_raw.shape[-1] for item in batch], dtype=torch.long)

    for i, item in enumerate(batch):
        tlen = item.z_raw.shape[-1]
        z_raw[i, :, :tlen] = item.z_raw
        if item.art_prior.dim() == 1:
            art_prior[i, :, :tlen] = item.art_prior.unsqueeze(-1)
        elif item.art_prior.dim() == 2:
            if item.art_prior.shape[-1] == tlen:
                art_prior[i, :, :tlen] = item.art_prior
            else:
                art_prior[i, :, :tlen] = item.art_prior.mean(dim=-1, keepdim=True)
        else:
            raise ValueError(f"unsupported art_prior shape: {tuple(item.art_prior.shape)}")

    return {
        "z_raw": z_raw,
        "art_prior": art_prior,
        "adj_matrix": adj,
        "y_emo": y_emo,
        "subj_id": subj_id,
        "lengths": lengths,
    }


def load_qvae_decoder(weight_path: str, device: torch.device, input_dim: int = 62, hidden_dim: int = 32, n_qubits: int = 6):
    if not HAS_QVAE_DEPS:
        raise ImportError("PyTorch/PennyLane dependencies missing for QVAE")
    model = QuantumEEGDenoiser(input_dim=input_dim, hidden_dim=hidden_dim, n_qubits=n_qubits).to(device)
    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def load_proxy_model(cfg: Dict, checkpoint_path: str, device: torch.device):
    model_type = cfg.get("model", {}).get("type", "EEG_DGCN")
    hidden_channels = cfg.get("model", {}).get("hidden_channels", 64)
    num_classes = cfg.get("eeg_semantics", {}).get("num_classes", 3)
    if model_type == "EEG_DGCN":
        model = EEG_DGCN(in_channels=5, hidden_channels=hidden_channels, num_classes=num_classes).to(device)
    else:
        model = EEG_GCN(in_channels=5, hidden_channels=hidden_channels, num_classes=num_classes).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    return model


def forward_proxy(proxy_model: torch.nn.Module, graph_batch, alpha: float = 1.0):
    if isinstance(proxy_model, EEG_DGCN):
        out = proxy_model(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr, graph_batch.batch, alpha=alpha)
    else:
        out = proxy_model(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr, graph_batch.batch)
    return out[0] if isinstance(out, tuple) else out


def _build_graphs_for_batch(node_de: torch.Tensor, adj_matrix: torch.Tensor, y_emo: torch.Tensor, subj_id: torch.Tensor, spec: TemporalGraphSpec, device: torch.device):
    graphs = []
    graph_labels = []
    for i in range(node_de.size(0)):
        trial_graphs = build_temporal_graphs_from_de(node_de[i], adj_matrix[i], y_emo[i], subj_id[i], spec=spec)
        graphs.extend(trial_graphs)
        if trial_graphs:
            graph_labels.extend([int(y_emo[i].item())] * len(trial_graphs))
    batch_graph = batch_graphs(graphs, device=device)
    labels = torch.tensor(graph_labels, dtype=torch.long, device=device) if graph_labels else None
    return batch_graph, labels


def _trim_decoded(x_hat: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    max_len = x_hat.size(-1)
    mask = torch.arange(max_len, device=x_hat.device).unsqueeze(0) < lengths.unsqueeze(1)
    return x_hat * mask.unsqueeze(1)


def train_collaborative_selector(
    qvae_decoder: QuantumEEGDenoiser,
    proxy_model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epochs_a2: int = 10,
    epochs_a3: int = 20,
    lambda_sparse: float = 0.1,
    lambda_smooth: float = 0.05,
    lambda_anticheat: float = 0.5,
    graph_spec: TemporalGraphSpec | None = None,
):
    graph_spec = graph_spec or TemporalGraphSpec()
    selector = ComponentSelector(in_channels=1, n_components=6).to(device)
    diff_stft = DifferentiableSTFTAndDE(fs=200.0, window_sec=2.0, hop_sec=1.0).to(device)

    for param in proxy_model.parameters():
        param.requires_grad = False
    proxy_model.eval()

    optimizer_sel = torch.optim.AdamW(selector.parameters(), lr=1e-3, weight_decay=1e-4)
    history = {"a2": [], "a3": []}

    for epoch in range(epochs_a2):
        selector.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in dataloader:
            z_raw = batch["z_raw"].to(device)
            art_prior = batch["art_prior"].to(device)
            y_emo = batch["y_emo"].to(device)
            adj_matrix = batch["adj_matrix"].to(device)
            subj_id = batch["subj_id"].to(device)
            lengths = batch["lengths"].to(device)

            optimizer_sel.zero_grad()
            p_mask = selector(z_raw)
            z_weighted = z_raw * p_mask
            x_pure_hat = qvae_decoder.decoder(z_weighted.transpose(1, 2)).transpose(1, 2)
            x_pure_hat = _trim_decoded(x_pure_hat, lengths)
            node_de = diff_stft(x_pure_hat)

            # 需修改接口签名，注入物理时序长度边界
            valid_bins = lengths // int(diff_stft.fs * 1.0) 
            graph_batch, graph_labels = _build_graphs_for_batch(node_de, adj_matrix, y_emo, subj_id, graph_spec, device, valid_bins)
            
            if graph_batch is None or graph_labels is None or graph_labels.numel() == 0:
                continue
            y_pred = forward_proxy(proxy_model, graph_batch)
            loss_task = F.cross_entropy(y_pred, graph_labels)
            loss_reg = compute_selector_loss(
                p_mask,
                art_prior,
                lambda_sparse=lambda_sparse,
                lambda_smooth=lambda_smooth,
                lambda_anticheat=lambda_anticheat,
            )
            loss_total = loss_task + loss_reg
            loss_total.backward()
            optimizer_sel.step()
            epoch_loss += float(loss_total.detach().cpu())
            n_batches += 1
        history["a2"].append(epoch_loss / max(n_batches, 1))

    for name, param in proxy_model.named_parameters():
        if "classifier" in name or "fc" in name:
            param.requires_grad = True
    proxy_model.train()
    optimizer_joint = torch.optim.AdamW([
        {"params": selector.parameters(), "lr": 1e-4},
        {"params": [p for p in proxy_model.parameters() if p.requires_grad], "lr": 1e-5},
    ])

    for epoch in range(epochs_a3):
        epoch_loss = 0.0
        n_batches = 0
        for batch in dataloader:
            z_raw = batch["z_raw"].to(device)
            art_prior = batch["art_prior"].to(device)
            y_emo = batch["y_emo"].to(device)
            adj_matrix = batch["adj_matrix"].to(device)
            subj_id = batch["subj_id"].to(device)
            lengths = batch["lengths"].to(device)

            optimizer_joint.zero_grad()
            p_mask = selector(z_raw)
            x_pure_hat = qvae_decoder.decoder((z_raw * p_mask).transpose(1, 2)).transpose(1, 2)
            x_pure_hat = _trim_decoded(x_pure_hat, lengths)
            node_de = diff_stft(x_pure_hat)
            graph_batch, graph_labels = _build_graphs_for_batch(node_de, adj_matrix, y_emo, subj_id, graph_spec, device)
            if graph_batch is None or graph_labels is None or graph_labels.numel() == 0:
                continue
            y_pred = forward_proxy(proxy_model, graph_batch, alpha=1.0)
            loss_total = F.cross_entropy(y_pred, graph_labels) + compute_selector_loss(
                p_mask,
                art_prior,
                lambda_sparse=lambda_sparse,
                lambda_smooth=lambda_smooth,
                lambda_anticheat=lambda_anticheat,
            )
            loss_total.backward()
            optimizer_joint.step()
            epoch_loss += float(loss_total.detach().cpu())
            n_batches += 1
        history["a3"].append(epoch_loss / max(n_batches, 1))

    return selector, history


def main():
    parser = argparse.ArgumentParser(description="Phase A collaborative selector training")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--proxy_ckpt", type=str, required=True)
    parser.add_argument("--qvae_ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, default="selector_phase_a.pt")
    parser.add_argument("--subjects", type=str, default=None, help="comma separated subject ids")
    args = parser.parse_args()

    cfg = _resolve_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.get("seed", 42))

    data_dir = args.data_dir or cfg.get("paths", {}).get("data_dir", "data/02_pure_features")
    subject_ids = None if not args.subjects else [int(x) for x in args.subjects.split(",") if x.strip()]
    dataset = PhaseASelectorDataset(data_dir, subject_ids=subject_ids)
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("phase_a", {}).get("batch_size", 4),
        shuffle=True,
        collate_fn=collate_selector_trials,
    )

    qvae = load_qvae_decoder(args.qvae_ckpt, device)
    proxy = load_proxy_model(cfg, args.proxy_ckpt, device)

    selector, history = train_collaborative_selector(
        qvae,
        proxy,
        loader,
        device,
        epochs_a2=cfg.get("phase_a", {}).get("epochs_a2", 10),
        epochs_a3=cfg.get("phase_a", {}).get("epochs_a3", 20),
        lambda_sparse=cfg.get("phase_a", {}).get("lambda_sparse", 0.1),
        lambda_smooth=cfg.get("phase_a", {}).get("lambda_smooth", 0.05),
        lambda_anticheat=cfg.get("phase_a", {}).get("lambda_anticheat", 0.5),
        graph_spec=TemporalGraphSpec(
            temporal_window=cfg.get("phase_a", {}).get("temporal_window", 3),
            edge_threshold=cfg.get("phase_a", {}).get("edge_threshold", 0.3),
            drop_initial_bins=cfg.get("phase_a", {}).get("drop_initial_bins", 20),
        ),
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"selector_state_dict": selector.state_dict(), "history": history}, out_path)
    with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"[+] selector saved to {out_path}")


if __name__ == "__main__":
    main()
