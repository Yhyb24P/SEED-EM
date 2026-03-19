"""
SEED GCN 训练主循环 (Training Loop Engine)
对齐 node_de / artifact_scores / retain_probs，并把 artifact-aware 正则并入真实训练入口。
"""
import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score

try:
    from engine_gnn.graph_operators import EEG_GCN, EEG_DGCN
except ImportError:
    from graph_operators import EEG_GCN, EEG_DGCN

try:
    from engine_gnn.cv_router import get_loso_loaders, get_loto_loaders
except ImportError:
    from cv_router import get_loso_loaders, get_loto_loaders


def resolve_config_path(path: str | None = None) -> str:
    candidates = []
    if path:
        candidates.append(path)
    candidates.extend([
        'configs/train_config.yaml',
        'train_config.yaml',
    ])
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f'未找到训练配置，已搜索: {candidates}')


def load_config(path: str | None = None) -> Dict:
    resolved = resolve_config_path(path)
    with open(resolved, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['_config_path'] = resolved
    return cfg


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.15) -> torch.Tensor:
    feats = F.normalize(features, dim=1)
    sim = torch.mm(feats, feats.T) / temperature
    eye = torch.eye(labels.size(0), device=labels.device)
    pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float() - eye
    sim_max = sim.max(dim=1, keepdim=True)[0].detach()
    log_prob = (sim - sim_max) - torch.log((torch.exp(sim - sim_max) * (1 - eye)).sum(dim=1, keepdim=True) + 1e-8)
    n_pos = pos_mask.sum(dim=1).clamp(min=1)
    return (-(pos_mask * log_prob).sum(dim=1) / n_pos).mean()


def compute_schedules(epoch: int, total_epochs: int) -> Tuple[float, float]:
    progress = float(epoch - 1) / float(max(total_epochs, 1))
    anneal_scalar = 2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0
    gamma_max = np.log(3.0) / np.log(15.0)
    alpha = 0.3 * anneal_scalar
    gamma = gamma_max * anneal_scalar
    return alpha, gamma


def masked_offdiag_cosine(a_emo: torch.Tensor, a_subj: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(a_emo.size(-1), device=a_emo.device).unsqueeze(0)
    a_emo_off = a_emo * (1 - eye)
    a_subj_off = a_subj * (1 - eye)
    num = torch.sum(a_emo_off * a_subj_off, dim=(1, 2))
    den = torch.norm(a_emo_off, p='fro', dim=(1, 2)) * torch.norm(a_subj_off, p='fro', dim=(1, 2)) + 1e-8
    return (num / den).mean()


def artifact_invariance_loss(z_emo: torch.Tensor, artifact_score: torch.Tensor | None) -> torch.Tensor:
    if artifact_score is None or artifact_score.numel() == 0:
        return z_emo.new_tensor(0.0)
    score = artifact_score.float()
    if score.dim() == 1:
        score = score.unsqueeze(1)
    score = torch.nan_to_num(score, nan=0.0)
    score = score.mean(dim=1, keepdim=True)
    score = (score - score.mean(dim=0, keepdim=True)) / (score.std(dim=0, keepdim=True) + 1e-8)
    z_norm = (z_emo - z_emo.mean(dim=0, keepdim=True)) / (z_emo.std(dim=0, keepdim=True) + 1e-8)
    return torch.mean(torch.abs(torch.mean(z_norm * score, dim=0)))


def unpack_outputs(model, batch, alpha: float):
    if isinstance(model, EEG_DGCN):
        return model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, alpha=alpha)
    return model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)


def compute_batch_loss(model, batch, criterion_cls, criterion_dom, epoch: int, total_epochs: int, lambda_artinv: float):
    alpha, gamma = compute_schedules(epoch, total_epochs)
    out = unpack_outputs(model, batch, alpha=alpha)

    if isinstance(out, tuple) and len(out) == 7:
        out_emo, out_trait, out_adv, z_emo, z_subj, a_emo, a_subj = out
        loss_emo = criterion_cls(out_emo, batch.y)
        loss_trait = criterion_dom(out_trait, batch.subj_id)
        loss_adv = criterion_dom(out_adv, batch.subj_id)
        loss_ortho = masked_offdiag_cosine(a_emo, a_subj)
        loss_supcon = supcon_loss(z_emo, batch.y)
        art_score = getattr(batch, 'artifact_score', None)
        loss_artinv = artifact_invariance_loss(z_emo, art_score)
        total = loss_emo + gamma * loss_trait + alpha * loss_adv + 0.1 * loss_ortho + 0.5 * loss_supcon + lambda_artinv * loss_artinv
        out_cls = out_emo
        stats = {
            'loss_emo': float(loss_emo.detach().cpu()),
            'loss_trait': float(loss_trait.detach().cpu()),
            'loss_adv': float(loss_adv.detach().cpu()),
            'loss_ortho': float(loss_ortho.detach().cpu()),
            'loss_supcon': float(loss_supcon.detach().cpu()),
            'loss_artinv': float(loss_artinv.detach().cpu()),
            'alpha': float(alpha),
            'gamma': float(gamma),
        }
        return total, out_cls, stats

    if isinstance(out, tuple) and len(out) == 5:
        out_emo, out_trait, out_adv, z_emo, _ = out
        loss_emo = criterion_cls(out_emo, batch.y)
        loss_trait = criterion_dom(out_trait, batch.subj_id)
        loss_adv = criterion_dom(out_adv, batch.subj_id)
        loss_artinv = artifact_invariance_loss(z_emo, getattr(batch, 'artifact_score', None))
        total = loss_emo + gamma * loss_trait + alpha * loss_adv + lambda_artinv * loss_artinv
        stats = {
            'loss_emo': float(loss_emo.detach().cpu()),
            'loss_trait': float(loss_trait.detach().cpu()),
            'loss_adv': float(loss_adv.detach().cpu()),
            'loss_ortho': 0.0,
            'loss_supcon': 0.0,
            'loss_artinv': float(loss_artinv.detach().cpu()),
            'alpha': float(alpha),
            'gamma': float(gamma),
        }
        return total, out_emo, stats

    out_cls = out if not isinstance(out, tuple) else out[0]
    loss = criterion_cls(out_cls, batch.y)
    return loss, out_cls, {
        'loss_emo': float(loss.detach().cpu()),
        'loss_trait': 0.0,
        'loss_adv': 0.0,
        'loss_ortho': 0.0,
        'loss_supcon': 0.0,
        'loss_artinv': 0.0,
        'alpha': float(alpha),
        'gamma': float(gamma),
    }


def train_epoch(model, loader, optimizer, criterion_cls, criterion_dom, device, epoch, total_epochs, lambda_artinv: float = 0.0):
    model.train()
    total_loss = 0.0
    stats_acc = None
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        loss, out_cls, stats = compute_batch_loss(model, batch, criterion_cls, criterion_dom, epoch, total_epochs, lambda_artinv)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        stats_acc = stats if stats_acc is None else {k: stats_acc[k] + stats[k] for k in stats_acc}

        preds = out_cls.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(batch.y.detach().cpu().numpy())

    epoch_loss = total_loss / max(len(loader.dataset), 1)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted') if all_labels else 0.0
    if stats_acc is None:
        stats_acc = {k: 0.0 for k in ['loss_emo', 'loss_trait', 'loss_adv', 'loss_ortho', 'loss_supcon', 'loss_artinv', 'alpha', 'gamma']}
    stats_mean = {k: v / max(len(loader), 1) for k, v in stats_acc.items()}
    return epoch_loss, epoch_f1, stats_mean


@torch.no_grad()
def evaluate_epoch(model, loader, criterion_cls, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        out = unpack_outputs(model, batch, alpha=1.0)
        out_cls = out[0] if isinstance(out, tuple) else out
        loss = criterion_cls(out_cls, batch.y)
        total_loss += loss.item() * batch.num_graphs
        preds = out_cls.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(batch.y.detach().cpu().numpy())

    epoch_loss = total_loss / max(len(loader.dataset), 1)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted') if all_labels else 0.0
    return epoch_loss, epoch_f1


def build_model(cfg: Dict, device: torch.device):
    model_type = cfg['model']['type']
    hidden_channels = int(cfg['model'].get('hidden_channels', 64))
    if model_type == 'EEG_DGCN':
        model = EEG_DGCN(in_channels=5, hidden_channels=hidden_channels, num_classes=3).to(device)
    else:
        model = EEG_GCN(in_channels=5, hidden_channels=hidden_channels, num_classes=3).to(device)
    return model


def compute_class_weights(train_loader, device: torch.device) -> torch.Tensor:
    labels = []
    for data in train_loader.dataset:
        labels.append(int(data.y.item()))
    class_counts = np.bincount(labels, minlength=3)
    smoothed_counts = class_counts + np.max(class_counts) * 0.05
    class_weights = 1.0 / smoothed_counts
    class_weights = class_weights / class_weights.sum() * 3.0
    print(f'标签分布: {class_counts}, 惩罚权重: {class_weights}')
    return torch.tensor(class_weights, dtype=torch.float32, device=device)


def main():
    parser = argparse.ArgumentParser(description='SEED classifier training with task-aware regularization')
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = cfg['paths']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    lambda_artinv = float(cfg.get('task_aware', {}).get('lambda_artinv', 0.2))
    seed_everything(int(cfg.get('seed', 42)))

    fold_summaries = []

    for test_subject_id in range(1, 16):
        print("\n=======================================================")
        print(f'=== 初始化 LOSO 物理隔离数据流 (Target: S{test_subject_id:02d}) ===')
        print('=======================================================')

        train_loader, val_loader, test_loader = get_loso_loaders(
            data_dir=cfg['paths']['data_dir'],
            test_subject_id=test_subject_id,
            batch_size=cfg['train']['batch_size'],
        )

        model = build_model(cfg, device)
        weight_tensor = compute_class_weights(train_loader, device)
        criterion_cls = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.1)
        criterion_dom = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg['train']['learning_rate'],
            weight_decay=cfg['train']['weight_decay'],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['train']['epochs'])

        best_val_f1 = -1.0
        patience_counter = 0
        best_path = os.path.join(checkpoint_dir, f'best_gcn_loso_S{test_subject_id:02d}.pth')

        for epoch in range(1, cfg['train']['epochs'] + 1):
            train_loss, train_f1, train_stats = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion_cls,
                criterion_dom,
                device,
                epoch,
                cfg['train']['epochs'],
                lambda_artinv=lambda_artinv,
            )
            val_loss, val_f1 = evaluate_epoch(model, val_loader, criterion_cls, device)
            scheduler.step()

            print(
                f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | "
                f"Val Loss: {val_loss:.4f} F1: {val_f1:.4f} | "
                f"ArtInv: {train_stats['loss_artinv']:.4f} alpha={train_stats['alpha']:.3f} gamma={train_stats['gamma']:.3f}"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'subject_id': test_subject_id,
                    'best_val_f1': best_val_f1,
                    'config_path': cfg['_config_path'],
                }, best_path)
                print('保存新的高置信度权重')
            else:
                patience_counter += 1
                if patience_counter >= cfg['train']['early_stopping_patience']:
                    print(f'=== Early Stopping Triggered at Epoch {epoch} ===')
                    break

        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        test_loss, test_f1 = evaluate_epoch(model, test_loader, criterion_cls, device)
        print(f'>>> Test | Loss: {test_loss:.4f} F1: {test_f1:.4f}')
        fold_summaries.append({
            'subject_id': test_subject_id,
            'best_val_f1': float(best_val_f1),
            'test_f1': float(test_f1),
            'checkpoint': best_path,
        })

        del model, optimizer, train_loader, val_loader, test_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = Path(checkpoint_dir) / 'loso_summary.json'
    summary_path.write_text(json.dumps({
        'config_path': cfg['_config_path'],
        'lambda_artinv': lambda_artinv,
        'folds': fold_summaries,
        'mean_test_f1': float(np.mean([x['test_f1'] for x in fold_summaries])) if fold_summaries else 0.0,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'已写入 LOSO 汇总: {summary_path}')


if __name__ == '__main__':
    main()
