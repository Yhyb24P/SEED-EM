"""
生理映射与特征归因审计器 (Audit Engine)
新增流形特征域聚类探测器 (UMAP Latent Space Projection)，联合 Saliency Topography 实现双重印证。
"""

import yaml
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import networkx as nx
import mne
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


try:
    import umap.umap_ as umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    warnings.warn("未检测到 umap-learn，将跳过隐空间降维投影渲染。执行 `pip install umap-learn` 激活此探针。")

from engine_gnn.graph_operators import EEG_GCN, EEG_DGCN
from engine_gnn.cv_router import get_loso_loaders

# 核心 16 导联拓扑流形顺序
CH_NAMES = [
    'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 
    'C3', 'CZ', 'C4', 'T8', 'PZ', 'O1', 'OZ', 'O2'
]

def load_config(path='configs/train_config.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_channel_positions():
    info = mne.create_info(ch_names=CH_NAMES, sfreq=200., ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage, on_missing='ignore')
    
    pos_dict = {}
    for idx, ch in enumerate(CH_NAMES):
        if ch in montage.ch_names:
            ch_idx = montage.ch_names.index(ch)
            pos_dict[idx] = montage.dig[ch_idx + 3]['r'][:2] 
        else:
            pos_dict[idx] = np.array([-0.05 if '1' in ch else 0.05, -0.1])
    return pos_dict

def evaluate_loso_fold(subject_id, cfg, device):
    checkpoint_path = os.path.join(cfg['paths']['checkpoint_dir'], f'best_gcn_loso_S{subject_id:02d}.pth')
    if not os.path.exists(checkpoint_path):
        return None, None, None, None

    # // 对齐嵌套交叉验证的三元组返回值签名字段
    _, _, test_loader = get_loso_loaders(
        data_dir=cfg['paths']['data_dir'],
        test_subject_id=subject_id, 
        
        batch_size=cfg['train']['batch_size']
    )

    model_type = cfg['model']['type']
    num_nodes = int(cfg['eeg_semantics'].get('num_channels', 16))
    if model_type == 'EEG_DGCN':
        model = EEG_DGCN(in_channels=5, hidden_channels=cfg['model']['hidden_channels'], num_classes=3, num_nodes=num_nodes)
    else:
        model = EEG_GCN(in_channels=5, hidden_channels=cfg['model']['hidden_channels'], num_classes=3)
        
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        print(f"\n[-] 严重张量异构拦截: S{subject_id:02d} 落盘权重与当前物理不匹配！")
        return None, None, None, None
        
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_features = []
    
    accumulated_adj = np.zeros((16, 16))
    batch_count = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            # // 解耦多态架构的正向推理与隐空间特征拾取
            if model_type == 'EEG_DGCN':
                out, _, _, features, _, A_emo, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, alpha=1.0)
                all_features.append(features.cpu().numpy())
                accumulated_adj += A_emo.mean(dim=0).cpu().numpy()
                batch_count += 1
            else:
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    avg_adj = accumulated_adj / max(batch_count, 1) if model_type == 'EEG_DGCN' else None
    features_cat = np.concatenate(all_features, axis=0) if all_features else None
    
    return np.array(all_labels), np.array(all_preds), avg_adj, features_cat

def plot_confusion_matrix(y_true, y_pred, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title('Global LOSO Confusion Matrix')
    plt.ylabel('True Emotion State')
    plt.xlabel('Predicted Emotion State')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Confusion_Matrix.png'), dpi=300)
    plt.close()

def plot_saliency_topography(global_adj, output_dir):
    if global_adj is None:
        return
        
    pos = get_channel_positions()
    G = nx.Graph()
    
    for i in range(16):
        G.add_node(i)

    # 取上界高激活能量边
    threshold = np.percentile(global_adj.flatten(), 98) 
    for i in range(16):
        for j in range(i+1, 16):
            if global_adj[i, j] > threshold:
                G.add_edge(i, j, weight=global_adj[i, j])

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=150, node_color='skyblue', alpha=0.8)
    nx.draw_networkx_labels(G, pos, labels={i: CH_NAMES[i] for i in range(16)}, font_size=8)
    
    edges = G.edges(data=True)
    if edges:
        weights = [data['weight'] for u, v, data in edges]
        max_weight = max(weights)
        norm_weights = [w / max_weight * 5.0 for w in weights]
        nx.draw_networkx_edges(G, pos, edge_color=weights, edge_cmap=plt.cm.Reds, 
                               width=norm_weights, alpha=0.7)

    plt.title('Dynamic GCN Saliency Topography (Top 2% Directed Routing Edges)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Saliency_Topography.png'), dpi=300)
    plt.close()

def plot_umap_projection(features, labels, subjects, output_dir):
    if not HAS_UMAP or features is None:
        return
    
    print("[*] 正在执行隐空间拓扑单纯复形投影 (UMAP Projection)...")
    reducer = umap.UMAP(n_components=2, metric='cosine', random_state=42)
    embedding = reducer.fit_transform(features)
    
    # 渲染图1：检验情感类别的流形可分性
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis', s=10, alpha=0.6)
    plt.title('UMAP Latent Space: Emotion Semantic Clustering')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    cbar = plt.colorbar(scatter)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(['Negative', 'Neutral', 'Positive'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'UMAP_Emotion_Cluster.png'), dpi=300)
    plt.close()

    # 渲染图2：检验跨被试局部仿射不变性 (Covariate Shift Alignment)
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=subjects, palette='tab20', s=15, alpha=0.7, legend='full')
    plt.title('UMAP Latent Space: Subject Domain Alignment')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Subject ID")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'UMAP_Subject_Domain.png'), dpi=300)
    plt.close()


def main():
    cfg = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'data/03_qa_reports/Evaluation'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=======================================================")
    print("=== 初始化全景物理映射与特征归因审计 (Audit Engine) ===")
    print("=======================================================")
    
    global_y_true = []
    global_y_pred = []
    global_features = []
    global_subjects = []
    
    global_dyn_adj_sum = np.zeros((16, 16))
    valid_subjects = 0

    for subj in range(1, 16):
        print(f"[*] 正在审计目标被试 S{subj:02d} 的泛化张量...")
        y_true, y_pred, avg_adj, feats = evaluate_loso_fold(subj, cfg, device)
        
        if y_true is not None:
            global_y_true.extend(y_true)
            global_y_pred.extend(y_pred)
            
            if feats is not None:
                global_features.append(feats)
                global_subjects.extend([subj] * len(y_true))
            
            fold_acc = accuracy_score(y_true, y_pred)
            fold_f1 = f1_score(y_true, y_pred, average='weighted')
            print(f"    └── S{subj:02d} 独立验证 | Acc: {fold_acc:.4f} | F1: {fold_f1:.4f}")
            
            if avg_adj is not None:
                global_dyn_adj_sum += avg_adj
                valid_subjects += 1

    if len(global_y_true) == 0:
        print("[-] 严重异常: 未能在 checkpoints/ 目录找到任何权重文件。")
        return

    global_acc = accuracy_score(global_y_true, global_y_pred)
    global_f1 = f1_score(global_y_true, global_y_pred, average='weighted')
    
    print("\n=======================================================")
    print(f"[+] 全局 LOSO (15-Fold) 物理隔离泛化性能:")
    print(f"    └── Accuracy : {global_acc:.4f}")
    print(f"    └── F1-Score : {global_f1:.4f}")
    
    plot_confusion_matrix(global_y_true, global_y_pred, output_dir)
    print(f"[+] 混淆矩阵已落盘 -> {output_dir}/Confusion_Matrix.png")
    
    if valid_subjects > 0:
        final_global_adj = global_dyn_adj_sum / valid_subjects
        plot_saliency_topography(final_global_adj, output_dir)
        print(f"[+] 显著性拓扑脑图已落盘 -> {output_dir}/Saliency_Topography.png")
        
    if HAS_UMAP and global_features:
        cat_features = np.concatenate(global_features, axis=0)
        cat_subjects = np.array(global_subjects)
        plot_umap_projection(cat_features, global_y_true, cat_subjects, output_dir)
        print(f"[+] UMAP 隐空间降维图谱已落盘 -> {output_dir}/UMAP_Emotion_Cluster.png & UMAP_Subject_Domain.png")
        print("    (审计提示: 检查不同被试的散点是否在代数空间均匀混叠，判断 Covariate Shift 抑制效果)")

if __name__ == '__main__':
    main()