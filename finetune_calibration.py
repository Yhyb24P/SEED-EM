"""
路线 B: 极小样本流形校准引擎 (Few-Shot Calibration)
与 LOSO 零样本训练环彻底解耦。纯净的 3-Trial 全域分层靶向自适应。
"""
import os
import random
import argparse
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader  # 恢复 PyG 动态批处理图语义
from sklearn.metrics import f1_score, accuracy_score
import yaml

from engine_gnn.graph_operators import EEG_DGCN
from configs.prep_config import FS

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CalibrationDataset(Dataset):
    """独立的靶向自适应数据泵，内置分层采样与严格防泄露归一化"""
    def __init__(self, data_dir, subject_id, mode='calib', calib_trials=None, subj_mean=None, subj_std=None):
        self.data_dir = data_dir
        self.subject_id = subject_id
        self.mode = mode
        
        # 强制特征抽取基准频段
        self.band_indices = {'delta': (2, 8), 'theta': (8, 16), 'alpha': (16, 26), 'beta': (26, 60), 'gamma': (60, 90)}
        
        # 1. 解析标签并执行分层采样
        label_path = os.path.join(data_dir, "../01_raw_mat/label.mat")
        if not os.path.exists(label_path):
            label_path = os.path.join(data_dir, "label.mat")
        self.labels = sio.loadmat(label_path)['label'][0] + 1  # 映射到 0, 1, 2
        
        # 2. 从 45 个 Trial 全域内执行绝对均匀的随机分层采样 (Stratified Sampling)
        if calib_trials is None:
            label_to_trials = {0: [], 1: [], 2: []}
            for day in range(3):
                for trial_idx in range(15):
                    lbl = int(self.labels[trial_idx])
                    label_to_trials[lbl].append((day, trial_idx))
            self.calib_trials = {lbl: random.choice(trials) for lbl, trials in label_to_trials.items()}
        else:
            self.calib_trials = calib_trials
            
        self.calib_set = set(self.calib_trials.values())
        
        # 3. 载入该被试的物理张量
        mat_path = os.path.join(data_dir, f"S{subject_id:02d}.mat")
        data = sio.loadmat(mat_path)
        self.node_de = data.get('node_de')
        self.stft = data.get('stft_features')
        self.adj = data['adj_matrix']
        
        # 4. 转导式无监督对齐 (Transductive Unsupervised Alignment)
        # 废除仅用 3-Trial 计算基线的短视行为。利用所有可见的无标签物理数据计算全局均值/方差
        # 恢复输入流形与源域预训练阶段 (45-Trial 级) 的尺度等价性。
        if mode == 'calib':
            all_feats = []
            for d in range(3):
                for t in range(15):
                    feat = self._get_raw_feat(d, t)
                    if feat is not None and feat.shape[-1] > 20:
                        all_feats.append(feat[:, :, 20:])
            concat_all = np.concatenate(all_feats, axis=-1)
            self.subj_mean = np.mean(concat_all, axis=-1, keepdims=True)
            self.subj_std = np.std(concat_all, axis=-1, keepdims=True) + 1e-8
        else:
            self.subj_mean = subj_mean
            self.subj_std = subj_std

        # 5. 构建非重叠切窗图样本
        self.data_list = self._build_graphs()

    def _unwrap_obj(self, arr, day, trial):
        """兼容 MATLAB 导出嵌套 Cell Array (3, 15) 与 (1, 3) 的异构变体"""
        if arr is None: return None
        day_arr = arr[day] if arr.shape[0] == 3 else arr[0, day]
        item = day_arr[trial]
        # 防止 NumPy 对象数组过度包装
        if isinstance(item, np.ndarray) and item.dtype == object and item.size == 1:
            return item[0]
        return item

    def _stft_to_de(self, stft_tensor):
        de_features = []
        for start, end in self.band_indices.values():
            band_energy = np.sum(np.square(stft_tensor[:, start:end, :]), axis=1)
            band_energy[band_energy <= 1e-8] = 1e-8
            de_features.append(np.log(band_energy))
        return np.stack(de_features, axis=1)

    def _get_raw_feat(self, day, trial):
        if self.node_de is not None:
            return self._unwrap_obj(self.node_de, day, trial)
        return self._stft_to_de(self._unwrap_obj(self.stft, day, trial))

    def _build_graphs(self):
        graphs = []
        T = 3
        for day in range(3):
            for trial in range(15):
                is_calib = (day, trial) in self.calib_set
                if (self.mode == 'calib' and not is_calib) or (self.mode == 'eval' and is_calib):
                    continue
                
                feat = self._get_raw_feat(day, trial)
                if feat.shape[-1] <= 20 + T:
                    continue
                
                # 截断前 20 帧，应用纯净基线归一化
                feat = feat[:, :, 20:]
                feat = (feat - self.subj_mean) / self.subj_std
                
                adj = np.asarray(self._unwrap_obj(self.adj, day, trial), dtype=np.float32)
                adj = adj - np.eye(adj.shape[0], dtype=np.float32)
                edge_indices = np.where(np.abs(adj) > 0.3)
                edge_index = torch.tensor(np.array(edge_indices), dtype=torch.long)
                edge_weight = torch.tensor(np.abs(adj[edge_indices]), dtype=torch.float32)
                label = self.labels[trial]
                
                # 非重叠切窗防止自相关过拟合
                n_seconds = feat.shape[2]
                for sec in range(0, n_seconds - T + 1, T):
                    x_slices = [feat[:, :, sec + t] for t in range(T)]
                    x_tensor = torch.tensor(np.concatenate(x_slices, axis=0), dtype=torch.float32)
                    
                    edge_indices_t = [edge_index + t * adj.shape[0] for t in range(T)]
                    edge_index_t = torch.cat(edge_indices_t, dim=1)
                    edge_weight_t = edge_weight.repeat(T)
                    
                    graphs.append(Data(
                        x=x_tensor, edge_index=edge_index_t, edge_attr=edge_weight_t,
                        y=torch.tensor([label], dtype=torch.long)
                    ))
        return graphs

    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx): return self.data_list[idx]


def load_proxy_model(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint.get('config_path', 'configs/train_config.yaml')
    with open(config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    num_nodes = int(cfg.get('eeg_semantics', {}).get('num_channels', 16))
    hidden_channels = int(cfg.get('model', {}).get('hidden_channels', 64))
    
    model = EEG_DGCN(in_channels=5, hidden_channels=hidden_channels, num_classes=3, num_nodes=num_nodes).to(device)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=True)
    return model

@torch.no_grad()
def evaluate_zeroshot(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, alpha=0.0)
        out_emo = out[0] if isinstance(out, tuple) else out
        preds = out_emo.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())
    return f1_score(all_labels, all_preds, average='weighted'), accuracy_score(all_labels, all_preds)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/02_pure_features")
    parser.add_argument("--ckpt_dir", type=str, default="data/05_checkpoints")
    parser.add_argument("--subject_id", type=int, required=True, help="Target outlier subject (e.g., 10)")
    parser.add_argument("--epochs", type=int, default=10, help="Fast calibration limits")
    parser.add_argument("--lr", type=float, default=1e-4) # 降级学习率防止极小样本瞬间塌缩
    args = parser.parse_args()
    
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ckpt_path = os.path.join(args.ckpt_dir, f"best_gcn_loso_S{args.subject_id:02d}.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"未找到 S{args.subject_id:02d} 的基准权重: {ckpt_path}")

    print(f"\n === 启动流形极小样本校准 (Few-Shot Calibration - Pure AdaLN) | Target: S{args.subject_id:02d} ===")
    
    # 1. 挂载隔离数据泵
    ds_calib = CalibrationDataset(args.data_dir, args.subject_id, mode='calib')
    ds_eval = CalibrationDataset(args.data_dir, args.subject_id, mode='eval', 
                                 calib_trials=ds_calib.calib_trials, 
                                 subj_mean=ds_calib.subj_mean, subj_std=ds_calib.subj_std)
    
    loader_calib = DataLoader(ds_calib, batch_size=32, shuffle=True)
    loader_eval = DataLoader(ds_eval, batch_size=64, shuffle=False)
    print(f"[*] 全域分层抽样锚点: {ds_calib.calib_trials} | 校准图: {len(ds_calib)} | 评估图: {len(ds_eval)}")

    # 2. 加载基准模型并执行零样本探针
    model = load_proxy_model(ckpt_path, device)
    f1_pre, acc_pre = evaluate_zeroshot(model, loader_eval, device)
    print(f"[-] 校准前 (Zero-Shot) -> F1: {f1_pre:.4f} | Acc: {acc_pre:.4f}")

    # 3. 彻底冻结物理图网络参数流形
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # 4. 执行特征空间非参数原型度量
    print("    └── 执行特征空间几何对齐 (Prototypical Alignment)")
    prototypes = {0: [], 1: [], 2: []}
    with torch.no_grad():
        for batch in loader_calib:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, alpha=0.0)
            # // 提取维度 3 的未投影特征张量
            # // 投影特征至单位超球面以抵消各向异性缩放
            z_emo = torch.nn.functional.normalize(out[3] if isinstance(out, tuple) and len(out) == 7 else out[0], p=2, dim=1)
            for i in range(batch.y.size(0)):
                lbl = batch.y[i].item()
                prototypes[lbl].append(z_emo[i])
        
        # // 聚合校准集原型中心
        C_k = {}
        for k in range(3):
            if not prototypes[k]:
                raise ValueError(f"缺失类别 {k} 的全域分层抽样锚点")
            # // 聚合原型中心并执行二次超球面投影
            C_k[k] = torch.nn.functional.normalize(torch.stack(prototypes[k]).mean(dim=0), p=2, dim=0)
            
        # // 显式计算并输出三类原型在超球面上的余弦相似度矩阵
        C_mat = torch.stack([C_k[0], C_k[1], C_k[2]])
        C_sim = torch.mm(C_mat, C_mat.T)
        print("    └── [诊断] 原型间余弦相似度矩阵 cos(C_i, C_j):")
        print(f"          | C_0 (Neg) | C_1 (Neu) | C_2 (Pos)")
        print(f"      C_0 |   {C_sim[0,0]:.4f}  |   {C_sim[0,1]:.4f}  |   {C_sim[0,2]:.4f}")
        print(f"      C_1 |   {C_sim[1,0]:.4f}  |   {C_sim[1,1]:.4f}  |   {C_sim[1,2]:.4f}")
        print(f"      C_2 |   {C_sim[2,0]:.4f}  |   {C_sim[2,1]:.4f}  |   {C_sim[2,2]:.4f}")
            
        # // 基于余弦相似度执行评估集推断
        all_preds, all_labels = [], []
        # // [新增] 用于统计分类置信度的累加器
        avg_max_sim = 0.0
        avg_margin = 0.0
        eval_count = 0
        for batch in loader_eval:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, alpha=0.0)
            # // 投影评估特征至单位超球面以对齐度量空间
            z_eval = torch.nn.functional.normalize(out[3] if isinstance(out, tuple) and len(out) == 7 else out[0], p=2, dim=1)
            sims = torch.stack([torch.nn.functional.cosine_similarity(z_eval, C_k[k].unsqueeze(0)) for k in range(3)], dim=1)
            preds = sims.argmax(dim=1)
            # // [新增] 记录最大相似度，反映特征空间聚类紧凑性
            max_sims = sims.max(dim=1).values
            avg_max_sim += max_sims.sum().item()
            
            # // 提取 Top-1 与 Top-2 相似度计算 Margin，量化决策边界拥挤度
            top2_sims, _ = torch.topk(sims, k=2, dim=1)
            avg_margin += (top2_sims[:, 0] - top2_sims[:, 1]).sum().item()
            eval_count += z_eval.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            
        f1_post = f1_score(all_labels, all_preds, average='weighted')
        acc_post = accuracy_score(all_labels, all_preds)
        # // [新增] 输出平均最大相似度，作为类内紧凑性的辅助指标
        print(f"    └── [调试信息] 平均最大余弦相似度 (置信度): {avg_max_sim / eval_count:.4f}")
        print(f"    └── Calib Finish | Eval F1: {f1_post:.4f} | Eval Acc: {acc_post:.4f}")
        print(f"    └── [诊断] 平均 Top-1/Top-2 Margin: {avg_margin / eval_count:.4f}")
        print(f"\n[+] 校准完成! S{args.subject_id:02d} F1-Score: {f1_pre:.4f} ──> Prototype F1: {f1_post:.4f}")

if __name__ == "__main__":
    main()