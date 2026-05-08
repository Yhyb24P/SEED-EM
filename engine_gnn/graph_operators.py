"""
图神经网络拓扑算子 (Graph Models)
引入文献 GC-VASE 的 Split-Latent Space 机制与 GNN Survey 的 Subspace Edge。
重构为 Dual-Stream Disentangled Graph Convolution，彻底从代数层面切断跨域污染。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GraphNorm

class GradReverse(torch.autograd.Function):
    """
    梯度反转算子 (Gradient Reversal Layer)
    [数学修正] 仅执行雅可比矩阵反转，将 alpha 标量退火的控制权上放至全局 Loss，
    消除 \alpha^2 重复缩放，确保分类器与判别器博弈处于同一量级。
    """
    @staticmethod
    def forward(ctx, x, alpha):
        # // 挂载退火因子至上下文流形维持计算图连贯性
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None
    
class BandRegionAttention(nn.Module):
    def __init__(self, num_nodes=16, num_bands=5):
        super().__init__()
        init_att = torch.zeros(num_nodes, num_bands)
        FRONTAL = [0, 1, 2, 3, 4, 5, 6]
        TEMPORAL = [7, 11]
        init_att[TEMPORAL, 1] = 1.0
        init_att[FRONTAL, 2] = 1.0
        self.attention = nn.Parameter(init_att)

    def forward(self, x_dense):
        # // 施加 Sigmoid 激活并广播频带空间权重
        att = torch.sigmoid(self.attention)
        return x_dense * att.unsqueeze(0)    

class EEG_GCN(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=32, num_classes=3):
        super(EEG_GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.norm1 = GraphNorm(hidden_channels)
        
        self.conv2 = GCNConv(hidden_channels, hidden_channels * 2)
        self.norm2 = GraphNorm(hidden_channels * 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),            
            
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_channels, num_classes)
        )

    def forward(self, x, edge_index, edge_weight, batch_index, alpha=1.0, return_features=False):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.norm1(x, batch_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        
        x = self.conv2(x, edge_index, edge_weight)
        x = self.norm2(x, batch_index)
        x = F.relu(x)
        
        x = global_mean_pool(x, batch_index)
        out = self.classifier(x)
        return out


class EEG_DGCN(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=32, num_classes=3, num_nodes=16):
        super(EEG_DGCN, self).__init__()
        self.num_nodes = num_nodes
        self.d_k = 16
        
        # // 挂载神经先验频带脑区注意力算子
        self.band_attn = BandRegionAttention(num_nodes=num_nodes, num_bands=in_channels)
        
        # === [Early Node Split: 早期节点正交基底生成] ===
        self.P_emo = nn.Linear(in_channels, hidden_channels)
        self.P_subj = nn.Linear(in_channels, hidden_channels)
        
        # === [双流独立拓扑生成器 (Dual-Stream Topology)] ===
        # 引入独立的全局先验参数，为动态网络提供宏观基底缓冲
        self.W_q_E = nn.Linear(hidden_channels, self.d_k)
        self.W_k_E = nn.Linear(hidden_channels, self.d_k)
        self.global_adj_logits_E = nn.Parameter(torch.randn(num_nodes, num_nodes))
        
        self.W_q_S = nn.Linear(hidden_channels, self.d_k)
        self.W_k_S = nn.Linear(hidden_channels, self.d_k)
        self.global_adj_logits_S = nn.Parameter(torch.randn(num_nodes, num_nodes))
        
        # === [双流独立通信块] ===
        # 使用 LayerNorm 阻断拼接数据集带来的全局动量 (running_mean/var) 污染
        # -- 情感流 (Emotion Stream) --
        self.lin1_E = nn.Linear(hidden_channels, hidden_channels)
        self.norm1_E = nn.LayerNorm(hidden_channels)
        self.lin2_E = nn.Linear(hidden_channels, hidden_channels * 2)
        self.norm2_E = nn.LayerNorm(hidden_channels * 2)
        
        # -- 物理特征流 (Subject Trait Stream) --
        self.lin1_S = nn.Linear(hidden_channels, hidden_channels)
        self.norm1_S = nn.LayerNorm(hidden_channels)
        self.lin2_S = nn.Linear(hidden_channels, hidden_channels * 2)
        self.norm2_S = nn.LayerNorm(hidden_channels * 2)
        
        # === [空间正交双射解耦头 (Orthogonal Bijection Heads)] ===
        self.classifier_emo = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_channels, num_classes)
        )
        
        self.classifier_trait = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_channels, 15)
        )
        
        self.classifier_adv = nn.Sequential(
            # // 引入 CDAN 条件特征维度拓展支持联合分布对齐
            nn.Linear(hidden_channels * 2 * num_classes, hidden_channels),            
            
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_channels, 15)
        )
        # // 挂载隐空间时序状态机捕获渐进性情感流形
        self.temporal_gru_E = nn.GRU(hidden_channels * 2, hidden_channels * 2, num_layers=1, batch_first=True)
        self.temporal_gru_S = nn.GRU(hidden_channels * 2, hidden_channels * 2, num_layers=1, batch_first=True)


    def forward(self, x, edge_index, edge_weight, batch_index, alpha=1.0, return_features=False):
        N_batch = x.size(0) // self.num_nodes
        x_dense = x.view(N_batch, self.num_nodes, -1)

        # // 执行先验特征加权提纯输入张量
        x_dense = self.band_attn(x_dense)        
        # === 1. 子空间正交投射 ===
        x_emo_init = self.P_emo(x_dense)   # (B, N, H)
        x_subj_init = self.P_subj(x_dense) # (B, N, H)
        
        # === 2. 独立隐图因果推断 ===
        # // 注入常数级对角线自环先验保障特征不坍缩
        I_prior = torch.eye(self.num_nodes, device=x.device).unsqueeze(0) * 0.5
        
        Q_E = self.W_q_E(x_emo_init)
        K_E = self.W_k_E(x_emo_init)
        A_attn_E = torch.bmm(Q_E, K_E.transpose(1, 2)) / (self.d_k ** 0.5)
        A_logits_E = A_attn_E + self.global_adj_logits_E.unsqueeze(0) + I_prior
        A_emo = torch.softmax(A_logits_E, dim=-1)
        
        Q_S = self.W_q_S(x_subj_init)
        K_S = self.W_k_S(x_subj_init)
        A_attn_S = torch.bmm(Q_S, K_S.transpose(1, 2)) / (self.d_k ** 0.5)
        A_logits_S = A_attn_S + self.global_adj_logits_S.unsqueeze(0) + I_prior
        A_subj = torch.softmax(A_logits_S, dim=-1)
        
        # === 3. 物理隔离的消息传递 ===
        # -- 情感流 --
        h1_E = self.lin1_E(x_emo_init)
        h1_E = torch.bmm(A_emo, h1_E)
        # // 增加恒等映射截断狄利克雷能量衰减
        h1_E = F.relu(self.norm1_E(h1_E)) + x_emo_init        
        h1_E = F.dropout(h1_E, p=0.2, training=self.training)
        
        h2_E = self.lin2_E(h1_E)
        h2_E = torch.bmm(A_emo, h2_E)
        h2_E = F.relu(self.norm2_E(h2_E)) 
        
        # -- 物理流 --
        h1_S = self.lin1_S(x_subj_init)
        h1_S = torch.bmm(A_subj, h1_S)
        h1_S = F.relu(self.norm1_S(h1_S)) + x_subj_init
        h1_S = F.dropout(h1_S, p=0.2, training=self.training)
        
        h2_S = self.lin2_S(h1_S)
        h2_S = torch.bmm(A_subj, h2_S)
        h2_S = F.relu(self.norm2_S(h2_S)) 
        
        # === 4. 时序空间重塑与 GRU 编码 ===
        # // 截取节点图索引张量极值推导真实物理 Batch Size，反推时序窗口 T_len 约束张量折叠
        B_real = batch_index.max().item() + 1
        T_len = N_batch // B_real
        
        z_emo_step = h2_E.mean(dim=1).view(B_real, T_len, -1)
        _, h_n_E = self.temporal_gru_E(z_emo_step)
        z_emo = h_n_E.squeeze(0)
        
        z_subj_step = h2_S.mean(dim=1).view(B_real, T_len, -1)
        _, h_n_S = self.temporal_gru_S(z_subj_step)
        z_subj = h_n_S.squeeze(0)
        
        # === 5. 正交头路由 ===
        out_emo = self.classifier_emo(z_emo)
        out_trait = self.classifier_trait(z_subj)
        
        # // 构造 CDAN 条件域特征防止跨类别语义混叠
        p_soft = F.softmax(out_emo.detach(), dim=-1)
        z_cdan = torch.bmm(z_emo.unsqueeze(2), p_soft.unsqueeze(1)).view(z_emo.size(0), -1)
        
        # // 对条件特征应用梯度反转算子
        z_cdan_rev = GradReverse.apply(z_cdan, alpha)
        out_adv = self.classifier_adv(z_cdan_rev)
        
        # 返回双流拉普拉斯矩阵供拓扑正交损失使用
        return out_emo, out_trait, out_adv, z_emo, z_subj, A_emo, A_subj