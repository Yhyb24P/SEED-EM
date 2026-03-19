from __future__ import annotations

"""
Phase A: Task-Aware Selector Collaborative Training Engine.

// 严格遵循重构设计稿 3.3 节的 A1 -> A2 -> A3 三段式优化。
// 动态接管 QVAE 解码器与 GNN Proxy 形成闭环。
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from engine_prep.task_aware_selector import ComponentSelector, DifferentiableSTFTAndDE, compute_selector_loss
from engine_quantum.qvae_net import QuantumEEGDenoiser
# 假设存在轻量级 Proxy GNN
# from engine_gnn.proxy_models import ProxyEmotionGCN 

def train_collaborative_selector(
    qvae_decoder: QuantumEEGDenoiser,
    proxy_model: torch.nn.Module, 
    dataloader: DataLoader,
    device: torch.device,
    epochs_a2: int = 10,
    epochs_a3: int = 20
):
    """三段式协同寻优引擎 (Phase A)"""
    
    selector = ComponentSelector(in_channels=1, n_components=6).to(device)
    diff_stft = DifferentiableSTFTAndDE(fs=200.0, window_sec=2.0).to(device)
    
    # ==========================================
    # Step A1: Proxy 预热 (假设 proxy_model 已完成旧特征预训练)
    # 此时冻结 Proxy 权重，仅通过它提取监督梯度
    # ==========================================
    for param in proxy_model.parameters():
        param.requires_grad = False
    proxy_model.eval()
    
    for param in qvae_decoder.parameters():
        param.requires_grad = False
    qvae_decoder.eval()

    optimizer_sel = torch.optim.AdamW(selector.parameters(), lr=1e-3, weight_decay=1e-4)

    # ==========================================
    # Step A2: 冻结 Proxy，独占训练 Selector
    # ==========================================
    print("[*] 启动 Step A2: 冻结 Proxy，训练 Task-Aware Selector...")
    selector.train()
    
    for epoch in range(epochs_a2):
        for batch in dataloader:
            # batch.z_raw: 未掩码的原始隐变量 (B, K, T)
            # batch.art_prior: Phase A 阶段的弱标签先验 (由 Pearson 提供)
            z_raw, art_prior, y_emo = batch.z_raw.to(device), batch.art_prior.to(device), batch.y_emo.to(device)
            
            optimizer_sel.zero_grad()
            
            # 1. Selector 打分
            p_mask = selector(z_raw)  # (B, K, 1)
            
            # 2. 加权流形并重构纯净物理信号 (可微)
            z_weighted = z_raw * p_mask
            x_pure_hat = qvae_decoder.decoder(z_weighted.transpose(1, 2)).transpose(1, 2)
            
            # 3. 可微频域降解
            node_de = diff_stft(x_pure_hat)
            
            # 4. 前向 Proxy 代理
            y_pred = proxy_model(node_de, batch.adj_matrix.to(device))
            
            # 5. 计算混合损失
            loss_task = F.cross_entropy(y_pred, y_emo)
            loss_reg = compute_selector_loss(p_mask, art_prior)
            loss_total = loss_task + loss_reg
            
            loss_total.backward()
            optimizer_sel.step()

    # ==========================================
    # Step A3: 解冻 Proxy，协同微调 (Collaborative Fine-tuning)
    # ==========================================
    print("[*] 启动 Step A3: 解冻 Proxy 顶层，开启微小学习率联训...")
    
    # 仅解冻 Proxy 最后两层，防止伪迹重新污染整个图表征
    for name, param in proxy_model.named_parameters():
        if "classifier" in name or "fc" in name:
            param.requires_grad = True
            
    proxy_model.train()
    optimizer_joint = torch.optim.AdamW([
        {'params': selector.parameters(), 'lr': 1e-4},
        {'params': proxy_model.parameters(), 'lr': 1e-5}  # 极小学习率约束
    ])
    
    for epoch in range(epochs_a3):
        for batch in dataloader:
            z_raw, art_prior, y_emo = batch.z_raw.to(device), batch.art_prior.to(device), batch.y_emo.to(device)
            optimizer_joint.zero_grad()
            
            p_mask = selector(z_raw)
            x_pure_hat = qvae_decoder.decoder((z_raw * p_mask).transpose(1, 2)).transpose(1, 2)
            node_de = diff_stft(x_pure_hat)
            y_pred = proxy_model(node_de, batch.adj_matrix.to(device))
            
            loss_total = F.cross_entropy(y_pred, y_emo) + compute_selector_loss(p_mask, art_prior)
            loss_total.backward()
            optimizer_joint.step()
            
    print("[+] Phase A 协同寻优结束，Task-Aware Selector 权重已准备落盘。")