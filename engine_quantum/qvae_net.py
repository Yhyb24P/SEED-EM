"""
深度模型架构
独立封装网络结构 (PyTorch + PennyLane)，解耦模型定义与预处理流。
"""
import warnings

# 尝试加载环境依赖
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import pennylane as qml
    HAS_QVAE_DEPS = True
except ImportError:
    HAS_QVAE_DEPS = False
    warnings.warn("未检测到 PyTorch 或 PennyLane，QVAE 组件将保持休眠状态。")

if HAS_QVAE_DEPS:
    class QuantumEEGDenoiser(nn.Module):
        """量子变分自编码器 (QVAE) 网络拓扑定义"""
        def __init__(self, input_dim=62, hidden_dim=32, n_qubits=6):
            super().__init__()
            self.n_qubits = n_qubits
            
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.fc_mu = nn.Linear(hidden_dim, n_qubits)
            self.fc_logvar = nn.Linear(hidden_dim, n_qubits)
            
            # self.dev = qml.device("default.qubit", wires=n_qubits)
            try:
                self.dev = qml.device("lightning.gpu", wires=n_qubits)
            except Exception:
                self.dev = qml.device("default.qubit", wires=n_qubits)
            
            @qml.qnode(self.dev, interface="torch")
            def q_circuit(inputs, weights):
                for i in range(n_qubits):
                    # qml.RY(inputs[i], wires=i)
                    # qml.RY(inputs[:, i], wires=i)
                    qml.RY(inputs[..., i], wires=i)
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
                return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
                
            self.q_circuit = q_circuit
            self.q_weights = nn.Parameter(torch.randn(2, n_qubits, 3))
            
            self.decoder = nn.Sequential(
                nn.Linear(n_qubits, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, input_dim)
            )

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x):
            h = self.encoder(x)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            z = self.reparameterize(mu, logvar)
            
            # # 剥离 torch.vmap 的黑盒映射，改用显式计算图遍历穿透 PennyLane 屏障
            # q_out_list = []
            # for i in range(z.size(0)):
            #     res_i = self.q_circuit(z[i], self.q_weights)
            #     if isinstance(res_i, (list, tuple)):
            #         q_out_list.append(torch.stack(res_i))
            #     else:
            #         q_out_list.append(res_i)
            # q_out = torch.stack(q_out_list).float()
            res = self.q_circuit(z, self.q_weights)
            q_out = torch.stack(res, dim=1).float() if isinstance(res, (list, tuple)) else res.float()
                
            recon = self.decoder(q_out)
            return recon, mu, logvar, q_out