SEED Dataset EEG Preprocessing & DL Feature Pipeline

本项目为基于上海交通大学 SEED（SJTU Emotion EEG Dataset）情绪脑电数据集的标准化预处理与深度学习特征提取流水线。项目实现了从 MATLAB 原始硬编码逻辑向现代 Python（基于 MNE-Python, SciPy, PyTorch 与 PennyLane）生态的完整、严谨的工程化迁移。

管线具备高度模块化的架构，支持高信噪比脑电信号的重构，并能并行提取适配图神经网络（GNN）与时空卷积网络（EEGNet/TSception）的先验特征矩阵。

目录结构规划

.
├── config.py                # 全局配置与元数据常量（采样率、通道拓扑等）
├── core_transforms.py       # 脑电核心信号处理无状态算子（CAR、滤波、截断等）
├── feature_extractors.py    # DL/GNN 特征提取器（图连通性、时频 STFT 等）
├── models.py                # 深度与量子模型架构定义（包含 QVAE 拓扑）
├── artifact_remover.py      # 伪迹隔离与多进程调度路由（支持 ICA / QVAE）
├── main.py                  # 流水线主入口
├── meta_info/               # 数据集实验材料与通道拓扑说明
├── matlab_legacy/           # 原始 MATLAB 算法参照归档
├── Data/                    # 数据流挂载目录（受 .gitignore 保护）
│   ├── Preprocessed_EEG/    # [Input] 存放原始 45 个被试 .mat 文件
│   └── EEG_pure/            # [Output] 自动化输出的纯净特征 .mat 文件
├── requirements.txt         # 依赖环境清单
├── .gitignore               # 版本控制忽略规则
└── README.md                # 项目文档


技术规格与处理节点

当前重构版本在保证数据处理严谨性的前提下，实施了端到端的性能优化与特征扩容：

1. 信号重组与频域净化 (Core Transforms)

硬编码拓扑修复：精确复现 SEED 实验日志，对特定被试异常通道实施物理均值修补。

频域截断与谐波抑制：采用 $1.0 \sim 45.0 \text{ Hz}$ 零相位带通滤波阻断低频漂移与高频肌电，串联 $50 \text{ Hz}$ IIR 陷波器（Notch Filter）彻底剥离工频干扰。

矢量化异常抑制：摒弃传统循环，使用 1D 布尔卷积算子对 $\pm 130\mu V$ 越界点进行高吞吐量的邻域膨胀截断。

2. 双重盲源去伪路由 (Artifact Rejection Routing)

Classic ICA (默认生产路径)：

采用 $1.0\text{ Hz}$ 专用高通副本进行解混矩阵拟合，彻底消除次低频漂移引发的方差倾斜与不收敛警告。

Picard 求解器优先：引入基于 L-BFGS 的 Picard 算法提供全局收敛保证，降级兜底至 FastICA。

Quantum VAE (前沿实验路径)：

构建集成 PyTorch 与 PennyLane 的变分量子线路（VQC）。

通过 torch.vmap 矢量化量子态模拟。计算隐空间与额极（FP1/FP2）眼电代理的皮尔逊相关系数，动态置零高伪迹关联的量子维度后重构信号。

3. 深度学习特征扩展 (DL Feature Extractors)

输出矩阵除纯净连续脑电外，原生附加两类高级特征矩阵：

Adjacency Matrix ($A \in \mathbb{R}^{62 \times 62}$)：无向皮尔逊全脑功能连通性矩阵，供 PyTorch Geometric (GraphConv/GCN) 构建 edge_index。

STFT Matrix ($Z \in \mathbb{C}^{F \times T}$)：$4\text{ Hz}$ 频域分辨率时频图谱，供 2D-CNN 或 TSception 提取时空节律特征。

4. 架构与内存优化

实施严格的内存流式释放（gc.collect(), .copy() 深拷贝提取），将 Day 3 高负载试次的内存峰值压降约 40%，阻断多字典键迭代引发的指针变异。

多进程锁保护：在 QVAE 模式下自动将 joblib 并发降维至 n_jobs=1，避免 PyTorch C++ 线程死锁；ICA 模式全速 n_jobs=-1 并发。

快速运行指南

1. 环境准备

推荐使用虚拟环境管理器配置依赖，核心需求包括 MNE, SciPy, Joblib。QVAE 及高效 ICA 求解器依赖 PyTorch, PennyLane 及 python-picard。

pip install numpy scipy pandas mne joblib scikit-learn
pip install python-picard torch pennylane


2. 数据就绪

在项目根目录构建 Data/Preprocessed_EEG，放入 SEED 数据集的 45 个实验原始 .mat 文件。系统会自动过滤混入的 label.mat。

3. 执行流水线

python main.py


4. 输出标准

纯净特征文件生成于 Data/EEG_pure/S{ID}.mat。其内部 MATLAB 字典层级为：

data_pure: $3 \times 15$ Object Array，纯净连续脑电数据。

adj_matrix: $3 \times 15$ Object Array，图连通性矩阵。

stft_features: $3 \times 15$ Object Array，时频变换矩阵。

sfreq: 采样率 ($200\text{ Hz}$)。

ch_names: $62$ 通道元数据列表。

系统状态与断言规范

为保证入模数据质量，管道设置了严格的输入形状与试次数量拦截器（通道数 $\neq 62$ 或 试次 $\neq 15$ 将直接抛出 ValueError 并阻断该被试）。