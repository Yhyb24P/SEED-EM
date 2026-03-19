# SEED-PhysioGCN: A Physics-Informed End-to-End EEG Emotion Recognition Pipeline

## 项目概述 (Project Overview)

**SEED-PhysioGCN** 是针对上海交通大学 SEED (SJTU Emotion EEG Dataset) 构建的端到端、具备严密物理与代数边界的情绪脑电识别系统。

本项目摒弃了传统“黑盒式”深度学习拼接范式，将原有的特征工程与图网络分类器进行了系统级统合。通过引入**微积分压摆率拦截**、**量子变分流形手术 (QVAE)**、**任务感知组件选择 (Task-Aware Selector)** 与**非平稳动态双流图卷积 (Dual-Stream DGCN)**，在底层代数架构上系统性解决了基线漂移、空间伪迹扩散、频域语义塌缩以及跨折叠验证中的泛化泄露问题。系统实现了从原始微伏级波动到宏观情绪状态 ($Y \in \{0, 1, 2\}$) 的满秩映射与严格溯源。

## 核心物理与数学防线 (Physics & Mathematical Defenses)

### 1. 物理源空间净化 (Calculus & Topological Surgery)

- $C^0$ **连续性确界**：废除绝对幅值“一刀切”，引入基于一阶导数的微积分压摆率限制器 (Slew Rate Limiter)，精准拦截 $>50\mu V/\text{sample}$ 的非生理硬件电涌，杜绝吉布斯高频宽带泄露。
    
- **任务感知流形正交掩码 (Task-Aware Masking)**：放弃静态的皮尔逊硬截断，引入基于下游情绪识别任务监督反馈的可学习组件选择器 (Component Selector)。通过可微的 STFT 算子桥接，在保留物理波形连续性的同时，实现伪迹的软隔离。
    

### 2. 马尔可夫拓扑连续性 (Markov Topological Continuity)

- **时态图直通 (Drop Graph)**：遵循“先时域后空域”逻辑，dFC/STFT 均基于绝对连续的时间流形运行。遇到高频爆音执行 Drop Graph 策略直接丢弃特定拓扑图，严格废除“残端拼接”引起的物理空间非自然跳变。
    

### 3. 时频语义对齐 (Semantic Frequency Alignment)

- 废除硬编码的错位频宽，严格锁定短时傅里叶变换 (STFT) 物理感受野为 $W_{stft}=2.0s$，产生绝对 $\Delta f = 0.5\text{Hz}$ 的频域底数。
    
- 微分熵 (DE) 降解算子在此基底上执行积分映射，无损重建 $\delta, \theta, \alpha, \beta, \gamma$ 经典生理频段，彻底消除低频能量塌缩与频带重叠。确保图卷积网络接收的时间切片严格满足 $1\text{ 帧} = 1\text{ 秒}$。
    

### 4. 双流正交图路由与谱正则化 (Dual-Stream Orthogonal Routing & Spectral Regularization)

- **双流解耦流形 (Dual-Stream Disentanglement)**：在图网络早期阶段 (Early Split) 生成正交基底，分离情感流与受试者物理流。结合梯度反转算子 (GradReverse) 的动态退火与拓扑图正交排斥损失 ($\mathcal{L}_{ortho}$)，从代数层面消除个体协变量偏移 (Covariate Shift)。
    
- **自环先验与残差阻尼 (Self-loop Prior & Residual Damping)**：在隐式拓扑偏置中强行注入常数级对角线先验 $\gamma I$，并结合跨层 Skip-Connection 恒等映射。截断狄利克雷能量衰减 (Dirichlet Energy Decay)，彻底阻断了马尔可夫游走中深层 GNN 的空间表征坍塌 (Over-smoothing)。
    

### 5. 嵌套泛化隔离 (Nested Cross-Validation Isolation)

- **阻断 Oracle Bias**：重构 `15-Fold LOSO` 为严格的 `13(Train) + 1(Val_Src) + 1(Test_Tgt)` 嵌套交叉验证路由。将 Early Stopping 的游标严格绑定至独立源域验证集，使目标域的时间轴与寻优完全代数正交，废除虚假高置信度。
    
- **基线锚点防泄露**：在 DataLoader 提取个体级 $Z-Score$ 分布基线 (`subj_mean`, `subj_std`) 时强制剥离 Test Trial，物理切断源域向目标域的时序统计量污染。
    
- **梯度保护**：引入极大值平滑锚点计算类别惩罚算子 $\text{smoothed\_counts} = c_i + \max(C) \times 0.05$，拦截 FP32 浮点下限，防止极端长尾样本引发的梯度雪崩与 Loss 放缩异常。
    

## 产出张量标准 (Tensor Standards)

预处理管线执行完毕后，生成的物理特征将落盘至 `data/02_pure_features/S{ID}.mat`。其数据遵循严密的类型定义与形状约束，可直接对接 PyTorch 图神经网络与序列模型：

||||
|---|---|---|
|**字段名**|**维度/格式**|**代数与生理说明**|
|`data_pure`|$3 \times 15$ Cell 数组 $\rightarrow 62 \times T$|剥离各类伪迹后的纯净 1D 脑电波形，维持原始时间长度以确保绝对马尔可夫连续性。|
|`adj_matrix`|$3 \times 15$ Cell 数组 $\rightarrow 62 \times 62$|静态皮尔逊连通矩阵，受自适应方差阈值过滤，精准指示全局情绪空间路由拓扑。|
|`dfc_matrix`|$3 \times 15$ Cell 数组 $\rightarrow K \times 62 \times 62$|动态时空功能连通性张量，自带极值方差熔断与 Drop Graph 机制，无 `NaN` 泄露危险。|
|`node_de`|$3 \times 15$ Cell 数组 $\rightarrow 62 \times 5 \times T_{\text{bins}}$|微分熵 (DE) 特征节点序列，适配于 Task-Aware Proxy 寻优与下游时空图网络。|
|`stft_features`|$3 \times 15$ Cell 数组 $\rightarrow 62 \times F \times T_{stft}$|STFT 频域包络（$0.5\text{Hz}$ 绝对分辨率），零相位漂移，直接兼容 DE 降解映射。|
|`qvae_latents`|$3 \times 15$ Cell 数组 $\rightarrow 6 \times T$|保留的纯净 $Z_{clean}$ 潜变量。抛弃皮层几何局限的极高信噪比流形模态。|
|`artifact_scores`|$3 \times 15$ Cell 数组 $\rightarrow M \times N$|用于审计与反作弊惩罚的伪迹成分先验强度得分。|
|`retain_probs`|$3 \times 15$ Cell 数组 $\rightarrow M \times N$|组件选择器 (Selector) 输出的连续保留概率软掩码。|
|`sfreq`|标量（$200.0$）|全局采样率物理元数据锁。|
|`ch_names`|62 元素列表|基于国际 10-20 系统的标准化电极空间映射顺序表。|

## 架构拓扑 (Architecture Topology)

项目遵循严格的“数据-算子-调度”三维解耦设计，阻断 I/O 竞态条件与时空变量污染：

```
SEED-PhysioGCN/
├── data/                            # [隔离挂载点] 统一张量流转中枢
│   ├── 01_raw_mat/                  # 原始 SEED 数据输入
│   ├── 02_pure_features/            # 预处理输出/图谱引擎输入 (物理同构界)
│   ├── 03_qa_reports/               # QA 与物理溯源探针落盘
│   ├── 04_weights/                  # 模型权重 (QVAE, Selector, Proxy)
│   └── 05_checkpoints/              # 15折 LOSO 泛化确界与分类器输出落盘
├── configs/                         # 注册表 (prep_config.py / train_config.yaml)
├── engine_prep/                     # [算子 1] 物理净化、特征提取与任务感知组件
│   ├── transforms.py                # 压摆率微积分、Safe-CAR
│   ├── extractors.py                # 连续时序 dFC/STFT
│   ├── artifact.py                  # BSS 分解与软掩码组装
│   └── task_aware_selector.py       # [新增] Task-Aware 神经网络与可微频域降解
├── engine_quantum/                  # [算子 2] PyTorch+PennyLane 流式预训练与投影
│   ├── qvae_net.py                  # 量子/变分强纠缠算子
│   └── pretrain_stream.py           # 流式懒加载预训练
├── engine_gnn/                      # [算子 3] GCN/DGCN 扩散拉普拉斯图层与 LOSO 路由
│   ├── dataloader.py                # 基线隔离与降解算子
│   ├── graph_operators.py           # 双流正交解耦动态图
│   ├── cv_router.py                 # 嵌套隔离CV路由引擎
│   └── classifier_taskaware_adapter.py # [新增] 任务感知损失聚合适配器
├── tools_audit/                     # [探针] 时空 1D/2D 绘图与全局混淆/显著性溯源
├── train_phase_a_selector.py        # [中台] Phase A 任务感知协同微调引擎
├── run_pipeline.py                  # 前端：触发预处理与多模态张量装箱
└── run_classifier.py                # 后端：触发 15-Fold 动态图卷积泛化寻优
```

### 核心算子文件映射对照表 (Module Registry)

用于追踪旧版零散代码向现代解耦范式（含 Phase A 演进）的映射路径：

||||
|---|---|---|
|**物理代数职能**|**规划后解耦文件**|**原始遗留文件溯源**|
|**配置** / 常数、采样率确界|`configs/prep_config.py`|`config.py`|
|**配置** / 超参数与频段字典|`configs/train_config.yaml`|`config.yaml`|
|**引擎** / 微积分拦截与 Safe-CAR|`engine_prep/transforms.py`|`core_transforms.py`|
|**引擎** / 连续时序 dFC/STFT|`engine_prep/extractors.py`|`feature_extractors.py`|
|**引擎** / BSS 软掩码保留|`engine_prep/artifact.py`|`artifact_remover.py`|
|**引擎** / 组件选择与可微特征桥接|`engine_prep/task_aware_selector.py`|[Phase A 新增架构]|
|**中台** / Phase A 三段式协同寻优|`train_phase_a_selector.py`|[Phase A 新增架构]|
|**量子** / 变分强纠缠算子|`engine_quantum/qvae_net.py`|`models.py`|
|**量子** / 流式懒加载预训练|`engine_quantum/pretrain_stream.py`|`train_qvae.py`|
|**图算** / 基线隔离与降解算子|`engine_gnn/dataloader.py`|`core/dataloader.py`|
|**图算** / 双流正交解耦动态图|`engine_gnn/graph_operators.py`|`core/graph_models.py`|
|**图算** / 嵌套隔离CV路由引擎|`engine_gnn/cv_router.py`|`core/cross_validation.py`|
|**图算** / 任务感知损失聚合适配|`engine_gnn/classifier_taskaware_adapter.py`|[Phase A 新增架构]|
|**探针** / 1D时域网格与2D时频热力图|`tools_audit/probe_1d2d.py`|`visualize_pipeline_stages.py`|
|**探针** / UMAP隐空间与显著性拓扑|`tools_audit/evaluate_loso.py`|`evaluate.py` / `visualize_advanced.py`|

## 部署与执行引擎 (Deployment & Execution)

### 1. 黎曼/量子流形环境配置

推荐使用 `conda` 构建强一致性的隔离计算图环境以保护代数精度：

```
conda create -n seedem python=3.11 -y
conda activate seedem
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install torch_geometric pennylane python-picard
pip install numpy scipy pandas mne joblib scikit-learn matplotlib seaborn pyyaml networkx tqdm umap-learn
```

### 2. 初始张量挂载

将官方 SEED 数据集内的 `S1.mat ~ S15.mat` 及 `label.mat` 部署至统一输入中枢 `data/01_raw_mat/` 目录下。

### 3. [Phase I] 量子基准预训练 (Quantum Pre-training)

由于全量载入引发的 $\mathcal{O}(NTC)$ RAM 爆炸，系统利用 `IterableDataset` 执行流式极小化博弈。提取跨个体的生理不变性子空间（本步骤全生命周期仅需执行一次）：

```
python -m engine_quantum.pretrain_stream --epochs 15 --batch_size 2000 --beta_max 0.5
```

_注：生成的泛化权重将自动落盘至 `data/04_weights/`。_

### 4. [Phase A] 任务感知协同微调 (Task-Aware Collaborative Fine-Tuning)

基于初始无监督权重，利用代理图网络提取下游任务监督信号，微调组件选择器，实现由任务效用主导的软掩码分离：

```
python train_phase_a_selector.py
```

### 5. [Phase II] 物理预处理管线 (Physics Pipeline)

执行端到端伪迹手术、一阶导数拦截与多模态流形张量装箱：

```
python run_pipeline.py
```

_注：该进程支持防灾断点续传。处理完毕的高保真原生多模态特征矩阵将留存于 `data/02_pure_features/`，并同步触发自动化质量审计 (QA Audit)。_

### 6. [Phase III] 图情绪状态分类 (Dynamic GCN Evaluation)

以预处理产出的确界张量为底层空间，执行基于嵌套交叉验证的跨个体独立推断：

```
python run_classifier.py
```

_系统自动调用的底层防线：类不平衡平滑计算_ $\rightarrow$ _嵌套 13+1+1 隔离路由_ $\rightarrow$ _隐空间动态对抗剥离_ $\rightarrow$ _VRAM 显存碎片主动回收。最优确界检查点归档于 `data/05_checkpoints/`。_

### 7. [Phase IV] 生理归因审计 (Physiological Audit)

载入交叉验证的最终分布边界，执行隐空间 UMAP 单纯复形投影与零样本逆向追踪，映射网络关注度至 2D 物理拓扑：

```
python -m tools_audit.evaluate_loso
```

> **审计验收依据**：有效的非平稳情绪映射（Saliency Topography）应在头皮前额叶（FP1, FP2）产生高亮拉普拉斯边连接，并在颞叶（T7, T8）展现显著的不对称性。若主导边聚集于顶枕区（O1, O2），应立刻重新核查基线截断与视觉诱发伪迹。UMAP 空间中应能观测到不同情绪类别的语义簇（Semantic Clusters）及受试者域的强制对齐。

## 张量拓扑流转路径 (Tensor Flow Dynamics)

为方便下游扩展研究，整个系统的张量演化轨迹归纳如下：

1. **载入级**：原始微伏张量 $X_{raw} \in \mathbb{R}^{62 \times T}$。
    
2. **截断级**：丢弃前 20 帧物理时间流形（等价于 20s 静息态视频引导），杜绝基线污染。
    
3. **软隔离级**：经 `task_aware_selector` 加权后输出软隔离物理张量，提取时频包络 $Z_{stft} \in \mathbb{R}^{62 \times F \times T_{bins}}$ 与 连通性拓扑 $A_{adj} \in \mathbb{R}^{62 \times 62}$。
    
4. **降解级**：执行 0.5Hz 字典积分映射，产生微分熵 DE 特征张量 $X_{de} \in \mathbb{R}^{62 \times 5 \times T_{valid}}$，执行跨域 Z-Score 映射。
    
5. **图通信级**：双流并行输入，通过对角线保护的稠密动态张量生成 $A_{emo}, A_{subj}$ 并进行含残差的拉普拉斯扩散。
    
6. **坍塌级**：全局均值池化（GAP）将 $62$ 空间维度坍塌，输出解耦隐空间流形 $Z_{emo}, Z_{subj}$。
    
7. **映射级**：经过对抗博弈的特征最终落入三元单纯形概率分布 $Y \in \{0, 1, 2\}$。
    

## 学术免责与约束声明

本项目作为 **SEED-PhysioGCN** 计算框架的最终重构实现，所有代数推演、基准拦截器、正交解耦损失以及截断阈值均旨在确保下游深度模型所学信息具备严格的神经生理学意义。若自行修改底层微积分梯度阈值（默认 $50.0\mu V$）、STFT 滑动窗长或移除嵌套隔离验证机制，请务必进行特征值谱弥散性与香农-奈奎斯特边界的重估。