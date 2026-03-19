# SEED Dataset EEG Preprocessing & Feature Engineering Pipeline

本项目是针对上海交通大学 **SEED (SJTU Emotion EEG Dataset)** 情绪脑电数据集的预处理与多模态特征工程流水线。项目将原始 MATLAB 脚本全面迁移至 Python 生态（MNE-Python, SciPy, PyTorch, PennyLane），并对经典信号处理流程进行了底层的拓扑重构与严格的数学边界保护，系统性解决了基线漂移、空间伪迹扩散、时间连续性断裂及跨计算框架的精度泄漏等工程级灾难。

## 核心特性

- **重构的马尔可夫拓扑连续性**：遵循“先时域后空域”逻辑，dFC/STFT 均基于绝对连续的时间流形运行。遇到高频爆音执行 Drop Graph 策略丢弃特定拓扑图，严格废除拼凑断崖造成的 Sinc 高频宽带泄露。
    
- **原生多模态特征矩阵装箱**：单次运行即可产出静态皮尔逊图 (GCN)、防 NaN 熔断的演化 dFC 张量 (ST-GCN)、STFT 时频热力包络 (2D-CNN) 以及已完成手术剔除的 6-qubits 纯净量子潜变量（$Z_{clean}$）。
    
- **自动化审计与动态探针 (QA Audit)**：旁路生成高密度可视化 PDF 学术报告，包含 1D 提纯前后时域网格比对、2D 频域分布热力图以及深度表征的高级拓扑连通矩阵审计。
    

## 目录结构

```
├── config.py                # 全局配置（全局常数、采样率、通道拓扑等硬性约束）
├── core_transforms.py       # 核心信号算子（双端 Safe-CAR、微积分压摆率限制器）
├── feature_extractors.py    # 特征引擎（集成 Drop Graph 与防 NaN 熔断的 dFC/STFT）
├── models.py                # 深度模型定义（QVAE 量子变分电路与张量解耦封装）
├── artifact_remover.py      # 伪迹隔离路由（支持 VRAM 时域微批次、正交掩码提取与物理逆投影）
├── train_qvae.py            # [预训练流] QVAE 流形映射权重生成器（含流式生成器与余弦退火）
├── visualize_advanced.py    # 高阶图谱探针（GNN 连通性拓扑与全局 PSD Welch's Audit）
├── visualize_pipeline_stages.py # 时空探针（管线各层级 1D / 2D 特征对比多页渲染）
├── main.py                  # 流水线主入口（断点续传、系统级多模态张量装箱）
├── Data/                    # 数据存储隔离区
│   ├── Preprocessed_EEG/    # [Input] 原始被试 .mat 文件
│   ├── QA_Reports/          # [QA Output] 自动化 PDF 质检报告与高阶审计
│   └── EEG_pure/            # [Output] 最终生成的多模态特征包 .mat 文件
└── requirements.txt         # 依赖清单
```

## 快速运行指南

### 1. 环境准备与依赖安装

推荐使用 `conda` 创建隔离计算环境：

```
conda create -n seedem python=3.11 -y
conda activate seedem
pip install numpy scipy pandas mne joblib scikit-learn matplotlib tqdm python-picard pennylane
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

### 2. 数据就绪

将 SEED 原始数据文件放置于 `Data/Preprocessed_EEG/` 目录下。

### 3. [必选] 执行流式泛化预训练

首次启动必须利用带噪数据生成非监督流形泛化权重，提供跨个体的通用解剖学推断基准：

```
# 自动通过 IterableDataset 流式消费 15 名被试的数据，防止 OOM
python train_qvae.py --epochs 15 --batch_size 2000 --beta_max 0.5
```

### 4. 执行全量多模态管线

```
# 激活自动推断并导出 .mat 多模态张量 (支持零成本断点续传)
python main.py
```

## 产出张量标准 (`Data/EEG_pure/S{ID}.mat`)

数据遵循严密的类型定义与形状约束，可直接对接 PyTorch 几何序列模型：

|   |   |   |
|---|---|---|
|**字段名**|**维度/格式**|**说明**|
|`data_pure`|$3 \times 15$ Cell 数组 $\rightarrow 62 \times T$|剥离各类伪迹后的纯净 1D 脑电波形，维持原始时间长度以确保绝对马尔可夫连续性。|
|`adj_matrix`|$3 \times 15$ Cell 数组 $\rightarrow 62 \times 62$|静态皮尔逊连通矩阵，受自适应阈值过滤计算，精准指示情绪空间路由拓扑。|
|`dfc_matrix`|$3 \times 15$ Cell 数组 $\rightarrow K \times 62 \times 62$|动态时空功能连通性张量，自带极值方差熔断与 Drop Graph 机制，无 `NaN` 泄露危险。|
|`qvae_latents`|$3 \times 15$ Cell 数组 $\rightarrow 6 \times T$|已执行拓扑手术的纯净 $Z_{clean}$ 潜变量。抛弃皮层几何局限，极高信噪比物理特征模态。|
|`stft_features`|$3 \times 15$ Cell 数组 $\rightarrow 62 \times F \times T_{stft}$|STFT 频域包络（$0.5\text{Hz}$ 分辨率），零相位漂移，直接兼容 2D-CNN 视觉提取模型。|
|`sfreq`|标量（$200.0$）|采样率元数据锁。|
|`ch_names`|62 元素列表|基于国际 10-20 系统的标准化电极空间映射顺序表。|