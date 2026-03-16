# SEED Dataset EEG Preprocessing & Feature Engineering Pipeline

本项目是针对上海交通大学 **SEED (SJTU Emotion EEG Dataset)** 情绪脑电数据集的预处理与多模态特征工程流水线。项目将原始 MATLAB 脚本全面迁移至 Python 生态（MNE-Python, SciPy, PyTorch, PennyLane），并对经典信号处理流程进行了底层的拓扑重构与严格的数学边界保护，系统性解决了基线漂移、空间伪迹扩散、时间连续性断裂及跨计算框架的精度泄漏等工程级灾难。

## 最新架构演进 (v3.0: Resilient Topology & Memory Safe)

本次重构在张量流转、代数适定性与物理时序约束上建立了严密的防线，全面提升了高阶深度学习特征的物理保真度：

- **时域拓扑连续性 (Time-Topology Preservation)**：彻底废除引发吉布斯频谱泄露的物理残端拼接。提取 dFC 与 STFT 时，引入 `Drop Graph` 掩码机制维持绝对连续的时间轴，严格保障序列网络 (ST-GCN/RNN) 的马尔可夫时序假设。
    
- **微积分压摆率拦截 (Calculus Slew Rate Limiter)**：引入基于一阶差分 ($\Delta V$) 的梯度拦截器，无损保留低频高振幅合法脑电波，精准插值电极松动引发的物理阶跃电涌。该算子已前置于全局解耦层，防止长窗全局方差污染。
    
- **适定性盲源分离 (Well-posed BSS)**：将滑动处理窗长延长至 **400.0 秒**（全 Trial 覆盖），确保经验协方差矩阵的特征值谱分布高度收敛，彻底杜绝短窗（如2.0s）引发的空间过拟合与情绪特征误杀。
    
- **跨框架精度防御与内存流式调度 (Precision & Iterable VRAM Safety)**：
    
    - 强制阻断 PennyLane (`float64`) 向 PyTorch (`float32`) 的张量精度泄漏。
        
    - 注入 `chunk_size=2000` 时域微批次前推，将空间复杂度由 $O(T)$ 降维至 $O(B_{chunk})$，彻底消灭 OOM。
        
    - 采用 **流式懒加载生成器 (Iterable Lazy-Loader)** 与 $\beta$ **退火上限截断**，支持全量数据（15名被试）进行无监督预训练泛化，消除个体颅骨解剖结构过拟合。
        

## 核心特性

- **重构的信号处理防线**：遵循“先时域后空域、先非线性后线性”的逻辑，1.0Hz 零相位 FIR 滤波器结合后置微积分伪迹拦截，将信号相位畸变降至最低。
    
- **原生多模态特征矩阵**：单次运行即可产出静态皮尔逊图 (GCN)、时序演化 dFC 张量 (ST-GCN)、STFT 频域热力包络 (2D-CNN) 及 6-qubits 非线性量子特征。
    
- **自动化审计探针 (QA Audit)**：旁路生成高密度可视化 PDF 报告，包含 1D 时域波形、2D 频域热力图、全局 PSD (Welch) 及图连通性矩阵。
    

## 目录结构

```
├── config.py                # 全局配置（400.0s全窗、采样率、通道拓扑等硬性约束）
├── core_transforms.py       # 核心信号算子（双端 Safe-CAR、微积分压摆率限制器）
├── feature_extractors.py    # 特征引擎（集成 Drop Graph 与防 NaN 熔断的 dFC/STFT）
├── models.py                # 深度模型定义（QVAE 量子变分电路与跨框架精度对齐）
├── artifact_remover.py      # 伪迹处理调度（支持 OOM 预防微批次与全局流形映射）
├── train_qvae.py            # [新增] QVAE 无监督预训练脚本（含流式张量生成器与 β 截断）
├── eeg_debugger.py          # 信号追踪探针（管线多阶段形态时域对比可视化）
├── visualize_pipeline_stages.py # 质检可视化（1D时域波形与 2D STFT）
├── main.py                  # 流水线主入口（多模态特征装箱、断点续传与进度追踪）
├── Data/                    # 数据存储隔离区
│   ├── Preprocessed_EEG/    # [Input] 原始被试 .mat 文件
│   ├── QA_Reports/          # [QA Output] 自动化 PDF 质检报告与高阶审计
│   └── EEG_pure/            # [Output] 最终生成的多模态特征包 .mat 文件
└── requirements.txt         # 依赖清单
```

## 快速运行指南

### 1. 环境准备与依赖安装

```
conda create -n seedem python=3.11 -y
conda activate seedem
pip install numpy scipy pandas mne joblib scikit-learn matplotlib tqdm python-picard pennylane
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

### 2. 数据就绪

将 SEED 原始数据文件放置于 `Data/Preprocessed_EEG/` 目录下。

### 3. [关键环节] 流式泛化预训练

首次运行必须生成可用的量子流形映射权重，否则系统将抛出 `FileNotFoundError` 阻断运行：

```
# 启动包含 IterableDataset 的流式预训练 (约耗时数十分钟)
python train_qvae.py --epochs 15 --batch_size 2000 --beta_max 0.5
```

### 4. 执行全量多模态管线

```
# 支持断点续传，随时 Ctrl+C 中断，再次运行将瞬间跳过已落盘的 Subject
python main.py
```

## 数据处理流水线与神经生理学映射

脑电矩阵 $X \in \mathbb{R}^{62 \times T}$ 遵循以下物理约束：

1. **去均值 (Zero-mean)**：移除时间序列直流偏置。
    
2. **坏导拓扑修复 (Topology Interpolation)**：基于日志球面样条重构已知脱落电极。
    
3. **频域净化 (FIR & Notch)**：$1.0 \sim 50.0 \text{Hz}$ 零相位滤波隔离极低频与市电。
    
4. **双端安全重参考 (Dual-Bound Safe CAR)**：执行 `1e-4 < std < 100.0` 掩码共模抑制。
    
5. **微积分极值拦截 (Slew Rate Limiter)**：先于盲源分离，一阶导数定位硬件爆音并实施缝合，保护全局方差。
    
6. **适定性盲源分离 (400.0s QVAE/ICA)**：全 Trial 单一流形解耦，提纯源空间动力学。
    
7. **多模态特征引擎 (Multimodal Extractor)**：生成 dFC 张量、STFT 与量子潜变量。
    

## 产出张量标准 (`Data/EEG_pure/S{ID}.mat`)

|字段名|维度/格式|说明|
|---|---|---|
|`data_pure`|$3 \times 15$ Cell 数组 $\rightarrow 62 \times T$|预处理后的纯净 1D 脑电波形，维持原始时间长度以防马尔可夫链断裂。|
|`adj_matrix`|$3 \times 15$ Cell 数组 $\rightarrow 62 \times 62$|静态皮尔逊连通矩阵，受自适应阈值过滤计算，适合静态 GCN。|
|`dfc_matrix`|$3 \times 15$ Cell 数组 $\rightarrow K \times 62 \times 62$|动态功能连通性张量，自带防 `NaN` 极值方差熔断，适合 ST-GCN。|
|`qvae_latents`|$3 \times 15$ Cell 数组 $\rightarrow 6 \times T$|QVAE 提取的高维量子特征投影流形（6-qubits 压缩），极高抗噪物理属性。|
|`stft_features`|$3 \times 15$ Cell 数组 $\rightarrow 62 \times F \times T_{stft}$|STFT 时频张量（$0.5\text{Hz}$ 频率分辨率），无吉布斯 Sinc 泄露，适合 2D-CNN。|
|`sfreq`|标量（$200.0$）|采样率元数据锁。|
|`ch_names`|62 元素列表|基于国际 10-20 系统的电极空间拓扑映射表。|