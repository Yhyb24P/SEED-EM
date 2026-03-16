# SEED Dataset EEG Preprocessing & Feature Engineering Pipeline

本项目是针对上海交通大学 **SEED (SJTU Emotion EEG Dataset)** 情绪脑电数据集的预处理与多模态特征工程流水线。项目将原始 MATLAB 脚本全面迁移至 Python 生态（MNE-Python, SciPy, PyTorch, PennyLane），并对经典信号处理流程进行了底层的拓扑重构与严格的数学边界保护，系统性解决了基线漂移、空间伪迹扩散、时间连续性断裂及跨计算框架的精度泄漏等工程级灾难。

## 最新架构演进 (v3.0: Resilient Topology & Memory Safe)

本次重构在张量流转、代数适定性与物理时序约束上建立了严密的防线，全面提升了高阶深度学习特征的物理保真度：

- **时域拓扑连续性 (Time-Topology Preservation)**：彻底废除引发吉布斯频谱泄露 (Gibbs Phenomenon) 的物理残端拼接。在提取动态功能连通性 (dFC) 与 STFT 时，引入 `Drop Graph` 掩码机制，维持绝对连续的时间轴，严格保障序列网络 (ST-GCN/RNN) 的马尔可夫时序演化假设。
    
- **微积分压摆率拦截 (Calculus Slew Rate Limiter)**：全面弃用绝对幅值截断。引入基于一阶差分 ($\Delta V$) 的梯度拦截器，无损保留低频高振幅合法脑电波（如 Delta 慢波），同时精准捕获并插值由电极松动引发的物理阶跃电涌。
    
- **双端安全重参考 (Dual-Bound Safe CAR)**：升级共平均参考算子，增加双向方差熔断锁 (`1e-4 < std < 100.0`)。同步隔离方差极小的“死导联”与高阻抗悬空的“天线导联”，切断恶性电磁噪声的全脑空间投毒 (Spatial Poisoning)。
    
- **适定性盲源分离与流形对齐 (Well-posed BSS & Global Manifold)**：
    
    - 将滑动处理窗长下调至 **2.0 秒**，显著提升 ICA/QVAE 盲源分离逆问题的数学适定性（样本/特征比达 6.4），完整覆盖眼电 (EOG) 单次周期，确保皮尔逊锚定掩码生效。
        
    - 注入全局统计量 ($global\_mean, global\_std$) 进行 Z-Score，消除局部批归一化引发的窗口拼接边界断崖伪迹 (Boundary Artifacts)。
        
- **跨框架精度防御与内存调度 (Precision & VRAM Safety)**：
    
    - 强制阻断 PennyLane (`float64`) 向 PyTorch (`float32`) 的张量精度泄漏，避免解码器权重底层类型阻断。
        
    - 优化 QVAE 实例的生命周期，显式解除 `qml.device` 句柄并在窗级处理后执行 IPC 与 CUDA 缓存清空，彻底根除 VRAM 碎片化与 OOM 泄漏。
        
    - 为 `np.corrcoef` 增加低方差下限熔断 ($<1e-4$)，杜绝死导联引发的 $0/0$ 极值溢出向 dFC 张量注入 `NaN` 梯度雪崩病毒。
        

## 核心特性

- **重构的信号处理防线**：遵循“先时域后空域、先非线性后线性”的逻辑，1.0Hz 零相位 FIR 滤波器结合后置的一阶导数伪迹拦截，将信号相位畸变降至最低。
    
- **原生多模态特征矩阵**：单次运行即可同步产出静态皮尔逊连通图 (GCN)、时序演化 dFC 张量 (ST-GCN)、STFT 频域热力包络 (2D-CNN) 及 6-qubits 非线性量子特征，为下游多模态融合模型提供零处理可用的标准化输入。
    
- **全局动态调度探针**：集成基于 `tqdm` 的全局 ETA 进度器与 UI 锁，平滑追踪多线程张量运算状态。
    
- **自动化审计探针 (QA Audit)**：旁路生成高密度可视化 PDF 报告，包含 1D 时域演变、2D 频域分布、全局 PSD (Welch's Method) 及图连通性热力图。
    

## 目录结构

```
├── config.py                # 全局配置（2.0s窗长、采样率、通道拓扑等硬性约束）
├── core_transforms.py       # 核心信号算子（双端 Safe-CAR、微积分压摆率限制器）
├── feature_extractors.py    # 特征引擎（集成 Drop Graph 与防 NaN 熔断的 dFC/STFT）
├── models.py                # 深度模型定义（QVAE 量子变分电路与跨框架精度对齐）
├── artifact_remover.py      # 伪迹处理调度（支持 OOM 预防与全局流形映射）
├── eeg_debugger.py          # 信号追踪探针（管线多阶段形态时域对比可视化）
├── visualize_pipeline_stages.py # 质检可视化（1D时域波形与 2D STFT）
├── main.py                  # 流水线主入口（多模态特征装箱与进度追踪）
├── Data/                    # 数据存储隔离区
│   ├── Preprocessed_EEG/    # [Input] 原始被试 .mat 文件
│   ├── QA_Reports/          # [QA Output] 自动化 PDF 质检报告与高阶审计
│   └── EEG_pure/            # [Output] 最终生成的多模态特征包 .mat 文件
└── requirements.txt         # 依赖清单
```

## 快速运行指南

### 1. 环境准备与依赖安装

推荐使用 `conda` 创建纯净的计算环境：

```
# 创建并激活 Python 3.11 独立环境
conda create -n seedem python=3.11 -y
conda activate seedem

# 安装基础科学计算、脑电处理与绘图库
pip install numpy scipy pandas mne joblib scikit-learn matplotlib tqdm

# 安装盲源分离算子与量子线路框架
pip install python-picard pennylane

# 安装 PyTorch (请根据实际 CUDA 版本调整 index-url，此为 CU121 示例)
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

### 2. 数据就绪

将 SEED 原始数据文件放置于 `Data/Preprocessed_EEG/` 目录下。程序具备容错机制，将自动忽略环境干扰文件（如 `label.mat` 或不可见配置文件）。

### 3. 执行管线

```
python main.py
```

_执行期间将通过进度条展示动态吞吐量。运行结束后，在 `Data/EEG_pure/` 获得特征张量，在 `Data/QA_Reports/` 获得诊断级 PDF 审计报告。_

## 数据处理流水线与神经生理学映射

脑电数据矩阵 $X \in \mathbb{R}^{62 \times T}$ 遵循以下严格的物理与数学约束：

1. **去均值 (Zero-mean)**：移除各通道时间序列直流偏置。
    
2. **坏导拓扑修复 (Topology Interpolation)**：基于 SEED 官方实验日志，使用球面样条或均值重构修复已知脱落电极，保障图网络节点几何完整性。
    
3. **频域净化 (FIR & Notch)**：执行 `1.0 ~ 50.0 Hz` 零相位 FIR 滤波，无损修正群延迟畸变，隔离呼吸极低频与市电 $50\text{Hz}$ 工频。
    
4. **双端安全重参考 (Dual-Bound Safe CAR)**：执行 `1e-4 < std < 100.0` 方差带通掩码的共模抑制。
    
5. **适定性盲源分离 (2.0s QVAE/ICA)**：在最优样本特征比下执行流形解耦。QVAE 在全局均值/方差约束下映射，同步提取降维量子潜变量序列 ($Z \in \mathbb{R}^{6 \times T_{clean}}$)。
    
6. **微积分极值拦截 (Slew Rate Limiter)**：使用一阶导数 $\Delta V/\Delta t$ 算子定位硬件爆音并实施局部线性缝合。
    
7. **多模态特征引擎 (Multimodal Extractor)**：基于 $120\mu V$ PtP 动态阈值生成 Drop Graph 掩码，在绝对连续时序上提取静态矩阵、dFC 张量与 STFT 频域包络。
    

## 产出张量标准

### 深度特征矩阵包 (`Data/EEG_pure/S{ID}.mat`)

数据遵循严密的类型定义与形状约束，直接对接 PyTorch DataLoader：

|   |   |   |
|---|---|---|
|**字段名**|**维度/格式**|**说明**|
|`data_pure`|$3 \times 15$ Cell 数组 $\rightarrow 62 \times T$|预处理后的纯净 1D 脑电波形，维持原始时间长度以防马尔可夫链断裂。|
|`adj_matrix`|$3 \times 15$ Cell 数组 $\rightarrow 62 \times 62$|静态皮尔逊连通矩阵，受自适应阈值过滤计算，适合静态 GCN。|
|`dfc_matrix`|$3 \times 15$ Cell 数组 $\rightarrow K \times 62 \times 62$|动态时空功能连通性张量，自带极值方差熔断器，无 `NaN` 泄露危险，适合 ST-GCN。|
|`qvae_latents`|$3 \times 15$ Cell 数组 $\rightarrow 6 \times T$|QVAE 提取的高维量子特征投影流形（6-qubits 压缩），若无依赖则为 None。|
|`stft_features`|$3 \times 15$ Cell 数组 $\rightarrow 62 \times F \times T_{stft}$|STFT 时频张量（$0.5\text{Hz}$ 物理频率分辨率），无吉布斯 Sinc 泄露，适合 2D-CNN。|
|`sfreq`|标量（$200.0$）|采样率元数据锁。|
|`ch_names`|62 元素列表|基于国际 10-20 系统的电极空间拓扑映射表。|