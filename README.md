# SEED Dataset EEG Preprocessing & Feature Engineering Pipeline

本项目是针对上海交通大学 **SEED (SJTU Emotion EEG Dataset)** 情绪脑电数据集的预处理与特征工程流水线。项目将原始 MATLAB 脚本全面迁移至 Python 生态（MNE-Python, SciPy, PyTorch, PennyLane），并对传统的信号处理流程进行了结构性优化，以解决基线漂移、滤波振铃效应及空间伪迹扩散等问题。

## 最新更新 (v2.0)

本次更新拓展了流水的特征提取能力，支持输出多种模态的深度学习特征：

- **引入 QVAE 潜变量特征 (Latent Extraction)**：修复了输入未归一化导致的 Tanh 激活层饱和问题。当前支持在去噪的同时，直接提取 6-qubits 量子潜变量序列作为降维后的非线性脑电特征。
    
- **动态功能连通性 (dFC Tensor)**：新增基于滑动窗口（默认 4s 窗长，1s 步长）的时变网络张量提取，适配 ST-GCN 等时空图神经网络。
    
- **自适应分段剔除 (Adaptive AutoReject)**：在计算功能连通性时，引入基于 95% 峰峰值 (PtP) 分位数的动态阈值扫描，自动剔除高方差的异常数据段，提升图邻接矩阵的稳健性。
    
- **多维质量质检 (Advanced QA Audit)**：新增全局功率谱密度 (PSD) 与 Pearson 邻接矩阵热力图的自动化评估与可视化功能。
    

## 核心特性

- **优化的信号处理流程**：遵循“先时域后空域、先非线性后线性”的原则，调整了滤波与极值截断算子的执行顺序，降低 IIR 滤波器面对阶跃信号时引发的振铃效应 (Ringing Effect)。
    
- **混合伪迹处理策略**：结合 1.0Hz FIR 高通滤波、动态方差过滤的 CAR (Common Average Reference) 以及 Post-ICA 时域极值拦截，系统性地处理低频漂移与非平稳的高频局部伪迹。
    
- **动态眼电参考 (Dynamic EOG Anchor)**：QVAE 模式内建方差掩码过滤机制，自适应选取信号质量良好的额叶通道作为眼电参考锚点，避免因电极物理脱落导致模型训练发散。
    

## 目录结构

```
├── config.py                # 全局配置（采样率、通道拓扑、阈值参数等）
├── core_transforms.py       # 核心信号算子（重参考、极值插值等）
├── feature_extractors.py    # 特征提取模块（集成 AutoReject 与 dFC 张量计算）
├── models.py                # 深度模型定义（QVAE 量子变分电路与编码器）
├── artifact_remover.py      # 伪迹处理调度（ICA/QVAE 引擎与 Z-score 归一化）
├── visualize_pipeline_stages.py # 质检可视化（1D时域波形与2D STFT频域热力图）
├── visualize_advanced.py    # 高阶质检可视化（PSD 频谱与图连通性热力图）
├── main.py                  # 流水线主入口（预处理调度与多模态特征保存）
├── Data/                    # 数据存储目录
│   ├── Preprocessed_EEG/    # [Input] 原始被试 .mat 文件存放路径
│   ├── QA_Reports/          # [QA Output] 自动化 PDF 质检报告输出路径
│   └── EEG_pure/            # [Output] 最终生成的特征包 .mat 文件输出路径
└── requirements.txt         # 依赖清单
```

## 快速运行指南

### 1. 环境准备与依赖安装

推荐使用 `conda` 或 `mamba` 管理依赖环境。针对具备 CUDA 加速的硬件环境：

```
# 创建并激活 Python 3.11 独立环境
conda create -n seedem python=3.11 -y
conda activate seedem

# 安装基础科学计算、脑电处理与绘图库
pip install numpy scipy pandas mne joblib scikit-learn matplotlib

# 安装独立成分分析算子与量子线路框架
pip install python-picard pennylane

# 安装 PyTorch (请根据实际 CUDA 版本调整 index-url)
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

### 2. 数据就绪

将 SEED 原始数据文件放置于 `Data/Preprocessed_EEG/` 目录下。程序会自动对文件进行自然排序，并忽略非脑电张量文件（如 `label.mat`）。

### 3. 执行管线

```
python main.py
```

_脚本将在后台进行多进程处理，运行结束后会在 `Data/EEG_pure/` 和 `Data/QA_Reports/` 目录下生成特征矩阵与 PDF 质检报告。若需生成高阶评估图表，可随后运行 `python visualize_advanced.py`。_

## 核心参数与算法设计说明

本管道的关键超参数基于脑电信号的生理特性与数据实际表现进行设定：

1. **FIR 高通截断频率 ( `l_freq = 1.0 Hz` )**
    
    - **说明**：SEED 原始数据中存在较强的低频基线漂移。常规的 0.25Hz IIR 滤波器在处理此类大振幅慢漂移或阶跃信号时易产生过冲（Overshoot）。改用 1.0Hz FIR 零相位滤波可有效移除基线漂移，同时避免相位畸变，为 ICA 算法提供更平稳的输入。
        
2. **Post-ICA 绝对幅值阈值 ( `threshold = 100.0 μV` )**
    
    - **说明**：在 ICA 之前执行幅值硬截断会破坏多通道信号间的线性投影关系，导致 ICA 无法有效分离眼电等成分。因此，本流程将幅值截断后置于 ICA 处理之后，专门用于处理 ICA 难以剥离的单通道、非平稳局部极值（如电极松动引起的瞬态高频伪迹）。
        
3. **自适应 AutoReject ( `Adaptive PtP < 95% Percentile` )**
    
    - **说明**：计算 Pearson 连通性矩阵时，局部的高频肌电或伪迹会显著影响整体协方差的估计。通过计算各 1s 分段的峰峰值 (PtP)，并取 95% 分位数作为剔除阈值，可自适应地过滤掉高方差分段，降低突发伪迹对图连通性矩阵的干扰。
        
4. **动态连通性窗口 ( `window_sec = 4.0, step_sec = 1.0` )**
    
    - **说明**：采用 4 秒窗长与 1 秒步长，用于捕捉大脑情绪微状态（Microstates）的动态演变，生成的 dFC 时空张量可直接作为 ST-GCN 等序列网络模型的输入。
        

## 数据处理流水线

脑电数据矩阵 $X \in \mathbb{R}^{62 \times T}$ 的标准化处理步骤如下：

1. **去均值 (Zero-mean)**：移除各通道时间序列均值，消除直流偏置。
    
2. **坏导插值 (Bad Channel Interpolation)**：基于官方实验日志，使用球面样条或均值插值修复已知脱落的电极。
    
3. **频域滤波 (Bandpass Filtering)**：执行 `1.0 ~ 50.0 Hz` FIR 带通滤波及 `50 Hz` 独立工频陷波处理。
    
4. **安全共平均参考 (Safe CAR)**：在计算全脑均值前，剔除方差趋于零的无效通道，避免异常通道干扰参考信号，随后执行 CAR 重参考。
    
5. **盲源分离与特征降维 (BSS / QVAE)**：基于 40s 滑动窗口分段。默认使用 ICA 提取并置零眼电/心电成分；或启用 QVAE 模块进行流形投影去噪，同时提取 6 维量子潜变量序列。
    
6. **时域极值拦截 (Post-Hoc Truncation)**：对重构后的纯净信号执行 $100\mu V$ 绝对幅值阈值插值，处理残余局部伪迹。
    
7. **多模态特征提取 (Feature Extraction)**：在执行自适应分段剔除后，计算静态图连通矩阵 (GCN)、动态功能连通性张量 dFC (ST-GCN)，以及基于 STFT 的多频段能量包络 (2D-CNN)。
    

## 产出数据标准

### 1. 深度学习特征包 (`Data/EEG_pure/S{ID}.mat`)

每个被试的数据最终被打包为单一的多模态 `.mat` 矩阵文件，可直接用于 PyTorch/TensorFlow 的数据加载，核心字段如下：

|   |   |   |
|---|---|---|
|**字段名**|**维度/格式**|**说明**|
|`data_pure`|$3 \times 15$ Cell 数组 $\rightarrow 62 \times T_{clean}$|预处理后的 1D 脑电时间序列数据。|
|`adj_matrix`|$3 \times 15$ Cell 数组 $\rightarrow 62 \times 62$|静态全窗皮尔逊连通性矩阵，已过滤高方差数据段，适配 GCN。|
|`dfc_matrix`|$3 \times 15$ Cell 数组 $\rightarrow K \times 62 \times 62$|动态时空功能连通性张量 ($K$ 为滑动窗数量)，适配时空动态图网络。|
|`qvae_latents`|$3 \times 15$ Cell 数组 $\rightarrow 6 \times T_{clean}$|QVAE 模型提取的 6 维潜变量序列（当设定 `method='ica'` 时该字段为空）。|
|`stft_features`|$3 \times 15$ Cell 数组 $\rightarrow 62 \times F \times T$|短时傅里叶变换时频包络图张量，表征各频段能量，适配 EEGNet/TSception。|
|`sfreq`|标量（$200.0$）|信号采样率。|
|`ch_names`|62 元素列表|基于 10-20 系统的标准电极名称列表。|

### 2. 自动化质检报告 (`Data/QA_Reports/`)

系统为每次试验 (Trial) 生成图表报告，用于人工评估预处理效果：

- **`Trial_XX_1D_Waveform.pdf`**：1D 时域波形对比，可观测基线漂移的消除与高频极值的处理情况。
    
- **`Trial_XX_2D_STFT.pdf`**：2D 频域热力图，展示 50Hz 工频干扰的消除与背景节律的能量分布。
    
- **`S{ID}_Advanced_Audit.pdf`**：综合拓扑与频域质检，包含静态邻接矩阵热力图，以及基于 Welch 方法的全局 PSD 曲线（用于验证信号的 1/f 特性）。
    

## 补充说明

- **内存管理**：处理较长的连续试次及进行并行 PDF 渲染时，建议环境具备 16GB 以上物理内存。程序内已集成 `gc.collect()` 以优化内存开销。
    
- **QVAE 并行限制**：当启用 `method='qvae'` 时，考虑到 PyTorch 张量操作与 PennyLane 量子模拟器在多进程下的上下文冲突，管线将自动禁用 joblib 的并行加速（自动设定 `n_jobs=1`）。