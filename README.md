# SEED Dataset EEG Preprocessing & Feature Engineering Pipeline

本项目是针对上海交通大学 **SEED (SJTU Emotion EEG Dataset)** 情绪脑电数据集的预处理与特征工程流水线。项目将原始 MATLAB 脚本全面迁移至 Python 生态（MNE-Python, SciPy, PyTorch, PennyLane），并对传统的信号处理流程进行了结构性优化，以解决基线漂移、滤波振铃效应及空间伪迹扩散等问题。

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
        

## 数据处理流水线与神经生理学推理 (Data Flow & Neurophysiological Reasoning)

脑电数据矩阵 $X \in \mathbb{R}^{62 \times T}$ 从原始采集状态到最终的纯净深度特征，需经历以下严谨的物理与数学阶段：

1. **去均值 (Zero-mean)**：
    
    - **操作**：移除各通道时间序列均值，消除直流偏置。
        
    - **生理学与DSP依据**：脑电放大器电极与头皮接触时会产生巨大的半电池电位（即直流偏置，可高达万微伏量级），这属于硬件噪声而非大脑皮层放电。去均值使信号强制锚定于 $0\mu V$ 的物理基准线，这是防止后续滤波器产生无限振铃（Ringing Effect）的绝对数学前提。
        
2. **坏导插值 (Bad Channel Interpolation)**：
    
    - **操作**：基于官方实验日志，使用球面样条或均值插值修复已知脱落的电极。
        
    - **生理学与DSP依据**：脑电信号在穿透脑膜与颅骨时具有高度的“容积传导效应（Volume Conduction）”，空间上高度连续。物理脱落的电极（表现为平直线或剧烈乱码）会导致空间协方差矩阵的秩塌陷。利用 10-20 系统的拓扑几何关系进行球面样条插值，可在不引入新噪声的前提下，完美修复大脑表面的空间流形，保障下游图神经网络（GNN）的节点完整性。
        
3. **频域滤波 (Bandpass Filtering)**：
    
    - **操作**：执行 `1.0 ~ 50.0 Hz` FIR 带通滤波及 `50 Hz` 独立工频陷波处理。
        
    - **生理学与DSP依据**：情绪认知的核心频段分布在 $\theta, \alpha, \beta, \gamma$ (约 4~50Hz 之间)。低于 1Hz 的极低频主要由受试者出汗（皮电响应）、呼吸等缓慢生理漂移构成；50Hz 则为典型的市电电磁干扰。面对高振幅漂移，经典 IIR 滤波器（如 Butterworth）的反馈环路会产生严重的时域畸变；而 FIR 滤波器是严格的零相位（Zero-phase）线性操作，不仅能有效抽干漂移，更能完美保留不同脑区之间用于计算连通性的“真实神经元放电相位差”。
        
4. **安全共平均参考 (Safe CAR)**：
    
    - **操作**：动态剔除方差趋于零或异常极大的无效通道，随后求取全脑均值并执行减法重参考。
        
    - **生理学与DSP依据**：传统的参考电极（如耳突）不可避免地会拾取环境噪声。CAR 的生理基础假设是“全脑瞬间电位积分趋近于零”，减去均值可以有效去除共模噪声。此处的**“安全（Safe）”**尤为关键：如果某个电极发生严重爆音，传统 CAR 会将该单点噪声除以通道数后“反向注入”给全脑所有健康脑区（引发空间投毒）。本管线内建的动态方差扫描锁彻底阻断了这一衍生污染。
        
5. **盲源分离与特征降维 (BSS / QVAE)**：
    
    - **操作**：基于 40s 滑动窗口分段。使用 ICA 提取并置零眼电/心电成分；或启用 QVAE 模块进行流形投影去噪，并提取 6 维量子潜变量。
        
    - **生理学与DSP依据**：眨眼（EOG）或心跳（ECG）是强大的物理电偶极子，其电位会呈线性扩散至整个头皮。ICA 利用非高斯性最大化原理，能够在多维空间中逆向解耦出这些非脑源性的独立成分。**量子变分自编码器 (QVAE)** 的物理意义更深：复杂的情绪脑电被认为生存在一个低维非线性流形上。将其强制压缩至 6-qubits，迫使网络只学习并保留与大脑认知相关的核心皮层放电模式，而随机的、非流形的肌电噪声会被量子解码器自然过滤。
        
6. **时域极值拦截 (Post-Hoc Truncation)**：
    
    - **操作**：对重构后的纯净信号执行 $100\mu V$ 绝对幅值阈值插值，处理残余局部伪迹。
        
    - **生理学与DSP依据**：真实的头皮脑电（即使在极度兴奋的高唤醒状态下）通常也在 $\pm 80\mu V$ 范围内。残余的瞬态高频突刺大多源于电极微小错位（Electrode Pops）或局部头皮肌肉抽搐（EMG）。此类伪迹高度非平稳且局限于单一导联，ICA 等空间算法极易漏判。**必须后置拦截的原因在于**：如果在 ICA 解耦前进行幅值截断，会破坏眼电等信号的空间线性投射规律（产生平头畸变导致 ICA 失效）。后置处理既保全了空间矩阵的有效性，又保障了最终时域波形的绝对纯净。
        
7. **多模态特征提取 (Multimodal Feature Extraction)**：
    
    - **操作**：在执行自适应分段剔除后，计算静态图连通矩阵 (GCN)、动态功能连通性张量 dFC (ST-GCN)，以及基于 STFT 的多频段能量包络 (2D-CNN)。
        
    - **生理学与DSP依据**：情绪产生本质上是全脑不同网络节点间的动态信息交互。在计算皮尔逊邻接矩阵时，微小的高频极值会导致公式的平方项爆炸，产生虚假的强相关性。**自适应 Sub-Epoch Rejection** 基于 95% 峰峰值分位数动态抛弃抖动纪元，确保 GNN 学习到的是真实的神经元集群同步节律。此外，通过滑动窗口提取的 **dFC (Dynamic Functional Connectivity)** 则精准刻画了大脑情绪微状态（Microstates）在不同时间切片上的演化拓扑轨迹。
        

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

## 自动化质检报告解读指南 (QA Audit Interpretation Guide)

管道运行后会在 `Data/QA_Reports/` 目录下生成一系列 PDF 报告，用于定量评估特征纯净度。以下为各可视化图表的科学解读基准：

### 1. 1D 时域波形图 (`Trial_XX_1D_Waveform.pdf`)

本图表对比了滤波前的原始信号（灰色）与流线输出的纯净信号（蓝色）。

- **零基线对齐 (Zero-Baseline Alignment)**：合格的蓝色信号应严格围绕 $0\mu V$ 中轴振荡。若蓝色波形呈现数秒级的缓慢偏离，表明 FIR 高通滤波未能有效清除游走基线 (Wandering Baseline)。
    
- **生理级振幅界限 (Amplitude Bound)**：健康脑电波的极值通常分布在 $\pm 20 \sim 80\mu V$ 区间。若蓝色波形仍存在 $> 150\mu V$ 的突刺，表明 ICA 遗漏了单通道的物理爆音 (Electrode Pop)。本管线已通过 Post-ICA 算子将其控制在安全范围内。
    
- **无硬性截顶 (No Flat-topping)**：蓝色波形的波峰应保持圆润连续。如果在图谱中观察到类似被“一刀切”的绝对水平直线，说明时域截断阈值设置过低，破坏了相位信息的连续性。
    

### 2. 2D STFT 时频热力图 (`Trial_XX_2D_STFT.pdf`)

本图表反映了 62 个通道在时间-频率维度的能量分布。

- **低频区纯净度 (0-5 Hz)**：若底部区域呈现出持续的、极高能量的“红色/黄色”横带，通常是阶跃信号经过滤波器后产生的吉布斯振铃效应 (Gibbs Phenomenon) 或未清洗干净的眼电干扰 (EOG)。理想状态应表现为正常的背景低能级。
    
- **50 Hz 陷波隔离带**：图谱顶部 $50\text{Hz}$ 处必须存在一条贯穿时间轴的平直暗带（深蓝色），代表市电工频干扰被有效抑制。
    
- **宽带高频伪迹 (Broadband EMG)**：若图中存在贯穿全频段（从 0 到 50Hz）的垂直高亮窄条纹，此为典型的瞬态肌肉收缩伪迹或电极抖动。
    

### 3. 全局功率谱密度 (`S{ID}_Advanced_Audit.pdf` - 右侧面板)

基于 Welch 方法计算的全局 PSD 曲线，用于验证信号的宏观频率特性。

- $1/f$ **幂律特性 (Pink Noise Law)**：红色的全局平均能量曲线应当呈现“左高右低”的平滑指数衰减规律。低频 $\delta, \theta$ 频段能量最高，向高频 $\beta, \gamma$ 频段单调递减。
    
- **陷波断崖**：曲线在 $50\text{Hz}$ 处应呈现尖锐的“V字形”跌落，验证空间中不存在残留的谐波能量。若高频尾部 ($>30\text{Hz}$) 不降反升或呈现扁平状，则提示严重的肌电 (EMG) 污染。
    

### 4. 皮尔逊连通性热力图 (`S{ID}_Advanced_Audit.pdf` - 左侧面板)

展示用于 GNN 训练的静态图连通性矩阵 ($A \in \mathbb{R}^{62 \times 62}$)。

- **容积传导效应区块 (Volume Conduction Blocks)**：矩阵在主对角线附近应呈现出明显的高正相关聚集区块（深红色），如额叶内部（Frontal-Frontal）和枕叶内部（Occipital-Occipital）。这反映了真实的大脑皮层局部耦合特征。
    
- **随机伪相关抑制**：若整个矩阵呈现出随机的“雪花噪点”，或某单行/单列呈现极端的全红/全蓝（全局强相关或强负相关），表明存在高方差伪迹未被剔除。本管线通过 `Adaptive AutoReject` 动态丢弃含极值的样本段，保障了矩阵的拓扑有效性。
    

## 补充说明

- **内存管理**：处理较长的连续试次及进行并行 PDF 渲染时，建议环境具备 16GB 以上物理内存。程序内已集成 `gc.collect()` 以优化内存开销。
    
- **QVAE 并行限制**：当启用 `method='qvae'` 时，考虑到 PyTorch 张量操作与 PennyLane 量子模拟器在多进程下的上下文冲突，管线将自动禁用 joblib 的并行加速（自动设定 `n_jobs=1`）。