# SEED Dataset EEG Preprocessing Pipeline

本项目为基于上海交通大学 SEED（SJTU Emotion EEG Dataset）情绪脑电数据集的标准化预处理流水线。项目实现了从 MATLAB 原始逻辑向现代 Python (`MNE-Python` + `SciPy`) 生态的完整、严格的工程化迁移。

流水线接收初始采集的连续脑电信号，执行自动化伪迹消除与空间滤波，输出具备高信噪比的纯净脑电特征矩阵，供后续情感状态分类与微分熵（DE）提取使用。

## 目录结构规划

```text
.
├── meta_info/                  # 数据集实验材料与通道拓扑说明
├── matlab_legacy/              # 原始 MATLAB (`preprocess_live.m`) 算法参照
├── Data/                       # 计算流挂载目录（由 .gitignore 排除）
│   ├── Preprocessed_EEG/       # [Input] 存放官网下载的原始 45 个 .mat 文件
│   ├── EEG_pure/               # [Output] 自动化输出的清洗后 .mat 文件
│   └── ExtractedFeatures/      # [Output] 预留特征提取目录
├── preprocess_seed.py          # 核心预处理算子与多核流水线
├── requirements.txt            # 依赖环境
└── README.md                   # 项目文档
```

## 技术规格与处理节点

当前 Python 重构版本在严格保证与原始 MATLAB 脚本**数学等效性**的前提下，对底层算子进行了计算性能与收敛性的深度优化：

1.  **不良通道硬插值 (Hardcoded Spherical Spline Interpolation)**
      - 严格映射 `SEED data check report.xlsx` 的实验日志，针对特定批次被试（Subject 1, 2, 13, 16, 37, 43, 45）实施球面样条插值，解决因硬件接触不良产生的坏导问题。
2.  **共平均参考 (Common Average Reference, CAR)**
      - 使用全脑均值重参考，消除系统性直流偏置。
3.  **零相位带通滤波 (Zero-phase Butterworth Bandpass Filter)**
      - $0.25 \sim 50 \text{ Hz}$，二阶 Butterworth 滤波器。通过 `scipy.signal.filtfilt` 消除相位偏移（Phase shift）。
4.  **线性基线漂移校正 (Detrending)**
      - 逐通道移除线性趋势。
5.  **矢量化超幅异常抑制 (Vectorized Artifact Rejection)**
      - 采用 1D 布尔卷积算子（$O(N)$ 复杂度），高效锁定超幅（$\pm130\mu V$）时间点并沿时间轴向外膨胀 `check_step=2` 个样本点进行物理截断。
6.  **双频带独立成分分析与并发去伪 (Dual-Band Windowed ICA with Joblib)**
      - 采用 $40\text{ s}$ 固定时间窗机制。
      - **收敛性优化**：构建 $1.0\text{ Hz}$ 高通副本专用于 `FastICA` 空间解混矩阵的拟合，彻底消除低频漂移引发的 `ConvergenceWarning`。
      - 自动选择额极通道（`FP1`, `FP2`）作为 EOG 代理，客观投射并剔除眼电与眨眼伪迹。
      - 引入 `joblib` CPU 级并发（`n_jobs=-1`），将 ICA 求解耗时压缩至原先的 30% 左右。

## 快速运行指南

### 1\. 环境准备

推荐使用 `conda` 或 `virtualenv` 管理依赖环境。需确保已安装 `scikit-learn` 以支持 MNE 的 FastICA 算法引擎。

```bash
pip install numpy scipy pandas mne joblib scikit-learn

```

*(可选) 您可以直接通过 requirements 导入：*

```bash
pip install -r requirements.txt
```

### 2\. 数据就绪

在项目根目录创建 `Data/Preprocessed_EEG` 目录，并将 SEED 数据集提供的 45 个实验原始 `.mat` 文件置入其中（无需包含 `label.mat`，脚本内建有自动隔离机制）。

### 3\. 执行预处理管线

启动并发清洗流：

```bash
python preprocess_seed.py
```

### 4\. 输出标准

清洗后的文件将以对应的被试编号生成至 `Data/EEG_pure/S{ID}.mat`。
输出 `.mat` 文件的内部层级结构如下：

  - `data_pure`: $3 \times 15$ 的对象数组 (Object Array)，严格映射至 MATLAB Cell 结构（天数 $\times$ 试次）。各单元内置 $62 \times T_{clean}$ 的纯净脑电信号。
  - `sfreq`: 采样率 ($200\text{ Hz}$)。
  - `ch_names`: $62$ 通道 10-20 系统标准命名顺序。

## 运行日志规范

在流水线运行期间，终端将提供进度追踪及边界验证反馈：

  - 若输入数据的通道数非 $62$ 或 试次数非 $15$，系统将主动抛出 `ValueError` 以阻断伪造数据或污染数据的进入。