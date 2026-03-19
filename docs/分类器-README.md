# SEED-GCN: Physics-Informed Dynamic Graph Learning for EEG Emotion Recognition

## 项目概述 (Project Overview)

本项目是基于上海交通大学 **SEED (SJTU Emotion EEG Dataset)** 的高保真深度学习分类器。接收上游预处理管线输出的无损脑电张量（静息连通性矩阵与 STFT 时频流形），通过微分熵 (DE) 严格频段降解与非平稳动态图卷积 (DGCN)，执行稳健的三分类情绪识别（积极、中性、消极）。

本系统在底层代数架构上进行了全面重构，系统性解决了传统脑电图网络中的图拉普拉斯负反馈干涉、频域语义塌缩以及折叠验证中的泛化泄露问题。

## 核心防御机制与数学重构 (Core Defenses & Mathematical Formulations)

为了确保情感识别的“高准确率”来源于真实的神经生理时空特征而非工程代码的张量泄露，本管线在数据流转与图消息传递中部署了严格的代数边界：

### 1. 物理语义对齐 (Semantic Frequency Alignment)

传统代码常硬编码 4Hz 步长切片，导致上下游张量错位。本系统自动将 STFT 映射为符合情绪加工特性的 **微分熵 (Differential Entropy, DE)**，并严格绑定 $W_{stft}=2.0s$ 产生的 $\Delta f = 0.5\text{Hz}$ 频域底数，无损聚合 $\delta, \theta, \alpha, \beta, \gamma$ 经典频段，根除了低频能量塌缩。

$$X_{DE} = \log\left( \sum_{f \in Band} Z_{stft}(f)^2 \right) \quad (\text{s.t. } Z_{stft} \ge 10^{-8})$$

### 2. 动态拓扑路由与半正定约束 (Dynamic Topology Routing)

针对静息皮尔逊矩阵 $A$ 无法捕捉短时情绪激化态的局限，提供 `EEG_DGCN` 算子。

- **静态保护**：对传入的经验协方差图执行 $\tilde{A} = |A| + I_N$，强制张量进入绝对值空间，保护图拉普拉斯算子的正向通信增益。
    
- **动态推断**：构建基于降秩投影 ($d=16$) 的稠密非对称状态机。通过张量内积 $A_{dyn} = \text{ReLU}( (X W_{dyn}) (X W_{dyn})^T )$ 生成跨脑区路由，在控制 $\mathcal{O}(N^2)$ 计算复杂度的同时，实现了自适应的空间功能重组。
    

### 3. 环境状态机与确界防线 (State Machine & Bounds)

- **浮点溢出截断**：引入极大值平滑锚点（$\text{smoothed\_counts} = c_i + \max(C) \times 0.05$）计算类惩罚权重，阻断 FP32 在极端长尾样本下的链式梯度爆炸。
    
- **流形初始化锁定**：内置 `seed_everything(42)` 与 CuDNN 确定性引擎，保证动态图的非凸寻优轨迹具有绝对的复现性。
    

### 4. 严格泛化隔离 (Leave-One-Subject-Out Physics Isolation)

废除随机切分，主训练流默认启动 **15-Fold LOSO** 循环验证。在物理层面上切断时间自相关性产生的虚假准确率。并在每折末尾执行 VRAM 显式释放 (`empty_cache`)，杜绝深层计算图碎片泄漏引发的 OOM。

## 目录结构规范

```
SEED_GCN_Classifier/
├── data/                         # [数据泵] 预处理管线产出的 S1~S15.mat 与 label.mat
├── checkpoints/                  # [落盘区] 15折交叉验证的最佳权重 (best_gcn_loso_S{id}.pth)
├── core/
│   ├── dataloader.py             # 核心: STFT->DE 0.5Hz映射与 1s 流形直通截断
│   ├── graph_models.py           # 算子: EEG_GCN 静态拉普拉斯与 EEG_DGCN 动态路由
│   └── cross_validation.py       # 隔离: LOTO/LOSO 迭代工厂
├── train.py                      # 训练流: 包含 15 折 LOSO 循环、平滑类别权重与梯度确界
├── evaluate.py                   # 审计流: 混淆矩阵与显著性图谱溯源
├── config.yaml                   # 注册表: 全局超参数、降维系数与频段确界
└── README.md                     # 本文档
```

## 快速启动 (Quick Start)

### 1. 环境依赖配置

推荐使用支持 CUDA 的环境，构建严格的代数运算底层：

```
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install torch_geometric
pip install numpy scipy pandas scikit-learn matplotlib seaborn pyyaml networkx
```

### 2. 张量摄入要求

请确保 `data/` 目录下放置了经受过上游微积分伪迹拦截器处理的纯净张量：

- `S1.mat ~ S15.mat`（需含 `adj_matrix` 与 `stft_features`，时间切片步长要求 $\Delta t = 1.0s$）
    
- `label.mat`（15 试次情绪向量，自然排序）
    

### 3. 启动全景折叠训练

直接执行 `train.py`。引擎将自动循环遍历 15 名受试者，执行完整的 LOSO 验证，并在每一折独立应用 Early Stopping 保存流形基准。

```
python train.py
```

_注：训练终端将实时输出每一折的加权经验风险极小化（Loss）与 F1-Score 泛化寻优轨迹。_

### 4. 零样本生理映射审计

待训练完毕，载入产生的权重簇，输出多维维度的脑网络可视化审查：

```
python evaluate.py
```

> **生理审计标准**：情绪分类（特别是中性 vs 其他）的高贡献图边应当呈现前额叶（FP1, FP2）与不对称的颞叶（T7, T8）激活。若拓扑矩阵在枕叶（O1, O2）发生重度聚集，提示网络已过拟合于视觉信号残留。

## 张量拓扑流转路径 (Tensor Flow Dynamics)

1. **载入级**：原始时频张量 $Z_{stft} \in \mathbb{R}^{62 \times 26 \times T_{bins}},\ A_{adj} \in \mathbb{R}^{62 \times 62}$。
    
2. **截断级**：丢弃前 20 帧物理时间流形（等价于 20s 静息态视频引导），杜绝基线污染。
    
3. **降解级**：执行 0.5Hz 字典积分映射，产生 DE 特征张量 $X_{de} \in \mathbb{R}^{62 \times 5 \times T_{valid}}$。
    
4. **图通信级**：经过绝对值半正定投影的静态网络 $\tilde{A}$ 或 稠密动态张量乘法产生的 $A_{dyn}$ 进行特征扩散。
    
5. **坍塌级**：全局均值池化（GAP）将 $62$ 空间维度坍塌，输出流形 $H_{graph} \in \mathbb{R}^{1 \times 64}$。
    
6. **映射级**：最终落入三元单纯形概率分布 $Y \in \{0, 1, 2\}$。