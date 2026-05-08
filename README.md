# SEED-PhysioGCN: Physics-Informed End-to-End EEG Emotion Recognition Pipeline

## 项目概述

**SEED-PhysioGCN** 是针对上海交通大学 SEED (SJTU Emotion EEG Dataset) 构建的端到端 EEG 情绪识别系统。系统将原始微伏级时间序列经过物理净化、频图张量构建、任务感知伪迹建模、双流动态图卷积分类与生理归因审计，收敛为一条可运行、可复现、可审计的完整工程链路。

项目强调三条底线：

- **时间轴连续性**：伪迹修复、窗口切分或滤波过程不得破坏物理时间轴。
- **拓扑一致性**：图拓扑、STFT、DE 与审计图必须共享一致的 16 导联节点口径与时间步长。
- **协议隔离性**：训练、验证、测试与目标被试校准遵守明确的嵌套泛化隔离协议，不混用统计量。

当前本地实现收口到 **16 导联核心拓扑**作为默认工程口径。

## 架构总览

```
SEED-PhysioGCN/
├── configs/
│   ├── train_config.yaml          # 主训练超参配置
│   └── prep_config.py             # 预处理全局常量 (采样率、通道拓扑)
├── engine_prep/                   # 预处理引擎
│   ├── transforms.py              # 信号级物理算子 (压摆率拦截、坏导插值)
│   ├── extractors.py              # 特征提取 (Pearson图、dFC、STFT)
│   ├── artifact.py                # 伪迹隔离调度 (QVAE / ICA 路由)
│   └── task_aware_selector.py     # Phase A ComponentSelector 定义
├── engine_quantum/                # 量子变分自编码器
│   ├── qvae_net.py                # QVAE 网络拓扑 (PyTorch + PennyLane)
│   └── pretrain_stream.py         # QVAE 流式泛化预训练
├── engine_gnn/                    # 图神经网络引擎
│   ├── graph_operators.py         # EEG_GCN / EEG_DGCN 模型定义
│   ├── dataloader.py              # 图数据泵 (SEEDGraphDataset)
│   ├── cv_router.py               # LOSO / LOTO 交叉验证路由
│   └── classifier_taskaware_adapter.py  # 任务感知适配器 (可微STFT→DE、损失聚合)
├── tools_audit/                   # 审计工具
│   ├── evaluate_loso.py           # 全局混淆矩阵、显著性拓扑、UMAP 审计
│   ├── probe_topology.py          # 特征拓扑 QA 看板
│   └── probe_1d2d.py              # 波形与 STFT 可视化 QA
├── run_pipeline.py                # Phase I 入口: 原始.mat → 16-ch 特征落盘
├── run_classifier.py              # Phase I 入口: LOSO 主训练循环
├── train_phase_a_selector.py      # Phase II 入口: 任务感知协同微调
├── finetune_calibration.py        # Phase III 入口: 极小样本超球面校准
├── preprocess_seed.py             # [遗留] 62-ch ICA 预处理脚本 (已被 run_pipeline.py 取代)
└── data/
    ├── 01_raw_mat/                # SEED 官方原始 .mat 文件
    ├── 02_pure_features/          # 预处理后特征落盘 (S01.mat ~ S15.mat)
    ├── 03_qa_reports/             # QA 可视化报告
    ├── 04_weights/                # QVAE 预训练权重
    └── 05_checkpoints/            # LOSO 模型 checkpoint
```

## 核心数据流

### 预处理管道 (`run_pipeline.py`)

```
原始 .mat (62-ch, ~200Hz)
  │
  ├─ fix_hardcoded_bads()          # SEED 官方实验日志坏导球面样条插值
  ├─ DC 剥离 (逐通道去均值)
  ├─ intercept_gradient_spikes()   # 一阶导数压摆率拦截 (阈值 50 μV/sample)
  ├─ 带通滤波 1-50 Hz (FIR, zero-phase)
  ├─ 50 Hz 工频陷波
  ├─ Safe-CAR (合法通道子集共同平均参考)
  ├─ 拓扑收口 → 16 导联核心子集
  │
  ├─ QVAE 窗口级伪迹清洗 (CPU 纯态量子模拟)
  │   ├─ z_raw:       QVAE 原始潜变量 (6 × T)
  │   ├─ z_clean:     伪迹感知加权后潜变量 (6 × T)
  │   ├─ artifact_score:  基于 frontal-EOG 相关性的伪迹先验 (6 × T)
  │   ├─ retain_prob:     组件保留概率 (6 × T)
  │   └─ hard_mask:       二值审计掩码 (6 × T)
  │
  ├─ compute_connectivity_matrix()  # 静态 Pearson 图 (16 × 16)
  ├─ compute_dfc_matrix()           # 动态功能连通图 (K × 16 × 16), Drop Graph 兜底
  ├─ compute_stft_features()        # STFT 包络 (2s 窗, 0.5Hz 分辨率)
  ├─ stft_to_node_de()              # 5 频带微分熵 (16 × 5 × T_bins)
  │
  └─ 落盘 S{ID}.mat (3 天 × 15 trials Cell Array)
```

### 落盘张量标准 (`data/02_pure_features/S{ID}.mat`)

| 字段 | 维度 | 含义 |
|---|---|---|
| `data_pure` | `3×15` Cell → `16×T` | 清洗后 16-ch 波形 |
| `adj_matrix` | `3×15` Cell → `16×16` | 静态 Pearson 连通图 |
| `dfc_matrix` | `3×15` Cell → `K×16×16` | 动态功能连通图 (窗口异常时注入单位矩阵) |
| `node_de` | `3×15` Cell → `16×5×T_bins` | **分类器主输入**: 5 频带微分熵 |
| `stft_features` | `3×15` Cell → `16×F×T_stft` | STFT 幅值包络 |
| `qvae_latents` / `qvae_latents_clean` | `3×15` Cell → `6×T` | 清洗后 QVAE 潜变量 |
| `qvae_latents_raw` | `3×15` Cell → `6×T` | 原始 QVAE 潜变量 (Phase A 用) |
| `artifact_scores` | `3×15` Cell → `6×T` | 伪迹先验强度 |
| `retain_probs` | `3×15` Cell → `6×T` | 组件保留概率 |
| `hard_masks` | `3×15` Cell → `6×T` | 二值审计掩码 |
| `ch_names` | 16 元素列表 | 核心导联拓扑名 |
| `sfreq` | 标量 200.0 | 采样率 |
| `band_names` | 5 元素列表 | `['delta', 'theta', 'alpha', 'beta', 'gamma']` |

### 频带定义

| 频带 | STFT bin 索引 | 物理频率范围 |
|---|---|---|
| delta | `[2, 8)` | 1-4 Hz |
| theta | `[8, 16)` | 4-8 Hz |
| alpha | `[16, 26)` | 8-13 Hz |
| beta | `[26, 60)` | 13-30 Hz |
| gamma | `[60, 90)` | 30-45 Hz |

bin 索引基于 2s 窗长 STFT (n_fft=400, freq_resolution=0.5Hz/bin)。

## 模型架构

### EEG_DGCN: 双流解耦动态图卷积网络

```
Input: x ∈ R^(B×N×5)  (N=16 节点, 5 频带 DE)
  │
  ├─ BandRegionAttention         # 脑区先验频带加权 (FRONTAL→alpha, TEMPORAL→theta)
  │
  ├─ Early Split ───────────────┬─ P_emo: Linear(5→H)   情绪子空间
  │                             └─ P_subj: Linear(5→H)  被试子空间
  │
  ├─ Dual-Stream Topology ─────┬─ 情绪流: Q_E/K_E + global_adj_logits_E + 0.5·I → A_emo
  │                             └─ 被试流: Q_S/K_S + global_adj_logits_S + 0.5·I → A_subj
  │
  ├─ Message Passing (×2) ─────┬─ 情绪流: Linear → A_emo·h → LayerNorm + Residual
  │                             └─ 被试流: Linear → A_subj·h → LayerNorm + Residual
  │
  ├─ GRU Temporal Encoding ────┬─ z_emo = GRU_E(h_E)   时序情感表征
  │                             └─ z_subj = GRU_S(h_S)  时序被试表征
  │
  ├─ Classification Heads ─────┬─ classifier_emo:  z_emo → 3 类情绪
  │                             ├─ classifier_trait: z_subj → 15 被试 (辅助)
  │                             └─ classifier_adv:  CDAN(z_emo⊗p, GradReverse) → 15 被试
  │
  └─ Output: (out_emo, out_trait, out_adv, z_emo, z_subj, A_emo, A_subj)
```

关键设计：
- **自环先验**: `0.5·I` 注入对角线，防止深层图扩散 over-smoothing
- **LayerNorm** (非 BatchNorm): 阻断跨被试拼接带来的全局动量污染
- **CDAN 条件对抗**: `z_emo ⊗ softmax(out_emo.detach())` 构造条件特征后经 GRL 反转
- **GRU 时序编码**: 将节点均值按时间窗口 reshape 后捕获渐进性情感流形

## 损失函数

`run_classifier.py` 中的主训练损失由 6 项组成：

```
L_total = L_emo                           # 情绪分类交叉熵 (label_smoothing=0.1)
        + γ · L_trait                      # 被试判别 (辅助任务, 对抗性)
        + α · L_adv                        # 被试混淆 (GRL 梯度反转)
        + 0.1 · L_ortho                    # 正交排斥 (off-diagonal cosine)
        + 0.5 · L_supcon                   # 监督对比损失 (SupCon, τ=0.15)
        + λ_artinv · L_artinv              # 伪迹不变性 (z_emo 与 artifact_score 去相关)
```

其中：
- `α` 和 `γ` 由退火调度控制: 3 epoch warmup 后以 sigmoid 曲线从 0 增长到 `α_max=0.15`, `γ_max=ln3/ln15`
- `λ_artinv` 默认 0.2 (YAML 未显式配置时由代码注入)
- `L_ortho` 仅对双流邻接矩阵的非对角元素计算余弦相似度
- `L_supcon` 对同类别样本的隐空间表征施加拉近约束

## 交叉验证协议

### LOSO (Leave-One-Subject-Out)

15-fold 嵌套交叉验证: `13(Train) + 1(Val) + 1(Test)`

```
For test_subject_id in 1..15:
    val_subject_id = (test_subject_id % 15) + 1
    Train: 13 subjects × 3 days × 15 trials
    Val:   1 subject  × 3 days × 15 trials  (Early Stopping 依据)
    Test:  1 subject  × 3 days × 15 trials
```

- 每个 trial 被切为非重叠的 3s 窗口图样本 (`temporal_window=3`)
- 边构建: `|adj[i,j]| > 0.3` 时保留边, 权重为 `|adj[i,j]|`
- `subj_mean / subj_std` 仅从训练+验证被试计算，测试被试统计量不参与
- 前 20 个 DE bins (warmup 阶段) 被截断

### 数据增强细节

`SEEDGraphDataset` 的图构建流程：
1. 从 `node_de` 加载 `(16, 5, T_bins)` 张量
2. 截断前 20 bins (warmup)
3. 按被试训练集统计量归一化
4. 每 3 个连续 bins 切一个图: `x = concat(de[:,:,t], de[:,:,t+1], de[:,:,t+2])` → `(48, 5)`
5. 边索引按时间窗口复制: `edge_index_t = cat([edge_index + t*16 for t in 0..2])`

## 训练流程

### Phase I: 泛化冷启动

```bash
# 1. QVAE 量子基准预训练 (CPU 纯态模拟)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m engine_quantum.pretrain_stream \
    --epochs 15 --batch_size 128 --beta_max 0.5

# 2. 预处理特征装箱 (QVAE 伪迹清洗)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_pipeline.py --artifact_method qvae

# 3. LOSO 主训练
python run_classifier.py
```

产出: `data/05_checkpoints/best_gcn_loso_S{01..15}.pth` + `loso_summary.json`

### Phase II: 任务感知协同微调

```bash
# 4. Phase A 协同微调 (Selector + Proxy 联合优化)
python train_phase_a_selector.py \
    --proxy_ckpt data/05_checkpoints/best_gcn_loso_S01.pth \
    --qvae_ckpt data/04_weights/qvae_pretrained.pt

# 5. 二次预处理 (使用更新后的 Selector)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_pipeline.py --artifact_method qvae

# 6. 最终 LOSO 训练
python run_classifier.py
```

Phase A 的两阶段微调:
- **A2 (Selector-only)**: 冻结 Proxy GNN，仅训练 `ComponentSelector` (10 epochs)
- **A3 (Joint)**: 解冻 Proxy 分类头，Selector 与 Proxy 联合优化 (20 epochs)

Selector 正则化损失: `L_reg = λ_sparse·L1(p) + λ_smooth·Var(p) + λ_anticheat·(p ⊙ art_prior)`

### Phase III: 极小样本超球面校准

```bash
# 7. 路线 B: 被试特异性原型对齐
python finetune_calibration.py --subject_id 10
```

协议 (与 LOSO 主训练完全解耦):
1. **分层采样**: 从目标被试 45 trials 中每类随机抽 1 个作为 calibration anchor
2. **转导式归一化**: 使用目标被试全部 45 trials 的物理数据计算 `subj_mean / subj_std`
3. **冻结网络**: 全部参数 `requires_grad=False`
4. **超球面投影**: 提取 `z_emo` → L2 归一化 → 计算类原型中心 `C_k`
5. **余弦推理**: `argmax_k cosine(z_eval, C_k)` 无参数分类

## 当前训练结果

### LOSO 主链 (15-Fold)

基于 `data/05_checkpoints/loso_summary.json`:

| 被试 | Val F1 | Test F1 |
|---|---|---|
| S01 | 0.4226 | 0.4613 |
| S02 | 0.5360 | 0.3871 |
| S03 | 0.6975 | 0.6823 |
| S04 | 0.7284 | 0.6882 |
| S05 | 0.6816 | 0.6028 |
| S06 | 0.7526 | 0.7613 |
| S07 | 0.9136 | 0.6135 |
| S08 | 0.7212 | 0.7903 |
| S09 | 0.4947 | 0.7435 |
| S10 | 0.7696 | 0.5026 |
| S11 | 0.5918 | 0.7090 |
| S12 | 0.7165 | 0.6359 |
| S13 | 0.6834 | 0.5356 |
| S14 | 0.6568 | 0.5983 |
| S15 | 0.4527 | 0.6067 |
| **Mean** | — | **0.6212** |

观察:
- 跨被试方差极大 (0.3871 ~ 0.7903)
- S07 存在严重过拟合 (val=0.91 vs test=0.61)
- 主误差集中在 Negative ↔ Neutral 双向混淆

### 路线 B 校准 (Target: S10)

- Zero-Shot F1: 0.5938
- Prototype Post-Calibration F1: 0.3073
- 原型间相似度: cos(C_0, C_2) ≈ 0.9944, cos(C_1, C_{0/2}) ≈ 0.60

结论: 单-trial 原型在冻结隐空间中不足以代表类别流形 (Trial-Specific Bias)，需考虑跨域协方差对齐等策略。

## 配置说明

### `configs/train_config.yaml`

```yaml
eeg_semantics:
  num_channels: 16        # 核心导联拓扑
  num_classes: 3          # 消极(0), 中性(1), 积极(2)
  fs: 200                 # 采样率

model:
  type: "EEG_DGCN"        # 主模型 (可选: EEG_GCN)
  hidden_channels: 64
  dropout_rate: 0.5

train:
  batch_size: 128
  epochs: 150
  learning_rate: 0.0001
  weight_decay: 0.0005
  early_stopping_patience: 20

phase_a:
  batch_size: 4
  epochs_a2: 10           # Selector-only 阶段
  epochs_a3: 20           # Joint 微调阶段
  temporal_window: 3      # 图时间窗口 (秒)
  edge_threshold: 0.3     # 边阈值
  drop_initial_bins: 20   # warmup 截断
```

注意: `lambda_artinv=0.2` 由 `run_classifier.py` 代码默认注入，YAML 中未显式配置。

### `configs/prep_config.py`

```python
FS = 200.0               # 采样率 (Hz)
WINDOW_SEC = 400.0        # QVAE 伪迹清洗窗口 (秒，覆盖完整 trial)
OUTLIER_THRESHOLD = 130   # 绝对幅值截断阈值 (μV)
TARGET_NODES = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7',
                'C3', 'CZ', 'C4', 'T8', 'PZ', 'O1', 'OZ', 'O2']
```

## 环境配置

```bash
conda create -n seedem python=3.11 -y
conda activate seedem
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric pennylane python-picard
pip install numpy scipy pandas mne joblib scikit-learn matplotlib seaborn pyyaml networkx tqdm umap-learn
```

将 SEED 官方 `S1.mat ~ S15.mat` 与 `label.mat` 放入 `data/01_raw_mat/`。

## 关键设计约束

修改以下任一参数必须重新评估整条链:

- 压摆率阈值 (默认 50.0 μV/sample)
- STFT 窗长 (默认 2.0s, 400 samples)
- DE 频段 bin 索引边界
- LOSO 的 13+1+1 隔离路由
- 16 导联核心节点集合
- 前 20 bins warmup 截断

## 遗留文件说明

- `preprocess_seed.py`: 早期 62 通道 ICA 预处理脚本 (MATLAB 对齐版本)，已被 `run_pipeline.py` 取代
- `docs/`: 内部设计文档 (`math.md`, `分类器-README.md`, `预处理-README.md`)
- `meta_info/`: SEED 官方通道坐标与实验元数据
- `weights/`: `pretrain_stream.py` 生成的兼容旧推理路径的 state_dict 副本
