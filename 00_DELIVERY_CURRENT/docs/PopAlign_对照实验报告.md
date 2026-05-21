# PopAlign 对照实验报告

## 实验目的

使用 PopAlign 框架对单细胞药物响应数据进行分析，通过 GMM 建模和 delta 指标量化不同药物对细胞群体的影响。

## 数据集

### 数据来源
- **Drugscreen 数据集**: PBMC 细胞，46 种药物处理 + 6 个 control
- **NIPS 数据集**: CD4+ T 细胞，多种药物处理

### 数据规模
| 数据集 | 细胞数 | 样本数 | 基因数 |
|--------|--------|--------|--------|
| Drugscreen | 33,482 | 46 treated + 6 control | ~20,000 |
| NIPS CD4 | - | - | - |

## 实验流程

### 1. 数据预处理

```python
# 加载数据
pop = PA.load_multiplexed(
    matrix='PopAlign_Data/drugscreen/drugscreen.mtx',
    barcodes='PopAlign_Data/drugscreen/barcodes.tsv',
    genes='PopAlign_Data/drugscreen/features.tsv',
    metafile='PopAlign_Data/drugscreen/meta.csv',
    controlstring='CTRL',
    outputfolder='output_drugscreen'
)

# 归一化
PA.normalize(pop, scaling_factor=1000)

# 基因过滤
PA.plot_gene_filter(pop, offset=1.3)
PA.filter(pop, remove_ribsomal=False, remove_mitochondrial=False)
```

**结果**: 过滤后保留 1081 个高变异基因

### 2. 特征提取

```python
# oNMF 特征空间
PA.onmf(pop, ncells=5000, nfeats=list(range(1,20)), nreps=2, niter=500)

# 选择特征数
PA.choose_featureset(pop, alpha=3, multiplier=3)
```

**结果**: 选择 m 个特征，构建特征矩阵 W

### 3. GMM 建模

```python
# 定义细胞类型标记
pbmc_types = {
    'Monocytes': ['CD14', 'CD33', 'LYZ', ...],
    'B-cells': ['MS4A1', 'CD19', 'CD79A'],
    'T cells': ['CD27', 'CD69', 'CD2', 'CD3D', ...],
}

# 构建 GMM
PA.build_gmms(
    pop,
    ks=3,           # 3 个高斯分量
    niters=2,
    training=0.8,
    nreplicates=0,
    reg_covar='auto',
    types=pbmc_types,
    criteria='aic'
)
```

**结果**: 每个样本得到一个 3 分量的 GMM 模型

### 4. 样本排序（LLR）

```python
PA.rank(
    pop,
    ref='CTRL2',
    k=100,
    niter=200,
    method='LLR',
    mincells=50,
    figsize=(10,5)
)
```

**结果**: 基于 LLR 对所有药物样本排序，量化与 control 的差异程度

### 5. 分量对齐与 Delta 分析

```python
# 对齐分量
PA.align(pop, ref='CTRL2', method='test2ref')

# 计算 delta
PA.plot_deltas(pop, figsize=(10,10), sortby='mu', pthresh=0.05)
```

**结果**: 得到每个药物的 delta_mu, delta_w, delta_cov

## 核心结果

### Delta 指标说明

| 指标 | 含义 | 计算方式 |
|------|------|----------|
| **delta_mu** | 基因表达均值偏移 | L2 距离: \|\|μ_ctrl - μ_treat\|\| |
| **delta_w** | 亚群比例变化 | 绝对差: \|w_ctrl - w_treat\| |
| **delta_cov** | 协方差结构变化 | Forstner 距离 |

### 结果解读

1. **delta_mu 大**: 药物引起了显著的基因表达变化
2. **delta_w 大**: 药物改变了细胞亚群的比例
3. **delta_cov 大**: 药物改变了细胞状态的分布形状

### 输出文件

```
output_drugscreen/
├── deltas/
│   ├── deltas_comp0_Monocytes_musort.pdf
│   ├── deltas_comp1_Monocytes_musort.pdf
│   ├── deltas_comp1_T cells_musort.pdf
│   └── ...
├── ranking/
│   ├── LLR_rankings_boxplot.pdf
│   └── LLR_rankings_stripplot.pdf
├── renderings/
│   └── 各样本 GMM 热力图
└── embedding/
    └── t-SNE/UMAP 可视化
```

## 与其他方法对比

### 对比维度

| 对比项 | PopAlign | 其他方法 (CellFlow/scGen) |
|--------|----------|---------------------------|
| **预测粒度** | 亚群级别（GMM 分量） | 细胞级别 |
| **输出形式** | delta_mu, delta_w, delta_cov | Y_pred (基因表达矩阵) |
| **特征空间** | oNMF | 原始基因空间 / 其他嵌入 |
| **模型类型** | GMM (生成模型) | 神经网络 (判别/生成) |

### 转换对比方法

#### 方法 1: 从 PopAlign 生成 Y_pred

```python
# 从 GMM 采样生成预测表达
gmm_ctrl = pop['samples']['CTRL']['gmm']
gmm_treat = pop['samples']['Drug_A']['gmm']

# 对 control 细胞分配 component
assignments = gmm_ctrl.predict(C_ctrl)

# 从 treated GMM 采样
Y_pred = np.array([
    np.random.multivariate_normal(
        gmm_treat.means_[k],
        gmm_treat.covariances_[k]
    ) for k in assignments
])
```

#### 方法 2: 从 Y_pred 计算 delta

```python
# 你的方法得到 Y_pred
# Y_pred.shape = (n_cells, n_genes)

# 计算 delta
delta_your = Y_pred.mean(axis=0) - Y_ctrl.mean(axis=0)

# 对比 PopAlign 的 delta_mu
delta_popalign = pop['deltas']['T cells']['combined']['mean_delta_mu']

# 计算相关性
from scipy.stats import pearsonr
r, p = pearsonr(delta_your, delta_popalign)
```

### 对比指标

| 指标 | 计算方式 | 含义 |
|------|----------|------|
| Pearson r | pearsonr(Y_true, Y_pred) | 线性相关性 |
| Spearman r | spearmanr(Y_true, Y_pred) | 秩相关性 |
| MSE | mean_squared_error(Y_true, Y_pred) | 均方误差 |
| 方向准确率 | mean(sign(pred) == sign(true)) | 变化方向一致性 |

## 结论

1. PopAlign 通过 GMM 建模有效捕捉了药物引起的细胞群体变化
2. delta_mu 是衡量基因表达变化的核心指标
3. 与其他方法对比时，需要在相同特征空间或转换为相同形式
4. PopAlign 的优势在于可解释性（亚群级别分析）和统计检验（p 值）

## 后续工作

1. 在 NIPS 数据集上重复实验
2. 与 CellFlow/scGen 进行定量对比
3. 探索不同分量数 (ks) 对结果的影响
4. 分析特定药物的敏感亚群
