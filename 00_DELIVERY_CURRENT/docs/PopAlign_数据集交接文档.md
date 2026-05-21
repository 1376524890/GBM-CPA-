# PopAlign 数据集交接文档

## 项目概述

本项目使用 PopAlign 框架分析单细胞药物响应数据，通过构建高斯混合模型（GMM）比较不同药物处理条件下的细胞群体变化。

## 目录结构

```
tutorials/
├── PopAlign_Tutorial_v1.ipynb    # 主实验 notebook
├── PopAlign_Tutorial_v1.py       # Python 脚本版本
├── convert_nips_h5ad_to_popalign.py  # 数据格式转换
├── extract_popalign_deltas.py    # 提取 delta 指标
├── extract_popalign_predictions.py   # 提取预测结果
├── fix_nips_ctrl.py              # 修复 control 数据
├── repair_nips_barcodes.py       # 修复 barcode
├── PopAlign_Data/                # 原始数据
│   ├── drugscreen/               # 药物筛选数据
│   ├── nips_source/              # NIPS 数据源
│   └── nips_popalign/            # 转换后的 PopAlign 格式
├── output_drugscreen/            # 药物筛选输出结果
├── output_nips_cd4/              # NIPS CD4 输出结果
└── docs/                         # 文档目录
```

## 运行流程

1. **数据加载**: `PA.load_multiplexed()` 加载 .mtx + barcodes + features + meta
2. **归一化**: `PA.normalize(pop, scaling_factor=1000)`
3. **基因过滤**: `PA.plot_gene_filter()` + `PA.filter()`
4. **特征提取**: `PA.onmf()` 计算 oNMF 特征空间
5. **特征选择**: `PA.choose_featureset()` 选择最优特征数
6. **GMM 建模**: `PA.build_gmms()` 为每个样本构建 GMM
7. **样本排序**: `PA.rank()` 使用 LLR 排序样本
8. **分量对齐**: `PA.align()` 对齐 control 和 treated 的 GMM 分量
9. **Delta 分析**: `PA.plot_deltas()` 计算并可视化 delta 指标

---

## 附录：pop 对象字段说明

### 顶层字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `pop['samples']` | dict | 所有样本数据，key 为样本名 |
| `pop['order']` | list | 样本名称列表（有序） |
| `pop['genes']` | list | 基因名称列表 |
| `pop['output']` | str | 输出目录路径 |
| `pop['controlstring']` | str | control 样本标识符（如 'CTRL'） |
| `pop['ref']` | str | 参考样本名称 |
| `pop['normed']` | bool | 是否已归一化 |
| `pop['scalingfactor']` | float | 归一化缩放因子 |
| `pop['original_mean']` | float | 归一化前的原始均值 |
| `pop['ncores']` | int | 并行计算核心数 |
| `pop['nreplicates']` | int | 重复实验次数 |

### 基因过滤相关字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `pop['nzidx']` | list | 非零基因索引 |
| `pop['filter_idx']` | list | 过滤后的基因索引 |
| `pop['filtered_genes']` | list | 过滤后的基因名称列表 |
| `pop['filtered_genes_set']` | set | 过滤后的基因名称集合 |
| `pop['genefiltering']` | dict | 基因过滤参数 |
| `pop['genefiltering']['lognzcv']` | array | log(CV) 值 |
| `pop['genefiltering']['lognzmean']` | array | log(mean) 值 |
| `pop['genefiltering']['slope']` | float | 拟合直线斜率 |
| `pop['genefiltering']['intercept']` | float | 拟合直线截距 |

### oNMF 特征空间字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `pop['W']` | array | oNMF 特征矩阵，形状 (n_features, n_genes) |
| `pop['featuretype']` | str | 特征类型，'onmf' 或 'pca' |
| `pop['errors']` | list | 不同特征数的重建误差 |
| `pop['chosen_m']` | int | 选择的特征数 |
| `pop['feat_labels']` | list | 每个特征的 GSEA 标签 |
| `pop['top_feat_labels']` | list | 每个特征的 top GSEA 标签 |

### 样本字段 (`pop['samples'][sample_name]`)

| 字段 | 类型 | 说明 |
|------|------|------|
| `M` | sparse matrix | 原始基因表达矩阵 (n_genes, n_cells)，已 log 变换 |
| `M_norm` | sparse matrix | 过滤后的基因表达矩阵 (n_filtered_genes, n_cells) |
| `C` | array | oNMF 系数矩阵 (n_cells, n_features) |
| `gmm` | GaussianMixture | 拟合的 GMM 模型 |
| `gmm_means` | array | GMM 各分量的均值 (n_components, n_features) |
| `gmm_types` | list | 各分量的细胞类型标签 |
| `replicates` | dict | 重复实验的 GMM 模型 |
| `cell_type` | array | 每个细胞的类型标签 |

### GMM 模型字段 (`pop['samples'][sample]['gmm']`)

| 字段 | 类型 | 说明 |
|------|------|------|
| `gmm.means_` | array | 各分量均值 (n_components, n_features) |
| `gmm.covariances_` | array | 各分量协方差矩阵 (n_components, n_features, n_features) |
| `gmm.weights_` | array | 各分量权重 (n_components,) |
| `gmm.n_components` | int | 分量数 |

### 对齐结果字段 (`pop['samples'][sample]['replicates'][rep]`)

| 字段 | 类型 | 说明 |
|------|------|------|
| `alignments` | array | 分量对齐结果 |
| `gmm` | GaussianMixture | 该重复的 GMM 模型 |
| `gmm_means` | array | 该重复的 GMM 均值 |
| `gmm_types` | array | 该重复的细胞类型标签 |

### Delta 分析字段 (`pop['deltas']`)

| 字段 | 类型 | 说明 |
|------|------|------|
| `pop['deltas']` | dict | 所有细胞类型的 delta 结果 |
| `pop['deltas'][cell_type]` | dict | 特定细胞类型的 delta |
| `pop['deltas'][cell_type]['combined']` | DataFrame | 合并的 delta 指标表 |
| `pop['deltas'][cell_type]['idx']` | array | 排序后的样本索引 |
| `pop['deltas'][cell_type]['orderedsamples']` | list | 排序后的样本名称 |

### combined DataFrame 列说明

| 列名 | 类型 | 说明 |
|------|------|------|
| `origidx` | int | 原始样本索引 |
| `orderedsamples` | str | 样本名称 |
| `mean_delta_mu` | float | 均值偏移（L2 距离），衡量基因表达变化 |
| `pvals_mu` | float | delta_mu 的 p 值 |
| `mean_delta_w` | float | 权重变化，衡量亚群比例变化 |
| `pvals_w` | float | delta_w 的 p 值 |
| `mean_delta_cov` | float | 协方差变化（Forstner 距离） |
| `pvals_cov` | float | delta_cov 的 p 值 |

### LLR 排序字段 (`pop['ranking']`)

| 字段 | 类型 | 说明 |
|------|------|------|
| `pop['ranking']` | dict | LLR 排序结果 |
| `pop['ranking']['scores']` | dict | 每个样本的 LLR 分数 |
| `pop['ranking']['ref']` | str | 参考样本名称 |

### 嵌入字段 (`pop['embedding']`)

| 字段 | 类型 | 说明 |
|------|------|------|
| `pop['embedding']` | dict | 降维嵌入结果 |
| `pop['embedding']['tsne']` | array | t-SNE 坐标 |
| `pop['embedding']['umap']` | array | UMAP 坐标 |

---

## 关键公式

### 归一化
```
g'_i = log(β * g_i / Σg_i + 1)
```

### LLR (Log-Likelihood Ratio)
```
LLR = (1/k) Σ log[L(g_i|θ_ctrl) / L(g_i|θ_sample)]
```

### Delta 指标
```
Δμ_i = ||μ_ref_i - μ_test_j||₂        # 均值偏移
Δw_i = |w_ref_i - w_test_j|            # 权重变化
ΔΣ_i = D_C(Σ_ref_i, Σ_test_j)         # 协方差变化（Forstner距离）
```

---

## 注意事项

1. **基因名称**: 确保提供有效的基因 ID，GSEA 和细胞类型注释依赖正确的基因名
2. **分量数选择**: `ks` 参数直接影响模型质量，建议先用默认值测试
3. **controlstring**: 必须正确设置，用于区分 control 和 treated 样本
4. **内存**: 大数据集建议设置 `ncells` 参数进行子采样
5. **随机性**: GMM 拟合有随机性，结果可能因随机种子不同而略有差异
