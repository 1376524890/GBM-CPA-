# GBM 单细胞药物扰动数据集交接文档

**数据集版本**: v1.0（NIPS-Ready）
**最后更新**: 2026-05-19
**数据负责人**: 裴立昆
**适用范围**: CPA、chemCPA、CRISP、CellOT、scGen、Biolord 等扰动建模方法的统一评估

---

## 1. 数据集概述

### 1.1 数据来源

本数据集来源于两项 GBM（胶质母细胞瘤）单细胞药物扰动研究：
- **GSE148842**: 包含 6 位患者（PW029-PW036, PW040），覆盖 6 种药物（Ana-12, Etoposide, Ispenisib, Panobinostat, RO4929097, Tazemetostat）+ DMSO control
- **GSE226202**: 包含额外患者（PW051xxx, PW052xxx, PW053xxx）和 GS 系列样本，添加 Temozolomide 药物

### 1.2 数据集规模

| 指标 | 数值 |
|------|------|
| 总细胞数 | 169,972 |
| 高变基因数 | 5,000 |
| 患者数 | 21 |
| 药物数 | 7（+ control） |
| Control 细胞 | 91,001 |
| Treated 细胞 | 78,971 |

### 1.3 任务定义

**核心任务**: 预测药物扰动后的基因表达变化（perturbation response prediction）。

给定:
- 某患者的 **control 细胞**的基因表达（X_ctrl）
- 目标**药物**信息（drug embedding / SMILES）

预测:
- 该患者接受该药物处理后细胞的**反事实基因表达**（X_pred）

### 1.4 OOD 设置

**OOD 目标**: PW034 患者 + Panobinostat 药物组合

- PW034 患者从未接受过 Panobinostat 治疗（在 train/valid 中仅接受 Etoposide）
- Panobinostat 在训练中可见（其他患者接受过），但 PW034 × Panobinostat 组合完全未见
- 这是一个 **未见患者-药物组合 OOD**（unseen combination），非未见药物 OOD

---

## 2. 推荐使用的最终文件

### 2.1 主数据文件

| 文件 | 路径 | 用途 |
|------|------|------|
| **GBM_NIPS_Ready.h5ad** | `GBM_NIPS_Ready.h5ad` | **唯一推荐使用的主数据文件** |

### 2.2 Embedding 文件

| 文件 | 路径 | 用途 |
|------|------|------|
| scGPT cell embeddings | 已嵌入 `adata.obsm["X_scGPT"]` | 细胞编码 (512d) |
| scGPT ctrl embeddings | 已嵌入 `adata.obsm["X_scGPT_ctrl"]` | 仅 control 细胞的 scGPT 编码 |
| scGPT pert embeddings | 已嵌入 `adata.obsm["X_scGPT_pert"]` | 仅 treated 细胞的 scGPT 编码 |
| MolFormer drug embeddings | 已嵌入 `adata.obsm["X_MolFormer"]` | 细胞级药物编码 (768d) |
| MolFormer drug-level | `GBM_molformer_drug_emb.parquet` | 药物级 embedding (6×768) |
| MolFormer metadata | `GBM_molformer_drug_emb.metadata.json` | SMILES↔药物名映射 |

### 2.3 Aligner 模型文件

| 文件 | 路径 | 用途 |
|------|------|------|
| Aligner (all cells) | `GBM_scGPT_aligner.pt` | scGPT→gene space MLP |
| Aligner (ctrl only) | `GBM_scGPT_aligner_ctrl.pt` | scGPT ctrl→gene space MLP |
| Aligner (pert only) | `GBM_scGPT_aligner_pert.pt` | scGPT pert→gene space MLP |

### 2.4 已训练模型和预测

详见第九节 CPA baseline 结果和文件清单。所有 CPA 变体模型和预测均可用作对照。

---

## 3. AnnData 结构说明

### 3.1 `.X` — log1p 归一化表达矩阵

- 格式: `scipy.sparse.csr_matrix`, dtype=float32
- 含义: **log1p 归一化的基因表达值**
- 值范围: [0.064, 9.083]
- 无 NaN/Inf

### 3.2 `.layers["counts"]` — 原始 UMI 计数

- 格式: `scipy.sparse.csr_matrix`, dtype=int32
- 含义: **原始 UMI counts**（非负整数）
- 值范围: [1, 5292]
- CPA baseline (M0, M4) 使用 counts 作为输入

### 3.3 `.obs` — 细胞元数据（关键字段见第 4 节）

### 3.4 `.obsm` — 细胞级 embedding

| Key | Shape | 含义 |
|-----|-------|------|
| `X_scGPT` | (169972, 512) | 所有细胞的 scGPT embedding |
| `X_scGPT_ctrl` | (169972, 512) | 仅 control 细胞非零的 scGPT embedding |
| `X_scGPT_pert` | (169972, 512) | 仅 treated 细胞非零的 scGPT embedding |
| `X_MolFormer` | (169972, 768) | 细胞级 MolFormer 药物 embedding；control 全零 |

### 3.5 `.uns` — 非结构化元数据

| Key | 含义 |
|-----|------|
| `top50_DEGs` | 每个 treated group 的 top 50 差异表达基因 (dict) |
| `drug_smiles` | 药物名→SMILES 映射 (dict, 7 entries) |
| `drug_embeddings` | 药物级 embedding 信息 |
| `hvg` | 高变基因选择信息 |
| `log1p` | log1p 变换记录 |
| `ood_split` | OOD split 定义 |

---

## 4. 关键字段说明表

### 4.1 数据组织字段

| 字段名 | 位置 | 类型 | 含义 | 必需 |
|--------|------|------|------|------|
| `perturbation` | obs | category | 药物名或 "control" | **是** |
| `condition` | obs | category | 清洗后的药物名（同 perturbation） | **是** |
| `cov_drug_name` | obs | category | `{patient}_{condition}` 组合键 | **是** — 用于 group 聚合 |
| `covariate_patient` | obs | category | 患者 ID | **是** |
| `cell_type` | obs | category | **当前为患者 ID**（非真实细胞类型） | 是（作为 covariate） |
| `dosage` | obs | float64 | 药物剂量（control=0.0） | 推荐 |

### 4.2 Split 和 Control 定义字段

| 字段名 | 位置 | 类型 | 含义 | 必需 |
|--------|------|------|------|------|
| `split` | obs | category | train / valid / ood | **是** |
| `neg_control` | obs | int64 | **1 = control, 0 = treated** | **是** |
| `is_control` | obs | bool | True = control, False = treated | 与 neg_control 完全一致 |
| `is_treated` | obs | category | "control" / "treated" | 辅助字段 |

### 4.3 药物信息字段

| 字段名 | 位置 | 类型 | 含义 | 必需 |
|--------|------|------|------|------|
| `SMILES` | obs | category | 原始 SMILES（control 为空） | 推荐 |
| `canonical_smiles` | obs | category | 规范化 SMILES（control 为空） | 推荐 |
| `drug_smiles` | uns | dict | 药物名→SMILES | 推荐 |
| `top50_DEGs` | uns | dict | `patient\|drug` → top50 DEG genes | **是** (评估) |

### 4.4 Embedding 字段

| Key | 位置 | 类型 | 含义 |
|-----|------|------|------|
| `X_scGPT` | obsm | float32 ndarray | 所有细胞的 scGPT embedding |
| `X_scGPT_ctrl` | obsm | float32 ndarray | 仅 control 非零的 scGPT |
| `X_scGPT_pert` | obsm | float32 ndarray | 仅 treated 非零的 scGPT |
| `X_MolFormer` | obsm | float32 ndarray | 药物 MolFormer embedding |

---

## 5. Split 与 OOD 设置

### 5.1 Split 定义

| Split | 细胞数 | 比例 | 内容 |
|-------|--------|------|------|
| train | 150,564 | 88.6% | 除 valid/ood 外的所有细胞 |
| valid | 16,729 | 9.8% | 随机保留的验证集 |
| ood | 2,679 | 1.6% | PW034 + Panobinostat |

### 5.2 OOD 详细设定

- **OOD group**: PW034_Panobinostat (2,679 cells, 全部 treated)
- **OOD 类型**: **未见患者-药物组合** (PW034 未曾接受 Panobinostat)
- PW034 在 train/valid 中仅出现为 control 和 Etoposide-treated
- Panobinostat 在 train/valid 中对其他患者可见
- **不是未见药物 OOD**，**不是未见患者 OOD**

### 5.3 OOD 的 control 来源

OOD split 中无 control 细胞。反事实预测的 control 来源为 **PW034_control** group（15,288 cells），分布在 train（12,202）和 valid（3,086）中。

### 5.4 重要规则

**禁止重新划分 split！** 所有方法必须使用预定义的 split 列进行 train/valid/ood 划分，以确保基线结果可复现。

---

## 6. Control/Treated 定义

### 6.1 核心规则

```
neg_control == 1  →  control 细胞
neg_control == 0  →  treated 细胞
```

`is_control` (bool) 与 `neg_control` 完全一致。

### 6.2 Matched Control 选择

每个 treated group `{patient}_{drug}` 的 matched control 为同患者的 control group:

```python
treated_group = "PW030_Panobinostat"
patient = treated_group.split("_")[0]  # "PW030"
control_group = f"{patient}_control"   # "PW030_control"
```

共 21 个 control groups（每个患者一个），与 treated groups 配对使用。

---

## 7. 药物与 SMILES 信息

### 7.1 药物列表

| Drug | Canonical SMILES | Treated Cells | 备注 |
|------|-----------------|---------------|------|
| Ana-12 | Cc1ccc(-c2nc3ccccc3n2-c2ccccc2)cc1 | 7,085 | |
| Etoposide | COc1cc2c(cc1O)C1C(=O)OCC1c1cc3c(cc1O2)OCO3 | 36,513 | 最多 treated 细胞 |
| Ispenisib | CC(C)N1CCN(c2ccnc(N3CCOCC3)n2)CC1 | 2,683 | ⚠️ 与 Tazemetostat SMILES 相同 |
| Panobinostat | C=Cc1ccc(-c2cnn(CCN3CCC(C(=O)NO)CC3)c2)cc1 | 22,220 | **OOD 目标药物** |
| RO4929097 | CC(C)(C)OC(=O)N1CCC(n2cnc3ccc(C(F)(F)F)cc32)CC1 | 4,806 | |
| Tazemetostat | CC(C)N1CCN(c2ccnc(N3CCOCC3)n2)CC1 | 4,128 | ⚠️ 与 Ispenisib SMILES 相同 |
| Temozolomide | CN1N=Nc2ncn(n2)C1=O | 1,536 | 仅 GS 系列患者 |

### 7.2 MolFormer Embedding 使用方式

1. **细胞级**: 直接使用 `adata.obsm["X_MolFormer"]`，形状 (169972, 768)
   - control 细胞: 全零向量
   - treated 细胞: 对应药物的 768d MolFormer embedding

2. **药物级**: 读取 `GBM_molformer_drug_emb.parquet`，形状 (6, 768)
   - 索引为 canonical SMILES 字符串（非药物名）
   - 通过 `GBM_molformer_drug_emb.metadata.json` 映射药物名→SMILES

### 7.3 ⚠️ 已知注意事项: Ispenisib 与 Tazemetostat

**Ispenisib 和 Tazemetostat 具有完全相同的 canonical SMILES**，因此 MolFormer embedding 完全相同。在分析这两种药物的结果时需要注意：

- 基于 MolFormer 的方法无法区分这两种药物
- 评估报告中两者应被标注为共享 embedding

---

## 8. Embedding 使用说明

### 8.1 X_scGPT (512d)

所有细胞（control + treated）使用 **blood-derived scGPT 模型权重**编码。存在 domain gap（blood vs GBM），但在扰动预测任务中仍能提供有用特征。

### 8.2 X_scGPT_ctrl (512d)

仅在 **control 细胞**上非零。适用于 CPA M2（control-only scGPT 输入）。treated 细胞位置为零向量。

### 8.3 X_scGPT_pert (512d)

仅在 **treated 细胞**上非零。适用于 CPA M3（perturbation-only scGPT 输入）。control 细胞位置为零向量。

### 8.4 X_MolFormer (768d)

细胞级药物表示。control 细胞为零向量，treated 细胞为对应药物的 MolFormer embedding。

### 8.5 Key 名称兼容性

| 本数据集的 key | NIPS/CRISP 兼容代码可能期望的 key | 处理方式 |
|---------------|----------------------------------|---------|
| `X_scGPT` | `XscGPT` | 在配置文件中指定或代码中添加 alias |
| `X_MolFormer` | `XMolFormer` | 在配置文件中指定或代码中添加 alias |
| `top50_DEGs` | `rank_genes_groups_cov` | 使用 top50_DEGs 并转换 key 格式 |

**不建议直接修改 h5ad 文件。** 各方法在自己的加载代码中处理 key 名称差异。

---

## 9. 不同对照方法如何使用本数据

### 9.1 CPA Baseline (M0)

- **细胞输入**: `layers["counts"]` (原始 UMI)
- **药物输入**: Learnable embedding（模型内部学习）
- **所需字段**: counts, condition, covariate_patient, split, neg_control
- **状态**: ✅ 可直接使用

### 9.2 CPA + scGPT (M1, M2, M3)

- **细胞输入**: `obsm["X_scGPT"]` (或 ctrl/pert) → Aligner MLP → 5000d gene space
- **药物输入**: Learnable embedding
- **所需文件**: aligner .pt 文件 + aligned .npy 文件
- **状态**: ✅ 可直接使用

### 9.3 CPA + MolFormer (M4)

- **细胞输入**: `layers["counts"]`
- **药物输入**: `obsm["X_MolFormer"]` 或 parquet 中的药物级 embedding
- **状态**: ✅ 可直接使用

### 9.4 CPA + scGPT + MolFormer (M5)

- **细胞输入**: `obsm["X_scGPT"]` → Aligner
- **药物输入**: `obsm["X_MolFormer"]`
- **状态**: ✅ 可直接使用

### 9.5 外部方法接入注意事项

| 方法 | 接入要点 |
|------|---------|
| **chemCPA** | 需要 drug SMILES（从 uns["drug_smiles"] 获取）；cell 输入可用 counts；注意 covariate 字段差异 |
| **CRISP** | 需要 DEG dict（使用 top50_DEGs，注意 `\|` vs `_` key 差异）；建议添加 `rank_genes_groups_cov` alias；需指定 control_key=neg_control |
| **CellOT** | 需要 source/target 分布（control → treated per group）；使用 counts 或 X 作为输入 |
| **scGen** | 需要 condition 标签 + counts；可能需要 one-hot 药物编码 |
| **Biolord** | 需要药品 attribute（可从 SMILES 构建）；需要 multi-label 格式 |
| **scVIDER** | 需要 counts + perturbation 标签 |

### 9.6 Embedding 替换矩阵

| 方案 | 细胞编码替换 | 药物编码替换 |
|------|------------|------------|
| CPA M0 | - | - |
| CPA M1 | counts → X_scGPT | - |
| CPA M4 | - | learnable → X_MolFormer |
| CPA M5 | counts → X_scGPT | learnable → X_MolFormer |

---

## 10. 统一评估协议

### 10.1 评估参数

| 参数 | 本数据集的值 |
|------|------------|
| Group key | `cov_drug_name` |
| Control key | `neg_control` (1=control, 0=treated) |
| DEG key | `top50_DEGs`（key 格式: `patient\|drug`） |
| Gene order | `adata.var_names`（固定顺序，5000 genes） |
| Split key | `split` |
| Aggregation | Unweighted macro average over valid groups |

### 10.2 Y_true / Y_pred / Y_ctrl 组织方式

```python
predictions = {
    group_name: {                          # e.g., "PW034_Panobinostat"
        "Y_true":  np.ndarray,             # treated cells × 5000 log1p expression
        "Y_pred":  np.ndarray,             # predicted cells × 5000 log1p expression
        "Y_ctrl":  np.ndarray,             # control cells × 5000 log1p expression
    }
}
```

### 10.3 指标计算协议

1. **Mean-profile metrics** (pearson, r2, mse, pearson_delta):
   - 先对每种细胞取 mean profile，再计算指标
   - Y_true_mean = mean(Y_true, axis=0)
   - Y_pred_mean = mean(Y_pred, axis=0)
   - Y_ctrl_mean = mean(Y_ctrl, axis=0)
   - pearson_delta = corr(Y_pred_mean - Y_ctrl_mean, Y_true_mean - Y_ctrl_mean)

2. **DEG-subset metrics** (pearson_delta_de, r2_de):
   - 先用 group 对应的 top50_DEGs 子集化矩阵，再计算

3. **sinkhorn_de**:
   - 使用 DEG-subset **细胞级矩阵**（不取 mean）
   - 直接计算 Y_pred[:, deg_genes] 与 Y_true[:, deg_genes] 的 Sinkhorn 距离

4. **Final score**: 所有 valid groups 的 **unweighted macro average**

### 10.4 Valid Group 过滤规则

```python
valid = (
    (treated_count > 5) &
    (~group_name.str.contains('dmso|control', case=False)) &
    (group_name in deg_dict) &
    (len(deg_genes) >= 2) &
    (matched_control_count >= 5)
)
```

当前数据集中有 **18 个 valid groups**（含 1 个 OOD group）。

---

## 11. 最小加载示例

```python
import numpy as np
import anndata as ad

# 加载数据
adata = ad.read_h5ad("GBM_NIPS_Ready.h5ad")
print(f"Cells: {adata.n_obs:,}, Genes: {adata.n_vars:,}")

# 读取 counts
counts = adata.layers["counts"]  # 原始 UMI, int32 sparse

# 读取 embedding
X_scGPT = adata.obsm["X_scGPT"]        # (169972, 512)
X_MolFormer = adata.obsm["X_MolFormer"] # (169972, 768)

# 选择 OOD 目标
ood_mask = adata.obs["split"] == "ood"
treated_mask = adata.obs["cov_drug_name"] == "PW034_Panobinostat"
ctrl_mask = adata.obs["cov_drug_name"] == "PW034_control"

# 构造 Y_true, Y_ctrl
Y_true = adata[ood_mask].X.toarray()
Y_ctrl = adata[ctrl_mask].X.toarray()

# 获取 DEG genes
deg_key = "PW034|Panobinostat"  # 注意：使用 | 而非 _
deg_genes = list(adata.uns["top50_DEGs"][deg_key])

print(f"Y_true: {Y_true.shape}, Y_ctrl: {Y_ctrl.shape}")
print(f"DEG genes: {len(deg_genes)}")
```

完整示例见 `load_gbm_example.py`。

---

## 12. 外部方法预测输出格式

### 12.1 Prediction h5ad 文件要求

```python
# 创建一个 prediction h5ad
pred_adata = ad.AnnData(
    X=Y_pred,                           # scipy.sparse.csr_matrix
    obs=control_cells_metadata,         # 预测对应的元数据
    var=adata.var                       # 必须与 GBM_NIPS_Ready 的 var 完全一致！
)
pred_adata.write("method_name_PW034_Panobinostat_pred.h5ad")
```

### 12.2 关键要求

1. **Gene order**: 预测矩阵的列顺序必须与 `GBM_NIPS_Ready.h5ad.var_names` 完全一致
2. **表达空间**: 预测值应为 log1p 归一化表达，与 adata.X 在同一空间
3. **obs 信息**: 应包含 `covariant_patient`, `condition`, `cov_drug_name` 以便追溯

### 12.3 Per-group Predictions Dict 格式

如果需要直接传递 numpy arrays：

```python
predictions = {
    "PW030_Panobinostat": {
        "Y_true": np.array(...),  # (n_treated, 5000)
        "Y_pred": np.array(...),  # (n_pred, 5000)
        "Y_ctrl": np.array(...),  # (n_ctrl, 5000)
    },
    # ... other valid groups
}
```

---

## 13. 常见错误

| # | 错误 | 正确做法 |
|---|------|---------|
| 1 | **重新划分 split** | 必须使用 adata.obs["split"] 的预设值 |
| 2 | **Gene order 不一致** | 预测矩阵列顺序必须与 adata.var_names 完全一致 |
| 3 | **Control 定义反了** | neg_control==1 是 control，==0 是 treated |
| 4 | **DEG key 对不上** | DEG key 使用 `\|` (如 `PW034\|Panobinostat`)，cov_drug_name 使用 `_` |
| 5 | **Embedding key 名不一致** | 本数据用 `X_scGPT`/`X_MolFormer`，非 `XscGPT`/`XMolFormer` |
| 6 | **使用全基因 Pearson 误判模型有效** | 必须使用 DEG-subset 的 pearson_delta_de，而非全基因 pearson |
| 7 | **对不同方法使用不同 valid groups** | 评估时所有方法必须使用相同的 18 valid groups |
| 8 | **混淆预测空间** | M0/M4 预测在 counts 空间，M1/M2/M3/M5 在 ~log1p 空间 |

---

## 14. 当前已知局限

1. **Single-group OOD**: 当前仅 1 个 OOD group (PW034_Panobinostat)，无法计算 OOD macro average。这是一个 case study 级别的 OOD 评估。

2. **scGPT domain gap**: 使用 blood-derived 权重编码 GBM 细胞，存在 domain mismatch。在 DNA 损伤响应等通路上的表示可能不够准确。

3. **Ispenisib/Tazemetostat SMILES 重复**: 两种药物共享相同 SMILES 和 MolFormer embedding，基于 MolFormer 的方法无法区分。

4. **Cell type 信息缺失**: `cell_type` 字段实际存储的是 patient ID。如果需要真实细胞类型注释，需要从原始数据重新获取或通过 marker gene annotation 推断。

5. **DEG 字典命名差异**: 本数据使用 `top50_DEGs` (key: `patient|drug`)，NIPS 标准协议期望 `rank_genes_groups_cov` (key: `patient_drug`)。功能上等价，但需要 key 名称转换。

6. **部分患者无 matched control**: 6 个 GSE226202 患者 (PW051704, PW052703, PW052706, PW052709, PW053707, PW053710) 只有 treated 细胞，没有对应 control。这使得这些 groups 无法参与评估。

7. **GS 系列患者**: 仅接受 Temozolomide，细胞数较少（192-768 cells），matched control 数量有限。

---

## 15. 联系与维护信息

| 项目 | 信息 |
|------|------|
| 数据负责人 | 裴立昆 |
| 最后更新 | 2026-05-19 |
| 文件版本 | v1.0 (NIPS-Ready) |
| 基础路径 | `/home/u2023312303/nature子刊/裴立昆实验` |
| Conda 环境 | `plknature` (anndata 0.12.11, scanpy 1.12.1, torch 2.5.1+cu121) |

### 后续修改规则

1. **请勿直接修改 GBM_NIPS_Ready.h5ad**。如需添加字段或修复，请使用 patch 脚本。
2. 添加新 embedding 时，写入 `obsm` 并使用一致的命名风格。
3. 添加新药物时，同步更新 `uns["drug_smiles"]` 和 MolFormer parquet。
4. 添加新患者时，确保 `cov_drug_name` 命名为 `{patient}_{condition}`，并计算对应的 DEG。
5. 任何修改后请更新本交接文档和 QC 报告。

---

## 附录: 文件清单

### 数据文件
- `GBM_NIPS_Ready.h5ad` — 主数据文件 (871 MB)
- `GBM_Universal_Perturbation_Ready.h5ad` — 预处理前数据 (223 MB)
- `GBM_scGPT_embeddings.h5ad` — 原始 scGPT embeddings (760 MB)
- `GBM_with_embeddings.h5ad` — 中间数据 (871 MB)
- `GBM_X_MolFormer.npy` — MolFormer 矩阵 (522 MB)
- `GBM_molformer_drug_emb.parquet` — 药物级 embedding (0.4 MB)
- `GBM_molformer_drug_emb.metadata.json` — SMILES 映射

### 模型文件
- `GBM_scGPT_aligner.pt` — Aligner MLP (all cells, 126 MB)
- `GBM_scGPT_aligner_ctrl.pt` — Aligner MLP (ctrl only, 126 MB)
- `GBM_scGPT_aligner_pert.pt` — Aligner MLP (pert only, 126 MB)
- `GBM_scGPT_aligned_XscGPT.npy` — 对齐后矩阵 (3400 MB)
- `GBM_scGPT_aligned_XscGPT_ctrl.npy` — 对齐后矩阵 (3400 MB)
- `GBM_scGPT_aligned_XscGPT_pert.npy` — 对齐后矩阵 (3400 MB)

### 模型目录
- `GBM_CPA_model/` — CPA M0 baseline
- `GBM_CPA_MolFormer_model/` — CPA M4 +MolFormer
- `GBM_CPA_scGPT_model/` — CPA M1 +scGPT
- `GBM_CPA_scGPT_ctrl_model/` — CPA M2 +scGPT ctrl
- `GBM_CPA_scGPT_pert_model/` — CPA M3 +scGPT pert
- `GBM_CPA_scGPT_MolFormer_model/` — CPA M5 +scGPT+MolFormer

### 评估结果
- `evaluation_results/` — 9 个 JSON 评估结果文件
- `GBM_CRISP_OOD_metrics.md` — CRISP 格式评估表
- `mlp_comparison_results/` — MLP 对比结果

### 参考文档
- `GBM_QC检查报告.md` — 本次 QC 报告
- `GBM_dataset_manifest.json` — 机器可读数据清单
- `load_gbm_example.py` — 最小加载示例
- `evaluate_gbm_predictions_protocol.md` — 评估接入协议
