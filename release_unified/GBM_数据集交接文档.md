# GBM 单细胞药物扰动数据集交接文档

**版本**: v1.1_unified_compatible
**最后更新**: 2026-05-19
**数据负责人**: 裴立昆
**唯一推荐文件**: `GBM_NIPS_Ready.h5ad`

---

## 0. 最容易误用的 8 点

在阅读本文件其余内容之前，请先确认以下关键事实：

1. **只使用 GBM_NIPS_Ready.h5ad 一个最终版本**。该文件已同时包含原始字段和兼容 alias 字段。不需要在"原始版"和"兼容版"之间选择或切换。

2. **OOD 是 PW034_Panobinostat 未见组合，不是完全未见患者**。PW034 的 control 和 Etoposide 细胞存在于 train/valid；Panobinostat 也在其他患者的 train/valid 中存在。本任务是 unseen patient-drug combination OOD。

3. **split 禁止重新划分**。必须使用 `adata.obs["split"]` 的预设 train/valid/ood 值。

4. **neg_control == 1 是 control，neg_control == 0 是 treated**。不要搞反。

5. **OOD split 中没有 control 细胞**。反事实预测的 matched control 为 train/valid 中的 PW034_control (15,288 cells)。

6. **rank_genes_groups_cov 和 top50_DEGs 都已存在**。前者 key 为 `patient_drug` (与 cov_drug_name 一致)，后者 key 为 `patient|drug`。内容完全一致，评估优先使用前者。

7. **X_scGPT 和 XscGPT、X_MolFormer 和 XMolFormer 都已存在**。两组 alias 内容完全一致。

8. **M0/M4 legacy prediction 是 counts space**。统一评估前必须有 `Y_pred_eval = np.log1p(np.maximum(Y_pred_counts, 0))`。论文主表使用 log1p space。

---

## 1. 数据集概述

### 1.1 数据来源

两项 GBM（胶质母细胞瘤）单细胞药物扰动研究的合并数据集：
- **GSE148842**: 6 位患者 (PW029-PW036, PW040)，6 种药物 + DMSO control
- **GSE226202**: 额外患者 (PW051xxx, PW052xxx, PW053xxx, GS 系列)，含 Temozolomide

### 1.2 规模

| 指标 | 数值 |
|------|------|
| 总细胞数 | 169,972 |
| 高变基因数 | 5,000 |
| 患者数 | 21 |
| 药物数 | 7 (+ control) |
| Control 细胞 | 91,001 |
| Treated 细胞 | 78,971 |

### 1.3 任务定义

预测药物扰动后的基因表达变化 (perturbation response prediction):
- 输入: 某患者的 control 细胞表达 + 目标药物信息
- 输出: 该患者接受该药物后的反事实基因表达

### 1.4 OOD 设置

**OOD 目标**: PW034_Panobinostat

当前 GBM OOD 目标为 PW034_Panobinostat，即 PW034 患者接受 Panobinostat 的组合在 train/valid 中未出现。PW034 患者本身不是完全未见，PW034 的 control 和 Etoposide 细胞存在于 train/valid；Panobinostat 药物本身也不是完全未见，其在其他患者中存在于 train/valid。因此本任务是 **unseen patient-drug combination OOD**，**不是 strict unseen patient OOD**，也**不是 strict unseen drug OOD**。

---

## 2. 唯一推荐数据文件

**GBM_NIPS_Ready.h5ad** 是本数据集的唯一推荐使用文件。

该文件已同时保留原始字段和兼容 alias 字段，所有方法统一使用此文件：

| 类别 | 原始 Key | 兼容 Alias Key | 说明 |
|------|---------|---------------|------|
| 细胞 embedding | `obsm["X_scGPT"]` | `obsm["XscGPT"]` | scGPT cell embedding (512d) |
| 细胞 embedding (ctrl) | `obsm["X_scGPT_ctrl"]` | - | 仅 control 非零 |
| 细胞 embedding (pert) | `obsm["X_scGPT_pert"]` | - | 仅 treated 非零 |
| 药物 embedding | `obsm["X_MolFormer"]` | `obsm["XMolFormer"]` | MolFormer drug embedding (768d) |
| DEG 字典 | `uns["top50_DEGs"]` | `uns["rank_genes_groups_cov"]` | key 分隔符不同，内容一致 |
| 版本元数据 | - | `uns["release_metadata"]` | 数据集版本信息 |

**不需要**在"原始 h5ad"和"兼容 h5ad"之间选择。所有方法统一加载 GBM_NIPS_Ready.h5ad。

---

## 3. AnnData 结构

### 3.1 `.X`

- 格式: `scipy.sparse.csr_matrix`, dtype=float32
- 含义: **log1p 归一化的基因表达值**
- 值范围: [0.064, 9.083] (非零存储值)
- 无 NaN/Inf
- **重要**: 是 CSR sparse matrix，对 sparse.data 的 min/max 仅反映非零存储值，不包含隐式零

### 3.2 `.layers["counts"]`

- 格式: `scipy.sparse.csr_matrix`, dtype=int32
- 含义: **原始 UMI counts** (非负整数)
- 值范围: [1, 5292] (非零存储值)
- 无 NaN/Inf
- CPA baseline (M0, M4) 使用 counts 作为输入

### 3.3 `.obsm`

| Key | Shape | 含义 |
|-----|-------|------|
| `X_scGPT` | (169972, 512) | 所有细胞的 scGPT embedding |
| `XscGPT` | (169972, 512) | ⬆ alias（内容完全一致） |
| `X_scGPT_ctrl` | (169972, 512) | 仅 control 细胞非零 |
| `X_scGPT_pert` | (169972, 512) | 仅 treated 细胞非零 |
| `X_MolFormer` | (169972, 768) | 细胞级 MolFormer 药物 embedding |
| `XMolFormer` | (169972, 768) | ⬆ alias（内容完全一致） |

### 3.4 `.uns`

| Key | 说明 |
|-----|------|
| `rank_genes_groups_cov` | DEG 字典 (key: `patient_drug`) — **推荐评估使用** |
| `top50_DEGs` | DEG 字典 (key: `patient\|drug`) — 保留兼容 |
| `drug_smiles` | 药物名→SMILES 映射 |
| `release_metadata` | 版本信息 |
| `hvg`, `log1p`, `ood_split` | 预处理记录 |

---

## 4. obs 字段说明

| 字段 | 类型 | 含义 | 必需 |
|------|------|------|------|
| `perturbation` | category | 药物名或 "control" | 是 |
| `condition` | category | 清洗后药物名（同 perturbation） | 是 |
| `cov_drug_name` | category | `{patient}_{condition}` 组合键 | 是 (group 聚合) |
| `covariate_patient` | category | 患者 ID | 是 |
| `cell_type` | category | **当前为 patient ID** (非生物细胞类型) | 是 (作为 covariate) |
| `split` | category | train / valid / ood | 是 |
| `neg_control` | int64 | 1 = control, 0 = treated | 是 |
| `is_control` | bool | True = control | 辅助 |
| `is_treated` | category | "control" / "treated" | 辅助 |
| `dosage` | float64 | 剂量 (control=0.0) | 推荐 |
| `SMILES` / `canonical_smiles` | category | 药物 SMILES (control 为空) | 推荐 |
| `dataset` | category | GSE148842 / GSE226202 | 参考 |

---

## 5. Split 与 OOD 设置

### 5.1 Split 分布

| Split | 细胞数 | Control | Treated |
|-------|--------|---------|---------|
| train | 150,564 | 81,927 | 68,637 |
| valid | 16,729 | 9,074 | 7,655 |
| ood | 2,679 | 0 | 2,679 |

**禁止重新划分 split。** 所有方法必须使用 `adata.obs["split"]` 的预设值。

### 5.2 OOD 详细设定

- OOD group: **PW034_Panobinostat** (2,679 cells)
- OOD 类型: **unseen patient-drug combination**
  - PW034 不是完全未见患者（其 control 和 Etoposide 细胞存在于 train/valid）
  - Panobinostat 不是完全未见药物（其他患者中存在于 train/valid）
  - PW034 × Panobinostat 组合在 train/valid 中从未出现
- OOD split 中无 control 细胞
- Matched control 来源: **PW034_control** (15,288 cells in train/valid)

---

## 6. Control/Treated 定义与 Matched Control

### 6.1 核心规则

```
neg_control == 1  →  control
neg_control == 0  →  treated
```

`is_control` (bool) 与 `neg_control` 完全一致。

### 6.2 Matched Control 获取

每个 treated group `{patient}_{drug}` 的 matched control 为同患者的 control group:

```python
patient = group_name.split("_")[0]
ctrl_group = f"{patient}_control"
ctrl_mask = adata.obs["cov_drug_name"] == ctrl_group
```

### 6.3 OOD 反事实预测的 Control 来源

OOD split 中没有 control。PW034_Panobinostat 的 matched control 来自 train/valid 中的 PW034_control (15,288 cells)。这意味着 OOD 反事实预测使用的是在 training 中见过的 control 细胞。

---

## 7. 表达空间、Counts 空间与预测空间

### 7.1 三个空间

| 空间 | 存储位置 | 格式 | 用途 |
|------|---------|------|------|
| **log1p 表达** | `adata.X` | float32 CSR | 统一评估、可视化 |
| **原始 counts** | `adata.layers["counts"]` | int32 CSR | CPA M0/M4 训练输入 |
| **预测输出** | 各 prediction h5ad | varies | 方法依赖 |

### 7.2 统一评估空间

**统一评估使用 log1p space。**

- Y_true: 默认来自 `adata.X` (log1p)
- Y_ctrl: 默认来自 `adata.X` (log1p)
- Y_pred: 如果模型输出 counts，先转换: `np.log1p(np.maximum(Y_pred_counts, 0))`

### 7.3 Legacy Prediction 空间标注

| Method | 预测空间 | 统一 log1p 评估 |
|--------|---------|----------------|
| CPA M0 baseline | **counts** | 需 log1p 转换 |
| CPA M4 +MolFormer | **counts** | 需 log1p 转换 |
| CPA M1 +scGPT | ~log1p | 直接使用 |
| CPA M2 +scGPT ctrl | ~log1p | 直接使用 |
| CPA M3 +scGPT pert | ~log1p | 直接使用 |
| CPA M5 +scGPT+MolFormer | ~log1p | 直接使用 |
| MeanShiftBaseline | ~log1p | 直接使用 |

**R2/MSE 对表达空间敏感**，不能把 counts-space 结果和 log1p-space 结果直接比较。论文主表建议使用统一 log1p evaluation。历史 legacy metrics 可保留但必须标注为 legacy/original pipeline metrics。

用 `release_unified/reevaluate_counts_predictions_log1p.py` 可对 M0/M4 重新评估。

---

## 8. 药物、SMILES 与 MolFormer

### 8.1 药物列表

| Drug | Canonical SMILES | Treated Cells | 备注 |
|------|-----------------|---------------|------|
| Ana-12 | Cc1ccc(-c2nc3ccccc3n2-c2ccccc2)cc1 | 7,085 | |
| Etoposide | COc1cc2c(cc1O)C1C(=O)OCC1c1cc3c(cc1O2)OCO3 | 36,513 | 最多 treated |
| Ispenisib | CC(C)N1CCN(c2ccnc(N3CCOCC3)n2)CC1 | 2,683 | ⚠️ 与 Tazemetostat SMILES 相同 |
| Panobinostat | C=Cc1ccc(-c2cnn(CCN3CCC(C(=O)NO)CC3)c2)cc1 | 22,220 | **OOD 目标** |
| RO4929097 | CC(C)(C)OC(=O)N1CCC(n2cnc3ccc(C(F)(F)F)cc32)CC1 | 4,806 | |
| Tazemetostat | CC(C)N1CCN(c2ccnc(N3CCOCC3)n2)CC1 | 4,128 | ⚠️ 与 Ispenisib SMILES 相同 |
| Temozolomide | CN1N=Nc2ncn(n2)C1=O | 1,536 | |

### 8.2 MolFormer 使用方式

1. **细胞级**: `adata.obsm["X_MolFormer"]` 或 `adata.obsm["XMolFormer"]` (二者等价)
   - control 细胞: 全零向量
   - treated 细胞: 对应药物的 768d embedding

2. **药物级**: `GBM_molformer_drug_emb.parquet`, shape (6, 768)
   - 索引为 canonical SMILES 字符串（6 行因 Ispenisib/Tazemetostat 共享 SMILES）
   - 通过 `GBM_molformer_drug_emb.metadata.json` 映射药物名→SMILES→parquet

### 8.3 ⚠️ Ispenisib/Tazemetostat 共享 SMILES

两种药物 canonical SMILES 完全相同，MolFormer embedding 完全相同。基于 MolFormer 的方法无法区分它们。论文中应作为 known limitation 标注。

---

## 9. scGPT / MolFormer Embedding 使用说明

### 9.1 Key 名称

本文件同时包含两组 key — 使用你偏好的那一组：

```python
# 以下两组等价
X_cell_raw = adata.obsm["X_scGPT"]
X_cell_raw = adata.obsm["XscGPT"]

X_drug_raw = adata.obsm["X_MolFormer"]
X_drug_raw = adata.obsm["XMolFormer"]
```

### 9.2 scGPT 说明

- 使用 blood-derived 预训练权重编码 GBM 细胞
- Domain gap 存在但可用 — GBM 和 blood 在 stress response 通路有重叠
- Aligner MLP (126 MB each): `GBM_scGPT_aligner.pt` / `_ctrl.pt` / `_pert.pt`
- 对齐后矩阵 (3400 MB each): `GBM_scGPT_aligned_XscGPT.npy`

---

## 10. DEG 字典与 Key 映射

### 10.1 两个 DEG Entry

| Entry | Key 格式 | 示例 | 用途 |
|-------|---------|------|------|
| `uns["rank_genes_groups_cov"]` | `patient_drug` | `PW034_Panobinostat` | **推荐评估使用**（与 cov_drug_name 一致） |
| `uns["top50_DEGs"]` | `patient\|drug` | `PW034\|Panobinostat` | 保留兼容 |

### 10.2 推荐读取方式

```python
group_name = "PW034_Panobinostat"
deg_genes = adata.uns["rank_genes_groups_cov"][group_name]
# 如果 rank_genes_groups_cov 不存在（不应发生），回退:
# deg_genes = adata.uns["top50_DEGs"][group_name.replace("_", "|")]
```

### 10.3 DEG 覆盖率

- 18 个 treated group 有 DEG（每 group 50 genes）
- 所有 DEG genes 存在于 adata.var_names
- 8 个 treated groups 无 DEG（GSE226202 额外患者: PW051704, PW052703, PW052706, PW052709, PW053707, PW053710 + 2 GS 系列无 DEG）

---

## 11. CPA Baseline 如何使用本数据

| Method | 细胞输入 | 药物输入 | 所需文件 | 状态 |
|--------|---------|---------|---------|------|
| M0 CPA | layers["counts"] | learnable emb | GBM_NIPS_Ready.h5ad | ✅ |
| M1 +scGPT all | X_scGPT → aligner | learnable emb | h5ad + aligner.pt | ✅ |
| M2 +scGPT ctrl | X_scGPT_ctrl → aligner_ctrl | learnable emb | h5ad + aligner_ctrl.pt | ✅ |
| M3 +scGPT pert | X_scGPT_pert → aligner_pert | learnable emb | h5ad + aligner_pert.pt | ✅ |
| M4 +MolFormer | layers["counts"] | X_MolFormer | h5ad | ✅ |
| M5 +scGPT+MolFormer | X_scGPT → aligner | X_MolFormer | h5ad + aligner.pt | ✅ |

**所有方法均使用同一个 GBM_NIPS_Ready.h5ad。** 不需要为不同方法准备不同的 h5ad 版本。

---

## 12. 外部方法接入注意事项

| 方法 | 接入要点 |
|------|---------|
| **chemCPA** | 使用 adata.uns["drug_smiles"] 获取 SMILES；cell 输入用 counts 或 X |
| **CRISP** | 使用 rank_genes_groups_cov；control_key=neg_control；注意 cov_drug_name 为 group key |
| **CellOT** | source=control cells, target=treated cells (per cov_drug_name)；推荐用 X |
| **scGen** | 用 condition 标签 + counts |
| **Biolord** | 可从 SMILES 构造 drug attribute |
| **scVIDER** | 用 counts + perturbation 标签 |

---

## 13. 统一评估协议摘要

### 13.1 评估参数

| 参数 | 值 |
|------|-----|
| Group key | `cov_drug_name` |
| Control key | `neg_control` (1=control, 0=treated) |
| DEG key | `rank_genes_groups_cov` (优先) |
| Gene order | `adata.var_names` (固定 5000 genes) |
| Eval space | **log1p** |
| Aggregation | Unweighted macro average over valid groups |

### 13.2 指标

- pearson, pearson_delta (全基因 mean-profile)
- pearson_delta_de (DEG 子集 mean-profile — **主要指标**)
- sinkhorn_de (DEG 子集 cell-level)
- direction_accuracy_de (DEG 子集 directional accuracy)
- r2score, mse (mean-profile)

### 13.3 Valid Groups

18 个 valid groups (含 1 个 OOD)。所有方法必须使用相同的 valid group 集合。如果某方法只覆盖部分 groups，必须报告 coverage，不可与 full coverage 方法直接比较。

详见 `evaluate_gbm_predictions_protocol.md`。

---

## 14. 预测文件格式

### 预测 Dict 格式

```python
predictions = {
    group_name: {
        "Y_true": np.ndarray,           # treated cells × 5000 log1p
        "Y_pred": np.ndarray,           # model prediction × 5000
        "Y_ctrl": np.ndarray,           # matched control × 5000 log1p
        "prediction_space": "log1p",    # "log1p" or "counts"
        "var_names": list[str],         # must match adata.var_names
        "metadata": dict,               # method info
    }
}
```

### 关键要求

- gene order 必须与 adata.var_names 完全一致
- 推荐使用 log1p 空间
- group_name 必须来自 adata.obs["cov_drug_name"]

---

## 15. 已有 Baseline 结果说明

### 15.1 Legacy Metrics（原始 pipeline）

存储于 `evaluation_results/`，包含 9 个 JSON 文件和 CRISP OOD metrics markdown。这些指标的预测空间和协议与当前统一标准可能存在差异。

### 15.2 统一 Log1p Metrics

**待重新生成。** 使用 `release_unified/reevaluate_counts_predictions_log1p.py` 对所有方法的 log1p-space 进行评估，生成可直接用于论文的指标。

**论文主表必须使用统一 log1p evaluation 的结果。**

---

## 16. 已知限制

1. **Single-group OOD**: 仅 1 个 OOD group，无法计算 OOD macro average。属于 case study 级别。
2. **scGPT domain gap**: blood-derived 权重编码 GBM 细胞，存在 domain mismatch。
3. **Ispenisib/Tazemetostat 重复**: 共享 SMILES 和 MolFormer embedding。
4. **Cell type 缺失**: cell_type 字段为 patient ID，非生物细胞类型。
5. **部分患者无 DEG**: 6 个额外患者无 DEG entry。
6. **OOD 无 control**: 反事实预测依赖 train/valid control。

---

## 17. 推荐论文表述

在论文中使用以下标准化表述：

- OOD: "an unseen patient-drug combination setting (PW034 × Panobinostat, not strict unseen patient or drug)"
- 数据: "GBM_NIPS_Ready, a unified 169,972-cell × 5,000-gene dataset with scGPT and MolFormer embeddings"
- 评估: "log1p-space evaluation with unweighted macro average over 18 valid covariate groups"
- 主要指标: "pearson_delta_de and sinkhorn_de"

---

## 18. 维护信息

| 项目 | 信息 |
|------|------|
| 数据负责人 | 裴立昆 |
| 版本 | v1.1_unified_compatible |
| 最后更新 | 2026-05-19 |
| 唯一推荐文件 | `GBM_NIPS_Ready.h5ad` |
| Conda 环境 | `plknature` |
| 备份位置 | `archive_internal/` |

### 修改规则

- 修改 h5ad 前先备份到 archive_internal/
- 新字段使用 patch 脚本添加
- 新 embedding 写入 obsm
- 更新后同步修改 manifest 和本文档
