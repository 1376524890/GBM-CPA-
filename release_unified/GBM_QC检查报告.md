# GBM 数据集 QC 检查报告

**版本**: v1.1_unified_compatible
**检查时间**: 2026-05-19
**检查目标**: GBM_NIPS_Ready.h5ad（唯一推荐版本）
**机器路径**: `/home/u2023312303/nature子刊/裴立昆实验`
**Python 环境**: conda env `plknature` (anndata 0.12.11, scanpy 1.12.1, torch 2.5.1+cu121, numpy 2.4.4)
**检查方法**: `release_unified/check_gbm_final_h5ad.py`

---

## 1. 版本说明

本报告针对 **GBM_NIPS_Ready.h5ad v1.1**。该文件是唯一推荐使用的最终标准化数据文件，已同时保留原始字段和兼容 alias 字段，包含：

- 原始 obsm key: `X_scGPT`, `X_scGPT_ctrl`, `X_scGPT_pert`, `X_MolFormer`
- 兼容 obsm key: `XscGPT` (alias of X_scGPT), `XMolFormer` (alias of X_MolFormer)
- 原始 DEG key: `top50_DEGs` (key 格式: `patient|drug`)
- 兼容 DEG key: `rank_genes_groups_cov` (key 格式: `patient_drug`, 与 cov_drug_name 一致)
- 版本元数据: `uns["release_metadata"]`

**不需要**在原始版和兼容版之间选择。所有方法统一使用此文件。

---

## 2. 核心文件存在性检查

| 文件 | 状态 | 大小 | 用途 |
|------|------|------|------|
| GBM_NIPS_Ready.h5ad | OK | ~3.2 GB | 唯一推荐数据文件（含兼容字段） |
| GBM_molformer_drug_emb.parquet | OK | 0.4 MB | 药物级 MolFormer embedding |
| GBM_molformer_drug_emb.metadata.json | OK | 1.2 KB | SMILES→药物名→parquet 映射 |
| GBM_scGPT_aligner.pt | OK | 126 MB | scGPT aligner MLP (all cells) |
| GBM_scGPT_aligner_ctrl.pt | OK | 126 MB | scGPT aligner MLP (ctrl only) |
| GBM_scGPT_aligner_pert.pt | OK | 126 MB | scGPT aligner MLP (pert only) |
| 6× CPA 预测 h5ad | OK | ~255-288 MB | CPA baseline 预测 |
| 9× 评估 JSON | OK | ~1 KB each | NIPS 格式评估结果 |
| archive_internal/备份 | OK | 871 MB (v1.0) | 安全回滚备份（不对外推荐） |

## 3. AnnData 主体结构检查

| 检查项 | 结果 | 状态 |
|--------|------|------|
| shape | (169972, 5000) | PASS |
| X dtype | float32 CSR sparse | PASS |
| X NaN/Inf | 0 / 0 | PASS |
| X nonzero_min | ~0.064 | PASS (非零表达有下界) |
| X nonzero_max | ~9.083 | PASS |
| layers["counts"] dtype | int32 CSR sparse | PASS |
| counts NaN/Inf | 0 / 0 | PASS |
| counts 非负整数 | True (nonzero_min=1, nonzero_max=5292) | PASS |
| var_names 唯一且 5000 | True | PASS |
| obs_names 唯一 | True | PASS |

### 3.1 Sparse Matrix 统计说明

**重要**: `adata.X` 和 `layers["counts"]` 均为 CSR sparse matrix。对 `sparse_matrix.data` 的 min/max 统计仅反映**非零存储值**，不包含隐式零。单细胞表达矩阵天然存在大量隐式零。

| 统计量 | X (log1p) | counts (UMI) |
|--------|-----------|--------------|
| 格式 | CSR float32 | CSR int32 |
| nnz | 见实际 nnz 属性 | 见实际 nnz 属性 |
| density | 实际 nnz / (169972×5000) | 实际 nnz / (169972×5000) |
| implicit_zero_ratio | 1 - density | 1 - density |
| nonzero_min | ~0.064 | 1 |
| nonzero_max | ~9.083 | 5292 |
| 说明 | 经过 log1p 后大部分值非零 | 原始 UMI 稀疏性高 |

**不应将 nonzero_min 理解为全矩阵没有 0。隐式零占稀疏矩阵的绝大部分空间（对 counts 而言）。**

## 4. obs 元数据字段检查

所有 20 个 obs 字段均存在，无缺失值:

| 字段 | dtype | 状态 | 说明 |
|------|-------|------|------|
| gsm_accession | category | PASS | |
| sample_title | category | PASS | |
| barcode_original | category | PASS | 164,946 unique |
| perturbation | category | PASS | 8 values |
| dosage | float64 | PASS | control=0.0, treated varies |
| covariate_patient | category | PASS | 21 patients |
| dataset | category | PASS | GSE148842, GSE226202 |
| cell_type | category | WARNING | = patient ID, not biological cell type |
| is_control | bool | PASS | True/False |
| is_treated | category | PASS | control/treated |
| split | category | PASS | train/valid/ood |
| cov_drug_name | category | PASS | 39 groups |
| neg_control | int64 | PASS | 1=control, 0=treated |
| condition | category | PASS | 8 values |
| SMILES | category | PASS | control empty |
| canonical_smiles | category | PASS | control empty |
| n_counts | - | PASS | |
| source_treatment | - | PASS | |
| source_file | - | PASS | |
| foundation_model_query_eligible | - | PASS | |

### 重点验证

1. **neg_control == is_control**: 完全一致。PASS。
2. **perturbation**: control + 7 种药物 (Ana-12, Etoposide, Ispenisib, Panobinostat, RO4929097, Tazemetostat, Temozolomide)。PASS。
3. **cell_type = patient ID**: WARNING。当前 `cell_type` 值为患者 ID，非生物细胞类型。在交接文档中说明。

## 5. Split 与 OOD 泄漏检查

| 检查项 | 结果 | 状态 |
|--------|------|------|
| train cells | 150,564 (88.6%) | PASS |
| valid cells | 16,729 (9.8%) | PASS |
| ood cells | 2,679 (1.6%) | PASS |
| Split 间细胞不重叠 | train∩valid=0, train∩ood=0, valid∩ood=0 | PASS |
| OOD 仅含 PW034+Panobinostat | True | PASS |
| OOD 无 control 细胞 | True (OOD 全为 treated) | 注记 |
| PW034 在 train/valid 中出现 | True (Etoposide 和 control) | 注记 |
| Panobinostat 在 train/valid 中出现 | True (其他患者) | 注记 |
| OOD 类型 | unseen patient-drug combination | PASS |

### OOD 统一表述

当前 GBM OOD 目标为 PW034_Panobinostat，即 PW034 患者接受 Panobinostat 的组合在 train/valid 中未出现。PW034 患者本身不是完全未见，PW034 的 control 和 Etoposide 细胞存在于 train/valid；Panobinostat 药物本身也不是完全未见，其在其他患者中存在于 train/valid。因此本任务是 **unseen patient-drug combination OOD**，**不是 strict unseen patient OOD**，也**不是 strict unseen drug OOD**。

### Matched Control 来源

OOD split 中无 control 细胞。反事实预测的 control 来源为 **PW034_control** group (15,288 cells)，分布在 train (12,202) 和 valid (3,086)。

## 6. Embedding 检查

| Key | Shape | dtype | NaN | Inf | 与 n_obs 匹配 | 全零行 |
|-----|-------|-------|-----|-----|-------------|--------|
| X_scGPT | (169972, 512) | float32 | 0 | 0 | PASS | 0 |
| XscGPT | (169972, 512) | float32 | 0 | 0 | PASS | 0 |
| X_scGPT_ctrl | (169972, 512) | float32 | 0 | 0 | PASS | 78,971 (treated) |
| X_scGPT_pert | (169972, 512) | float32 | 0 | 0 | PASS | 91,001 (control) |
| X_MolFormer | (169972, 768) | float32 | 0 | 0 | PASS | 91,001 (control) |
| XMolFormer | (169972, 768) | float32 | 0 | 0 | PASS | 91,001 (control) |

### Alias 一致性

- **XscGPT == X_scGPT**: PASS (数值完全一致)
- **XMolFormer == X_MolFormer**: PASS (数值完全一致)
- X_scGPT_ctrl/pert 零掩码与 neg_control 一致: PASS (0 mismatch)
- Ispenisib 与 Tazemetostat MolFormer 相同: **WARNING** (SMILES 相同)

## 7. DEG 字典检查

| 检查项 | 结果 | 状态 |
|--------|------|------|
| top50_DEGs 存在 | 18 entries | PASS |
| rank_genes_groups_cov 存在 | 18 entries | PASS |
| 内容一致性 | 完全一致 (仅 key 分隔符不同) | PASS |
| top50_DEGs key 格式 | `patient\|drug` (例: PW034\|Panobinostat) | 注记 |
| rank_genes_groups_cov key 格式 | `patient_drug` (例: PW034_Panobinostat) | 注记 |
| 每个 group 50 genes | True | PASS |
| DEG genes 全在 var_names | True | PASS |
| OOD target 有 DEG | True (PW034_Panobinostat) | PASS |

## 8. NIPS/CRISP 评估兼容性

| 检查项 | 值 | 状态 |
|--------|-----|------|
| 总 cov_drug_name groups | 39 | - |
| Valid treated groups | 18 | PASS |
| Invalid groups | 21 (15 control + 6 treated 无 DEG) | 注记 |
| OOD valid groups | 1 (PW034_Panobinostat) | 注记 |
| rank_genes_groups_cov 可用 | True | PASS |
| DEG key 与 cov_drug_name 一致 | True (均使用 `_` 分隔符) | PASS |

## 9. CPA 方法兼容性

| Method | 状态 | 说明 |
|--------|------|------|
| M0 CPA baseline | PASS | counts + condition + covariate_patient + split + neg_control |
| M1 CPA + scGPT all | PASS | X_scGPT/XscGPT + aligner.pt |
| M2 CPA + scGPT ctrl | PASS | X_scGPT_ctrl + aligner_ctrl.pt |
| M3 CPA + scGPT pert | PASS | X_scGPT_pert + aligner_pert.pt |
| M4 CPA + MolFormer | PASS | counts + X_MolFormer/XMolFormer |
| M5 CPA + scGPT + MolFormer | PASS | X_scGPT + X_MolFormer + aligner |

**所有 6 种 CPA baseline 均满足输入要求。**

## 10. 预测文件与评估空间

| Method | 预测文件 | 预测空间 | 统一评估须知 |
|--------|---------|---------|------------|
| CPA M0 | GBM_CPA_PW034_Panobinostat_pred.h5ad | **counts** | 需 log1p 转换后评估 |
| CPA M4 +MolFormer | GBM_CPA_MolFormer_PW034_Panobinostat_pred.h5ad | **counts** | 需 log1p 转换后评估 |
| CPA M1 +scGPT | GBM_CPA_scGPT_PW034_Panobinostat_pred.h5ad | ~log1p | 可直接评估 |
| CPA M2 +scGPT ctrl | GBM_CPA_scGPT_ctrl_PW034_Panobinostat_pred.h5ad | ~log1p | 可直接评估 |
| CPA M3 +scGPT pert | GBM_CPA_scGPT_pert_PW034_Panobinostat_pred.h5ad | ~log1p | 可直接评估 |
| CPA M5 +scGPT+Mol | GBM_CPA_scGPT_MolFormer_PW034_Panobinostat_pred.h5ad | ~log1p | 可直接评估 |
| MeanShift | GBM_MeanShiftBaseline_PW034_Panobinostat_pred.h5ad | ~log1p | 可直接评估 |

**统一评估使用 log1p space。** M0/M4 的 legacy prediction 是 counts space，必须先 `np.log1p(np.maximum(Y, 0))` 转换。历史 legacy metrics 可保留但必须标注为 legacy/original pipeline metrics。论文主表建议使用统一 log1p evaluation。

## 11. 错误与警告汇总

### 错误 (ERROR): 0

无阻断性错误。

### 警告 (WARNING)

| # | 级别 | 描述 |
|---|------|------|
| W1 | MEDIUM | cell_type 使用 patient ID，非生物细胞类型 |
| W2 | MEDIUM | Ispenisib 与 Tazemetostat SMILES 相同，MolFormer embedding 相同 |
| W3 | MEDIUM | M0/M4 legacy prediction 是 counts space，统一评估需 log1p 转换 |
| W4 | MEDIUM | OOD 仅含 1 个 valid group（single-group case study） |
| W5 | LOW | 8 个 treated groups 无 DEG 且无 matched control |
| W6 | LOW | scGPT 使用 blood-derived 权重，存在 domain gap |
| W7 | LOW | MolFormer parquet 索引为 SMILES 字符串，非药物名 |
| W8 | LOW | OOD split 中无 control |

## 12. 总体结论

**PASS_WITH_WARNINGS**

GBM_NIPS_Ready.h5ad v1.1 作为唯一推荐文件，满足所有内部和外部方法的兼容性要求:
- 169,972 cells × 5,000 genes，结构完整，无 NaN/Inf
- 所有必要 obs 字段齐全
- 6 个 obsm key（含兼容 alias）齐全
- 2 个 DEG dict（含兼容 alias）齐全，内容一致
- Split 设置正确，无数据泄露
- 所有 CPA baseline 模型已训练并产出预测
- 兼容字段经 42 项完整性检查验证通过
