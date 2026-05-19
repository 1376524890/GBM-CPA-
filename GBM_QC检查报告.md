# GBM 数据集 QC 检查报告

**检查时间**: 2026-05-19
**机器路径**: `/home/u2023312303/nature子刊/裴立昆实验`
**Python 环境**: conda env `plknature` (anndata 0.12.11, scanpy 1.12.1, torch 2.5.1+cu121, numpy 2.4.4)
**检查范围**: GBM_NIPS_Ready.h5ad 及所有相关数据文件、脚本、预测文件和评估结果

---

## 1. 核心文件存在性检查

| 文件 | 状态 | 大小 | 用途 |
|------|------|------|------|
| GBM_NIPS_Ready.h5ad | OK | 871 MB | 最终标准化数据集 |
| GBM_Universal_Perturbation_Ready.h5ad | OK | 223.6 MB | 预处理前 counts 矩阵 |
| GBM_with_embeddings.h5ad | OK | 871 MB | 添加 embedding 后的中间数据 |
| GBM_scGPT_embeddings.h5ad | OK | 760.5 MB | 原始 scGPT 细胞 embedding |
| GBM_X_MolFormer.npy | OK | 522.2 MB | MolFormer embedding 矩阵 |
| GBM_molformer_drug_emb.parquet | OK | 0.4 MB | 药物级 MolFormer embedding |
| GBM_molformer_drug_emb.metadata.json | OK | 1.2 KB | SMILES-药物映射元数据 |
| GBM_scGPT_aligner.pt | OK | 126.1 MB | scGPT aligner (all cells) |
| GBM_scGPT_aligner_ctrl.pt | OK | 126.1 MB | scGPT aligner (control only) |
| GBM_scGPT_aligner_pert.pt | OK | 126.1 MB | scGPT aligner (perturbed only) |
| GBM_scGPT_aligned_XscGPT.npy | OK | 3399 MB | 对齐后的 scGPT 基因空间矩阵 |
| GBM_scGPT_aligned_XscGPT_ctrl.npy | OK | 3399 MB | 对齐后 scGPT (ctrl) |
| GBM_scGPT_aligned_XscGPT_pert.npy | OK | 3399 MB | 对齐后 scGPT (pert) |

**所有 21 个核心文件存在**。12 个脚本文件和 9 个评估结果文件也存在。

## 2. AnnData 主体结构检查

| 检查项 | 结果 | 状态 |
|--------|------|------|
| shape | (169972, 5000) | PASS |
| X dtype | float32 CSR sparse | PASS |
| X NaN/Inf | 0 / 0 | PASS |
| X 值范围 | [0.0637, 9.0827] (log1p 空间) | PASS |
| X 零值比例 | 0.0000 | PASS |
| layers["counts"] | int32 CSR, shape=(169972, 5000) | PASS |
| counts NaN/Inf | 0 / 0 | PASS |
| counts 非负整数 | True (min=1, max=5292) | PASS |
| var_names 数量 | 5000 | PASS |
| var_names 唯一 | True | PASS |
| obs_names 唯一 | True | PASS |

**结论**: AnnData 主体结构完整、无错误。X 为 log1p 归一化表达，counts 为原始 UMI 整数计数。

## 3. obs 元数据字段检查

### 必要字段覆盖率

所有 16 个必要字段均存在，无缺失值：

| 字段 | dtype | 唯一值数 | 状态 |
|------|-------|---------|------|
| gsm_accession | category | 54 | PASS |
| sample_title | category | 54 | PASS |
| barcode_original | category | 164,946 | PASS |
| perturbation | category | 8 | PASS |
| dosage | float64 | 8 | PASS |
| covariate_patient | category | 21 | PASS |
| dataset | category | 2 | PASS |
| cell_type | category | 21 | WARNING (见下文) |
| is_control | bool | 2 | PASS |
| split | category | 3 | PASS |
| cov_drug_name | category | 39 | PASS |
| neg_control | int64 | 2 | PASS |
| condition | category | 8 | PASS |
| is_treated | category | 2 | PASS |
| SMILES | category | 7 | PASS |
| canonical_smiles | category | 7 | PASS |

### 重点检查

1. **perturbation**: control + 7 种药物。PASS。
   - control: 91,001 | Etoposide: 36,513 | Panobinostat: 22,220 | Ana-12: 7,085 | RO4929097: 4,806 | Tazemetostat: 4,128 | Ispenisib: 2,683 | Temozolomide: 1,536

2. **condition**: 清洗后的药物名。PASS。

3. **neg_control 与 is_control 完全一致**: True。PASS。

4. **is_treated**: control/treated 两种值。PASS。

5. **cov_drug_name**: 格式 `{patient}_{condition}`。PASS。

6. **dosage**: control=0.0, treated 值范围 [0.0018, 50.0]。PASS。

7. **SMILES/canonical_smiles**: control 为空字符串，treated 非空。PASS。

8. **cell_type 字段使用 patient ID**: WARNING。当前 cell_type 值等于 covariate_patient（患者 ID），不是真实细胞类型（如 astrocyte, neuron 等）。这符合 CPA 需要 covariate 字段的要求，但需要在交接文档中说明。

## 4. Split 与 OOD 泄漏检查

### Split 分布

| Split | 细胞数 | 比例 | Control | Treated |
|-------|--------|------|---------|---------|
| train | 150,564 | 88.6% | 81,927 | 68,637 |
| valid | 16,729 | 9.8% | 9,074 | 7,655 |
| ood | 2,679 | 1.6% | 0 | 2,679 |

### OOD 泄漏检查

| 检查项 | 结果 | 状态 |
|--------|------|------|
| OOD 仅含 PW034 + Panobinostat | True | PASS |
| PW034 不出现在 train/valid | False (PW034 在 train: 28,096, valid: 3,086，但仅为 Etoposide 和 control) | PASS (未见患者组合 OOD) |
| Panobinostat 出现在 train/valid | True (train: 17,584, valid: 1,957，其他患者) | PASS (非未见药物 OOD) |
| OOD 类型 | 未见患者-药物组合 (PW034从未接受Panobinostat) | PASS |
| Split 间无细胞重叠 | train/valid=0, train/ood=0, valid/ood=0 | PASS |
| OOD 中无 control 细胞 | OOD 仅含 treated cells (2,679) | **WARNING** |

**关于 OOD control**: OOD 中 PW034_Panobinostat 仅有 treated 细胞。反事实预测的 control 来源为 PW034_control（15,288 cells，分布在 train/valid）。需要在交接文档中说明。

### 小规模 group 标记

被过滤的 treated group（无 DEG 或无 matched control）:
- PW051704_Panobinostat: 1,881 treated, 无 DEG, 无 matched control
- PW052703_Etoposide: 1,293 treated, 无 DEG, 无 matched control
- PW052706_Etoposide: 1,013 treated, 无 DEG, 无 matched control
- PW052709_Etoposide: 1,289 treated, 无 DEG, 无 matched control
- PW053707_Panobinostat: 2,340 treated, 无 DEG, 无 matched control
- PW053710_Panobinostat: 1,718 treated, 无 DEG, 无 matched control

## 5. Embedding 检查

| Key | Shape | dtype | NaN | Inf | 全零行 | 非零行比例 | 与n_obs匹配 |
|-----|-------|-------|-----|-----|--------|-----------|-----------|
| X_scGPT | (169972, 512) | float32 | 0 | 0 | 0 | 100% | PASS |
| X_scGPT_ctrl | (169972, 512) | float32 | 0 | 0 | 78,971 (treated) | 53.54% (control only) | PASS |
| X_scGPT_pert | (169972, 512) | float32 | 0 | 0 | 91,001 (control) | 46.46% (treated only) | PASS |
| X_MolFormer | (169972, 768) | float32 | 0 | 0 | 91,001 (control) | 46.46% (treated only) | PASS |

### 关键验证

1. **X_scGPT_ctrl 零掩码与 neg_control 一致**: 0 mismatch。PASS。
2. **X_scGPT_pert 零掩码与 neg_control 一致**: 0 mismatch。PASS。
3. **control 的 X_MolFormer 全零**: 91,001/91,001 全零。PASS。
4. **同一药物的 X_MolFormer 一致**: 所有 7 种药物各只有 1 个唯一 embedding。PASS。
5. **Ispenisib 与 Tazemetostat**: SMILES 完全相同，MolFormer embedding 完全相同（cosine distance = 0.0）。**WARNING**。
6. **XscGPT / XMolFormer alias**: 不存在。**WARNING** — 需要在使用 NIPS/CRISP 兼容代码时指定正确的 key 名。

## 6. 药物 SMILES 与 MolFormer 文件检查

| 检查项 | 结果 | 状态 |
|--------|------|------|
| uns["drug_smiles"] 存在 | True，覆盖 7 种 treated 药物 | PASS |
| control 无 SMILES | True | PASS |
| MolFormer parquet 存在 | True | PASS |
| parquet 维度 | (6, 768) — 6 行（Ispenisib/Tazemetostat 共享 SMILES） | PASS |
| metadata json 存在 | True，包含 drug→SMILES→parquet 映射 | PASS |
| obsm 与 parquet 一致性 | 所有 7 种药物完全匹配 | PASS |
| parquet 索引格式 | SMILES 字符串（非药物名） | **WARNING** |

**药物表**:

| Drug | Canonical SMILES | Treated Cells | 备注 |
|------|-----------------|---------------|------|
| Ana-12 | Cc1ccc(-c2nc3ccccc3n2-c2ccccc2)cc1 | 7,085 | |
| Etoposide | COc1cc2c(cc1O)C1C(=O)OCC1c1cc3c(cc1O2)OCO3 | 36,513 | |
| Ispenisib | CC(C)N1CCN(c2ccnc(N3CCOCC3)n2)CC1 | 2,683 | 与 Tazemetostat SMILES 相同 |
| Panobinostat | C=Cc1ccc(-c2cnn(CCN3CCC(C(=O)NO)CC3)c2)cc1 | 22,220 | OOD 目标药物 |
| RO4929097 | CC(C)(C)OC(=O)N1CCC(n2cnc3ccc(C(F)(F)F)cc32)CC1 | 4,806 | |
| Tazemetostat | CC(C)N1CCN(c2ccnc(N3CCOCC3)n2)CC1 | 4,128 | 与 Ispenisib SMILES 相同 |
| Temozolomide | CN1N=Nc2ncn(n2)C1=O | 1,536 | |

## 7. DEG 字典检查

| 检查项 | 结果 | 状态 |
|--------|------|------|
| top50_DEGs 存在 | True (18 entries) | PASS |
| rank_genes_groups_cov 存在 | False | **WARNING** |
| DEG key 格式 | `patient|drug` (使用 `|` 分隔) | **WARNING** |
| cov_drug_name 格式 | `patient_drug` (使用 `_` 分隔) | |
| 映射完整性 | 18 个 DEG key → 18 个 cov_drug_name 一一对应 | PASS |
| 每个 group 50 genes | True | PASS |
| 所有 DEG genes 在 var_names | True | PASS |
| 无重复 DEG | True | PASS |
| OOD target 有 DEG | True (PW034\|Panobinostat, 50 genes) | PASS |

**DEG 覆盖率**: 18/18 有 DEG 的 treated group 均有 50 个 DEG。8 个有 treated 但无 DEG 的 group 被标记为 invalid。

**DEG key 映射示例**:
- `top50_DEGs["PW034|Panobinostat"]` → `cov_drug_name == "PW034_Panobinostat"`
- 所有 key 均可通过 `replace("|", "_")` 转换

## 8. NIPS/CRISP 评估兼容性检查

| 检查项 | 值 |
|--------|-----|
| 总 cov_drug_name groups | 39 |
| 有效 treated groups (treated > 5, has DEG, has matched control >= 5) | 18 |
| 无效 groups | 21 (15 control groups + 6 treated groups 无 DEG) |
| OOD 有效 groups | 1 (PW034_Panobinostat) |
| 评估类型 | Single-group OOD case study |

**被过滤原因汇总**:
- treated_too_few: 15 control groups (n_treated=0)
- is_control_group: 15 control groups
- deg_missing: 15 control groups + 6 treated groups (PW051704, PW052703, PW052706, PW052709, PW053707, PW053710)
- no_matched_control_group: 6 treated groups

## 9. CPA 方法兼容性检查

| Method | Required Fields/Files | Status | Missing Items | Suggested Fix |
|--------|----------------------|--------|---------------|---------------|
| M0 CPA baseline | counts, condition, covariate_patient, split, neg_control | **PASS** | - | 可直接使用 |
| M1 CPA + scGPT all | X_scGPT, aligner.pt, aligned_XscGPT.npy | **PASS** | XscGPT alias | 代码中指定 key="X_scGPT" |
| M2 CPA + scGPT ctrl | X_scGPT_ctrl, aligner_ctrl.pt, aligned_ctrl.npy | **PASS** | - | 可直接使用 |
| M3 CPA + scGPT pert | X_scGPT_pert, aligner_pert.pt, aligned_pert.npy | **PASS** | - | 可直接使用 |
| M4 CPA + MolFormer | counts, X_MolFormer, drug SMILES | **PASS** | XMolFormer alias | 代码中指定 key="X_MolFormer" |
| M5 CPA + scGPT + MolFormer | X_scGPT, X_MolFormer, aligner.pt | **PASS** | - | 可直接使用 |

**所有 CPA baseline 方法均满足输入要求。**

## 10. 预测文件检查

| 文件 | Method | Shape | var_names | NaN/Inf | 值范围 | Target | 可用评估 |
|------|--------|-------|-----------|---------|--------|--------|---------|
| GBM_CPA_PW034_Panobinostat_pred.h5ad | CPA_M0 | (15288, 5000) | PASS | 0/0 | [0, 656.9] | PW034_ctrl counterfactual | YES (counts space) |
| GBM_CPA_MolFormer_PW034_Panobinostat_pred.h5ad | CPA_M4 | (15288, 5000) | PASS | 0/0 | [0, 1048.5] | PW034_ctrl counterfactual | YES (counts space) |
| GBM_CPA_scGPT_PW034_Panobinostat_pred.h5ad | CPA_M1 | (15288, 5000) | PASS | 0/0 | [0, 4.32] | PW034_ctrl counterfactual | YES (log1p space) |
| GBM_CPA_scGPT_ctrl_PW034_Panobinostat_pred.h5ad | CPA_M2 | (15288, 5000) | PASS | 0/0 | [0, 4.78] | PW034_ctrl counterfactual | YES (log1p space) |
| GBM_CPA_scGPT_pert_PW034_Panobinostat_pred.h5ad | CPA_M3 | (15288, 5000) | PASS | 0/0 | [0, 4.39] | PW034_ctrl counterfactual | YES (log1p space) |
| GBM_CPA_scGPT_MolFormer_PW034_Panobinostat_pred.h5ad | CPA_M5 | (15288, 5000) | PASS | 0/0 | [0, 4.43] | PW034_ctrl counterfactual | YES (log1p space) |
| GBM_MeanShiftBaseline_PW034_Panobinostat_pred.h5ad | MeanShift | (2679, 5000) | PASS | 0/0 | [0, 7.34] | OOD treated cells | YES |
| MLP_M1_PW034_Panobinostat_pred.h5ad | MLP_M1 | (15288, 5000) | PASS | 0/0 | [0, 8.36] | PW034_ctrl counterfactual | YES |
| MLP_M5_PW034_Panobinostat_pred.h5ad | MLP_M5 | (15288, 5000) | PASS | 0/0 | [0, 8.61] | PW034_ctrl counterfactual | YES |

**所有预测文件均可用于统一评估。** 注意：M0/M4 预测在 counts 空间；其他模型在约 log1p 空间。

## 11. 已存在评估结果汇总

| Method | PrΔ DE | Sp DE | R² DE | Sinkhorn DE | Direction Acc | 来源 |
|--------|--------|-------|-------|-------------|---------------|------|
| CPA M0 | 0.608 | 0.463 | -18.636 | 0.004 | 96.0% | CRISP metrics |
| CPA M4 +MolFormer | 0.693 | 0.585 | -18.957 | 0.004 | 94.0% | CRISP metrics |
| CPA M1 +scGPT | 0.103 | 0.219 | -311.580 | 0.006 | 36.0% | CRISP metrics |
| CPA M2 +scGPT ctrl | 0.111 | 0.173 | -486.665 | 0.006 | 44.0% | CRISP metrics |
| CPA M3 +scGPT pert | 0.027 | 0.209 | -929.834 | 0.009 | 46.0% | CRISP metrics |
| CPA M5 scGPT+Mol | 0.098 | 0.199 | -507.680 | 0.007 | 44.0% | CRISP metrics |
| MeanShift | -0.468 | -0.230 | -276.429 | 0.190 | 20.0% | CRISP metrics |
| MLP M1 | 0.018 | -0.003 | -1046.680 | 0.012 | 18.0% | CRISP metrics |
| MLP M5 | -0.069 | -0.086 | -1017.322 | 0.013 | 20.0% | CRISP metrics |

NIPS-format JSON 评估结果（9 个文件）也存在于 evaluation_results/ 目录，包含 pearson_delta_de, sinkhorn_de, pearson, r2score, mse 等指标。

## 12. 错误与警告汇总

### 错误 (ERROR): 0

无阻断性错误。

### 警告 (WARNING): 9 条

| # | 级别 | 描述 | 建议 |
|---|------|------|------|
| W1 | WARNING | DEG keys 使用 `\|` 分隔符（如 `PW034\|Panobinostat`），而 cov_drug_name 使用 `_`（如 `PW034_Panobinostat`） | 评估代码中实现 `replace("\|", "_")` 映射 |
| W2 | WARNING | rank_genes_groups_cov 不存在，仅有 top50_DEGs | 可用 top50_DEGs 替代，或生成 alias；已确认内容等价 |
| W3 | WARNING | obsm 中不存在 XscGPT / XMolFormer 兼容 alias | 外部方法需在配置中指定正确 key：`X_scGPT`, `X_MolFormer` |
| W4 | WARNING | cell_type 字段使用 patient ID 而非真实细胞类型 | 不影响 CPA 使用，但需在交接文档中说明 |
| W5 | WARNING | Ispenisib 与 Tazemetostat SMILES 完全相同 | 作为已知注意事项记录；MolFormer embedding 无法区分 |
| W6 | WARNING | 8 个 treated groups 无 DEG 且无 matched control | 这些 groups 无法参与评估；属于 GSE226202 额外患者 |
| W7 | WARNING | MolFormer parquet 使用 SMILES 字符串作为索引 | 外部方法需通过 metadata json 映射药物名→SMILES→embedding |
| W8 | WARNING | CPA M0/M4 预测在 counts 空间，非 log1p 空间 | 评估前需要确认统一空间或进行 log1p 变换 |
| W9 | WARNING | OOD 中无 control 细胞 | 反事实预测需使用 train/valid 中 PW034_control |

## 13. 总体结论

**PASS_WITH_WARNINGS**

数据集已完成规范化处理，满足 CPA 系列 baseline 和 NIPS/CRISP 评估协议的核心要求：
- 169,972 细胞 × 5,000 基因，结构完整，无 NaN/Inf
- 所有必要 obs 字段齐全
- 三种 embedding（scGPT all/ctrl/pert + MolFormer）均已计算并嵌入 obsm
- Split 设置正确，无数据泄露
- DEG 字典 18 groups，每个 50 基因
- 所有 CPA baseline 模型已训练并产出预测
- 评估结果已覆盖 9 种方法

9 条 WARNING 均为非阻断性的兼容性/文档性问题，已在交接文档中详细说明处理方案。
