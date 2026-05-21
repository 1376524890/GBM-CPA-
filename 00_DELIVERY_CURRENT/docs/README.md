# GBM 单细胞药物扰动数据集 — 统一发布版 v1.1

## 重要：仅一个推荐文件

`00_DELIVERY_CURRENT/dataset/GBM_NIPS_Ready.h5ad` 是本数据集的唯一推荐使用文件。

该文件已同时包含原始字段和兼容 alias 字段，**不需要**在 "原始版" 和 "兼容版" 之间切换或选择。

## 快速开始

```bash
# 1. 激活环境
conda activate plknature

# 2. 完整性检查
python 00_DELIVERY_CURRENT/code/cpa_nips/check_gbm_final_h5ad.py

# 3. 加载示例
python 00_DELIVERY_CURRENT/code/cpa_nips/load_gbm_example.py

# 4. 对 counts-space 预测重新评估（log1p 空间）
python 00_DELIVERY_CURRENT/code/cpa_nips/reevaluate_counts_predictions_log1p.py
```

## 绝对不允许的操作

1. **不要重新划分 split** — 使用 `adata.obs["split"]` 的预设值（train/valid/ood）
2. **不要重排 genes** — 所有预测列必须与 `adata.var_names` 顺序完全一致
3. **不要把 control 定义反了** — `neg_control == 1` 是 control，`== 0` 是 treated
4. **不要把 OOD 写成完全未见患者** — PW034 的 control 和 Etoposide 细胞存在于 train/valid
5. **不要混用 counts/log1p 空间** — 统一评估使用 log1p 空间
6. **不要为不同方法使用不同 valid groups** — 统一使用 18 个 valid groups
7. **不要在文档中区分 "原始 h5ad" 和 "兼容 h5ad"** — 只有一个推荐文件

## OOD 定义（必须统一使用）

当前 GBM OOD 目标为 **PW034_Panobinostat**，即 PW034 患者接受 Panobinostat 的组合在 train/valid 中未出现。

- PW034 患者本身不是完全未见（PW034 的 control 和 Etoposide 细胞存在于 train/valid）
- Panobinostat 药物本身也不是完全未见（在其他患者中存在于 train/valid）
- 本任务是 **unseen patient-drug combination OOD**，**不是 strict unseen patient OOD**，也不是 **strict unseen drug OOD**
- OOD split 中没有 control 细胞；反事实预测的 matched control 来源为 **PW034_control**（train/valid 中的 15,288 个细胞）

## 表达空间

- `adata.X` = **log1p 归一化表达**（统一评估推荐使用）
- `adata.layers["counts"]` = **原始 UMI counts**（CPA M0/M4 训练用）
- CPA M0/M4 的已有预测是 **counts space**，log1p 统一评估前必须先转换：
  ```python
  Y_pred_eval = np.log1p(np.maximum(Y_pred_counts, 0))
  ```
- M1/M2/M3/M5、MeanShift、MLP 的预测更接近 **log1p space**

## 文件清单

| 文件 | 用途 |
|------|------|
| `00_DELIVERY_CURRENT/dataset/GBM_NIPS_Ready.h5ad` | 唯一推荐主数据文件 |
| `00_DELIVERY_CURRENT/code/cpa_nips/` | NIPS 原始字段 CPA 训练、预测、评估代码 |
| `00_DELIVERY_CURRENT/models/GBM_CPA_NIPS_model/` | 当前 NIPS CPA 模型 |
| `00_DELIVERY_CURRENT/predictions/GBM_CPA_NIPS_PW034_Panobinostat_pred.h5ad` | 当前 NIPS CPA OOD 预测 |
| `00_DELIVERY_CURRENT/evaluation/nips/CPA_NIPS_ood_metrics.json` | 当前 NIPS CPA 评估结果 |
| `00_DELIVERY_CURRENT/evaluation/unified_log1p/` | 统一 log1p 评估结果 |
| `00_DELIVERY_CURRENT/docs/` | 交接、QC、评估协议文档 |
| `00_DELIVERY_CURRENT/dataset/GBM_dataset_manifest.json` | 机器可读数据清单 |

## Compatibility Aliases

最终 GBM_NIPS_Ready.h5ad 同时包含以下字段（原始 + 兼容 alias）：

| 类别 | 原始 Key | 兼容 Alias Key |
|------|---------|---------------|
| 细胞 embedding | `obsm["X_scGPT"]` | `obsm["XscGPT"]` |
| 药物 embedding | `obsm["X_MolFormer"]` | `obsm["XMolFormer"]` |
| DEG 字典 | `uns["top50_DEGs"]` | `uns["rank_genes_groups_cov"]` |

两组 key 内容完全一致，外部方法按需选用。

## DEG Key 格式说明

- `top50_DEGs` key 格式: `patient|drug`（例: `PW034|Panobinostat`）
- `rank_genes_groups_cov` key 格式: `patient_drug`（例: `PW034_Panobinostat`）
- 两者内容完全一致（50 genes each），仅 key 分隔符不同
- 推荐 NIPS/CRISP 评估优先使用 `rank_genes_groups_cov`

## 环境要求

- conda env: `plknature`
- anndata >= 0.12, scanpy, numpy, scipy, pandas
- torch (for loading .pt models)
- geomloss (optional, for sinkhorn_de evaluation)
