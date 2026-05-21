# File Management Standard

整理日期：2026-05-21  
整理前快照：`management/original_structure_20260521_173411_lightweight/`

## 目录职责

| 目录 | 职责 |
| --- | --- |
| `00_DELIVERY_CURRENT/` | 当前可交付内容，只放最终版数据、当前 CPA NIPS 代码、当前模型/预测/评估和交付文档 |
| `01_REUSABLE_ASSETS/` | 当前版仍复用的源数据、中间 h5ad、embedding、aligner、药物表 |
| `02_RUNTIME_RESULTS/` | 训练日志、旧预测、旧模型、临时评估、pycache、core dump |
| `03_ARCHIVE_HISTORY/` | 不再作为当前流程入口的历史版本，按版本归档 |
| `04_DOCS_REFERENCES/` | 背景资料、旧报告、旧协议说明 |
| `05_CODE/` | 非交付入口的预处理、历史对比和工具脚本 |
| `management/` | 目录快照、文件管理规范、后续迭代记录 |

## 当前交付版

版本：GBM NIPS unified v1.1  
唯一推荐数据集：`00_DELIVERY_CURRENT/dataset/GBM_NIPS_Ready.h5ad`  
适用于 NIPS 原始字段的 CPA 代码：`00_DELIVERY_CURRENT/code/cpa_nips/`

当前交付内容包括：

| 类别 | 路径 |
| --- | --- |
| 最终数据集 | `00_DELIVERY_CURRENT/dataset/GBM_NIPS_Ready.h5ad` |
| 数据清单 | `00_DELIVERY_CURRENT/dataset/GBM_dataset_manifest.json` |
| CPA NIPS 训练/预测/评估代码 | `00_DELIVERY_CURRENT/code/cpa_nips/` |
| 当前 CPA NIPS 模型 | `00_DELIVERY_CURRENT/models/GBM_CPA_NIPS_model/` |
| 当前 CPA NIPS 预测 | `00_DELIVERY_CURRENT/predictions/GBM_CPA_NIPS_PW034_Panobinostat_pred.h5ad` |
| 当前 CPA NIPS 指标 | `00_DELIVERY_CURRENT/evaluation/nips/CPA_NIPS_ood_metrics.json` |
| 统一 log1p 指标 | `00_DELIVERY_CURRENT/evaluation/unified_log1p/` |
| 文档 | `00_DELIVERY_CURRENT/docs/` |

## 当前仍复用但不作为交付入口的内容

这些内容不归档到历史版本，因为后续重建、复现或扩展仍会复用：

| 类别 | 路径 |
| --- | --- |
| 通用扰动 h5ad | `01_REUSABLE_ASSETS/preprocessed_data/GBM_Universal_Perturbation_Ready.h5ad` |
| embedding 合并中间 h5ad | `01_REUSABLE_ASSETS/preprocessed_data/GBM_with_embeddings.h5ad` |
| scGPT embedding h5ad | `01_REUSABLE_ASSETS/preprocessed_data/GBM_scGPT_embeddings.h5ad` |
| scGPT counts 输入 | `01_REUSABLE_ASSETS/preprocessed_data/GBM_counts_for_scgpt.h5ad` |
| MolFormer cell-level 矩阵 | `01_REUSABLE_ASSETS/preprocessed_data/GBM_X_MolFormer.npy` |
| MolFormer drug-level 表 | `01_REUSABLE_ASSETS/embeddings/GBM_molformer_drug_emb.parquet` |
| MolFormer metadata | `01_REUSABLE_ASSETS/embeddings/GBM_molformer_drug_emb.metadata.json` |
| scGPT aligner 和对齐矩阵 | `01_REUSABLE_ASSETS/embeddings/GBM_scGPT_*` |
| GEO 源数据和 CPA-ready 数据 | `01_REUSABLE_ASSETS/source_geo/GBM_dataset/` |

## 历史归档

| 版本 | 目录 | 内容 |
| --- | --- | --- |
| v0 | `03_ARCHIVE_HISTORY/v0_tcga_mri_pipeline/` | 早期 TCGA/MRI 多模态数据流程 |
| v1.0 | `03_ARCHIVE_HISTORY/v1_0_cpa_outputs/` | 旧 CPA PW034/Panobinostat 输出包 |
| v1.0 | `03_ARCHIVE_HISTORY/v1_0_pre_alias_backup/` | alias 兼容前备份和旧 alias h5ad |
| v1.0 | `03_ARCHIVE_HISTORY/v1_0_previous_model/` | previous CPA baseline 模型 |

## 后续迭代规则

1. 开始整理或大规模移动前，先在 `management/original_structure_YYYYMMDD_HHMMSS/` 记录目录快照和 `git status --short`。
2. 新实验先写入 `02_RUNTIME_RESULTS/`，不得直接覆盖 `00_DELIVERY_CURRENT/`。
3. 只有经过 QC、路径复查和文档更新的结果才能提升到 `00_DELIVERY_CURRENT/`。
4. 旧交付版下线时，整体移动到 `03_ARCHIVE_HISTORY/vX_Y_<short_name>/`，同版本数据、模型、代码、指标和说明放在一起。
5. 仍被新版复用的文件不要归档；保留在 `01_REUSABLE_ASSETS/`，并在本文“当前仍复用”表中列明。
6. 每次路径调整后运行 `rg` 检查代码中的旧路径引用，并至少运行语法检查。
7. 文档中凡是作为当前入口的路径，必须指向 `00_DELIVERY_CURRENT/`；历史说明可以保留旧文件名，但要标注为历史。

## 路径复查命令

```bash
rg -n "GBM_NIPS_Ready|GBM_Universal|GBM_CPA_|evaluation_results|release_unified|scripts/" \
  00_DELIVERY_CURRENT/code 05_CODE 04_DOCS_REFERENCES

python -m compileall 00_DELIVERY_CURRENT/code 05_CODE
```
