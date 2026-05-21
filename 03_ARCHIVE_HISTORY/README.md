# Historical Archive

历史内容按版本或来源归档，同一版本的文档、模型和输出必须放在同一个子目录下。

| 目录 | 版本/来源 | 说明 |
| --- | --- | --- |
| `v0_tcga_mri_pipeline/` | v0 早期 TCGA/MRI 多模态流程 | 与当前 NIPS GBM 单细胞交付无直接依赖 |
| `v1_0_cpa_outputs/` | v1.0 CPA PW034/Panobinostat 输出包 | 旧输出包、旧 manifest、旧报告 |
| `v1_0_pre_alias_backup/` | v1.0 alias 兼容前备份 | `GBM_NIPS_Ready_before_compatible_alias_backup.h5ad` 和旧 alias 文件 |
| `v1_0_previous_model/` | v1.0 previous CPA model | `GBM_CPA_model.previous/` |
| `v1_1_old_layout_release_manifest/` | v1.1 旧目录布局清单 | 整理前 `release_unified/GBM_dataset_manifest.json` |

仍被当前交付版复用的源数据、embedding、aligner 和中间 h5ad 不放在这里，统一放入 `01_REUSABLE_ASSETS/`。
