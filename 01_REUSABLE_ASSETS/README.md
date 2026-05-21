# Reusable Assets

这里存放一直复用到当前交付版、但不是交付入口的资产。

| 类别 | 内容 |
| --- | --- |
| `preprocessed_data/` | `GBM_Universal_Perturbation_Ready.h5ad`、`GBM_with_embeddings.h5ad`、`GBM_scGPT_embeddings.h5ad`、`GBM_counts_for_scgpt.h5ad`、`GBM_X_MolFormer.npy` |
| `embeddings/` | MolFormer 药物 embedding、scGPT aligner 权重、scGPT 对齐矩阵 |
| `source_geo/GBM_dataset/` | GEO 下载、映射和 CPA-ready 源数据 |

这些文件不归档为历史版，因为当前交付版仍依赖它们进行复现、重建或对比实验。
