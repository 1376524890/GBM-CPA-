# Current Delivery

当前交付版为 GBM NIPS 协议统一发布版 v1.1，入口文件如下：

| 类别 | 路径 |
| --- | --- |
| 最终数据集 | `dataset/GBM_NIPS_Ready.h5ad` |
| 数据清单 | `dataset/GBM_dataset_manifest.json` |
| NIPS CPA 代码 | `code/cpa_nips/` |
| NIPS CPA 模型 | `models/GBM_CPA_NIPS_model/` |
| NIPS CPA 预测 | `predictions/GBM_CPA_NIPS_PW034_Panobinostat_pred.h5ad` |
| NIPS CPA 评估 | `evaluation/nips/CPA_NIPS_ood_metrics.json` |
| 统一 log1p 评估 | `evaluation/unified_log1p/` |
| 交付文档 | `docs/` |

推荐命令：

```bash
conda activate plknature
python 00_DELIVERY_CURRENT/code/cpa_nips/check_gbm_final_h5ad.py
python 00_DELIVERY_CURRENT/code/cpa_nips/load_gbm_example.py
python 00_DELIVERY_CURRENT/code/cpa_nips/run_cpa_nips.py --skip-eval
python 00_DELIVERY_CURRENT/code/cpa_nips/evaluate_nips_gbm.py \
  --predicted 00_DELIVERY_CURRENT/predictions/GBM_CPA_NIPS_PW034_Panobinostat_pred.h5ad \
  --method CPA_NIPS
```
