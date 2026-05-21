# Code

这里存放非当前交付入口的复现、预处理和历史对比脚本。

| 目录 | 内容 |
| --- | --- |
| `preprocessing/` | GEO/embedding/NIPS 格式生成脚本，默认读写 `01_REUSABLE_ASSETS/` 和 `00_DELIVERY_CURRENT/dataset/` |
| `comparison/` | 历史对比实验脚本，默认读写 `01_REUSABLE_ASSETS/` 和 `02_RUNTIME_RESULTS/` |
| `utilities/` | 通用工具脚本 |

当前 NIPS 原始字段 CPA 交付代码在 `00_DELIVERY_CURRENT/code/cpa_nips/`。
