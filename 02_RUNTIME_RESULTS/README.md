# Runtime Results

这里存放训练、预测、评估和崩溃转储等运行产物。

| 类别 | 内容 |
| --- | --- |
| `predictions/legacy_comparison/` | M0/M1/M2/M3/M4/M5、MeanShift、MLP 等旧对比预测 |
| `models/legacy_comparison/` | 旧对比实验训练出的 CPA 模型 |
| `evaluation/legacy_nips/` | 旧对比实验 NIPS/CRISP 指标 |
| `logs/` | 训练日志、预测日志、scVI/lightning 日志、pycache |
| `crash_dumps/` | `core` 等崩溃转储 |

新实验运行时，先写入本目录；确认成为交付版后，再提升到 `00_DELIVERY_CURRENT/` 并更新 `management/FILE_MANAGEMENT.md`。
