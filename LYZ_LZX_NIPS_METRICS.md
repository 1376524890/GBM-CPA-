# NIPS 统一评估协议：数据划分、预测格式与指标计算说明

```text
github：https://github.com/LiuYZ2024/CRISP-PRO/tree/develop
最后更新：2026-05-18
负责人：刘耀泽/龙泽兴
```

## 1. 文档目的

本文档用于统一 NIPS/NeurIPS 数据集上的 evaluation protocol，使复现不同工作、不同 baseline 或不同 paper method 的同学，能够在同一套数据 split、perturbation group、DEG subset、metric definition 和 aggregation 方式下报告结果。

本文档的目标不是只复现当前 CRISP 模型，也不是要求所有方法都使用当前 checkpoint。更准确地说，本协议回答的是：

```text
我复现了一个 baseline 或其他 paper method，现在如何用与你们一致的方式计算 NIPS 指标？
```

核心原则：

- 任意方法只要能输出 NIPS 上的 predicted treated expression，就可以转换到本协议下评估。
- 当前 `CRISP/eval.py::evaluate(...)` 是 reference implementation，而不是唯一可用入口。
- 如果外部方法无法直接调用 `evaluate(...)`，也必须严格复刻本文档中的 valid group filtering、DEG lookup、mean-profile construction、metric definitions 和 group-level `macro average`。
- 严格对齐时，应使用同一个最终 NIPS `h5ad`、同一个 `split_key`、同一个 `rank_genes_groups_cov`，以及相同的 gene order。

一句话协议：

```text
对每个 valid cov_drug_name group，收集该 group 的真实 treated cells、同 split 同 cell_type 的 control cells、以及某个方法给出的 predicted treated cells；
除 sinkhorn_de 外，先对每个 group 构造 true/pred/control mean profile 再计算指标；
sinkhorn_de 使用 DEG subset 上的 cell-level matrices；
最后对所有 valid groups 做不加权 group-level macro average。
```

## 2. 统一评估必须固定的对象

复现其他工作时，以下对象必须固定，否则不同方法之间的结果不可公平比较。

| Object | Required protocol | Why it matters |
| --- | --- | --- |
| `h5ad` | 使用同一个最终 NIPS `h5ad`。 | 固定表达矩阵、metadata、split columns、DEG dictionary 和 gene set。 |
| `split_key` | 使用同一个 `split` / `split2` / `split3`。 | 固定 IID/OOD 评估集；不同 split key 是不同测试集。 |
| control definition | `adata.obs["neg_control"] == 1` 为 control，非 1 为 treated。 | 固定 treated/control 划分。 |
| perturbation group | 使用 `cov_drug_name`。 | 固定 group-level evaluation unit。 |
| DEG dictionary | 使用 `adata.uns["rank_genes_groups_cov"]`。 | 固定 `*_de` metrics 的 gene subset。 |
| gene order | 保持 `adata.var_names` 顺序不变。 | 保证 prediction columns、DEG indices 和 expression columns 一致。 |
| aggregation | 使用 unweighted group-level `macro average`。 | 保证汇总口径一致，不受 group cell 数影响。 |

特别强调：如果某个 baseline 原论文使用了自己的 split、自己的 DEGs 或自己的 aggregation，仍应在本文档定义的 NIPS protocol 上重新评估，才能与当前结果对齐。

## 3. 对任意模型的预测输出要求

这是接入外部 baseline 或 paper method 时最关键的部分。本协议不要求模型结构与 CRISP 相同，只要求最终预测能整理成统一格式。

### 接入方式 A：直接接入当前 `evaluate(...)`

如果外部方法可以包装成与当前 CRISP model 类似的接口，使其在每个 group 下从 same-cell-type control cells 生成 `Y_pred`，可以直接调用 reference implementation：

```python
evaluate(model, treated_dataset, control_dataset)
```

这里的关键不是模型必须是 CRISP，而是 `evaluate(...)` 在内部会按照统一规则：

1. 遍历 `treated_dataset.pert_categories` 中的 groups；
2. 找到同 split、同 cell type 的 control cells；
3. 调用模型生成 predicted treated cells；
4. 使用 `calc_metrics(...)` 计算 per-group metrics；
5. 对 valid groups 做 unweighted macro average。

如果能够复用这一入口，最容易避免 group filtering、DEG lookup、R2 clipping、NaN handling 和 aggregation 的细节偏差。

### 接入方式 B：外部模型已经生成预测矩阵

很多复现工作不会使用当前 `evaluate(...)`，而是已经通过自己的 pipeline 得到了 predicted expression matrix。此时请将预测整理为如下 per-group 格式：

```python
predictions = {
    group_name: {
        "Y_true": np.ndarray,  # real treated cells, shape [n_treated, n_genes]
        "Y_pred": np.ndarray,  # predicted treated cells, shape [n_pred, n_genes]
        "Y_ctrl": np.ndarray,  # matched control cells, shape [n_control, n_genes]
    }
}
```

要求：

- `group_name` 必须对应 `adata.obs["cov_drug_name"]`。
- `Y_true` 必须是该 group 在指定 `split_key` / setting 中的真实 treated cells。
- `Y_ctrl` 必须是同一 evaluation split、同一 `cell_type` 的 control cells，即 `neg_control == 1` 的 cells。
- `Y_pred` 是该方法对该 group 的 predicted treated expression。
- `Y_pred` 的行数可以等于 matched control cells 数，也可以等于 treated cells 数；mean-profile metrics 只使用均值，因此不要求一一配对。
- 如果要计算 `sinkhorn_de`，`Y_pred` 的行数、采样策略和分布会影响结果，因此必须明确报告 prediction sampling strategy。
- 所有矩阵的 gene columns 必须与 `adata.var_names` 完全一致，包括顺序。
- 所有方法必须使用同一批 valid groups；不能为某个方法自定义筛选 group 子集。

## 4. NIPS split 和 group 定义

NIPS 评估使用 `split_key` 指定的 AnnData split column。该列包含：

- `train`：训练集。
- `test`：IID evaluation 使用的 held-out samples。
- `ood`：OOD evaluation 使用的 held-out samples。

在当前 reference implementation 中：

```text
IID = test_treated + test_control
OOD = ood_treated + ood_control
```

对应的子集由 `CRISP/data.py` 和 `CRISP/trainer.py` 构造：

```python
"test_treated" = dataset.subset("test", "treated")
"test_control" = dataset.subset("test", "control")
"ood_treated" = dataset.subset("ood", "treated")
"ood_control" = dataset.subset("ood", "control")
```

NIPS preprocessing notebook 中包含三套不同 OOD cell-type settings：

```python
split_dataset(adata, ["Myeloid cells", "T regulatory cells"], "split")
split_dataset(adata, ["T cells CD4+", "B cells"], "split2")
split_dataset(adata, ["T cells CD8+", "NK cells"], "split3")
```

注意事项：

- `split`、`split2`、`split3` 是三套不同测试设定，不能直接当作同一个测试集比较。
- 如果要比较多个方法，所有方法必须使用同一个 `split_key`。
- 即使某篇原论文使用自己的 split，也应在本文档定义的 NIPS split 上重新评估，才能与当前协议结果对齐。
- 严格复现时，不要重新从 raw data 生成 split。NIPS notebook 中的 `group.sample(frac=...)` 没有固定 `random_state`，重新生成可能得到不同逐细胞划分。

评估 group 定义：

```text
cov_drug_name = cell_type + "_" + condition
```

其中 `condition` 是清洗后的药物名。group-level 指标以 `cov_drug_name` 为基本单位。

## 5. Valid group filtering

所有方法都必须使用相同的 valid group filtering。否则，即使 per-group metric 完全一致，最终 `macro average` 也不可比较。

| Filtering rule | Required behavior | Why it matters |
| --- | --- | --- |
| treated cell count > 5 | 当前 group 的真实 treated cells 数量必须大于 5。 | 避免不稳定的 group mean。 |
| group name 不包含 `dmso` 或 `control` | case-insensitive 检查；包含则跳过。 | control group 本身不是 perturbation response。 |
| group 在 `rank_genes_groups_cov` 中有 DEG entry | 必须能用 group key 找到 DEG genes。 | `*_de` metrics 需要固定 DEG subset。 |
| DEG count >= 2 | DEG indices 少于 2 的 group 跳过。 | R2 / Pearson 至少需要足够 gene 数。 |
| matched control cell count >= 5 | 同 split、同 cell type 的 control cells 数量必须至少为 5。 | 避免 control mean 和预测分布不稳定。 |

重要原则：不同方法不能自行选择 group 子集。例如，不能因为某个 baseline 对某些 drug/cell type 没有预测就从 aggregate 中删除这些 group；这会改变 `macro average` 的含义。若某方法无法覆盖所有 valid groups，需要在结果表中明确说明 coverage，并避免与 full-coverage 方法直接比较。

## 6. Mean-profile construction

对每个 valid group，统一构造：

```python
y = mean(Y_true, axis=0)  # true treated mean profile
p = mean(Y_pred, axis=0)  # predicted treated mean profile
c = mean(Y_ctrl, axis=0)  # matched control mean profile
```

然后按如下方式计算：

- `r2score`、`pearson`、`mse`、`pearson_delta` 在 all genes 的 `y`、`p`、`c` 上计算。
- `r2score_de`、`pearson_de`、`mse_de`、`pearson_delta_de` 在 DEG subset `D` 上计算。
- `sinkhorn_de` 不使用 mean profile，而是使用 cell-level matrices：`Y_true[:, D]` 和 `Y_pred[:, D]`。

本协议不是逐细胞配对评估协议。除 `sinkhorn_de` 外，不要求 `Y_true` 和 `Y_pred` 有相同行数，也不要求一一配对。关键是每个 group 的 `mean profile` 和 matched control definition 一致。

## 7. Metric definitions

记号：

- `G`：全部 genes。
- `D`：当前 group 的 DEG subset，来自 `rank_genes_groups_cov[group]`。
- `Y_true`：真实 treated expression matrix。
- `Y_pred`：预测 treated expression matrix。
- `Y_ctrl`：matched control expression matrix。
- `y = mean(Y_true, axis=0)`。
- `p = mean(Y_pred, axis=0)`。
- `c = mean(Y_ctrl, axis=0)`。

### `r2score`

- Input: `y_G`, `p_G`。
- Definition:

```text
R2(y, p) = 1 - sum_j (y_j - p_j)^2 / sum_j (y_j - mean(y))^2
r2score = max(R2(y_G, p_G), 0)
```

- Protocol behavior: R2 负值截断为 0；reference implementation 使用 `torchmetrics.R2Score`，并将 prediction clamp 到 `[-3e12, 3e12]`。
- Direction: higher is better。

### `r2score_de`

- Input: `y_D`, `p_D`。
- Definition:

```text
r2score_de = max(R2(y_D, p_D), 0)
```

- Protocol behavior: 与 `r2score` 相同，但只在 DEG subset 上计算。
- Direction: higher is better。
- Notes: DEG subset 必须来自 `rank_genes_groups_cov`。

### `pearson`

- Input: `y_G`, `p_G`。
- Definition:

```text
pearson = corr(y_G, p_G)
```

- Protocol behavior: Pearson 结果为 `NaN` 时置为 0。
- Direction: higher is better。

### `pearson_de`

- Input: `y_D`, `p_D`。
- Definition:

```text
pearson_de = corr(y_D, p_D)
```

- Protocol behavior: Pearson 结果为 `NaN` 时置为 0；如果 true 或 pred 的 DEG mean vector 总和为 0，则先将第一个元素加 `1e-6`，这是 reference implementation 中的 zero-sum guard。
- Direction: higher is better。
- Notes: 必须保留 zero-sum guard 才能与 `CRISP/eval.py::calc_metrics(...)` 精确一致。

### `mse`

- Input: `y_G`, `p_G`。
- Definition:

```text
mse = mean_j (y_j - p_j)^2
```

- Protocol behavior: reference implementation 使用 `sklearn.metrics.mean_squared_error`。
- Direction: lower is better。

### `mse_de`

- Input: `y_D`, `p_D`。
- Definition:

```text
mse_de = mean_j_in_D (y_j - p_j)^2
```

- Protocol behavior: 在 DEG subset 上计算 MSE。
- Direction: lower is better。

### `pearson_delta`

- Input: `(y_G - c_G)`, `(p_G - c_G)`。
- Definition:

```text
true_delta = y_G - c_G
pred_delta = p_G - c_G
pearson_delta = corr(true_delta, pred_delta)
```

- Protocol behavior: Pearson 结果为 `NaN` 时置为 0。
- Direction: higher is better。
- Notes: 比较的是 treated-control perturbation effect，不是原始 expression similarity。

### `pearson_delta_de`

- Input: `(y_D - c_D)`, `(p_D - c_D)`。
- Definition:

```text
true_delta_de = y_D - c_D
pred_delta_de = p_D - c_D
pearson_delta_de = corr(true_delta_de, pred_delta_de)
```

- Protocol behavior: Pearson 结果为 `NaN` 时置为 0。
- Direction: higher is better。
- Notes: 这是 DEG-restricted perturbation-effect correlation。

### `sinkhorn_de`

- Input: `Y_true[:, D]`, `Y_pred[:, D]`。
- Definition:

```text
sinkhorn_de = SamplesLoss(loss="sinkhorn", blur=0.05)(
    Y_true[:, D],
    Y_pred[:, D],
)
```

- Protocol behavior: 使用 `geomloss.SamplesLoss(loss="sinkhorn", blur=0.05)`；如果 `Y_true[:, D]` 和 `Y_pred[:, D]` 的总和都为 0，则记为 0。
- Direction: lower is better。
- Notes: 这是当前唯一启用的 cell-level matrix metric；`Y_pred` 的采样策略会影响该指标。

## 8. Aggregation

所有方法都必须先对每个 valid group 分别计算 metric，再对 group 做 unweighted `macro average`：

```text
aggregate_metric = mean(metric(group_1), ..., metric(group_K))
```

其中 `K` 是所有方法共同采用的 valid group 数量。

禁止的替代做法：

- 不要按 cell 数加权。
- 不要先拼接所有 cells 后统一计算 Pearson/MSE/R2。
- 不要为某个方法自定义 group subset 后再 aggregate。
- 不要把不同 `split_key` 的结果当作同一个测试集汇总。

这个规则保证每个 perturbation group 在最终结果中权重相同，避免结果被大 cell-count group 主导。

## 9. Reference implementation in this repository

当前仓库提供了一套 reference implementation，可用于直接评估 CRISP，也可作为外部 evaluator 的实现参考。

| 文件或输出 | 作用 |
| --- | --- |
| `CRISP/eval.py::evaluate(...)` | 遍历 groups、生成/收集预测、过滤 valid groups、调用 `calc_metrics(...)` 并做 macro average。 |
| `CRISP/eval.py::calc_metrics(...)` | 计算 `r2score`、`pearson`、`mse`、`pearson_delta`、`sinkhorn_de` 及其 DEG 版本。 |
| `experiments/configs/nips.yaml` | NIPS 字段配置，包含 `pert_category: cov_drug_name`、`control_key: neg_control`、`degs_key: rank_genes_groups_cov`、`FM_key: X_scGPT`、`pc_cov: type_donor` 等。 |
| `CRISP/data.py` | 定义 `Dataset` / `SubDataset`，并构造 train/test/ood 与 treated/control 子集。 |
| `CRISP/trainer.py` | 在训练过程中调用 `evaluate(...)`，并保存评估输出。 |
| `eval_stats.pkl` | aggregate metrics，结构通常为 `{"iid": {...}, "ood": {...}}`。 |
| `eval_stats_all.pkl` | per-group metrics，结构通常为 `{"iid": {group: {...}}, "ood": {group: {...}}}`。 |
| `pred_mean.pkl` | per-group `true` / `pred` / `ctrl` mean profile；不能复算 `sinkhorn_de`。 |

如果复现方法可以接入当前代码，推荐直接调用 reference implementation。如果不能，也必须复刻同样的 filtering、DEG lookup、mean-profile metrics、`sinkhorn_de` 和 aggregation 逻辑。

注意：当前 `CRISP/eval.py` 中存在 hard-coded `.to("cuda")` 路径。CPU-only 环境可能需要 CPU-compatible patch，但任何 patch 都必须保持评估逻辑不变，并在报告中说明。

## 10. Minimal evaluator pseudocode

下面展示外部 `predictions` 如何按照统一协议计算指标。它是协议伪代码，不是替代 `CRISP/eval.py::calc_metrics(...)` 的精确实现。

```python
metrics = {}

for group in valid_groups:
    Y_true = predictions[group]["Y_true"]
    Y_pred = predictions[group]["Y_pred"]
    Y_ctrl = predictions[group]["Y_ctrl"]

    D = deg_dict[group]  # convert gene names to adata.var_names indices

    y = Y_true.mean(axis=0)
    p = Y_pred.mean(axis=0)
    c = Y_ctrl.mean(axis=0)

    metrics[group] = calc_metrics(
        y=y,
        p=p,
        c=c,
        Y_true=Y_true,
        Y_pred=Y_pred,
        D=D,
    )

aggregate = {
    metric_name: np.mean([metrics[g][metric_name] for g in metrics])
    for metric_name in metric_names
}
```

精确复现时，请参考 `CRISP/eval.py::calc_metrics(...)` 中的 R2 clipping、Pearson NaN-to-zero、`pearson_de` zero-sum guard 和 `sinkhorn_de` 计算方式。

## 11. 给复现其他工作的同学的 checklist

- [ ] 我是否使用了同一个最终 NIPS `h5ad`？
- [ ] 我是否使用了同一个 `split_key`（`split` / `split2` / `split3`）？
- [ ] 我是否使用了同一个 `cov_drug_name` group 定义？
- [ ] 我是否使用了同一个 `neg_control` control 定义？
- [ ] 我是否使用了同一个 `rank_genes_groups_cov`？
- [ ] 我的 prediction gene order 是否与 `adata.var_names` 完全一致？
- [ ] 我的 `Y_ctrl` 是否来自同 split、同 `cell_type` 的 control cells？
- [ ] 我的 mean-profile 指标是否先 group mean 再计算？
- [ ] 我的 aggregate 是否是 group-level unweighted `macro average`？
- [ ] 我的 R2 / Pearson / Sinkhorn 细节是否与 reference implementation 一致？
- [ ] 我是否报告了 `split_key`、method name、valid group count 和 prediction sampling strategy？

## 12. 结果报告模板

该表用于比较不同方法，例如 CRISP、baseline A、baseline B、reproduced paper method 等。

| Dataset | Split Key | Method | Setting | #Groups | Pearson ↑ | Pearson DE ↑ | Delta Pearson ↑ | Delta Pearson DE ↑ | R2 ↑ | R2 DE ↑ | MSE ↓ | MSE DE ↓ | Sinkhorn DE ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NIPS | split | CRISP | iid | `len(eval_stats_all["iid"])` | `eval_stats["iid"]["pearson"]` | `eval_stats["iid"]["pearson_de"]` | `eval_stats["iid"]["pearson_delta"]` | `eval_stats["iid"]["pearson_delta_de"]` | `eval_stats["iid"]["r2score"]` | `eval_stats["iid"]["r2score_de"]` | `eval_stats["iid"]["mse"]` | `eval_stats["iid"]["mse_de"]` | `eval_stats["iid"]["sinkhorn_de"]` |
| NIPS | split | baseline A | ood | `<# valid groups>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` | `<value>` |

说明：

- `Setting` 可以是 `iid` 或 `ood`。
- `#Groups` 必须来自统一 valid group filtering。
- 若使用 reference implementation 输出，metric values 来自 `eval_stats.pkl`，`#Groups` 来自 `len(eval_stats_all[setting])`。
- 若外部方法 coverage 不完整，必须单独标注，不应直接与 full valid-group coverage 的方法比较。

## 13. Common pitfalls

### Pitfall: 使用论文原始 split 而不是当前 NIPS split

Symptom: 方法结果与当前协议下结果无法对齐。

Cause: 原论文 split 与本协议 `split` / `split2` / `split3` 不同。

Fix: 在最终 NIPS `h5ad` 的指定 `split_key` 上重新评估所有方法。

### Pitfall: 使用自己的 DEG

Symptom: `*_de` metrics 与 reference 差异很大。

Cause: 重新计算或替换了 `rank_genes_groups_cov`。

Fix: 使用最终 `h5ad` 中保存的 `adata.uns["rank_genes_groups_cov"]`。

### Pitfall: 使用 cell-weighted average

Symptom: per-group metrics 接近，但 aggregate metrics 不一致。

Cause: 按 cell 数加权，而不是 group-level unweighted `macro average`。

Fix: 对 valid groups 的 per-group metrics 做简单平均。

### Pitfall: 直接逐细胞算 Pearson/MSE

Symptom: Pearson/MSE 与协议结果系统性不同。

Cause: 本协议除 `sinkhorn_de` 外使用 group `mean profile`，不是 cell-level paired metrics。

Fix: 先构造 `y = mean(Y_true)`、`p = mean(Y_pred)`、`c = mean(Y_ctrl)`，再计算指标。

### Pitfall: gene order 不一致

Symptom: 所有指标异常，尤其 DEG metrics 不可靠。

Cause: prediction columns 与 `adata.var_names` 顺序不一致。

Fix: 在评估前按 `adata.var_names` 重排预测矩阵列。

### Pitfall: `Y_ctrl` 选择不一致

Symptom: `pearson_delta` / `pearson_delta_de` 明显不一致。

Cause: control cells 没有来自同 split、同 cell type，或 control definition 不是 `neg_control == 1`。

Fix: 使用同 evaluation split、同 `cell_type`、`neg_control == 1` 的 cells 作为 `Y_ctrl`。

### Pitfall: 从 mean profile 复算 `sinkhorn_de`

Symptom: 无法复现 reference `sinkhorn_de`。

Cause: `sinkhorn_de` 需要 `Y_true[:, D]` 和 `Y_pred[:, D]` cell-level matrices。

Fix: 保存 cell-level predictions 或直接使用 `eval_stats_all.pkl` / reference evaluator 输出。

### Pitfall: 不同 `split_key` 直接比较

Symptom: `split`、`split2`、`split3` 结果被混在一个 benchmark 中。

Cause: 三者对应不同 OOD cell-type settings。

Fix: 分开报告不同 `split_key`，或只在同一 `split_key` 内比较方法。

### Pitfall: 自定义过滤 group

Symptom: 某方法 aggregate 更高，但 valid group count 更少。

Cause: 该方法跳过了难预测或无预测的 groups。

Fix: 所有方法使用同一 valid group set；如果 coverage 不完整，必须单独报告并避免直接公平比较。

### Pitfall: group key 中包含额外 underscore

Symptom: DEG lookup 失败或 cell type 解析错误。

Cause: reference implementation 使用 `split("_")` 解析 group key。

Fix: NIPS 使用保存好的 `cov_drug_name` 和清洗后的 `condition`；新数据需要避免额外 `_` 或同步记录修改后的解析逻辑。

### Pitfall: CPU 环境运行 reference implementation 失败

Symptom: 出现 `.to("cuda")` 相关错误。

Cause: 当前 `CRISP/eval.py` 中存在 CUDA-specific transfer。

Fix: 使用 CUDA 环境，或应用 CPU-compatible patch；patch 后必须确认评估逻辑未改变。

## 14. 需要发送给合作者的材料

| 材料 | 用途 |
| --- | --- |
| repository link / reference implementation | 提供 `CRISP/eval.py`、`CRISP/data.py`、`experiments/configs/nips.yaml` 等参考实现。 |
| code commit hash | 固定 reference implementation 版本。 |
| `NIPS_METRICS.md` | 本统一评估协议文档。 |
| final NIPS `h5ad` path/version | 固定 expression matrix、metadata、split columns、gene order 和 DEG dictionary。 |
| `experiments/configs/nips.yaml` | 固定字段映射，如 `pert_category: cov_drug_name`、`control_key: neg_control`、`degs_key: rank_genes_groups_cov`、`FM_key: X_scGPT`、`pc_cov: type_donor`。 |
| DEG dictionary already in h5ad | 即 `adata.uns["rank_genes_groups_cov"]`，用于所有 `*_de` metrics。 |
| example `eval_stats.pkl` | 示例 aggregate metrics，可用于检查读取格式和结果表。 |
| example `eval_stats_all.pkl` | 示例 per-group metrics，可用于检查 valid group set。 |
| example `pred_mean.pkl` | 示例 mean profiles，可用于理解 mean-profile 指标；不能复算 `sinkhorn_de`。 |
| optional evaluator script | 如果另行封装了外部 predictions evaluator，应一并发送。 |
| `requirement.txt` 或 environment file | 固定 package versions，尤其是 `torchmetrics`、`scipy`、`scikit-learn`、`geomloss`。 |
| optional CUDA availability notes | 说明 reference implementation 是否需要 CUDA，或是否提供 CPU-compatible patch。 |

发送这些材料的目的，是让合作者可以把任意方法的预测结果接入同一套 NIPS evaluation protocol，而不是只复现某一个 CRISP checkpoint。
