# GBM 扰动预测统一评估接入协议

**版本**: v1.0
**最后更新**: 2026-05-19
**适用范围**: 所有需要在 GBM 数据集上进行扰动预测评估的方法

---

## 1. 评估概述

本协议定义了如何将任何方法的预测结果转换为统一的评估格式。遵循此协议后，可直接使用 `scripts/evaluate_nips_gbm.py` 或 `scripts/evaluate_crisp_ood.py` 进行评估。

---

## 2. 评估数据加载

```python
import anndata as ad
import numpy as np

BASE = "/home/u2023312303/nature子刊/裴立昆实验"
adata = ad.read_h5ad(f"{BASE}/GBM_NIPS_Ready.h5ad")
```

---

## 3. 预测结果格式

### 3.1 Per-Group Predictions Dict

所有方法必须输出如下格式的 predictions dict:

```python
predictions = {
    group_name: {
        "Y_true": np.ndarray,   # (n_treated_cells, 5000) float32
        "Y_pred": np.ndarray,   # (n_pred_cells, 5000) float32
        "Y_ctrl": np.ndarray,   # (n_ctrl_cells, 5000) float32
    }
}
```

### 3.2 格式要求

| 要求 | 说明 |
|------|------|
| `group_name` | 必须来自 `adata.obs["cov_drug_name"]` |
| `Y_true` | 真实 treated 细胞的 log1p 表达 |
| `Y_ctrl` | 同患者 control 细胞的 log1p 表达 |
| `Y_pred` | 模型预测的表达（应与 adata.X 在同一空间） |
| 列顺序 | 必须与 `adata.var_names` 完全一致 |
| 形状 | (n_cells, 5000) |

### 3.3 构造示例

```python
def build_prediction_dict(adata, model_predict_fn):
    """为所有 valid groups 构造 predictions dict."""
    predictions = {}
    
    valid_groups = get_valid_groups(adata)  # 参见 4.1 节
    
    for group_name in valid_groups:
        # 获取 treated 细胞
        treated_mask = adata.obs["cov_drug_name"] == group_name
        Y_true = adata[treated_mask].X.toarray()
        
        # 获取 matched control
        patient = group_name.split("_")[0]
        ctrl_group = f"{patient}_control"
        ctrl_mask = adata.obs["cov_drug_name"] == ctrl_group
        Y_ctrl = adata[ctrl_mask].X.toarray()
        
        # 模型预测（使用 control 细胞作为输入）
        Y_pred = model_predict_fn(
            control_cells=adata[ctrl_mask],
            target_drug=group_name.split("_")[1]
        )
        
        predictions[group_name] = {
            "Y_true": Y_true,
            "Y_pred": Y_pred,
            "Y_ctrl": Y_ctrl,
        }
    
    return predictions
```

---

## 4. Valid Group 过滤

### 4.1 过滤规则

```python
def get_valid_groups(adata):
    """获取所有可用于评估的 valid group."""
    deg_dict = adata.uns["top50_DEGs"]
    groups = []
    skipped = {
        "treated_too_few": [],
        "is_control_group": [],
        "deg_missing": [],
        "deg_too_few": [],
        "ctrl_too_few": [],
    }
    
    for group_name in adata.obs["cov_drug_name"].unique():
        # 跳过 control groups
        if "control" in group_name.lower():
            skipped["is_control_group"].append(group_name)
            continue
        
        # 检查 treated 细胞数量
        n_treated = (adata.obs["cov_drug_name"] == group_name).sum()
        if n_treated <= 5:
            skipped["treated_too_few"].append(group_name)
            continue
        
        # 检查 DEG
        deg_key = group_name.replace("_", "|")  # 注意 key 格式转换
        if deg_key not in deg_dict:
            skipped["deg_missing"].append(group_name)
            continue
        
        deg_genes = deg_dict[deg_key]
        if len(deg_genes) < 2:
            skipped["deg_too_few"].append(group_name)
            continue
        
        # 检查 matched control 数量
        patient = group_name.split("_")[0]
        ctrl_group = f"{patient}_control"
        n_ctrl = (adata.obs["cov_drug_name"] == ctrl_group).sum()
        if n_ctrl < 5:
            skipped["ctrl_too_few"].append(group_name)
            continue
        
        groups.append(group_name)
    
    return groups, skipped
```

### 4.2 当前 Valid Groups (18)

```
GS359_Temozolomide, GS772_Temozolomide, GS785_Temozolomide, GS789_Temozolomide,
PW029_Etoposide,
PW030_Ana-12, PW030_Etoposide, PW030_Ispenisib, PW030_Panobinostat,
PW030_RO4929097, PW030_Tazemetostat,
PW032_Etoposide, PW032_Panobinostat,
PW034_Etoposide, PW034_Panobinostat,       # PW034_Panobinostat = OOD
PW036_Etoposide, PW036_Panobinostat,
PW040_Panobinostat
```

---

## 5. 指标计算协议

### 5.1 Mean-Profile Metrics

对于每个 group，先计算均值再计算指标：

```python
def compute_mean_profile_metrics(Y_true, Y_pred, Y_ctrl, deg_genes=None):
    """Return dict of mean-profile metrics."""
    if deg_genes is not None:
        # 子集化到 DEG genes
        deg_idx = [list(adata.var_names).index(g) for g in deg_genes]
        Y_true = Y_true[:, deg_idx]
        Y_pred = Y_pred[:, deg_idx]
        Y_ctrl = Y_ctrl[:, deg_idx]
    
    # 均值 profile
    mu_true = Y_true.mean(axis=0)
    mu_pred = Y_pred.mean(axis=0)
    mu_ctrl = Y_ctrl.mean(axis=0)
    
    # Pearson (全基因 or DEG)
    pearson = np.corrcoef(mu_true, mu_pred)[0, 1]
    
    # Pearson delta (treatment effect)
    delta_true = mu_true - mu_ctrl
    delta_pred = mu_pred - mu_ctrl
    pearson_delta = np.corrcoef(delta_true, delta_pred)[0, 1]
    
    # R2
    ss_res = np.sum((mu_true - mu_pred) ** 2)
    ss_tot = np.sum((mu_true - mu_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    # MSE
    mse = np.mean((mu_true - mu_pred) ** 2)
    
    return {
        "pearson": pearson,
        "pearson_delta": pearson_delta,
        "r2score": r2,
        "mse": mse,
    }
```

### 5.2 Sinkhorn DE (Cell-Level DEG)

Sinkhorn 距离必须在 **DEG 子集**的**细胞级矩阵**上计算：

```python
def compute_sinkhorn_de(Y_true, Y_pred, deg_genes):
    """Sinkhorn distance on DEG-subset cell-level matrices."""
    import geomloss
    deg_idx = [list(adata.var_names).index(g) for g in deg_genes]
    
    Y_true_de = Y_true[:, deg_idx]
    Y_pred_de = Y_pred[:, deg_idx]
    
    # L2 normalize
    Y_true_de = Y_true_de / np.linalg.norm(Y_true_de, axis=1, keepdims=True)
    Y_pred_de = Y_pred_de / np.linalg.norm(Y_pred_de, axis=1, keepdims=True)
    
    sinkhorn = geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=0.05)
    return sinkhorn(Y_true_de, Y_pred_de).item()
```

### 5.3 指标命名约定

| 全基因指标 | DEG 子集指标 | 说明 |
|-----------|-------------|------|
| pearson | pearson_de | Mean-profile Pearson correlation |
| pearson_delta | pearson_delta_de | Mean-profile Pearson delta (treatment effect) |
| r2score | r2score_de | Mean-profile R² |
| mse | mse_de | Mean-profile MSE |
| - | sinkhorn_de | Cell-level Sinkhorn distance on DEGs |

**注意**: `pearson_de` 和 `r2score_de` 在本数据集中通常为 0（50 genes 不足以提供有意义的全基因相关性）。主要关注 **pearson_delta_de** 和 **sinkhorn_de**。

---

## 6. 最终分数聚合

### 6.1 Macro Average

```python
def macro_average(per_group_metrics):
    """Unweighted mean across all valid groups."""
    n = len(per_group_metrics)
    avg = {}
    for key in per_group_metrics[0].keys():
        avg[key] = np.mean([m[key] for m in per_group_metrics])
    return avg
```

**不加权**：每个 group 权重相同，不按细胞数加权。

### 6.2 OOD 评估

当前仅 1 个 OOD group (PW034_Panobinostat)，OOD 评估即为该 group 的单点指标。

---

## 7. 预测输出文件规范

### 7.1 H5AD 格式（推荐）

```python
pred_adata = ad.AnnData(
    X=scipy.sparse.csr_matrix(Y_pred),   # float32, (n_cells, 5000)
    obs=obs_metadata,                     # 至少包含 covariate_patient, condition
    var=adata.var.copy(),                 # 必须与主数据 var 完全一致！
)
pred_adata.write("MethodName_PW034_Panobinostat_pred.h5ad")
```

### 7.2 Numpy 格式

```python
np.save("MethodName_Y_pred.npy", Y_pred)  # (n_cells, 5000) float32
# 同时保存 var_names 索引用于验证
np.save("MethodName_var_names.npy", adata.var_names.values)
```

---

## 8. 常见陷阱

### 8.1 基因顺序不一致

```python
# 永远使用主数据的 var_names 作为参考
ref_genes = list(adata.var_names)

# 在保存预测前验证
assert list(pred_adata.var_names) == ref_genes, "Gene order mismatch!"
```

### 8.2 表达式空间不一致

所有预测应与 `adata.X` 在同一空间（log1p 归一化）。如果模型输出在 counts 空间：

```python
Y_pred_log1p = np.log1p(Y_pred_counts)
```

### 8.3 DEG Key 格式

```python
# DEG dict 使用 | 分隔
deg_key = "PW034|Panobinostat"  # 正确

# cov_drug_name 使用 _ 分隔
group_name = "PW034_Panobinostat"  # 正确

# 转换
deg_key = group_name.replace("_", "|")
```

### 8.4 Control 定义

```python
# neg_control == 1 → control
# neg_control == 0 → treated
# 不要反了！
```

---

## 9. 评估脚本使用

### 9.1 NIPS 格式评估

```bash
python scripts/evaluate_nips_gbm.py \
    --h5ad GBM_NIPS_Ready.h5ad \
    --predictions predictions_dict.pkl \
    --output evaluation_results/MyMethod_ood_metrics.json
```

### 9.2 CRISP OOD 评估

```bash
python scripts/evaluate_crisp_ood.py \
    --h5ad GBM_NIPS_Ready.h5ad \
    --predictions predictions_dict.pkl \
    --output GBM_MyMethod_CRISP_OOD_metrics.md
```

---

## 10. 评估结果 JSON 格式

```json
{
  "method": "MethodName",
  "setting": "ood",
  "macro_avg": {
    "r2score": 0.0,
    "pearson": 0.95,
    "pearson_delta": 0.45,
    "pearson_delta_de": 0.30,
    "sinkhorn_de": 0.005,
    "n_valid_groups": 1,
    "skipped": {
      "treated_too_few": 0,
      "dmso_control": 0,
      "deg_missing": 0,
      "deg_too_few": 0,
      "ctrl_too_few": 0
    }
  },
  "per_group": {
    "PW034_Panobinostat": {
      "pearson_delta_de": 0.30,
      "sinkhorn_de": 0.005,
      "n_treated": 2679,
      "n_ctrl": 15288,
      "n_pred": 15288
    }
  }
}
```

---

## 版本历史

| 版本 | 日期 | 修改 |
|------|------|------|
| v1.0 | 2026-05-19 | 初始版本，基于 GBM_NIPS_Ready.h5ad |
