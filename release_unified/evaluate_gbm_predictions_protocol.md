# GBM 扰动预测统一评估接入协议

**版本**: v1.1 (基于 GBM_NIPS_Ready.h5ad unified)
**最后更新**: 2026-05-19
**唯一主文件**: `GBM_NIPS_Ready.h5ad`

---

## 1. 概述

本协议定义如何将任意方法的预测结果转换为统一评估格式。遵循此协议后，所有方法的结果可公平对比。

**核心原则**:
- 所有方法使用同一个 GBM_NIPS_Ready.h5ad
- 统一使用 log1p 评估空间
- 统一使用 18 个 valid groups
- 统一使用 rank_genes_groups_cov 作为 DEG 来源

---

## 2. 数据加载

```python
import anndata as ad
import numpy as np
import scipy.sparse as sp

adata = ad.read_h5ad("GBM_NIPS_Ready.h5ad")

# 验证
assert adata.shape == (169972, 5000)
assert "rank_genes_groups_cov" in adata.uns
assert "XscGPT" in adata.obsm or "X_scGPT" in adata.obsm
assert "XMolFormer" in adata.obsm or "X_MolFormer" in adata.obsm
```

---

## 3. 预测输入格式

### 3.1 Per-Group Predictions Dict

```python
predictions = {
    group_name: {
        "Y_true": np.ndarray or sparse matrix,    # (n_treated, 5000)
        "Y_pred": np.ndarray or sparse matrix,    # (n_pred, 5000)
        "Y_ctrl": np.ndarray or sparse matrix,    # (n_ctrl, 5000)
        "prediction_space": "log1p" | "counts",   # 预测空间标注
        "var_names": list[str],                   # 必须与 adata.var_names 一致
        "metadata": {                             # 方法信息
            "method": "method_name",
            "description": "...",
        }
    }
}
```

### 3.2 格式要求

| 要求 | 说明 |
|------|------|
| `group_name` | 必须来自 `adata.obs["cov_drug_name"]` |
| `Y_true` | 真实 treated 细胞 log1p 表达 (来自 adata.X) |
| `Y_ctrl` | 同患者 control 细胞 log1p 表达 (来自 adata.X) |
| `Y_pred` | 模型预测表达 |
| `prediction_space` | 标明 Y_pred 是 "log1p" 还是 "counts" |
| `var_names` | 必须与 `adata.var_names` 顺序完全一致 |
| 形状 | (n_cells, 5000) |

### 3.3 Counts-Space 预测转换

```python
def normalize_to_log1p(Y, prediction_space):
    if prediction_space == "counts":
        return np.log1p(np.maximum(Y, 0))
    elif prediction_space == "log1p":
        return np.asarray(Y, dtype=np.float64)
    else:
        raise ValueError(f"Unknown prediction_space: {prediction_space}")
```

---

## 4. DEG 读取

```python
def get_deg_genes(adata, group_name):
    """优先使用 rank_genes_groups_cov."""
    rg = adata.uns["rank_genes_groups_cov"]
    if group_name in rg:
        return list(rg[group_name])
    # 回退 (不应发生)
    t50 = adata.uns["top50_DEGs"]
    legacy_key = group_name.replace("_", "|")
    if legacy_key in t50:
        return list(t50[legacy_key])
    raise KeyError(f"No DEG for {group_name}")
```

---

## 5. Valid Group 过滤

```python
def get_valid_groups(adata):
    deg = adata.uns["rank_genes_groups_cov"]
    groups = []
    skipped = {"treated_too_few": [], "control": [], "deg_missing": [],
               "deg_too_few": [], "ctrl_too_few": []}

    for gn in adata.obs["cov_drug_name"].unique():
        if "control" in gn.lower():
            skipped["control"].append(gn)
            continue
        nt = (adata.obs["cov_drug_name"] == gn).sum()
        if nt <= 5:
            skipped["treated_too_few"].append(gn)
            continue
        if gn not in deg:
            skipped["deg_missing"].append(gn)
            continue
        if len(deg[gn]) < 2:
            skipped["deg_too_few"].append(gn)
            continue
        patient = gn.split("_")[0]
        nc = (adata.obs["cov_drug_name"] == f"{patient}_control").sum()
        if nc < 5:
            skipped["ctrl_too_few"].append(gn)
            continue
        groups.append(gn)
    return groups, skipped
```

当前 18 个 valid groups:
```
GS359_Temozolomide, GS772_Temozolomide, GS785_Temozolomide, GS789_Temozolomide,
PW029_Etoposide, PW030_Ana-12, PW030_Etoposide, PW030_Ispenisib,
PW030_Panobinostat, PW030_RO4929097, PW030_Tazemetostat,
PW032_Etoposide, PW032_Panobinostat, PW034_Etoposide, PW034_Panobinostat,
PW036_Etoposide, PW036_Panobinostat, PW040_Panobinostat
```

---

## 6. 指标计算

### 6.1 Mean-Profile Metrics

```python
def mean_profile_metrics(Y_true, Y_pred, Y_ctrl, deg_genes, var_names):
    mu_t = Y_true.mean(axis=0)
    mu_p = Y_pred.mean(axis=0)
    mu_c = Y_ctrl.mean(axis=0)

    results = {}
    results["pearson"] = np.corrcoef(mu_t, mu_p)[0, 1]
    results["mse"] = np.mean((mu_t - mu_p)**2)
    ss_res = np.sum((mu_t - mu_p)**2)
    ss_tot = np.sum((mu_t - mu_t.mean())**2)
    results["r2score"] = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    d_t = mu_t - mu_c
    d_p = mu_p - mu_c
    results["pearson_delta"] = np.corrcoef(d_t, d_p)[0, 1]

    # DEG subset
    deg_idx = [list(var_names).index(g) for g in deg_genes if g in var_names]
    if deg_idx:
        results["pearson_de"] = np.corrcoef(mu_t[deg_idx], mu_p[deg_idx])[0, 1]
        results["mse_de"] = np.mean((mu_t[deg_idx] - mu_p[deg_idx])**2)
        results["r2score_de"] = ...  # analogous on DEG subset
        results["pearson_delta_de"] = np.corrcoef(d_t[deg_idx], d_p[deg_idx])[0, 1]
    return results
```

### 6.2 Sinkhorn DE (Cell-Level, DEG Subset)

```python
def compute_sinkhorn_de(Y_true, Y_pred, deg_genes, var_names):
    import geomloss
    deg_idx = [list(var_names).index(g) for g in deg_genes if g in var_names]
    Yt = Y_true[:, deg_idx]
    Yp = Y_pred[:, deg_idx]
    Yt = Yt / (np.linalg.norm(Yt, axis=1, keepdims=True) + 1e-8)
    Yp = Yp / (np.linalg.norm(Yp, axis=1, keepdims=True) + 1e-8)
    loss = geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=0.05)
    return loss(Yt.astype(np.float32), Yp.astype(np.float32)).item()
```

### 6.3 Direction Accuracy (DEG Subset)

```python
def compute_direction_accuracy(Y_true, Y_pred, Y_ctrl, deg_genes, var_names):
    deg_idx = [list(var_names).index(g) for g in deg_genes if g in var_names]
    delta_true = Y_true[:, deg_idx] - Y_ctrl.mean(axis=0)[deg_idx]
    delta_pred = Y_pred[:, deg_idx] - Y_ctrl.mean(axis=0)[deg_idx]
    sign_match = np.sign(delta_true.mean(axis=0)) == np.sign(delta_pred.mean(axis=0))
    return float(sign_match.mean())
```

### 6.4 完整指标清单

| 指标 | 空间 | 计算方式 |
|------|------|---------|
| `pearson` | 全基因 5000 | mean profile 后 corr |
| `pearson_de` | DEG subset | mean profile 后 corr (小 DEG 集可能不显著) |
| `pearson_delta` | 全基因 5000 | mean profile delta 后 corr |
| `pearson_delta_de` | DEG subset | **主要指标** — mean profile delta 后 corr |
| `r2score` | 全基因 5000 | mean profile R² |
| `r2score_de` | DEG subset | mean profile R² |
| `mse` | 全基因 5000 | mean profile MSE |
| `mse_de` | DEG subset | mean profile MSE |
| `sinkhorn_de` | DEG subset | **cell-level** Sinkhorn distance |
| `direction_accuracy_de` | DEG subset | directional sign match |

---

## 7. 聚合方式

```python
def macro_average(per_group_results):
    """Unweighted mean across valid groups."""
    keys = per_group_results[0].keys()
    avg = {}
    for k in keys:
        vals = [r[k] for r in per_group_results if not np.isnan(r.get(k, float('nan')))]
        avg[k] = np.mean(vals) if vals else float('nan')
    return avg
```

**不加权**: 每个 group 权重相同，不按细胞数加权。

**Coverage**: 如果方法只覆盖部分 valid groups，必须报告 coverage = n_covered / 18，不可直接与 full coverage 方法比较 macro average。

---

## 8. 评估输出格式

```json
{
  "method": "MethodName",
  "setting": "ood",
  "eval_space": "log1p",
  "converted_from": "counts" | null,
  "coverage": {"n_valid": 18, "n_covered": 18},
  "macro_avg": {
    "pearson": 0.95, "pearson_delta": 0.45,
    "pearson_delta_de": 0.30, "sinkhorn_de": 0.005,
    "n_valid_groups": 18,
    "skipped": {}
  },
  "per_group": {
    "PW034_Panobinostat": {
      "pearson_delta_de": 0.30, "sinkhorn_de": 0.005,
      "n_treated": 2679, "n_ctrl": 15288, "n_pred": 15288
    }
  }
}
```

---

## 9. Counts-Space 预测重新评估

```bash
python release_unified/reevaluate_counts_predictions_log1p.py \
    --h5ad GBM_NIPS_Ready.h5ad \
    --output-dir evaluation_results_unified_log1p
```

该脚本读取 legacy prediction h5ad 文件，对 CPA M0/M4 进行 log1p 转换后重新计算指标，生成与 log1p-space 方法可比较的评估结果。

---

## 10. 禁止事项

1. 不要重新划分 split
2. 不要重排 genes
3. 不要把 neg_control 理解反
4. 不要把 OOD 写成完全未见患者
5. 不要混用 counts/log1p 空间
6. 不要为不同方法使用不同 valid groups
7. 不要在"原始 h5ad"和"兼容 h5ad"之间选择 — 只有一个推荐文件
