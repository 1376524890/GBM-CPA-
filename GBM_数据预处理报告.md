# GBM 数据集预处理报告

## 概述

对 GBM 数据集 `GBM_Universal_Perturbation_Ready.h5ad` 使用预训练编码器（scGPT / MolFormer）提取 embedding，生成可直接用于对比实验的数据文件。

**执行日期**：2026-05-15

---

## 一、原始数据概况

| 属性 | 值 |
|------|-----|
| 文件 | `GBM_Universal_Perturbation_Ready.h5ad` |
| 规模 | 169,972 细胞 × 5,000 高变基因 |
| 扰动种类 | 8 种（control + 7 种药物） |
| 药物 | Ana-12, Etoposide, Ispenisib, Panobinostat, RO4929097, Tazemetostat, Temozolomide |
| 患者数 | 21 |
| 分割 | train (150,564) / valid (16,729) / ood (2,679) |
| X 矩阵 | log1p 归一化（sparse CSR） |
| counts 层 | 原始整数计数（int32） |
| SMILES 存储 | `uns["drug_smiles"]` 字典（不在 obs 中） |
| 原始 obsm | 空 |

---

## 二、编码器信息

| 编码器 | 模态 | 输入 | 输出维度 | 模型路径 |
|--------|------|------|---------|---------|
| **scGPT** | 细胞 scRNA-seq | 基因表达 counts 矩阵 | 512d | `/home/u2023312303/nature子刊/zyq/encoder/scGPT_blood/` |
| **MolFormer** | 药物 SMILES | SMILES 字符串 | 768d | HuggingFace `ibm/MoLFormer-XL-both-10pct` |

- scGPT 基因匹配率：4547/5000（91%），未匹配基因自动补零
- MolFormer 编码 6 个唯一 SMILES（Ispenisib 与 Tazemetostat SMILES 相同）
- 运行环境：`conda activate nature`

---

## 三、输出文件

### 3.1 主输出

| 文件 | 大小 | 说明 |
|------|------|------|
| **`GBM_with_embeddings.h5ad`** | ~831 MB | 原始数据 + 4 个 embedding 键 + SMILES/condition 列 |
| **`GBM_molformer_drug_emb.parquet`** | ~385 KB | 药物级 MolFormer embedding（SMILES 索引，768 整数列名） |
| `GBM_molformer_drug_emb.metadata.json` | 1.3 KB | 药物-SMILES-canonical 映射元数据 |

### 3.2 新增 obs 列

| 列名 | 类型 | 说明 |
|------|------|------|
| `SMILES` | str | 每个细胞对应的药物 SMILES（control 为空字符串） |
| `condition` | str | `"control"` 或 `"treated"`（兼容 neurips 示例数据格式） |
| `canonical_smiles` | str | RDKit 规范化的 SMILES |

### 3.3 新增 obsm 键

| 键 | 维度 | 说明 | 验证状态 |
|----|------|------|---------|
| `X_scGPT` | (169972, 512) | 全部细胞的 scGPT embedding | ✓ |
| `X_scGPT_ctrl` | (169972, 512) | 仅对照组保留 scGPT，扰动组全零 | ✓ |
| `X_scGPT_pert` | (169972, 512) | 仅扰动组保留 scGPT，对照组全零 | ✓ |
| `X_MolFormer` | (169972, 768) | 每个细胞对应的药物 MolFormer embedding（control 为零向量） | ✓ |

### 3.4 Parquet 格式

`GBM_molformer_drug_emb.parquet` 格式与示例数据 `nips_molformer_drug_emb.parquet` 完全一致：

- 索引名：`SMILES`
- 索引值：规范化 SMILES 字符串（6 个）
- 列名：整数 0-767（共 768 维）
- 数据类型：float32

---

## 四、对比方法数据使用指南

根据预处理产出的 embedding，可按以下方式配置不同的对比方法：

### 4.1 方法对照表

| 编号 | 方法 | 细胞编码 | 药物编码 | 细胞侧数据源 | 药物侧数据源 |
|------|------|---------|---------|-------------|-------------|
| M0 | **Baseline（原始 CPA）** | 原始基因表达 (5000d) | 可学习 Embedding | `adata.X` / `layers["counts"]` | `pert_encoder` (内置) |
| M1 | **scGPT（全部细胞）** | scGPT (512d) | 可学习 Embedding | `obsm["X_scGPT"]` | `pert_encoder` (内置) |
| M2 | **scGPT（仅对照组）** | scGPT 仅对照组 (512d) | 可学习 Embedding | `obsm["X_scGPT_ctrl"]` | `pert_encoder` (内置) |
| M3 | **scGPT（仅扰动组）** | scGPT 仅扰动组 (512d) | 可学习 Embedding | `obsm["X_scGPT_pert"]` | `pert_encoder` (内置) |
| M4 | **MolFormer** | 原始基因表达 (5000d) | MolFormer (768d) | `adata.X` / `layers["counts"]` | `obsm["X_MolFormer"]` + parquet |
| M5 | **scGPT + MolFormer** | scGPT (512d) | MolFormer (768d) | `obsm["X_scGPT"]` | `obsm["X_MolFormer"]` + parquet |

### 4.2 加载示例

```python
import scanpy as sc
import pandas as pd

# 加载数据
adata = sc.read_h5ad("GBM_with_embeddings.h5ad")

# 细胞 embedding（按需选择）
scgpt_all = adata.obsm["X_scGPT"]         # M1：全部细胞 scGPT
scgpt_ctrl = adata.obsm["X_scGPT_ctrl"]   # M2：仅对照组 scGPT
scgpt_pert = adata.obsm["X_scGPT_pert"]   # M3：仅扰动组 scGPT

# 药物 embedding
molformer = adata.obsm["X_MolFormer"]     # M4/M5：每细胞药物 embedding
drug_table = pd.read_parquet("GBM_molformer_drug_emb.parquet")  # 药物级表

# 区分 control / perturbation
ctrl_mask = adata.obs["condition"] == "control"
pert_mask = adata.obs["condition"] == "treated"

# 按 split 选取
train_mask = adata.obs["split"] == "train"
valid_mask = adata.obs["split"] == "valid"
ood_mask = adata.obs["split"] == "ood"
```

### 4.3 MLP 维度对齐

当下游模型输入维度与 embedding 维度不匹配时，使用 `scripts/align_embeddings_mlp.py`：

```bash
# scGPT 512d → 5000d（对齐基因表达空间）
python scripts/align_embeddings_mlp.py align \
    --input scgpt_embeddings.npy \
    --output scgpt_aligned_5000.npy \
    --target-dim 5000

# MolFormer 768d → 201d（对齐 RDKit 指纹空间）
python scripts/align_embeddings_mlp.py align \
    --input molformer_embeddings.npy \
    --output molformer_aligned_201.npy \
    --target-dim 201
```

也可编程调用：
```python
from scripts.align_embeddings_mlp import Aligner, train_aligner
aligner = Aligner(input_dim=512, output_dim=5000, hidden_dims=[1024, 2048])
# 如有目标嵌入可训练：train_aligner(aligner, source, target)
```

---

## 五、数据完整性验证

### 5.1 形状验证

| 检查项 | 预期 | 实际 | 状态 |
|--------|------|------|------|
| 细胞数 | 169,972 | 169,972 | ✓ |
| 基因数 | 5,000 | 5,000 | ✓ |
| X_scGPT 维度 | (169972, 512) | (169972, 512) | ✓ |
| X_MolFormer 维度 | (169972, 768) | (169972, 768) | ✓ |
| drug parquet 维度 | (6, 768) | (6, 768) | ✓ |

### 5.2 掩码验证

| 检查项 | 预期 | 实际 | 状态 |
|--------|------|------|------|
| X_scGPT_ctrl：对照组非零 | 91,001/91,001 | 91,001/91,001 | ✓ |
| X_scGPT_ctrl：扰动组全零 | True | True | ✓ |
| X_scGPT_pert：扰动组非零 | 78,971/78,971 | 78,971/78,971 | ✓ |
| X_scGPT_pert：对照组全零 | True | True | ✓ |
| X_MolFormer：对照组全零 | True | True | ✓ |
| X_MolFormer：扰动组非零 | 78,971/78,971 | 78,971/78,971 | ✓ |

### 5.3 原始数据保护

| 检查项 | 状态 |
|--------|------|
| `GBM_Universal_Perturbation_Ready.h5ad` 未被修改 | ✓（mtime、size 不变） |
| 所有 embedding 写入新文件 | ✓ |
| CPA 源码（site-packages）未被修改 | ✓ |

---

## 六、脚本清单

所有脚本位于 `scripts/` 目录下：

| 脚本 | 运行环境 | 功能 |
|------|---------|------|
| `encode_gbm_cells_scgpt.py` | `conda activate nature` | 用 scGPT 编码全部细胞 → 512d embedding + 掩码变体 |
| `encode_gbm_drugs_molformer.py` | `conda activate nature` | 用 MolFormer 编码药物 SMILES → 768d embedding + parquet |
| `align_embeddings_mlp.py` | 任意 | MLP 维度对齐工具（对齐器 + 训练函数） |
| `prepare_gbm_with_embeddings.py` | `conda activate plknature` | 主入口：合并所有 embedding 到最终 h5ad |

### 执行流程

```
1. plknature环境
   └─→ prepare_gbm...py 内部自动提取 counts → GBM_counts_for_scgpt.h5ad

2. nature环境
   ├─→ encode_gbm_cells_scgpt.py    → GBM_scGPT_embeddings.h5ad
   └─→ encode_gbm_drugs_molformer.py → GBM_X_MolFormer.npy + parquet

3. plknature环境
   └─→ prepare_gbm_with_embeddings.py --scgpt ... --molformer ... → GBM_with_embeddings.h5ad
```

---

## 七、环境兼容性说明

- **nature 环境**（anndata 0.11.4）：可运行 scGPT 和 MolFormer 编码器，但无法直接读写新版 AnnData 格式。编码脚本通过 h5py 直接读取原始数据绕过此限制。
- **plknature 环境**（anndata 0.12.11）：用于数据合并和格式转换，可正常读写所有 h5ad 文件。
- 中间文件 `GBM_counts_for_scgpt.h5ad` 使用 plknature 导出、nature 兼容的格式。

---

## 八、已知问题

1. **Ispenisib 与 Tazemetostat SMILES 相同**：两者映射到同一个 SMILES `CC(C)N1CCN(CC1)C2=NC(=NC=C2)N3CCOCC3`，可能与原始数据准备中的 PubChem 查询有关。不影响编码结果（两个药物的 MolFormer embedding 相同）。

2. **scGPT 基因匹配率 91%**：GBM 数据 5,000 个 HVG 中有 4,547 个在 scGPT 血液模型词汇表中有匹配。未匹配的 453 个基因在编码时补零，对 embedding 质量影响有限。
