# GBM 药物扰动预测 — 对比方法实验完整文档

> 最后更新：2026-05-19
> 负责人：裴立昆实验组
> 数据：GBM scRNA-seq (169,972 细胞 × 5,000 基因 × 7 药物 × 21 患者)

---

## 目录

1. [项目概述](#1-项目概述)
2. [名词解释](#2-名词解释)
3. [数据处理流水线](#3-数据处理流水线)
4. [对比方法体系](#4-对比方法体系)
5. [技术架构](#5-技术架构)
6. [实验脚本清单](#6-实验脚本清单)
7. [评估协议](#7-评估协议)
8. [实验结果](#8-实验结果)
9. [文件索引](#9-文件索引)

---

## 1. 项目概述

### 1.1 科学问题

给定一个**未见过的患者**（PW034）和一种**药物**（Panobinostat），能否仅基于其他患者的数据，预测该患者在接受药物治疗后的基因表达变化？

这是一个 **OOD（Out-of-Distribution）零样本药物扰动预测** 问题。

### 1.2 核心思路

使用 **预训练编码器**（scGPT 编码细胞状态，MolFormer 编码药物结构）提取多模态特征，结合 **CPA（Compositional Perturbation Autoencoder）** 架构进行反事实预测，并系统对比不同编码器组合的效果。

### 1.3 实验意义

| 层面 | 意义 |
|------|------|
| **方法学** | 验证预训练大模型（scGPT、MolFormer）在药物扰动预测中的迁移能力 |
| **生物学** | 为 GBM（胶质母细胞瘤）个性化用药提供计算预测工具 |
| **工程学** | 建立统一的对比实验框架，确保不同方法在相同条件下公平比较 |
| **可复现性** | 遵循 NIPS 统一评估协议，保证结果可与他组对比 |

---

## 2. 名词解释

### 2.1 数据相关

| 术语 | 英文 | 解释 |
|------|------|------|
| **h5ad** | AnnData HDF5 | scanpy/anndata 的标准数据格式，存储表达矩阵 + 元数据 + embedding |
| **HVG** | Highly Variable Genes | 高变基因，筛选后的特征基因子集（本实验 5,000 个） |
| **counts** | Raw counts | 原始整数 UMI 计数，CPA 使用 NB（负二项）损失时需要 |
| **log1p** | log(1+x) | 对数归一化表达值，用于可视化和部分指标计算 |
| **SMILES** | Simplified Molecular Input Line Entry System | 用字符串表示化学分子结构的标准格式 |
| **cell_type** | 细胞类型/患者 | 本数据中实际是患者 ID（如 PW034），非传统细胞类型 |
| **perturbation** | 扰动/药物处理 | 细胞接受的药物处理（8 种：control + 7 药物） |
| **cov_drug_name** | 协变量-药物组合 | NIPS 协议的 group key，格式 `cell_type_drug` |
| **neg_control** | 阴性对照标记 | 1=control 细胞, 0=treated 细胞（NIPS 标准命名） |

### 2.2 方法相关

| 术语 | 全称 | 解释 |
|------|------|------|
| **CPA** | Compositional Perturbation Autoencoder | 组合扰动自编码器，可预测未见药物/患者的基因表达 |
| **scGPT** | single-cell Generative Pre-trained Transformer | 基于 Transformer 的单细胞基础模型，将基因表达编码为 512d 向量 |
| **MolFormer** | Molecular Transformer | 基于 Transformer 的化学分子基础模型，将 SMILES 编码为 768d 向量 |
| **MLP** | Multi-Layer Perceptron | 多层感知机，用于维度对齐的神经网络 |
| **CRISP** | — | 对比方法统一评估框架（刘耀泽/龙泽兴 提供） |
| **OOD** | Out-of-Distribution | 分布外预测，模型在未见过的患者/药物上做预测 |
| **IID** | Independent and Identically Distributed | 分布内预测，模型在见过的患者上进行 held-out 评估 |

### 2.3 评估相关

| 术语 | 英文 | 解释 |
|------|------|------|
| **DEG** | Differentially Expressed Genes | 差异表达基因，药物处理后显著变化的基因子集 |
| **top50_DEGs** | Top 50 DEGs | 每个 (患者, 药物) 组合的 Top50 差异表达基因 |
| **rank_genes_groups_cov** | — | NIPS 数据中的完整 DEG 字典（599 entries） |
| **r2score** | R² Score | 决定系数，衡量预测与真实的拟合程度，NIPS 协议截断为 ≥0 |
| **pearson** | Pearson Correlation | 皮尔逊相关系数，衡量线性相关性 |
| **pearson_delta** | Delta Pearson | delta = treated − control，衡量扰动效应的相关性 |
| **pearson_delta_de** | Delta Pearson on DEGs | 在 DEG 子集上的扰动效应 Pearson（**核心指标**） |
| **mse** | Mean Squared Error | 均方误差 |
| **sinkhorn_de** | Sinkhorn Distance on DEGs | 最优传输距离，衡量预测与真实分布的一致性 |
| **macro average** | 宏平均 | 所有 group 的 per-group 指标做不加权平均 |

### 2.4 技术相关

| 术语 | 解释 |
|------|------|
| **AnnData** | anndata 库的核心数据结构，`.X` 存表达矩阵，`.obs` 存细胞元数据，`.obsm` 存 embedding |
| **backed mode** | AnnData 的只读模式，数据保留在磁盘，按需加载（节省内存） |
| **NB loss** | 负二项损失（Negative Binomial），适用于整数 count 数据 |
| **Gauss loss** | 高斯损失（MSE），适用于连续值数据 |
| **pert_encoder** | CPA 内部的药物编码器，将药物名映射为可学习的 Embedding 向量 |
| **covars_encoder** | CPA 内部的协变量编码器，将患者 ID 映射为 Embedding 向量 |
| **counterfactual** | 反事实预测：如果某患者的 control 细胞接受了某药物，表达会怎样？ |

---

## 3. 数据处理流水线

### 3.1 总体流程

```
GBM GEO 原始数据 (GSE148842 + GSE226202)
    │
    ├─→ GBM_dataset/download_geo_to_cpa.py  ──→ cpa_ready.h5ad (per dataset)
    │
    ├─→ scripts/prepare_gbm_universal.py    ──→ GBM_Universal_Perturbation_Ready.h5ad
    │                                            169,972 细胞 × 5,000 HVG
    │                                            split: train/valid/ood
    │                                            top50_DEGs: 18 entries
    │
    ├─→ scripts/encode_gbm_cells_scgpt.py   ──→ GBM_scGPT_embeddings.h5ad
    │    (conda activate nature)                  X_scGPT:       (169972, 512)
    │                                             X_scGPT_ctrl:  (169972, 512) 仅对照组非零
    │                                             X_scGPT_pert:  (169972, 512) 仅扰动组非零
    │
    ├─→ scripts/encode_gbm_drugs_molformer.py──→ GBM_X_MolFormer.npy + parquet
    │    (conda activate nature)                  X_MolFormer: (169972, 768)
    │
    ├─→ scripts/prepare_gbm_with_embeddings.py ──→ GBM_with_embeddings.h5ad (~831 MB)
    │    (conda activate plknature)
    │
    └─→ scripts/fix_gbm_nips_format.py ──────────→ GBM_NIPS_Ready.h5ad (~831 MB)
         (conda activate plknature)                 + cov_drug_name (39 groups)
                                                    + neg_control (0/1)
                                                    + condition = 药物名
```

### 3.2 编码器详情

#### scGPT（细胞编码器）

| 属性 | 值 |
|------|-----|
| 模型 | scGPT blood (预训练) |
| 权重路径 | `/home/u2023312303/nature子刊/zyq/encoder/scGPT_blood/` |
| 输入 | 基因表达 counts 矩阵 |
| 输出 | 512 维 float32 embedding |
| 基因匹配率 | 4547/5000 (91%)，未匹配基因自动补零 |
| 环境 | `conda activate nature` |
| 3 种变体 | `X_scGPT`（全部细胞）、`X_scGPT_ctrl`（仅对照组有值）、`X_scGPT_pert`（仅扰动组有值） |

#### MolFormer（药物编码器）

| 属性 | 值 |
|------|-----|
| 模型 | IBM MoLFormer-XL-both-10pct |
| 来源 | HuggingFace `ibm/MoLFormer-XL-both-10pct` |
| 输入 | SMILES 字符串 |
| 输出 | 768 维 float32 embedding（pooler_output） |
| 编码药物数 | 6 个唯一 SMILES（Ispenisib 与 Tazemetostat SMILES 相同） |
| 环境 | `conda activate nature` |

### 3.3 MLP 维度对齐器

| 属性 | 值 |
|------|-----|
| 用途 | scGPT 512d → 基因表达 5000d 的维度投影 |
| 架构 | 512 → 1024 → 2048 → 4096 → 5000（含 BatchNorm + Dropout） |
| 参数量 | 31.5M |
| 训练数据 | 150,564 train + 16,729 valid 细胞 |
| 训练 epochs | 85（early stop） |
| 验证 Pearson r | 0.9997（均值 profile 几乎完美对齐） |
| 输出文件 | `GBM_scGPT_aligner.pt` + `GBM_scGPT_aligned_XscGPT.npy` |

---

## 4. 对比方法体系

### 4.1 方法矩阵

```
                药物编码 →
细胞编码 ↓     可学习 Embedding    MolFormer (768d, 冻结)
─────────────────────────────────────────────────
基因表达(5000d)     M0 ★              M4 ★
scGPT 全细胞(512d)  M1                M5
scGPT ctrl(512d)    M2                —
scGPT pert(512d)    M3                —
```

### 4.2 各方法详细定义

| 方法 | 细胞输入 | 药物输入 | CPA recon_loss | 需要 MLP? | 状态 |
|------|---------|---------|:---:|:---:|:---:|
| **M0** | `adata.X` counts (5000d) | pert_encoder (可学习 Embedding, 32d) | nb | 否 | ✅ |
| **M1** | `X_scGPT` → MLP → 5000d | pert_encoder (可学习 Embedding) | gauss | 是 | 🔄 |
| **M2** | `X_scGPT_ctrl` → MLP → 5000d | pert_encoder (可学习 Embedding) | gauss | 是 | ⬜ |
| **M3** | `X_scGPT_pert` → MLP → 5000d | pert_encoder (可学习 Embedding) | gauss | 是 | ⬜ |
| **M4** | `adata.X` counts (5000d) | MolFormer 768d → Linear → 32d (冻结) | nb | 否 | ✅ |
| **M5** | `X_scGPT` → MLP → 5000d | MolFormer 768d → Linear → 32d (冻结) | gauss | 是 | ⬜ |
| **MeanShift** | — (非参数) | — (非参数) | — | — | ✅ |

### 4.3 各方法意义

| 方法 | 要验证的科学假设 |
|------|----------------|
| **M0 vs MeanShift** | CPA 自编码器架构是否优于简单的均值偏移 baseline |
| **M4 vs M0** | 预训练药物表示（MolFormer）是否优于任务特异性学习的药物表示 |
| **M1 vs M0** | 预训练细胞表示（scGPT）是否可以替代原始基因表达作为 CPA 输入 |
| **M2 vs M1** | 仅对 control 细胞编码（去掉扰动信号）的效果 |
| **M3 vs M1** | 仅对扰动细胞编码（去掉 control 信号）的效果 |
| **M5 vs M1/M4** | 双预训练编码器（scGPT + MolFormer）的协同效应 |

---

## 5. 技术架构

### 5.1 CPA 模型结构

```
输入层:
  adata.layers["counts"] ──→ Encoder ──→ z_basal (32d, 细胞基底状态)
  pert_encoder[drug]     ──→ PerturbationNetwork ──→ z_pert (32d, 药物扰动效应)
  covars_encoder[patient]──→ Embedding ──→ z_covs (32d, 患者协变量)

潜空间:
  z = z_basal + z_pert + z_covs (32d, 最终潜表示)

输出层:
  z ──→ Decoder ──→ 重建的基因表达 (5000d)

训练目标:
  NB loss: 负二项对数似然 (用于整数值 counts)
  Gauss loss: MSE (用于连续值，如 scGPT 对齐后的值)
  + 对抗损失 (perturbation classifier)
```

### 5.2 药物编码器变体

```
M0/M1/M2/M3: 可学习 Embedding
  药物名 ──→ nn.Embedding(9, 32) ──→ 32d latent
  9 = <PAD> + control + 7 drugs

M4/M5: MolFormer (冻结)
  SMILES ──→ MolFormer(预训练,冻结) ──→ 768d ──→ nn.Linear(768, 32) ──→ 32d latent
```

### 5.3 细胞编码器变体

```
M0/M4: 原始基因表达
  counts (5000d) ──→ Encoder(5000→256→256→256→32) ──→ z_basal (32d)

M1/M2/M3/M5: scGPT 对齐
  scGPT (512d) ──→ MLP Aligner(冻结) ──→ aligned (5000d) ──→ Encoder ──→ z_basal (32d)
```

### 5.4 预测流程（反事实推理）

```
1. 取目标患者的 control 细胞 (如 PW034 + control)
2. 将 perturbation 标签改为目标药物 (Panobinostat)
3. 将 dosage 设为 1.0
4. 通过 CPA 前向传播: control 基底 + 药物扰动 = 预测表达
5. 输出: 反事实基因表达矩阵 (15288 细胞 × 5000 基因)
```

---

## 6. 实验脚本清单

| 脚本 | 环境 | 功能 |
|------|------|------|
| `scripts/prepare_gbm_universal.py` | plknature | 合并 GEO 数据集，构建 Universal h5ad |
| `scripts/encode_gbm_cells_scgpt.py` | nature | scGPT 编码细胞 → 512d + 掩码变体 |
| `scripts/encode_gbm_drugs_molformer.py` | nature | MolFormer 编码药物 → 768d + parquet |
| `scripts/prepare_gbm_with_embeddings.py` | plknature | 合并所有 embedding 到 GBM_with_embeddings.h5ad |
| `scripts/fix_gbm_nips_format.py` | plknature | 添加 NIPS 列 → GBM_NIPS_Ready.h5ad |
| `scripts/train_scgpt_aligner.py` | plknature | 训练 MLP: scGPT 512d → 5000d |
| `scripts/train_cpa_ood.py` | plknature | **M0**: 训练 CPA baseline |
| `scripts/train_cpa_molformer.py` | plknature | **M4**: 训练 CPA + MolFormer |
| `scripts/train_cpa_scgpt.py` | plknature | **M1/M2/M3**: 训练 CPA + scGPT 对齐 |
| `scripts/predict_mean_shift_baseline.py` | plknature | MeanShift 非参数 baseline |
| `scripts/predict_mlp_comparison.py` | plknature | MLP 直接预测（M1/M5 简化版） |
| `scripts/evaluate_crisp_ood.py` | plknature | 原始单 group 5 指标评估 |
| `scripts/evaluate_nips_gbm.py` | plknature | NIPS 协议 9 指标多 group 评估 |

---

## 7. 评估协议

### 7.1 原始评估（evaluate_crisp_ood.py）

对单 group (PW034|Panobinostat) 的 **Top50 DEG** 基因计算 5 个指标：

| 指标 | 含义 | 方向 |
|------|------|:---:|
| PrΔ DE | 预测 vs 真实 delta logFC 的 Pearson r | ↑ |
| Sp DE | 预测 vs 真实 delta logFC 的 Spearman ρ | ↑ |
| R² score DE | 预测 vs 真实处理后均值的 R² | ↑ |
| Sinkhorn DE | 预测与真实分布的 Sinkhorn 距离 | ↓ |
| Direction Accuracy | delta 方向一致的基因比例 | ↑ |

### 7.2 NIPS 协议评估（evaluate_nips_gbm.py）

遵循 `LYZ_LZX_NIPS_METRICS.md` 的标准化协议，对每个 valid group 计算 9 个指标后 macro average：

| 指标 | 计算方式 | 方向 |
|------|---------|:---:|
| r2score | max(R²(y, p), 0)，全基因 | ↑ |
| r2score_de | max(R²(y_D, p_D), 0)，DEG 子集 | ↑ |
| pearson | corr(y, p)，全基因 | ↑ |
| pearson_de | corr(y_D, p_D)，DEG 子集 | ↑ |
| mse | mean((y-p)²)，全基因 | ↓ |
| mse_de | mean((y_D-p_D)²)，DEG 子集 | ↓ |
| pearson_delta | corr(y-c, p-c)，全基因 | ↑ |
| pearson_delta_de | corr(y_D-c_D, p_D-c_D)，DEG 子集 | ↑ |
| sinkhorn_de | Sinkhorn(Y_true[:,D], Y_pred[:,D]) | ↓ |

其中 y=mean(Y_true), p=mean(Y_pred), c=mean(Y_ctrl), D=DEG indices。

---

## 8. 实验结果

### 8.1 原始评估结果（单 Group OOD）

| Method | PrΔ DE ↑ | Sp DE ↑ | R² DE ↑ | Sinkhorn DE ↓ | Direction ↑ |
|---|---|---|---|---|---|
| MeanShiftBaseline | -0.468 | -0.230 | -276.429 | 0.190 | 20.0% |
| MLP (M1: scGPT) | 0.018 | -0.003 | -1046.680 | 0.012 | 18.0% |
| MLP (M5: scGPT+MolFormer) | -0.069 | -0.086 | -1017.322 | 0.013 | 20.0% |
| **CPA (M0: baseline)** | **0.608** | **0.463** | **-18.636** | **0.004** | **96.0%** |
| **CPA (M4: +MolFormer)** | **0.693** | **0.585** | **-18.957** | **0.004** | **94.0%** |
| CPA (M1: +scGPT) | (训练中) | | | | |

### 8.2 NIPS 协议评估结果

| Metric | MeanShift | MLP M1 | MLP M5 | **CPA M0** | **CPA M4** |
|---|---|---|---:|---:|---:|
| r2score ↑ | 0.820 | 0.859 | 0.836 | 0.000 | 0.000 |
| pearson ↑ | 0.942 | 0.946 | 0.933 | 0.861 | 0.862 |
| mse ↓ | 0.007 | 0.005 | 0.006 | 0.174 | 0.170 |
| **pearson_delta_de ↑** | 0.000 | 0.018 | 0.000 | **0.608** | **0.693** |
| **sinkhorn_de ↓** | 0.190 | 0.012 | 0.013 | **0.004** | 0.004 |

### 8.3 关键发现

1. **CPA 碾压简单 baseline**：pearson_delta_de 0.608-0.693 vs 0.000-0.018
2. **MolFormer 提升 CPA**：pearson_delta_de +14%（0.608→0.693）
3. **简单方法失败原因**：全基因 pearson ~0.94（预测 control 状态即可），但 DEG delta 接近 0（无法预测扰动特异性效应）
4. **scGPT 对齐器极高质量**：val_r=0.9997，说明 scGPT 保留了足够的细胞状态信息

---

## 9. 文件索引

### 9.1 数据文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `GBM_Universal_Perturbation_Ready.h5ad` | ~213 MB | 原始数据，无 embedding |
| `GBM_with_embeddings.h5ad` | ~831 MB | 预处理后，含 X_scGPT + X_MolFormer |
| `GBM_NIPS_Ready.h5ad` | ~831 MB | NIPS 格式，含 cov_drug_name + neg_control |
| `GBM_molformer_drug_emb.parquet` | ~385 KB | 药物级 MolFormer embedding 表 |
| `GBM_scGPT_aligned_XscGPT.npy` | ~3.2 GB | scGPT→基因 对齐后的矩阵 |
| `GBM_scGPT_aligner.pt` | ~126 MB | 训练好的 MLP 对齐器权重 |

### 9.2 模型文件

| 文件 | 说明 |
|------|------|
| `GBM_CPA_model/` | M0: CPA baseline 模型 |
| `GBM_CPA_MolFormer_model/` | M4: CPA + MolFormer 模型 |
| `GBM_CPA_scGPT_model/` | M1: CPA + scGPT 模型（训练中） |

### 9.3 预测文件

| 文件 | 方法 | 形状 |
|------|------|------|
| `GBM_CPA_PW034_Panobinostat_pred.h5ad` | M0 | (15288, 5000) |
| `GBM_CPA_MolFormer_PW034_Panobinostat_pred.h5ad` | M4 | (15288, 5000) |
| `GBM_MeanShiftBaseline_PW034_Panobinostat_pred.h5ad` | MeanShift | (15288, 5000) |
| `mlp_comparison_results/MLP_M1_*.h5ad` | MLP M1 | (15288, 5000) |
| `mlp_comparison_results/MLP_M5_*.h5ad` | MLP M5 | (15288, 5000) |

### 9.4 评估文件

| 文件 | 说明 |
|------|------|
| `GBM_CRISP_OOD_metrics.md` | 原始 5 指标汇总表 |
| `evaluation_results/*_ood_metrics.json` | NIPS 9 指标 per-group + macro average |
| `GBM_对比方法实验报告.md` | 中文实验报告 |
| `GBM_数据预处理报告.md` | 预处理流程详细报告 |
| `LYZ_LZX_NIPS_METRICS.md` | NIPS 统一评估协议（刘耀泽/龙泽兴） |

### 9.5 参考数据（zyq 目录）

| 文件 | 说明 |
|------|------|
| `/home/.../zyq/data/nips_pp_scFM_MolFormer.h5ad` | NIPS 范例数据（6.6 GB） |
| `/home/.../zyq/data/nips_molformer_drug_emb.parquet` | NIPS 药物 embedding 表 |
| `/home/.../zyq/data/load_nips_molformer.py` | NIPS 数据通用加载器 |
| `/home/.../zyq/encoder/scGPT.py` | scGPT 编码示例 |
| `/home/.../zyq/encoder/molformer.py` | MolFormer 编码示例 |

---

## 附录：环境配置

```bash
# 编码器环境 (scGPT / MolFormer)
conda activate nature    # anndata 0.11.4, scgpt, transformers

# 训练/评估环境 (CPA)
conda activate plknature  # anndata 0.12.11, scvi-tools, cpa, pytorch-lightning

# GPU: NVIDIA GeForce RTX 4090 × 8 (24 GB each)
# 本实验使用 GPU 0-3
```
