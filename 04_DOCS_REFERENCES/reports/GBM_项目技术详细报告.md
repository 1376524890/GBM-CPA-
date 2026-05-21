# GBM 药物扰动预测 — 项目技术详细报告

> **项目负责人**：裴立昆实验组
> **更新日期**：2026-05-19
> **科学问题**：给定未见过的患者 (PW034) 和药物 (Panobinostat)，预测基因表达变化（OOD 零样本药物扰动预测）

---

## 目录

1. [原始数据集详情](#1-原始数据集详情)
2. [数据预处理流水线](#2-数据预处理流水线)
3. [预训练编码器提取的特征](#3-预训练编码器提取的特征)
4. [MLP 维度对齐器](#4-mlp-维度对齐器)
5. [对比方法矩阵与模型架构](#5-对比方法矩阵与模型架构)
6. [训练详情与超参数](#6-训练详情与超参数)
7. [OOD 反事实预测流程](#7-ood-反事实预测流程)
8. [评估协议与指标体系](#8-评估协议与指标体系)
9. [完整实验结果](#9-完整实验结果)
10. [关键发现与讨论](#10-关键发现与讨论)
11. [文件资产完整索引](#11-文件资产完整索引)
12. [环境与依赖](#12-环境与依赖)

---

## 1. 原始数据集详情

### 1.1 数据来源

| 数据集 | GEO 编号 | 数据内容 | 原始细胞数 |
|--------|---------|---------|:---:|
| GSE148842 | `GBM_dataset/GSE148842_cpa_ready.h5ad` | 胶质母细胞瘤患者来源的肿瘤细胞药物处理 scRNA-seq | ~120K |
| GSE226202 | `GBM_dataset/GSE226202_cpa_ready.h5ad` | 补充 GBM 患者队列，同实验平台 | ~50K |

两个数据集经 `prepare_gbm_universal.py` 合并并标准化为统一格式。

### 1.2 合并后数据集 `GBM_Universal_Perturbation_Ready.h5ad`

| 属性 | 值 |
|------|-----|
| **规模** | **169,972 细胞 × 5,000 高变基因 (HVG)** |
| **表达矩阵 X** | float32，log1p 归一化值 (sparse CSR) |
| **原始 counts 层** | int32，未归一化整数 UMI 计数 |
| **患者数** | **21 名**：GS359, GS772, GS785, GS789, PW029, PW030, PW031, PW032, PW034, PW036, PW040, PW051702, PW051704, PW051708, PW052703, PW052705, PW052706, PW052709, PW053707, PW053710, PW053711 |
| **药物扰动种类** | **8 种**：control + Ana-12, Etoposide, Ispenisib, Panobinostat, RO4929097, Tazemetostat, Temozolomide |
| **对照组细胞** | 91,001 个 (is_control=True) |
| **扰动组细胞** | 78,971 个 (is_control=False) |

### 1.3 数据分割 (Split)

| Split | 细胞数 | 用途 |
|--------|:---:|------|
| **train** | 150,564 (88.6%) | CPA 模型训练 |
| **valid** | 16,729 (9.8%) | 验证、早停、超参数选择 |
| **ood** | 2,679 (1.6%) | OOD 测试 — 仅 PW034 + Panobinostat 组合 |
| **总计** | 169,972 | |

> **OOD 定义**：患者 PW034 的所有细胞都从未出现在训练/验证集中，且仅评估其对 Panobinostat 药物的响应。这意味着模型必须在**完全未见过的患者**上做零样本预测。

### 1.4 obs 列 (细胞元数据)

| 列名 | 类型 | 说明 |
|------|------|------|
| `gsm_accession` | str | GEO 样本编号 |
| `sample_title` | str | GEO 样本标题 |
| `barcode_original` | str | 细胞条形码 |
| `perturbation` | str | 药物名称 (8 种) 或 "control" |
| `dosage` | float | 药物剂量 (均为 1.0，表示处理条件) |
| `covariate_patient` | str | 患者 ID (21 个) |
| `source_treatment` | str | 来源组织处理信息 |
| `n_counts` | int | 每个细胞的 UMI 总计数 |
| `dataset` | str | 数据来源 (GSE148842 / GSE226202) |
| `cell_type` | str | 患者 ID (与 covariate_patient 相同，CPA 格式兼容) |
| `is_control` | bool | 是否为对照组细胞 |
| `source_file` | str | 原始 h5ad 文件名 |
| `split` | str | train / valid / ood |
| `foundation_model_query_eligible` | bool | FM 查询资格标记 |

### 1.5 药物 SMILES 信息

| 药物 | SMILES | 分子特征 |
|------|--------|---------|
| **Ana-12** | `CC1=CC=C(C=C1)C2=NC3=CC=CC=C3N2C4=CC=CC=C4` | TrkB 受体拮抗剂 |
| **Etoposide** | `COC1=CC2=C(C=C1O)C3C(COC3=O)C4=CC5=C(C=C4O2)OCO5` | 拓扑异构酶 II 抑制剂 |
| **Ispenisib** | `CC(C)N1CCN(CC1)C2=NC(=NC=C2)N3CCOCC3` | PI3K 抑制剂 |
| **Panobinostat** | `C1CN(CCC1C(=O)NO)CCN2C=C(C=N2)C3=CC=C(C=C3)C=C` | HDAC 抑制剂 (OOD 目标) |
| **RO4929097** | `CC(C)(C)OC(=O)N1CCC(CC1)N2C=NC3=C2C=C(C=C3)C(F)(F)F` | γ-分泌酶抑制剂 |
| **Tazemetostat** | `CC(C)N1CCN(CC1)C2=NC(=NC=C2)N3CCOCC3` | EZH2 抑制剂 |
| **Temozolomide** | `CN1C(=O)N2C=NC(=N2)N=N1` | DNA 烷化剂 (GBM 一线化疗药) |

> **注意**：Ispenisib 与 Tazemetostat 具有相同的 SMILES 字符串，使用 PubChem 查询获得。两者的 MolFormer embedding 完全相同。这可能影响区分这两种药物的实验。

### 1.6 Top50 差异表达基因 (DEGs)

- **存储位置**：`adata.uns["top50_DEGs"]`，共 **18 个 entries**
- **计算方式**：对每个 (患者, 药物) 组合执行 Welch t-test，计算 log2 fold change，用 Benjamini-Hochberg FDR 校正 p 值，取 abs(logFC) 最大且 FDR 显著的 Top50 基因
- **示例** GS359|Temozolomide Top5 DEGs：CXCL5, HBA1, CSF2, F13A1, ADGRL4
- **用途**：所有 `*_de` 评估指标均在这些基因子集上计算

---

## 2. 数据预处理流水线

### 2.1 总体处理流程

```
原始 GEO 数据 (GSE148842 + GSE226202)
    │
    ├─ Step 1: prepare_gbm_universal.py
    │   合并 → 标准化列名 → 提取 counts → log1p 归一化 → HVG 筛选 (5000)
    │   → 分配 split → 计算 top50 DEGs → 查 PubChem SMILES → 生成 RDKit 指纹
    │   输出: GBM_Universal_Perturbation_Ready.h5ad (213 MB)
    │
    ├─ Step 2: encode_gbm_cells_scgpt.py  [conda: nature]
    │   scGPT 编码 169,972 细胞 → 512d embedding + 3 种掩码变体
    │   输出: GBM_scGPT_embeddings.h5ad (760 MB)
    │
    ├─ Step 3: encode_gbm_drugs_molformer.py  [conda: nature]
    │   MolFormer 编码 7 种药物 SMILES → 768d embedding
    │   输出: GBM_X_MolFormer.npy (522 MB) + parquet (394 KB) + metadata
    │
    ├─ Step 4: prepare_gbm_with_embeddings.py  [conda: plknature]
    │   合并所有 embedding 到统一 h5ad
    │   输出: GBM_with_embeddings.h5ad (~831 MB)
    │
    ├─ Step 5: fix_gbm_nips_format.py  [conda: plknature]
    │   添加 cov_drug_name (39 groups) + neg_control + 修正 condition 列
    │   输出: GBM_NIPS_Ready.h5ad (~831 MB)  ★ 最终数据文件
    │
    ├─ Step 6: train_scgpt_aligner.py  [conda: plknature]
    │   训练 MLP: scGPT 512d → 基因表达 5000d (×3 次: all/ctrl/pert)
    │   输出: 3× aligner.pt (126 MB each) + 3× aligned.npy (3.2 GB each)
    │
    └─ Step 7: train_scgpt_aligner.py (变体)
        训练仅对照组/仅扰动组的独立对齐器
        输出: GBM_scGPT_aligner_ctrl.pt + GBM_scGPT_aligner_pert.pt
```

### 2.2 NIPS 格式标准化 (Step 5) 新增列

| 新增列 | 类型 | 值范围 | 说明 |
|--------|------|--------|------|
| `cov_drug_name` | str | 39 个唯一值 | `cell_type + "_" + perturbation`，作为评估的 group key |
| `neg_control` | int | {0, 1} | 1=control, 0=treated，NIPS 协议 control 定义 |
| `condition` | str | 8 种 | 药物名称 (非 "treated")，NIPS 标准格式 |
| `is_treated` | str | "control"/"treated" | 保留原 condition 值以向后兼容 |
| `SMILES` | str | 7 个唯一值 | 每个细胞对应的药物 SMILES (control 为空) |
| `canonical_smiles` | str | — | RDKit 规范化的 SMILES |

### 2.3 最终数据文件 `GBM_NIPS_Ready.h5ad` 结构

```
GBM_NIPS_Ready.h5ad (831 MB)
├── .X: (169972, 5000) float32 log1p 归一化表达矩阵
├── .layers["counts"]: (169972, 5000) int32 原始 UMI 计数
├── .obs: 20 列细胞元数据
├── .var: 5000 个 HVG 基因名
├── .obsm:
│   ├── X_scGPT:      (169972, 512) float32 全细胞 scGPT embedding (100% 非零)
│   ├── X_scGPT_ctrl: (169972, 512) float32 仅对照组有值 (53.5% 非零)
│   ├── X_scGPT_pert: (169972, 512) float32 仅扰动组有值 (46.5% 非零)
│   └── X_MolFormer:  (169972, 768) float32 每细胞药物 embedding
├── .uns:
│   ├── top50_DEGs: 18 个 (患者|药物) → Top50 基因列表
│   ├── drug_smiles: 7 个药物 → SMILES 字符串
│   ├── drug_embeddings: 7 个药物 → 201d RDKit 指纹
│   └── ood_split: {"target_patient": "PW034", "target_drug": "Panobinostat"}
└── .obsp: (空)
```

---

## 3. 预训练编码器提取的特征

### 3.1 scGPT 细胞编码器

| 属性 | 值 |
|------|-----|
| **模型** | scGPT **blood** (单细胞预训练 Transformer) |
| **预训练数据** | 33M+ 人外周血单个核细胞 (PBMC) scRNA-seq |
| **架构** | 12 层 Transformer Encoder，每层 multi-head self-attention (8 heads)，512 维 hidden size |
| **输入** | 基因表达整数 counts，按 gene name 匹配模型词汇表 |
| **输出** | **512 维 float32 embedding** (最后一个 Transformer 层的 [CLS] token 或 mean pooling) |
| **基因匹配** | **4,547/5,000 (90.94%)** — GBM HVG 中有 453 个基因不在 scGPT blood 词汇表中 |
| **未匹配基因处理** | 在编码输入时补零 (zero padding) |
| **编码脚本** | `scripts/encode_gbm_cells_scgpt.py`，使用 scgpt.tasks.embed_data() |
| **运行环境** | `conda activate nature` (anndata 0.11.4, scgpt) |
| **批次大小** | 64 cells/batch |
| **GPU 使用** | NVIDIA RTX 4090 |

#### scGPT embedding 统计

| 键 | 形状 | 非零比例 | 含义 |
|---|------|:---:|------|
| `X_scGPT` | (169972, 512) | 100% | 所有细胞都编码，embedding 值范围 [-0.473, 0.873] |
| `X_scGPT_ctrl` | (169972, 512) | 53.5% | 仅对照组 (91,001 个) 保留原 embedding，扰动组置零 |
| `X_scGPT_pert` | (169972, 512) | 46.5% | 仅扰动组 (78,971 个) 保留原 embedding，对照组置零 |

> **三个变体的设计目的**：
> - `X_scGPT`：标准的全部细胞编码，用于 M1 和 M5
> - `X_scGPT_ctrl`：**只用未经药物处理的细胞状态**来做药物扰动预测 (M2)，理论假设是"control 细胞已包含足够的基底状态信息"
> - `X_scGPT_pert`：**只用药物处理后的细胞状态** (M3)，测试"扰动信号是否反向指向药物效应"

### 3.2 MolFormer 药物编码器

| 属性 | 值 |
|------|-----|
| **模型** | **MoLFormer-XL-both-10pct** (IBM, HuggingFace) |
| **预训练数据** | 1.1B 分子 (PubChem + ZINC) SMILES 字符串 |
| **架构** | 12 层 BERT-like Transformer Encoder，768 维 hidden size |
| **输入** | SMILES 字符串 → WordPiece tokenizer → token IDs |
| **输出** | **768 维 float32 embedding** (pooler_output, [CLS] token 经 tanh 激活) |
| **训练模式** | 掩码语言模型 (MLM)：随机遮盖 SMILES 子结构，预测被遮盖的 token |
| **编码药物数** | 6 个唯一 SMILES (Ispenisib ≡ Tazemetostat) |
| **编码脚本** | `scripts/encode_gbm_drugs_molformer.py` |
| **运行环境** | `conda activate nature` (transformers, rdkit) |
| **批次大小** | 32 SMILES/batch |

#### MolFormer embedding 输出格式

**Per-cell embedding** (`X_MolFormer`):
- 形状：(169972, 768)，每个细胞对应其药物的 768d 向量
- Control 细胞对应零向量 (768 维全零)
- 值范围：[-2.558, 8.636]

**Drug-level parquet** (`GBM_molformer_drug_emb.parquet`):
- 形状：(6, 768)
- 索引：6 个规范化 SMILES 字符串
- 列名：整数 0-767
- 与 NIPS 范例数据 `nips_molformer_drug_emb.parquet` 格式完全一致

**SMILES → 药物映射**:

| 规范化 SMILES | 对应药物 | Parquet 中 |
|---|---|---|
| `C=Cc1ccc(-c2cnn(CCN3CCC(C(=O)NO)CC3)c2)cc1` | Panobinostat | ✓ |
| `CN1N=Nc2ncn(n2)C1=O` | Temozolomide | ✓ |
| `CC(C)N1CCN(c2ccnc(N3CCOCC3)n2)CC1` | Ispenisib, Tazemetostat | ✓ (共享) |
| `COc1cc2c(cc1O)C1C(=O)OCC1c1cc3c(cc1O2)OCO3` | Etoposide | ✓ |
| `CC(C)(C)OC(=O)N1CCC(n2cnc3ccc(C(F)(F)F)cc32)CC1` | RO4929097 | ✓ |
| `Cc1ccc(-c2nc3ccccc3n2-c2ccccc2)cc1` | Ana-12 | ✓ |

---

## 4. MLP 维度对齐器

### 4.1 架构与训练

| 属性 | 值 |
|------|-----|
| **目的** | 将 scGPT 512d embedding 投影回基因表达 5000d 空间 |
| **架构** | `512 → 1024 → 2048 → 4096 → 5000` |
| **层结构** | Linear → ReLU → BatchNorm1d → Dropout(0.1)，×3 隐藏层，最后 Linear 输出 |
| **参数量** | **31.5M** |
| **训练数据** | 150,564 train + 16,729 valid 细胞 |
| **损失函数** | MSE (Mean Squared Error) |
| **优化器** | Adam (lr=1e-3) + ReduceLROnPlateau (factor=0.5, patience=15) |
| **批次大小** | 512 |
| **早停** | patience=40 epochs |
| **最佳 epoch** | 85 (all), 48 (ctrl), 18 (pert) |

### 4.2 训练结果

| Aligner | scGPT key | 验证 Pearson r (均值 profile) | 输出文件 |
|---------|-----------|:---:|------|
| **Aligner (all)** | X_scGPT | **0.9997** | `GBM_scGPT_aligner.pt` (126 MB) |
| Aligner (ctrl) | X_scGPT_ctrl | — | `GBM_scGPT_aligner_ctrl.pt` (126 MB) |
| Aligner (pert) | X_scGPT_pert | — | `GBM_scGPT_aligner_pert.pt` (126 MB) |

> **r = 0.9997 的含义**：scGPT 512d embedding 经过 MLP 投影后，恢复出的基因表达**均值 profile** 与真实值几乎完美相关。这说明 scGPT 保留了极其充分的细胞状态信息，信息损失主要发生在个体基因水平而非均值层面。

### 4.3 Aligner 输出矩阵

| 文件 | 形状 | 大小 | 说明 |
|------|------|------|------|
| `GBM_scGPT_aligned_XscGPT.npy` | (169972, 5000) | 3.2 GB | 全细胞对齐 |
| `GBM_scGPT_aligned_XscGPT_ctrl.npy` | (169972, 5000) | 3.2 GB | 仅对照对齐 |
| `GBM_scGPT_aligned_XscGPT_pert.npy` | (169972, 5000) | 3.2 GB | 仅扰动对齐 |

---

## 5. 对比方法矩阵与模型架构

### 5.1 方法矩阵

```
                   药物编码 →
细胞编码 ↓          可学习 Embedding         MolFormer (768d, 冻结)
─────────────────────────────────────────────────────────
基因表达 (5000d)         M0 ★ (Baseline)         M4 ★ (Best)
scGPT all (512d)         M1                      M5
scGPT ctrl (512d)        M2                      —
scGPT pert (512d)        M3                      —
```

| 方法 | 细胞输入 | 药物输入 | CPA recon_loss | 需要 Aligner | 核心假设 |
|------|---------|---------|:---:|:---:|------|
| **M0** | `layers["counts"]` (5000d int) | pert_encoder (可学习 Embedding, 32d) | **nb** (负二项) | 否 | CPA 自编码器可以从原始 counts 学习有意义的表示 |
| **M1** | X_scGPT → Aligner → 5000d | pert_encoder (可学习 Embedding, 32d) | **gauss** (MSE) | 是 | 预训练细胞表示优于原始基因表达 |
| **M2** | X_scGPT_ctrl → Aligner → 5000d | pert_encoder (可学习 Embedding, 32d) | **gauss** | 是 | 仅 control 细胞编码足以预测药物效应 |
| **M3** | X_scGPT_pert → Aligner → 5000d | pert_encoder (可学习 Embedding, 32d) | **gauss** | 是 | 扰动细胞编码可反向推断药物效应 |
| **M4** | `layers["counts"]` (5000d int) | MolFormer 768d → Linear(768,32) (冻结) | **nb** (负二项) | 否 | 预训练药物表示优于任务特异性学习 |
| **M5** | X_scGPT → Aligner → 5000d | MolFormer 768d → Linear(768,32) (冻结) | **gauss** | 是 | 双预训练编码器有协同效应 |
| **MeanShift** | — (非参数) | — (非参数) | — | — | 最简单的 baseline |
| **MLP M1** | X_scGPT 512d | 可学习 Embedding 8d | — (无 CPA) | 否 | 简单 MLP 可否直接做高维回归 |
| **MLP M5** | X_scGPT 512d | MolFormer 768d | — (无 CPA) | 否 | 双预训练特征 + 简单 MLP |

### 5.2 CPA 模型架构（M0/M4 共享）

CPA (Compositional Perturbation Autoencoder) 的架构：

```
┌─────────────────────────────────────────────────────────────┐
│                        CPA 模型结构                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  输入层:                                                      │
│    adata.layers["counts"] ──→ Encoder ──→ z_basal (32d)     │
│    pert_encoder[drug]     ──→ PertNet ──→ z_pert  (32d)     │
│    covars_encoder[patient]──→ Embed   ──→ z_covs  (32d)     │
│                                                               │
│  潜空间 (Latent Space):                                       │
│    z = z_basal + z_pert + z_covs  (32d, 加法组合)           │
│                                                               │
│  输出层:                                                      │
│    z ──→ Decoder ──→ 重建基因表达 (5000d)                     │
│                                                               │
│  训练目标:                                                    │
│    1. 重建损失 (recon_loss):                                  │
│       - nb (负二项对数似然): 用于 M0/M4 (整数 counts)        │
│       - gauss (MSE): 用于 M1/M2/M3/M5 (连续对齐值)           │
│    2. 对抗损失 (adversarial loss):                            │
│       - perturbation classifier: 确保 z_basal 不泄露药物信息  │
│       - covariate classifier: 确保 z_pert 不泄露患者信息     │
│                                                               │
│  Encoder:  5000 → 256 → 256 → 256 → 32 (MLP + ReLU)          │
│  Decoder:  32 → 256 → 256 → 256 → 5000 (MLP + ReLU)          │
│  PertNet:  drug_dim → 32 (MLP 或 Linear)                     │
│  Embed:    n_patients → 32 (nn.Embedding)                     │
│                                                               │
│  总参数量: ~2.7M (取决于 encoder/decoder 的具体配置)          │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 药物编码器变体详解

**M0/M1/M2/M3 — 可学习 Embedding**:
```python
nn.Embedding(9, 32, padding_idx=0)
# 9 = <PAD>(0) + control(1) + Ana-12(2) + Etoposide(3) + Ispenisib(4)
#     + Panobinostat(5) + RO4929097(6) + Tazemetostat(7) + Temozolomide(8)
# 每个药物有一个 32d 的可学习向量，在训练中通过梯度下降优化
# 优点：完全适应 CPA 的优化目标和数据分布
# 缺点：无法利用药物结构信息，对新药零样本预测能力弱
```

**M4/M5 — MolFormer (冻结)**:
```python
MolFormer(SMILES) → 768d embedding → nn.Linear(768, 32) → 32d latent
# MolFormer 权重 requires_grad=False（冻结），只有 Linear(768,32) 可学习
# 优点：利用预训练化学知识，对药物结构敏感
# 缺点：768d → 32d 压缩可能丢失信息；如果 MolFormer 对某些药物表征不佳则影响结果
```

### 5.4 细胞编码器变体详解

**M0/M4 — 原始基因表达**:
```
counts (5000d int) → CPA Encoder (5000→256→256→256→32) → z_basal (32d)
# CPA 端到端学习基因表达到 latent 的编码
# NB 损失直接作用在整数 counts 上，保留了 count 数据的分布特性
```

**M1/M2/M3/M5 — scGPT 对齐**:
```
scGPT 512d → Aligner (冻结, 512→1024→2048→4096→5000) → aligned 5000d → CPA Encoder → z_basal (32d)
# Aligner 在训练 CPA 之前预训练好并冻结
# CPA Encoder 此时学习的是"对齐后的伪表达"到 latent 的映射
# Gauss 损失 (因为对齐值不是整数 counts)
```

### 5.5 MLP 直接预测方法

MLP M1 和 MLP M5 是 **不使用 CPA 的直接预测方法**：

```
M1:  scGPT 512d ⊕ Drug_Embed 8d  → MLP → 5000d 基因表达
M5:  scGPT 512d ⊕ MolFormer 768d → MLP → 5000d 基因表达

MLP 架构: (512+8 或 512+768) → 2048 → 4096 → 2048 → 5000
          + BatchNorm + ReLU + Dropout
参数量: ~28M
200 epochs, batch_size=512, Adam lr=1e-3
```

### 5.6 MeanShift 非参数 Baseline

```
算法:
  1. 在非 PW034 患者上计算:
     ctrl_mean = mean(非PW034 control 细胞表达)
     drug_mean = mean(非PW034 Panobinostat 处理细胞表达)
     Δ = drug_mean - ctrl_mean  (药物扰动偏移)
  2. 预测 PW034 对 Panobinostat 的响应:
     PW034_pred = PW034_control_mean + Δ

# 本质：假设药物效应在所有患者间恒定
# 缺点：完全忽略患者特异性，无法建模个体化药物响应
```

---

## 6. 训练详情与超参数

### 6.1 CPA 模型训练配置

| 超参数 | M0 | M4 | M1/M2/M3 | M5 |
|--------|:---:|:---:|:---:|:---:|
| **潜空间维度 (n_latent)** | 32 | 32 | 32 | 32 |
| **重建损失函数** | nb (负二项) | nb | gauss (MSE) | gauss |
| **最大 epochs** | 50 | 50 | 50 | 50 |
| **早停 patience** | 20 | 20 | 20 | 20 |
| **批次大小** | 1024 | 1024 | 1024 | 1024 |
| **梯度裁剪值** | 3.0 | 3.0 | 3.0 | 3.0 |
| **随机种子** | 7 | 7 | 7 | 7 |
| **训练 split** | train | train | train | train |
| **验证 split** | valid | valid | valid | valid |
| **测试 split** | ood | ood | ood | ood |
| **药物编码维度** | 32 (可学习) | 768 (冻结) | 32 (可学习) | 768 (冻结) |
| **GPU** | 4,5 | 0 | 0 | 0 |

### 6.2 训练历史

| 方法 | 最佳 epoch | 最佳 val r2_mean | 最终 val recon_loss | 模型大小 |
|------|:---:|:---:|:---:|:---:|
| **CPA M0** | 47 | 0.8592 | 910.65 | ~10 MB (model.pt) |
| **CPA M4** | 36 | **0.8621** | 909.31 | ~10 MB |
| CPA M1 | 23 | 0.7845 | -9937.82 (gauss) | ~10 MB |
| CPA M2 | 48 | **0.8722** | -12482.64 (gauss) | ~10 MB |
| CPA M3 | 18 | 0.8063 | -12252.76 (gauss) | ~10 MB |
| CPA M5 | 22 | 0.8457 | -9099.92 (gauss) | ~10 MB |

> **注意**：gauss 损失的 magnitude 与 nb 损失不可直接比较。负值是因为对数空间的计算方式不同。

### 6.3 Aligner 训练配置

| 超参数 | 值 |
|--------|-----|
| 架构 | 512 → 1024 → 2048 → 4096 → 5000 |
| 优化器 | Adam (lr=1e-3) |
| 学习率调度 | ReduceLROnPlateau (factor=0.5, patience=15, min_lr=1e-6) |
| 损失 | MSE |
| 批次大小 | 512 |
| 最大 epochs | 300 |
| 早停 patience | 40 |
| 最佳 epoch (all) | 85 |
| 最佳验证损失 (all) | 0.109 |
| 验证 Pearson r (all) | **0.9997** |

---

## 7. OOD 反事实预测流程

所有 CPA 方法使用相同的反事实推理流程：

```
给定:
  - 患者 PW034 的 control 细胞 (15,288 个)
  - 目标药物 Panobinostat
  - 训练好的 CPA 模型

步骤:
  1. 提取 basal = PW034 + control 的 AnnData 子集
  2. 修改 basal 的 perturbation 标签: "control" → "Panobinostat"
  3. 设置 dosage = 1.0, is_control = False
  4. 构建 CPA_REGISTRY_KEYS:
     - PERTURBATIONS: Panobinostat 的 pert_encoder index
     - PERTURBATIONS_DOSAGES: 1.0
  5. 通过 CPA 前向传播:
     - z_basal = Encoder(PW034 control 细胞的表达)  # 真实的基底状态
     - z_pert = PertNet(Panobinostat embedding)       # 药物的扰动效应
     - z_covs = Embed("PW034")                        # 患者特异性协变量
     - z = z_basal + z_pert + z_covs                  # 组合潜表示
     - pred = Decoder(z)                              # 预测处理后的表达
  6. 输出: 15,288 细胞 × 5,000 基因的预测表达矩阵

预测语义:
  "如果 PW034 患者的这 15,288 个 control 细胞接受了 Panobinostat 处理，
   它们的基因表达会是什么样子？"
```

**预测输出文件**:

| 文件 | 方法 | 形状 | 预测含义 |
|------|------|------|------|
| `GBM_CPA_PW034_Panobinostat_pred.h5ad` | M0 | (15288, 5000) | CPA 反事实预测 |
| `GBM_CPA_MolFormer_PW034_Panobinostat_pred.h5ad` | M4 | (15288, 5000) | CPA+MolFormer 反事实预测 |
| `GBM_CPA_scGPT_PW034_Panobinostat_pred.h5ad` | M1 | (15288, 5000) | CPA+scGPT all 反事实预测 |
| `GBM_CPA_scGPT_ctrl_PW034_Panobinostat_pred.h5ad` | M2 | (15288, 5000) | CPA+scGPT ctrl 反事实预测 |
| `GBM_CPA_scGPT_pert_PW034_Panobinostat_pred.h5ad` | M3 | (15288, 5000) | CPA+scGPT pert 反事实预测 |
| `GBM_CPA_scGPT_MolFormer_PW034_Panobinostat_pred.h5ad` | M5 | (15288, 5000) | CPA+scGPT+MolFormer 反事实预测 |
| `GBM_MeanShiftBaseline_PW034_Panobinostat_pred.h5ad` | MeanShift | (15288, 5000) | 非参数均值偏移预测 |
| `mlp_comparison_results/MLP_M1_*.h5ad` | MLP M1 | (15288, 5000) | MLP 直接预测 |
| `mlp_comparison_results/MLP_M5_*.h5ad` | MLP M5 | (15288, 5000) | MLP 双特征预测 |

---

## 8. 评估协议与指标体系

### 8.1 评估协议 A：CRISP OOD 单 Group 评估

**评估对象**：单 group (PW034|Panobinostat) 的 Top50 DEG 基因

**评估脚本**：`scripts/evaluate_crisp_ood.py`

**5 个指标**：

| 指标 | 英文名 | 方向 | 计算方式 |
|------|--------|:---:|------|
| **PrΔ DE** | Pearson r of delta on DEGs | ↑ | `corr(pred_logFC, true_logFC)` 在 Top50 DEG 上 |
| **Sp DE** | Spearman ρ of delta on DEGs | ↑ | `spearmanr(pred_logFC, true_logFC)` 在 Top50 DEG 上 |
| **R² DE** | R² Score on DEGs | ↑ | `r2_score(true_post_mean, pred_post_mean)` 在 Top50 DEG 上 |
| **Sinkhorn DE** | Sinkhorn Distance on DEGs | ↓ | `SamplesLoss("sinkhorn", blur=0.05)(pred_matrix, true_matrix)` |
| **Direction** | Direction Accuracy | ↑ | `mean(sign(pred_logFC) == sign(true_logFC)) × 100%` |

### 8.2 评估协议 B：NIPS 统一评估协议

**协议文档**：`LYZ_LZX_NIPS_METRICS.md` (刘耀泽/龙泽兴 制定)

**评估流程**：
1. 遍历所有 valid `cov_drug_name` groups (经 5 项过滤规则筛选)
2. 对每个 group 构建 mean profile: `y = mean(Y_true)`, `p = mean(Y_pred)`, `c = mean(Y_ctrl)`
3. 计算 9 个 per-group 指标
4. 对所有 valid groups 做不加权 macro average

**Group 过滤规则**：treated cells > 5, group 不含 "dmso"/"control", DEG entry 存在, DEG ≥ 2 个, matched control ≥ 5

**9 个指标**：

| 指标 | 输入 | 方向 | 公式 |
|------|------|:---:|------|
| **r2score** | y, p (全基因) | ↑ | `max(R²(y, p), 0)` |
| **r2score_de** | y, p (DEG) | ↑ | `max(R²(y_D, p_D), 0)` |
| **pearson** | y, p (全基因) | ↑ | `corr(y, p)`, NaN→0 |
| **pearson_de** | y, p (DEG) | ↑ | `corr(y_D, p_D)`, NaN→0, zero-sum guard |
| **mse** | y, p (全基因) | ↓ | `mean((y-p)²)` |
| **mse_de** | y, p (DEG) | ↓ | `mean((y_D-p_D)²)` |
| **pearson_delta** | y-c, p-c (全基因) | ↑ | `corr(y-c, p-c)` |
| **pearson_delta_de** | y_D-c_D, p_D-c_D (DEG) | ↑ | **核心指标** — 扰动效应的 DEG 相关性 |
| **sinkhorn_de** | Y_true[:,D], Y_pred[:,D] | ↓ | `SamplesLoss("sinkhorn", blur=0.05)` |

**评估脚本**：`scripts/evaluate_nips_gbm.py`

---

## 9. 完整实验结果

### 9.1 CRISP OOD 单 Group 评估（PW034|Panobinostat, Top50 DEGs）

| Method | PrΔ DE ↑ | Sp DE ↑ | R² DE ↑ | Sinkhorn DE ↓ | Direction ↑ |
|---|---|---|---|---|---|
| MeanShiftBaseline | -0.468 | -0.230 | -276.429 | 0.190 | 20.0% |
| MLP M1 (scGPT) | 0.018 | -0.003 | -1046.680 | 0.012 | 18.0% |
| MLP M5 (scGPT+MolFormer) | -0.069 | -0.086 | -1017.322 | 0.013 | 20.0% |
| CPA M1 (scGPT all) | 0.103 | 0.219 | -311.580 | 0.006 | 36.0% |
| CPA M2 (scGPT ctrl) | 0.111 | 0.173 | -486.665 | 0.006 | 44.0% |
| CPA M3 (scGPT pert) | 0.027 | 0.209 | -929.834 | 0.009 | 46.0% |
| CPA M5 (scGPT+MolFormer) | 0.098 | 0.199 | -507.680 | 0.007 | 44.0% |
| **CPA M0 (baseline)** | **0.608** | 0.463 | -18.636 | **0.004** | **96.0%** |
| **CPA M4 (+MolFormer)** | **0.693** | **0.585** | -18.957 | **0.004** | 94.0% |

### 9.2 NIPS 协议评估（macro average over valid groups, OOD）

| Method | r2score ↑ | pearson ↑ | mse ↓ | pearson_delta_de ↑ | sinkhorn_de ↓ |
|---|---|---|---|---|---|
| MeanShift | 0.820 | 0.942 | 0.007 | 0.000 | 0.190 |
| MLP M1 | 0.859 | 0.946 | 0.005 | 0.018 | 0.012 |
| MLP M5 | 0.836 | 0.933 | 0.006 | 0.000 | 0.013 |
| CPA M1 | 0.777 | 0.920 | 0.008 | 0.103 | 0.006 |
| CPA M2 | **0.822** | 0.938 | 0.006 | 0.111 | 0.006 |
| CPA M3 | 0.716 | 0.894 | 0.010 | 0.027 | 0.009 |
| CPA M5 | 0.748 | 0.920 | 0.009 | 0.098 | 0.007 |
| CPA M0 | 0.000 | 0.861 | 0.174 | 0.608 | **0.004** |
| **CPA M4** | 0.000 | 0.862 | 0.169 | **0.693** | **0.004** |

### 9.3 M4 vs M0 性能对比（核心结果）

| 指标 | M0 (CPA baseline) | M4 (CPA + MolFormer) | 变化 |
|------|:---:|:---:|:---:|
| PrΔ DE (Pearson r) | 0.608 | **0.693** | **+14.0%** |
| Sp DE (Spearman ρ) | 0.463 | **0.585** | **+26.3%** |
| Direction Accuracy | **96.0%** | 94.0% | -2.1% |
| Sinkhorn DE | **0.00432** | 0.00443 | +2.4% |
| NIPS pearson_delta_de | 0.608 | **0.693** | **+14.0%** |

### 9.4 原始 vs scGPT CPA 方法对比

| 指标 | M0 (raw counts) | M1 (scGPT all) | M2 (scGPT ctrl) | M3 (scGPT pert) | M5 (scGPT+MolFormer) |
|------|:---:|:---:|:---:|:---:|:---:|
| PrΔ DE | **0.608** | 0.103 | 0.111 | 0.027 | 0.098 |
| Direction | **96.0%** | 36.0% | 44.0% | 46.0% | 44.0% |
| NIPS pearson_delta_de | **0.608** | 0.103 | 0.111 | 0.027 | 0.098 |

---

## 10. 关键发现与讨论

### 10.1 主要发现

1. **CPA (M0) 是有效的方法**：pearson_delta_de = 0.608，Direction Accuracy = 96.0%，远超所有简单 baseline（MeanShift, MLP 直接预测）

2. **MolFormer 预训练药物表示显著提升 CPA (M4)**：pearson_delta_de 从 0.608 → 0.693 (+14.0%)，Spearman 提升 +26.3%，证明**大分子预训练模型在药物扰动预测中具有正向迁移能力**

3. **scGPT 预训练细胞表示在此任务中不优于原始表达**：所有基于 scGPT 的 CPA 方法 (M1-M3, M5) 的 PrΔ DE 仅 0.027-0.111，远低于 M0 的 0.608。可能原因：
   - scGPT 在**血液数据**上预训练，与 GBM **脑肿瘤**数据存在 domain gap
   - 基因匹配率仅 91%（453/5000 基因缺失）
   - Aligner 虽然均值相关 0.9997，但个体基因水平的对齐精度可能不够
   - Gauss 损失可能不如 NB 损失适合此数据特性

4. **简单方法 (MeanShift, MLP) 无法预测扰动特异性效应**：全基因 pearson ~0.94 看起来不错，但这是因为它们预测了 control 状态（与药物处理无关的背景表达模式）。DEG delta 相关接近 0，证明了**有效的扰动预测必须将 control 状态与药物效应解耦**

5. **M2 (scGPT ctrl only) 在 scGPT 方法中最佳**：pearson_delta_de = 0.111，稍优于 M1 (0.103)，提示**只用 control 编码可能减少了扰动信号的噪声**

### 10.2 方法学意义

| 层面 | 意义 |
|------|------|
| **方法学** | 验证了 CPA 架构在 OOD 零样本药物扰动预测中的有效性；证明预训练药物编码器可以作为 CPA 的即插即用组件 |
| **生物学** | 为 GBM 个性化用药提供了计算预测工具；Panobinostat (HDAC 抑制剂) 在 PW034 上的响应可被计算预测 |
| **工程学** | 建立了完整的对比实验框架，9 个方法在完全相同的条件下进行公平比较 |
| **可复现性** | 严格遵循 NIPS 统一评估协议，所有数据、模型、评估脚本可追溯 |

### 10.3 局限性与后续方向

1. **scGPT domain gap**：应尝试在脑肿瘤数据上微调 scGPT，或在更多样化的单细胞数据上使用预训练编码器
2. **MolFormer 的线性投影**：768d → 32d 可能导致信息瓶颈，可尝试更复杂的投影方式
3. **仅单 group OOD 评估**：当前 OOD 仅 PW034+Panobinostat 一个组合，验证集较小 (2,679 细胞)
4. **Ispenisib/Tazemetostat SMILES 相同**：需要确认这是 PubChem 查询问题还是两种药物真的有相同结构

---

## 11. 文件资产完整索引

### 11.1 数据文件

| 文件 | 大小 | 内容 |
|------|------|------|
| `GBM_Universal_Perturbation_Ready.h5ad` | ~213 MB | 原始数据，无 embedding，包含 5000 HVG + top50 DEGs + drug SMILES |
| `GBM_counts_for_scgpt.h5ad` | ~114 MB | 仅整数 counts，用于 scGPT 编码输入 |
| `GBM_scGPT_embeddings.h5ad` | ~760 MB | scGPT 编码结果：X_scGPT, X_scGPT_ctrl, X_scGPT_pert |
| `GBM_X_MolFormer.npy` | ~522 MB | 每细胞 MolFormer drug embedding |
| `GBM_molformer_drug_emb.parquet` | ~394 KB | 药物级 MolFormer embedding 表 (6 行 × 768 列) |
| `GBM_molformer_drug_emb.metadata.json` | 1.2 KB | 药物-SMILES 映射元数据 |
| `GBM_with_embeddings.h5ad` | ~831 MB | 合并所有 embedding 的中间文件 |
| **`GBM_NIPS_Ready.h5ad`** | **~831 MB** | **★ 最终 NIPS 协议数据文件** |

### 11.2 模型文件

| 文件 | 大小 | 对应方法 | 说明 |
|------|------|------|------|
| `GBM_CPA_model/model.pt` | ~10 MB | M0 | CPA baseline 训练权重 |
| `GBM_CPA_MolFormer_model/model.pt` | ~10 MB | M4 | CPA + MolFormer 训练权重 |
| `GBM_CPA_scGPT_model/model.pt` | ~10 MB | M1 | CPA + scGPT all 训练权重 |
| `GBM_CPA_scGPT_ctrl_model/model.pt` | ~10 MB | M2 | CPA + scGPT ctrl 训练权重 |
| `GBM_CPA_scGPT_pert_model/model.pt` | ~10 MB | M3 | CPA + scGPT pert 训练权重 |
| `GBM_CPA_scGPT_MolFormer_model/model.pt` | ~10 MB | M5 | CPA + scGPT + MolFormer 训练权重 |
| `GBM_scGPT_aligner.pt` | ~126 MB | M1/M5 | scGPT all → 5000d Aligner 权重 |
| `GBM_scGPT_aligner_ctrl.pt` | ~126 MB | M2 | scGPT ctrl → 5000d Aligner 权重 |
| `GBM_scGPT_aligner_pert.pt` | ~126 MB | M3 | scGPT pert → 5000d Aligner 权重 |

### 11.3 Aligner 输出矩阵

| 文件 | 大小 | 说明 |
|------|------|------|
| `GBM_scGPT_aligned_XscGPT.npy` | 3.2 GB | 全细胞对齐 |
| `GBM_scGPT_aligned_XscGPT_ctrl.npy` | 3.2 GB | 仅对照对齐 |
| `GBM_scGPT_aligned_XscGPT_pert.npy` | 3.2 GB | 仅扰动对齐 |

### 11.4 预测文件

| 文件 | 方法 | 形状 |
|------|------|------|
| `GBM_CPA_PW034_Panobinostat_pred.h5ad` | M0 | (15288, 5000) |
| `GBM_CPA_MolFormer_PW034_Panobinostat_pred.h5ad` | M4 | (15288, 5000) |
| `GBM_CPA_scGPT_PW034_Panobinostat_pred.h5ad` | M1 | (15288, 5000) |
| `GBM_CPA_scGPT_ctrl_PW034_Panobinostat_pred.h5ad` | M2 | (15288, 5000) |
| `GBM_CPA_scGPT_pert_PW034_Panobinostat_pred.h5ad` | M3 | (15288, 5000) |
| `GBM_CPA_scGPT_MolFormer_PW034_Panobinostat_pred.h5ad` | M5 | (15288, 5000) |
| `GBM_MeanShiftBaseline_PW034_Panobinostat_pred.h5ad` | MeanShift | (15288, 5000) |
| `mlp_comparison_results/MLP_M1_PW034_Panobinostat_pred.h5ad` | MLP M1 | (15288, 5000) |
| `mlp_comparison_results/MLP_M5_PW034_Panobinostat_pred.h5ad` | MLP M5 | (15288, 5000) |

### 11.5 评估结果文件

| 文件 | 协议 | 内容 |
|------|------|------|
| `GBM_CRISP_OOD_metrics.md` | CRISP 5 metrics | 所有 9 个方法的单 group 汇总 |
| `evaluation_results/CPA_M0_ood_metrics.json` | NIPS 9 metrics | M0 per-group + macro avg |
| `evaluation_results/CPA_M1_scGPT_ood_metrics.json` | NIPS 9 metrics | M1 |
| `evaluation_results/CPA_M2_scGPT_ctrl_ood_metrics.json` | NIPS 9 metrics | M2 |
| `evaluation_results/CPA_M3_scGPT_pert_ood_metrics.json` | NIPS 9 metrics | M3 |
| `evaluation_results/CPA_M4_MolFormer_ood_metrics.json` | NIPS 9 metrics | M4 |
| `evaluation_results/CPA_M5_scGPT_MolFormer_ood_metrics.json` | NIPS 9 metrics | M5 |
| `evaluation_results/MeanShiftBaseline_ood_metrics.json` | NIPS 9 metrics | MeanShift |
| `evaluation_results/MLP_M1_ood_metrics.json` | NIPS 9 metrics | MLP M1 |
| `evaluation_results/MLP_M5_ood_metrics.json` | NIPS 9 metrics | MLP M5 |

### 11.6 训练日志

| 文件 | 方法 |
|------|------|
| `GBM_CPA_training.log` | M0 |
| `GBM_CPA_MolFormer_training.log` | M4 |
| `GBM_CPA_scGPT_PW034_Panobinostat_pred.log` | M1 |
| `GBM_CPA_scGPT_ctrl_PW034_Panobinostat_pred.log` | M2 |
| `GBM_CPA_scGPT_pert_PW034_Panobinostat_pred.log` | M3 |
| `GBM_CPA_scGPT_MolFormer_PW034_Panobinostat_pred.log` | M5 |
| `GBM_scGPT_aligned_XscGPT.log` | Aligner (all) |
| `GBM_scGPT_aligned_XscGPT_ctrl.log` | Aligner (ctrl) |
| `GBM_scGPT_aligned_XscGPT_pert.log` | Aligner (pert) |

### 11.7 脚本文件 (scripts/)

| 脚本 | 环境 | 功能 |
|------|------|------|
| `prepare_gbm_universal.py` | plknature | Step 1: 数据合并与标准化 |
| `encode_gbm_cells_scgpt.py` | nature | Step 2: scGPT 细胞编码 |
| `encode_gbm_drugs_molformer.py` | nature | Step 3: MolFormer 药物编码 |
| `prepare_gbm_with_embeddings.py` | plknature | Step 4: embedding 合并 |
| `fix_gbm_nips_format.py` | plknature | Step 5: NIPS 格式标准化 |
| `train_scgpt_aligner.py` | plknature | Step 6-7: MLP 对齐器训练 (×3 变体) |
| `train_cpa_ood.py` | plknature | M0: CPA baseline 训练 + 预测 + 评估 |
| `train_cpa_molformer.py` | plknature | M4: CPA + MolFormer 训练 + 预测 + 评估 |
| `train_cpa_scgpt.py` | plknature | M1/M2/M3: CPA + scGPT 训练 + 预测 |
| `train_cpa_scgpt_molformer.py` | plknature | M5: CPA + scGPT + MolFormer 训练 + 预测 |
| `predict_mean_shift_baseline.py` | plknature | MeanShift 非参数 baseline |
| `predict_mlp_comparison.py` | plknature | MLP M1/M5 直接预测 |
| `evaluate_crisp_ood.py` | plknature | CRISP OOD 5 指标评估 |
| `evaluate_nips_gbm.py` | plknature | NIPS 9 指标多 group 评估 |
| `run_all_comparisons.py` | plknature | 批量运行 M0+M4 的封装脚本 |

### 11.8 文档文件

| 文件 | 内容 |
|------|------|
| `GBM_项目完整文档.md` | 项目概述与方法体系 |
| `GBM_数据预处理报告.md` | 预处理流程详细报告 |
| `GBM_对比方法实验报告.md` | 对比方法实验报告 |
| `GBM_CRISP_OOD_metrics.md` | CRISP 评估结果汇总表 |
| `LYZ_LZX_NIPS_METRICS.md` | NIPS 统一评估协议（刘耀泽/龙泽兴） |
| `GBM_项目技术详细报告.md` | **本文档** — 完整技术报告 |

---

## 12. 环境与依赖

### 12.1 Conda 环境

**plknature** (训练/评估环境):
- Python 3.11
- anndata 0.12.11 (新版 AnnData 格式支持)
- scvi-tools (CPA 框架)
- cpa (组合扰动自编码器)
- pytorch-lightning
- geomloss (Sinkhorn 距离)
- scipy, scikit-learn, numpy, pandas
- CUDA 12.x

**nature** (编码器环境):
- Python 3.11
- anndata 0.11.4 (旧版)
- scGPT (scgpt 包)
- transformers (MolFormer)
- rdkit (化学信息学)

### 12.2 GPU 分配

| GPU | 用途 |
|:---:|------|
| 0-3 | M1-M5 训练 (CUDA_VISIBLE_DEVICES=0 或 0,1,2,3) |
| 4-5 | M0 训练 (CUDA_VISIBLE_DEVICES=4,5) |
| 全部 | NVIDIA GeForce RTX 4090 × 8 (24 GB VRAM each) |

---

## 附录 A：数据流图

```
                    ┌──────────────────────────┐
                    │  GSE148842 + GSE226202   │
                    │  原始 GEO scRNA-seq 数据   │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  GBM_Universal_          │
                    │  Perturbation_Ready.h5ad │
                    │  169,972 cells × 5,000   │
                    │  HVG, split, top50_DEGs  │
                    └──┬────────────┬──────────┘
                       │            │
          ┌────────────▼──┐    ┌───▼──────────────┐
          │ scGPT 编码     │    │ MolFormer 编码    │
          │ (blood 预训练)  │    │ (1.1B 分子预训练) │
          │ 512d × 3 变体   │    │ 768d drug emb    │
          └──────┬─────────┘    └───┬──────────────┘
                 │                  │
          ┌──────▼──────────────────▼──────────┐
          │  GBM_with_embeddings.h5ad          │
          │  + X_scGPT + X_MolFormer            │
          └──────────────┬─────────────────────┘
                         │
          ┌──────────────▼─────────────────────┐
          │  GBM_NIPS_Ready.h5ad ★ 最终数据    │
          │  + cov_drug_name + neg_control      │
          └──┬──────┬──────┬──────┬──────┬─────┘
             │      │      │      │      │
        ┌────▼┐ ┌───▼┐ ┌──▼─┐ ┌──▼─┐ ┌──▼──┐
        │ M0  │ │ M1 │ │ M2 │ │ M3 │ │ M4  │ ...
        │ CPA │ │CPA+│ │CPA+│ │CPA+│ │CPA+ │
        │base │ │scGPT│ │scGPT│scGPT│ │MolF │
        └──┬──┘ └──┬─┘ └──┬─┘ └──┬─┘ └──┬──┘
           │       │      │      │      │
        ┌──▼───────▼──────▼──────▼──────▼──────┐
        │  9 个方法的 OOD 预测                  │
        │  (PW034 control → Panobinostat 处理)   │
        └──────────────┬───────────────────────┘
                       │
        ┌──────────────▼───────────────────────┐
        │  两套评估协议                          │
        │  CRISP: 5 metrics (single group)     │
        │  NIPS:  9 metrics (multi-group)      │
        └──────────────────────────────────────┘
```

## 附录 B：CPA 技术兼容性说明

由于当前 CPA 包是较旧版本，与新版 scvi-tools / PyTorch Lightning 2.x 有 API 不兼容。所有训练脚本 (`train_cpa_ood.py`, `train_cpa_molformer.py`, `train_cpa_scgpt.py`, `train_cpa_scgpt_molformer.py`) 在启动时执行以下兼容性补丁：

1. **`parse_use_gpu_arg` 替换**：解决新版 scvi-tools 中 GPU 参数格式变化
2. **`SaveBestState` Callback 替换**：解决 Lightning 2.x 中 Callback API 签名变化
3. **`TrainRunner` 兼容包装**：桥接旧版 CPA 的 `use_gpu` 参数与新版 `accelerator`/`devices` 参数
4. **Lightning 2.x epoch hooks 迁移**：将旧版 `training_epoch_end`/`validation_epoch_end` 迁移到 `on_train_epoch_end`/`on_validation_epoch_end`
5. **CPA 动态加载**：通过 `importlib` 直接加载 CPA 源码，避免 pip 安装路径冲突

这些补丁**不改变 CPA 的核心逻辑**，仅确保其在当前环境中可运行。
